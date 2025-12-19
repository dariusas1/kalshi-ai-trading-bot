"""
Machine Learning Price Predictions

This module implements short-term price predictions using historical market data:
- Features: Momentum (1h, 6h, 24h), Trend (linear regression), Mean Reversion, Volatility
- Model: Weighted linear regression / trend extrapolation
- Output: Predicted prices and confidence levels for AI analysis

Used as context for AI trading decisions to improve entry/exit timing.
"""

import numpy as np
from scipy import stats
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from src.clients.kalshi_client import KalshiClient
from src.utils.logging_setup import get_trading_logger


@dataclass
class MLPrediction:
    """Short-term price prediction for a market."""
    market_id: str
    current_price: float
    predicted_price_1h: float
    predicted_price_4h: float
    predicted_price_12h: float
    confidence: float
    trend_direction: str  # "up", "down", "neutral"
    momentum_score: float  # -1.0 to 1.0
    volatility_score: float
    mean_reversion_signal: float  # distance from rolling mean
    last_updated: float


class MLPricePredictor:
    """
    Predicts short-term price movements using historical data.
    
    This implementation uses statistical features and regression to forecast
    future prices, providing quantitative context for the AI reasoning model.
    """
    
    def __init__(
        self,
        kalshi_client: KalshiClient,
        lookback_hours: int = 168,  # 7 days for long-term trend
        min_data_points: int = 10
    ):
        self.kalshi_client = kalshi_client
        self.lookback_hours = lookback_hours
        self.min_data_points = min_data_points
        self.logger = get_trading_logger("ml_predictor")
        
        # Cache for predictions
        self._prediction_cache: Dict[str, MLPrediction] = {}
        self._cache_ttl = 1800  # 30 minutes
    
    async def get_prediction(self, market_id: str) -> Optional[MLPrediction]:
        """
        Generate price prediction for a market.
        """
        try:
            # Check cache
            if market_id in self._prediction_cache:
                pred = self._prediction_cache[market_id]
                if time.time() - pred.last_updated < self._cache_ttl:
                    return pred
            
            # Fetch history
            end_ts = int(time.time())
            start_ts = end_ts - (self.lookback_hours * 3600)
            
            history_data = await self.kalshi_client.get_market_history(
                ticker=market_id,
                start_ts=start_ts,
                end_ts=end_ts,
                limit=1000
            )
            
            if not history_data:
                return None
            
            # Extract prices and timestamps
            history = history_data.get('history', history_data.get('prices', []))
            if not isinstance(history, list) or len(history) < self.min_data_points:
                return None
            
            # Process history into numpy arrays
            times, prices = self._process_history(history)
            if len(prices) < self.min_data_points:
                return None
            
            # Current price
            current_price = prices[-1]
            
            # Calculate features
            momentum = self._calculate_momentum(prices)
            volatility = self._calculate_volatility(prices)
            trend_slope, trend_confidence = self._calculate_trend(times, prices)
            mean_rev = self._calculate_mean_reversion(prices)
            
            # Generate predictions
            pred_1h = self._predict_future(current_price, trend_slope, 1)
            pred_4h = self._predict_future(current_price, trend_slope, 4)
            pred_12h = self._predict_future(current_price, trend_slope, 12)
            
            # Determine direction
            if abs(trend_slope) < 0.0005:  # Very shallow slope
                direction = "neutral"
            else:
                direction = "up" if trend_slope > 0 else "down"
            
            # Create prediction object
            prediction = MLPrediction(
                market_id=market_id,
                current_price=current_price,
                predicted_price_1h=pred_1h,
                predicted_price_4h=pred_4h,
                predicted_price_12h=pred_12h,
                confidence=trend_confidence,
                trend_direction=direction,
                momentum_score=momentum,
                volatility_score=volatility,
                mean_reversion_signal=mean_rev,
                last_updated=time.time()
            )
            
            # Cache it
            self._prediction_cache[market_id] = prediction
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting for {market_id}: {e}")
            return None
    
    def _process_history(self, history: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert history list to numpy arrays of times and normalized prices."""
        times = []
        prices = []
        
        for p in history:
            ts = p.get('ts', p.get('timestamp'))
            val = p.get('yes_price', p.get('price', p.get('yes_bid')))
            
            if ts and val is not None:
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp()
                    except:
                        continue
                
                # Normalize price to 0-1
                if val > 1:
                    val = val / 100
                
                times.append(float(ts))
                prices.append(float(val))
        
        # Sort by time
        sorted_indices = np.argsort(times)
        return np.array(times)[sorted_indices], np.array(prices)[sorted_indices]
    
    def _calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate momentum score based on various windows."""
        if len(prices) < 20:
            return 0.0
        
        # Short-term (last 5 points)
        st_mom = (prices[-1] - prices[-5]) / max(0.01, prices[-5]) if len(prices) >= 5 else 0
        # Med-term (last 20 points)
        mt_mom = (prices[-1] - prices[-20]) / max(0.01, prices[-20]) if len(prices) >= 20 else st_mom
        
        # Weighted momentum
        momentum = (st_mom * 0.7 + mt_mom * 0.3)
        return float(np.clip(momentum, -1.0, 1.0))
    
    def _calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate recent volatility."""
        if len(prices) < 10:
            return 0.05
        returns = np.diff(prices) / np.maximum(0.01, prices[:-1])
        return float(np.std(returns))
    
    def _calculate_trend(self, times: np.ndarray, prices: np.ndarray) -> Tuple[float, float]:
        """Calculate trend slope and confidence using linear regression."""
        # Normalize times to hours from start
        x = (times - times[0]) / 3600
        y = prices
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Confidence based on R-squared and sample size
        r_sq = r_value ** 2
        confidence = r_sq * (1 - p_value)  # Simple confidence metric
        
        return float(slope), float(np.clip(confidence, 0.1, 0.95))
    
    def _calculate_mean_reversion(self, prices: np.ndarray) -> float:
        """Calculate distance from rolling mean."""
        if len(prices) < 50:
            avg = np.mean(prices)
        else:
            avg = np.mean(prices[-50:])
        
        dist = prices[-1] - avg
        return float(dist)
    
    def _predict_future(self, current_price: float, slope: float, hours: int) -> float:
        """Predict future price based on slope, with dampening."""
        # Dampen the slope for longer periods to avoid extreme predictions
        dampening = 1.0 / (1.0 + (hours * 0.05))
        pred = current_price + (slope * hours * dampening)
        return float(np.clip(pred, 0.01, 0.99))
    
    def format_for_prompt(self, prediction: MLPrediction) -> str:
        """Format prediction data for an AI prompt."""
        return (
            f"MODEL PREDICTIONS:\n"
            f"- Trend: {prediction.trend_direction} (Confidence: {prediction.confidence:.1%})\n"
            f"- Predicted prices: 1h: {prediction.predicted_price_1h:.2f}, "
            f"4h: {prediction.predicted_price_4h:.2f}, 12h: {prediction.predicted_price_12h:.2f}\n"
            f"- Momentum: {prediction.momentum_score:.2f} | Volatility: {prediction.volatility_score:.2f}\n"
            f"- Mean Reversion: {'Overbought' if prediction.mean_reversion_signal > 0.05 else 'Oversold' if prediction.mean_reversion_signal < -0.05 else 'Neutral'}"
        )
