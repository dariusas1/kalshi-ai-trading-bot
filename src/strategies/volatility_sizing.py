"""
Volatility-Adjusted Position Sizing

This module implements dynamic position sizing based on market volatility:
- Scale positions UP in low-volatility markets (more predictable)
- Scale positions DOWN in high-volatility markets (more risky)

Uses historical price data from Kalshi API to calculate rolling volatility
and adjust Kelly Criterion fractions accordingly.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

from src.clients.kalshi_client import KalshiClient
from src.utils.logging_setup import get_trading_logger


@dataclass
class VolatilityMetrics:
    """Volatility analysis results for a market."""
    market_id: str
    current_volatility: float  # Recent volatility (e.g., last 24h)
    historical_volatility: float  # Longer-term volatility (e.g., 7 days)
    volatility_ratio: float  # current / historical
    is_low_volatility: bool
    is_high_volatility: bool
    position_multiplier: float  # Multiplier for position sizing
    regime: str  # "low", "normal", "high"
    sample_size: int  # Number of data points used


class VolatilityAnalyzer:
    """
    Analyzes market volatility and provides position sizing adjustments.
    
    Key principles:
    - Low volatility = more predictable outcomes = larger positions
    - High volatility = uncertain outcomes = smaller positions
    - Baseline volatility is calibrated to prediction market norms (~15%)
    """
    
    def __init__(
        self,
        kalshi_client: KalshiClient,
        baseline_volatility: float = 0.15,
        multiplier_min: float = 0.5,
        multiplier_max: float = 2.0,
        lookback_short_hours: int = 24,
        lookback_long_hours: int = 168  # 7 days
    ):
        self.kalshi_client = kalshi_client
        self.logger = get_trading_logger("volatility_analyzer")
        
        # Configuration
        self.baseline_volatility = baseline_volatility
        self.multiplier_min = multiplier_min
        self.multiplier_max = multiplier_max
        self.lookback_short_hours = lookback_short_hours
        self.lookback_long_hours = lookback_long_hours
        
        # Cache for volatility calculations (avoid repeated API calls)
        self._volatility_cache: Dict[str, Tuple[VolatilityMetrics, float]] = {}
        self._cache_ttl_seconds = 900  # 15 minutes
    
    async def get_volatility_metrics(self, market_id: str) -> Optional[VolatilityMetrics]:
        """
        Calculate volatility metrics for a market.
        
        Returns VolatilityMetrics with position sizing multiplier.
        """
        try:
            # Check cache first
            cached = self._get_cached_metrics(market_id)
            if cached:
                return cached
            
            # Fetch historical price data
            price_history = await self._fetch_price_history(market_id)
            
            if not price_history or len(price_history) < 5:
                self.logger.debug(f"Insufficient price history for {market_id}")
                return self._default_metrics(market_id)
            
            # Calculate volatilities
            current_vol = self._calculate_volatility(
                price_history, 
                hours=self.lookback_short_hours
            )
            historical_vol = self._calculate_volatility(
                price_history,
                hours=self.lookback_long_hours
            )
            
            # Determine regime and multiplier
            metrics = self._compute_metrics(
                market_id, 
                current_vol, 
                historical_vol,
                len(price_history)
            )
            
            # Cache the result
            self._cache_metrics(market_id, metrics)
            
            self.logger.info(
                f"Volatility for {market_id}: "
                f"current={current_vol:.3f}, historical={historical_vol:.3f}, "
                f"regime={metrics.regime}, multiplier={metrics.position_multiplier:.2f}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility for {market_id}: {e}")
            return self._default_metrics(market_id)
    
    async def get_position_multiplier(self, market_id: str) -> float:
        """
        Get the position sizing multiplier for a market.
        
        Returns a value between multiplier_min and multiplier_max:
        - > 1.0: Scale up position (low volatility)
        - < 1.0: Scale down position (high volatility)
        - = 1.0: Normal position size
        """
        metrics = await self.get_volatility_metrics(market_id)
        if metrics:
            return metrics.position_multiplier
        return 1.0  # Default: no adjustment
    
    async def _fetch_price_history(self, market_id: str) -> List[Dict]:
        """Fetch historical price data from Kalshi API."""
        try:
            # Calculate timestamps
            end_ts = int(time.time())
            start_ts = end_ts - (self.lookback_long_hours * 3600)
            
            response = await self.kalshi_client.get_market_history(
                ticker=market_id,
                start_ts=start_ts,
                end_ts=end_ts,
                limit=500  # Get enough data points
            )
            
            if not response:
                return []
            
            # Extract history from response
            history = response.get('history', [])
            if not history:
                history = response.get('prices', [])
            if not history:
                history = response if isinstance(response, list) else []
            
            return history
            
        except Exception as e:
            self.logger.error(f"Error fetching price history for {market_id}: {e}")
            return []
    
    def _calculate_volatility(
        self, 
        price_history: List[Dict], 
        hours: int
    ) -> float:
        """
        Calculate rolling volatility from price history.
        
        Uses standard deviation of price returns.
        """
        try:
            if not price_history:
                return self.baseline_volatility
            
            # Filter to relevant time window
            cutoff_ts = int(time.time()) - (hours * 3600)
            recent_prices = []
            
            for point in price_history:
                # Handle different API response formats
                ts = point.get('ts', point.get('timestamp', 0))
                if isinstance(ts, str):
                    try:
                        ts = int(datetime.fromisoformat(ts.replace('Z', '+00:00')).timestamp())
                    except:
                        continue
                
                if ts >= cutoff_ts:
                    # Get price (could be yes_price, price, etc.)
                    price = point.get('yes_price', point.get('price', point.get('yes_bid', 0)))
                    if isinstance(price, (int, float)) and price > 0:
                        # Normalize to 0-1 range if in cents
                        if price > 1:
                            price = price / 100
                        recent_prices.append(price)
            
            if len(recent_prices) < 3:
                return self.baseline_volatility
            
            # Calculate returns
            prices = np.array(recent_prices)
            returns = np.diff(prices) / prices[:-1]
            
            # Handle edge cases
            returns = returns[np.isfinite(returns)]
            if len(returns) < 2:
                return self.baseline_volatility
            
            # Calculate annualized volatility
            # Prediction markets typically resolve in days/weeks, so adjust accordingly
            volatility = np.std(returns) * np.sqrt(len(returns))  # Scale by observations
            
            return max(0.01, min(1.0, volatility))  # Cap between 1% and 100%
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return self.baseline_volatility
    
    def _compute_metrics(
        self,
        market_id: str,
        current_vol: float,
        historical_vol: float,
        sample_size: int
    ) -> VolatilityMetrics:
        """Compute full volatility metrics and position multiplier."""
        
        # Calculate volatility ratio (current vs historical)
        if historical_vol > 0:
            volatility_ratio = current_vol / historical_vol
        else:
            volatility_ratio = 1.0
        
        # Determine regime
        if current_vol < self.baseline_volatility * 0.7:
            regime = "low"
            is_low = True
            is_high = False
        elif current_vol > self.baseline_volatility * 1.5:
            regime = "high"
            is_low = False
            is_high = True
        else:
            regime = "normal"
            is_low = False
            is_high = False
        
        # Calculate position multiplier
        # Low volatility -> higher multiplier (scale up)
        # High volatility -> lower multiplier (scale down)
        if current_vol > 0:
            raw_multiplier = self.baseline_volatility / current_vol
        else:
            raw_multiplier = 1.0
        
        # Apply additional adjustment based on volatility ratio
        # If current vol is spiking compared to historical, be more cautious
        if volatility_ratio > 1.5:
            raw_multiplier *= 0.8  # Extra caution during vol spikes
        elif volatility_ratio < 0.7:
            raw_multiplier *= 1.1  # Extra confidence during calm periods
        
        # Clamp to configured bounds
        position_multiplier = max(
            self.multiplier_min,
            min(self.multiplier_max, raw_multiplier)
        )
        
        return VolatilityMetrics(
            market_id=market_id,
            current_volatility=current_vol,
            historical_volatility=historical_vol,
            volatility_ratio=volatility_ratio,
            is_low_volatility=is_low,
            is_high_volatility=is_high,
            position_multiplier=position_multiplier,
            regime=regime,
            sample_size=sample_size
        )
    
    def _default_metrics(self, market_id: str) -> VolatilityMetrics:
        """Return default metrics when calculation fails."""
        return VolatilityMetrics(
            market_id=market_id,
            current_volatility=self.baseline_volatility,
            historical_volatility=self.baseline_volatility,
            volatility_ratio=1.0,
            is_low_volatility=False,
            is_high_volatility=False,
            position_multiplier=1.0,
            regime="normal",
            sample_size=0
        )
    
    def _get_cached_metrics(self, market_id: str) -> Optional[VolatilityMetrics]:
        """Get cached metrics if still valid."""
        if market_id in self._volatility_cache:
            metrics, cached_time = self._volatility_cache[market_id]
            if time.time() - cached_time < self._cache_ttl_seconds:
                return metrics
        return None
    
    def _cache_metrics(self, market_id: str, metrics: VolatilityMetrics):
        """Cache volatility metrics."""
        self._volatility_cache[market_id] = (metrics, time.time())
    
    def clear_cache(self):
        """Clear the volatility cache."""
        self._volatility_cache.clear()


async def apply_volatility_adjustment(
    kelly_fraction: float,
    market_id: str,
    kalshi_client: KalshiClient,
    analyzer: Optional[VolatilityAnalyzer] = None
) -> Tuple[float, VolatilityMetrics]:
    """
    Apply volatility adjustment to a Kelly fraction.
    
    Args:
        kelly_fraction: The base Kelly fraction
        market_id: Market to analyze
        kalshi_client: Kalshi API client
        analyzer: Optional pre-configured analyzer
        
    Returns:
        Tuple of (adjusted_kelly, volatility_metrics)
    """
    if analyzer is None:
        analyzer = VolatilityAnalyzer(kalshi_client)
    
    metrics = await analyzer.get_volatility_metrics(market_id)
    
    if metrics:
        adjusted_kelly = kelly_fraction * metrics.position_multiplier
        return adjusted_kelly, metrics
    
    return kelly_fraction, analyzer._default_metrics(market_id)
