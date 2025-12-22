
import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
from src.strategies.ml_predictions import MLPricePredictor
from src.strategies.portfolio_optimization import AdvancedPortfolioOptimizer, MarketOpportunity

class TestDirectionalStrategy:
    """Verify directional trading strategy logic"""
    
    @pytest_asyncio.fixture
    async def ml_predictor(self, mock_kalshi):
        return MLPricePredictor(mock_kalshi)

    @pytest_asyncio.fixture
    async def portfolio_optimizer(self, db_manager, mock_kalshi):
        return AdvancedPortfolioOptimizer(db_manager, mock_kalshi, AsyncMock())

    def test_ml_trend_calculation(self, ml_predictor):
        """Should calculate trend slope correctly"""
        # Linear uptrend
        times = np.array([0, 3600, 7200]) # 0, 1h, 2h
        prices = np.array([0.50, 0.51, 0.52]) 
        
        slope, conf = ml_predictor._calculate_trend(times, prices)
        
        # Slope should be positive (0.01 per hour)
        assert slope > 0
        assert abs(slope - 0.01) < 0.001

    def test_ml_momentum_calculation(self, ml_predictor):
        """Should calculate momentum correctly"""
        # Strong uptrend
        prices = np.linspace(0.40, 0.60, 20)
        momentum = ml_predictor._calculate_momentum(prices)
        assert momentum > 0.5 # Strong positive momentum

    @pytest.mark.asyncio
    async def test_kelly_fraction_calculation(self, portfolio_optimizer):
        """Should calculate Kelly fraction correctly"""
        # Setup opportunity: High edge, high confidence
        opp = MarketOpportunity(
            market_id="TEST", market_title="Test", category="Test",
            predicted_probability=0.8,
            market_probability=0.5,
            confidence=0.9,
            edge=0.3,
            volatility=0.1,
            expected_return=0.6,
            max_loss=1.0,
            time_to_expiry=10,
            correlation_score=0,
            kelly_fraction=0, fractional_kelly=0, risk_adjusted_fraction=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown_contribution=0
        )
        
        fractions = await portfolio_optimizer._calculate_kelly_fractions([opp])
        kelly = fractions["TEST"]
        
        assert kelly > 0
        # Standard Kelly: p=0.8, b=1.0 (evens). f = (1*0.8 - 0.2)/1 = 0.6
        # With cleanup and adjustments, it should still be significant but < 0.6 due to safety caps
        assert kelly < 0.6
        assert kelly > 0.05 # At least some meaningful size

    @pytest.mark.asyncio
    async def test_portfolio_diversification(self, portfolio_optimizer):
        """Should penalize high correlation in allocation"""
        # Two identical opportunities
        opp1 = MarketOpportunity(
            market_id="M1", market_title="M1", category="Tech",
            predicted_probability=0.7, market_probability=0.5, confidence=0.8, edge=0.2,
            volatility=0.1, expected_return=0.4, max_loss=1.0, time_to_expiry=10,
            correlation_score=0, kelly_fraction=0, fractional_kelly=0, risk_adjusted_fraction=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown_contribution=0
        )
        opp2 = MarketOpportunity(
            market_id="M2", market_title="M2", category="Tech",
            predicted_probability=0.7, market_probability=0.5, confidence=0.8, edge=0.2,
            volatility=0.1, expected_return=0.4, max_loss=1.0, time_to_expiry=10,
            correlation_score=0, kelly_fraction=0, fractional_kelly=0, risk_adjusted_fraction=0,
            sharpe_ratio=0, sortino_ratio=0, max_drawdown_contribution=0
        )
        
        # Fake high correlation matrix
        with patch.object(portfolio_optimizer, '_estimate_correlation_matrix', return_value=np.array([[1.0, 0.9], [0.9, 1.0]])):
            # Also mock _calculate_kelly_fractions to return base values
            with patch.object(portfolio_optimizer, '_calculate_kelly_fractions', return_value={"M1": 0.1, "M2": 0.1}):
                
                adjusted = portfolio_optimizer._apply_correlation_adjustments({"M1": 0.1, "M2": 0.1}, np.array([[1.0, 0.9], [0.9, 1.0]]))
                
                # Should reduce allocation due to high correlation
                assert adjusted["M1"] < 0.1
                assert adjusted["M2"] < 0.1
