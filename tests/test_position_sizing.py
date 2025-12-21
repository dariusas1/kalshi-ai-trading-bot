
import pytest
import pytest_asyncio
from src.utils.position_limits import PositionLimitsManager
from src.utils.database import Position
from datetime import datetime

class TestPositionSizing:
    """Verify position sizing never exceeds limits - prevents account blowup"""
    
    @pytest.mark.asyncio
    async def test_max_position_size_enforced(self, db_manager, mock_kalshi):
        """Single position should NEVER exceed max_position_size_pct of portfolio"""
        manager = PositionLimitsManager(db_manager, mock_kalshi)
        
        # Mock portfolio value = $10,000
        # max_position_size_pct = 3% (from settings default) => $300
        
        # Test valid size
        result_valid = await manager.check_position_limits(200.0, portfolio_value=10000.0)
        assert result_valid.can_trade is True
        
        # Test invalid size
        result_invalid = await manager.check_position_limits(400.0, portfolio_value=10000.0)
        assert result_invalid.can_trade is False
        assert "exceeds limit" in result_invalid.reason
        
    @pytest.mark.asyncio
    async def test_max_positions_limit_enforced(self, mock_db_manager, mock_kalshi):
        """Should NEVER open more than max_positions concurrent positions"""
        manager = PositionLimitsManager(mock_db_manager, mock_kalshi)
        manager.max_positions = 5
        
        # Mock 5 existing positions
        mock_db_manager.get_open_positions.return_value = [1, 2, 3, 4, 5] # Dummy list of length 5
        
        result = await manager.check_position_limits(100.0, portfolio_value=10000.0)
        assert result.can_trade is False
        assert "Position count" in result.reason
        
    @pytest.mark.asyncio
    async def test_position_size_with_zero_balance(self, db_manager, mock_kalshi):
        """Should handle zero balance gracefully"""
        manager = PositionLimitsManager(db_manager, mock_kalshi)
        result = await manager.check_position_limits(100.0, portfolio_value=0.0)
        assert result.can_trade is False
        
    @pytest.mark.asyncio
    async def test_position_size_with_negative_balance(self, db_manager, mock_kalshi):
        """Should refuse to trade with negative balance"""
        manager = PositionLimitsManager(db_manager, mock_kalshi)
        result = await manager.check_position_limits(100.0, portfolio_value=-100.0)
        assert result.can_trade is False

    @pytest.mark.asyncio
    async def test_total_portfolio_usage_check(self, mock_db_manager, mock_kalshi):
        """Test total portfolio usage limits"""
        manager = PositionLimitsManager(mock_db_manager, mock_kalshi)
        
        # Setup: $10,000 portfolio, $500 cash availability
        mock_kalshi.balance = 500.0
        
        # Portfolio value $10000. Cash $500. Usage = ($9500 / 10000) = 95%
        # If we try to add another 3% ($300), usage goes to 98%
        # Limit in code is 100% (based on file content I read: "RELAXED FOR FULL PORTFOLIO USE")
        
        # So it should PASS if < 100%
        result = await manager.check_position_limits(300.0, portfolio_value=10000.0)
        
        # Wait, inside check_position_limits:
        # projected_usage = current_usage + proposed_position_pct
        # current_usage = (used_capital / portfolio_value) * 100
        # used_capital = portfolio_value - available_cash
        
        # If balance is 500, used is 9500. 95%.
        # proposed 300 is 3%.
        # total 98%.
        
        # Code says: if projected_usage > 100: values
        
        assert result.can_trade is True
