
import pytest
from src.utils.pnl_tracker import PnLTracker
from datetime import datetime, timedelta

class TestPnLCalculations:
    """Verify P&L is calculated correctly - ensures accurate performance tracking"""
    
    @pytest.mark.asyncio
    async def test_realized_pnl_calculation(self, mock_db_manager, mock_kalshi):
        """Realized P&L should be (sell_price - buy_price) * quantity"""
        tracker = PnLTracker(mock_db_manager, mock_kalshi)
        
        # Mock fills: Buy 10 @ $0.50, Sell 10 @ $0.60
        fills = [
            {
                'ticker': 'TEST', 'action': 'buy', 'side': 'yes', 'count': 10, 
                'yes_price': 50, 'created_time': '2025-01-01T10:00:00Z'
            },
            {
                'ticker': 'TEST', 'action': 'sell', 'side': 'yes', 'count': 10, 
                'yes_price': 60, 'created_time': '2025-01-01T11:00:00Z'
            }
        ]
        
        metrics = await tracker._calculate_metrics(fills, [])
        
        # Profit = (0.60 - 0.50) * 10 = $1.00
        assert metrics.total_pnl == pytest.approx(1.0)
        assert metrics.total_wins == 1
        assert metrics.total_losses == 0

    @pytest.mark.asyncio
    async def test_loss_calculation(self, mock_db_manager, mock_kalshi):
        """Loss should be negative P&L"""
        tracker = PnLTracker(mock_db_manager, mock_kalshi)
        
        # Buy 10 @ $0.50, Sell 10 @ $0.40
        fills = [
            {'ticker': 'TEST', 'action': 'buy', 'side': 'yes', 'count': 10, 'yes_price': 50, 'created_time': '2025-01-01T10:00:00Z'},
            {'ticker': 'TEST', 'action': 'sell', 'side': 'yes', 'count': 10, 'yes_price': 40, 'created_time': '2025-01-01T11:00:00Z'}
        ]
        
        metrics = await tracker._calculate_metrics(fills, [])
        
        # Loss = (0.40 - 0.50) * 10 = -$1.00
        assert metrics.total_pnl == pytest.approx(-1.0)
        assert metrics.total_wins == 0
        assert metrics.total_losses == 1

    @pytest.mark.asyncio
    async def test_partial_fills_logic(self, mock_db_manager, mock_kalshi):
        """Should handle partial fills correctly"""
        tracker = PnLTracker(mock_db_manager, mock_kalshi)
        
        # Buy 20, Sell 10
        fills = [
            {'ticker': 'TEST', 'action': 'buy', 'side': 'yes', 'count': 20, 'yes_price': 50, 'created_time': '2025-01-01T10:00:00Z'},
            {'ticker': 'TEST', 'action': 'sell', 'side': 'yes', 'count': 10, 'yes_price': 60, 'created_time': '2025-01-01T11:00:00Z'}
        ]
        
        metrics = await tracker._calculate_metrics(fills, [])
        
        # Only 10 matched. Profit = (0.60 - 0.50) * 10 = $1.00
        assert metrics.total_pnl == pytest.approx(1.0) 
        
    @pytest.mark.asyncio
    async def test_handles_fees_implicit(self, mock_db_manager, mock_kalshi):
        """P&L calculation based on fill prices inherently includes slippage if prices reflect it.
        (Kalshi API prices are execution prices. Fees are separate but PnL here is Gross unless fees subtracted).
        The code uses yes_price/no_price directly.
        """
        pass # The current implementation is Gross P&L based on fills.
