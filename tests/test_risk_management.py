
import pytest
from unittest.mock import MagicMock, patch
from src.utils.risk_manager import RiskManager
from src.utils.database import Position
from datetime import datetime

class TestRiskManagement:
    """Verify all risk limits are enforced - prevents runaway losses"""
    
    @pytest.fixture
    def risk_manager(self, mock_db_manager, mock_kalshi):
        return RiskManager(mock_db_manager, mock_kalshi)

    @pytest.mark.asyncio
    async def test_calculates_reduction_correctly(self, risk_manager):
        """Should calculate correct reduction factor based on violations"""
        
        # Mock results
        results = MagicMock()
        results.portfolio_volatility = 0.20 # 20%
        results.max_portfolio_drawdown = 0.10 # 10%
        results.correlation_score = 0.50
        
        # Patch config to have limits
        with patch('src.utils.risk_manager.settings') as mock_settings:
            risk_manager.config = MagicMock()
            risk_manager.config.max_portfolio_volatility = 0.10
            risk_manager.config.max_drawdown_limit = 0.05
            
            # 1. Volatility violation (20% vs 10% limit) -> +5% reduction
            # 2. Drawdown violation (10% vs 5% limit) -> +5% reduction
            # Base 5% + 3% (2 violations) + 5% (vol) + 5% (dd) = ~18%
            
            factor = risk_manager._calculate_reduction_factor(
                results, 
                ['Vol', 'Drawdown'], 
                has_vol=True, 
                has_dd=True, 
                has_corr=False
            )
            
            assert factor >= 0.10 # At least 10%
            assert factor <= 0.25 # Cap at 25%

    @pytest.mark.asyncio
    async def test_prioritizes_risky_positions(self, risk_manager):
        """Should prioritize closing risky positions first"""
        
        # Create dummy positions
        p1 = Position(
            id=1, market_id="RISKY", side="yes", quantity=100, 
            entry_price=0.5,
            timestamp=datetime.now(), status="open",
            confidence=0.5, strategy="market_making",
            stop_loss_price=None # Risky!
        ) # Score High
        
        p2 = Position(
            id=2, market_id="SAFE", side="yes", quantity=100,
            entry_price=0.5,
            timestamp=datetime.now(), status="open",
            confidence=0.9, strategy="directional",
            stop_loss_price=0.45
        ) # Score Low
        
        positions = await risk_manager._prioritize_positions_for_reduction(
            [p1, p2], True, True, False
        )
        
        assert len(positions) == 2
        assert positions[0].market_id == "RISKY" # Risky should be first

    @pytest.mark.asyncio
    async def test_closes_position_in_live_mode(self, risk_manager):
        """Should place sell order in live mode"""
        
        position = Position(
            id=1, market_id="TICKER", side="yes", quantity=10, 
            entry_price=0.5,
            timestamp=datetime.now(), status="open"
        )
        
        with patch('src.config.settings.settings.trading') as mock_trading:
            mock_trading.live_trading_enabled = True
            
            # Seed the position in mock client so it can be sold
            risk_manager.kalshi_client.positions.append({
                'ticker': 'TICKER',
                'side': 'yes',
                'position': 10,
                'fees_paid': 0,
                'cost_basis': 5.0
            })
            
            success = await risk_manager._close_position_for_risk(position)
            
            assert success is True
            # Verify order placed
            assert len(risk_manager.kalshi_client.orders) == 1
            assert risk_manager.kalshi_client.orders[0]['action'] == 'sell'
            assert risk_manager.kalshi_client.orders[0]['ticker'] == 'TICKER'
