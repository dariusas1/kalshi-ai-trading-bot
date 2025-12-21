
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.jobs.execute import execute_position
from src.utils.database import Position
from datetime import datetime

class TestOrderExecution:
    """Verify orders are placed correctly - prevents buying when should sell"""
    
    @pytest.fixture
    def position(self):
        return Position(
            id=1, market_id="TICKER", side="yes", quantity=10, 
            entry_price=0.5, timestamp=datetime.now(), status="pending"
        )
    
    @pytest.mark.asyncio
    async def test_buy_order_creates_correct_side(self, position, mock_db_manager, mock_kalshi):
        """BUY decision should create BUY order parameters correctly"""
        mock_db_manager.store_idempotency_key = AsyncMock(return_value=True) # Success
        mock_db_manager.update_idempotency_result = AsyncMock()
        mock_db_manager.update_position_to_live = AsyncMock()
        
        # Setup mocks
        mock_kalshi.place_order = AsyncMock(return_value={
            'order': {'order_id': '123', 'status': 'filled', 'yes_price': 50}
        })
        mock_kalshi.get_market = AsyncMock(return_value={
            'market': {'yes_ask': 50, 'yes_bid': 49}
        })
        
        # Ensure paper trading is OFF for live execution test
        with patch('src.config.settings.settings.trading') as mock_trading:
            mock_trading.paper_trading_mode = False
            mock_trading.algorithmic_execution = False
            
            await execute_position(position, True, mock_db_manager, mock_kalshi)

        
        # Verify call args
        args = mock_kalshi.place_order.call_args[1]
        assert args['ticker'] == "TICKER"
        assert args['side'] == "yes"
        assert args['action'] == "buy"
        assert args['count'] == 10
        
    @pytest.mark.asyncio
    async def test_high_edge_uses_ioc(self, position, mock_db_manager, mock_kalshi):
        """High edge deals should use IOC if configured"""
        mock_db_manager.store_idempotency_key = AsyncMock(return_value=True)
        mock_kalshi.place_order = AsyncMock(return_value={'order': {'order_id': '123', 'status': 'filled'}})
        mock_kalshi.get_market = AsyncMock(return_value={'market': {}})
        
        with patch('src.config.settings.settings.trading') as mock_trading:
            mock_trading.algorithmic_execution = True
            mock_trading.paper_trading_mode = False
            
            # Edge > 0.10 => IOC
            await execute_position(position, True, mock_db_manager, mock_kalshi, edge=0.15)
            
            args = mock_kalshi.place_order.call_args[1]
            assert args.get('time_in_force') == 'immediate_or_cancel'
            
    @pytest.mark.asyncio
    async def test_duplicate_order_prevention(self, position, mock_db_manager, mock_kalshi):
        """Should not place duplicate orders"""
        # Store key returns False (exists)
        mock_db_manager.store_idempotency_key = AsyncMock(return_value=False) 
        mock_db_manager.check_idempotency_key = AsyncMock(return_value={'status': 'pending'})
        
        # Ensure place_order is a Mock
        mock_kalshi.place_order = AsyncMock()
        
        result = await execute_position(position, True, mock_db_manager, mock_kalshi)
        
        assert result is False
        mock_kalshi.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_blocks_kxmv_markets(self, mock_db_manager, mock_kalshi):
        """Should block combo markets"""
        position = Position(
            id=1, market_id="KXMV-COMBO", side="yes", quantity=10, 
            entry_price=0.5, timestamp=datetime.now(), status="pending"
        )
        
        mock_kalshi.place_order = AsyncMock()
        result = await execute_position(position, True, mock_db_manager, mock_kalshi)
        assert result is False
        mock_kalshi.place_order.assert_not_called()

