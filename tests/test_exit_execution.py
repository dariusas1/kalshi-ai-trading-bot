import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from src.jobs.track import should_exit_position, run_tracking
from src.utils.database import Position, TradeLog

@pytest.mark.asyncio
async def test_should_exit_position_stop_loss():
    # Setup a YES position with a stop loss
    position = Position(
        market_id="test-market",
        side="YES",
        entry_price=0.50,
        quantity=10,
        timestamp=datetime.now(),
        stop_loss_price=0.40  # Trigger if price <= 0.40
    )
    
    # Test case 1: Price above stop loss
    should_exit, reason, price = await should_exit_position(
        position, current_yes_price=0.45, current_no_price=0.55, market_status="active"
    )
    assert not should_exit
    
    # Test case 2: Price hits stop loss
    should_exit, reason, price = await should_exit_position(
        position, current_yes_price=0.40, current_no_price=0.60, market_status="active"
    )
    assert should_exit
    assert "stop_loss_triggered" in reason

@pytest.mark.asyncio
async def test_should_exit_position_take_profit():
    # Setup a NO position with a take profit
    position = Position(
        market_id="test-market",
        side="NO",
        entry_price=0.50,
        quantity=10,
        timestamp=datetime.now(),
        take_profit_price=0.70  # Profit if NO price rises to 0.70
    )
    
    # Test case 1: NO price at 0.70 (YES=0.30)
    should_exit, reason, price = await should_exit_position(
        position, current_yes_price=0.30, current_no_price=0.70, market_status="active"
    )
    assert should_exit
    assert reason == "take_profit"

@pytest.mark.asyncio
async def test_run_tracking_executes_sell():
    import datetime as dt
    # Mock dependencies
    db_manager = MagicMock()
    
    # Setup mock position that should exit
    position = Position(
        id=1,
        market_id="exit-market",
        side="YES",
        entry_price=0.50,
        quantity=10,
        timestamp=datetime.now() - dt.timedelta(hours=1),
        stop_loss_price=0.45,
        live=True,
        status="open",
        strategy="test"
    )
    
    db_manager.get_open_live_positions = AsyncMock(return_value=[position])
    db_manager.update_position_status = AsyncMock()
    db_manager.add_trade_log = AsyncMock()
    
    # Mock market data showing price hit stop loss
    mock_market_data = {
        'market': {
            'ticker': 'exit-market',
            'yes_price': 40,
            'no_price': 60,
            'status': 'active'
        }
    }
    
    # Patch KalshiClient AND the internal functions
    with patch('src.jobs.track.KalshiClient') as mock_client_class, \
         patch('src.jobs.execute.place_profit_taking_orders', new_callable=AsyncMock) as mock_profit, \
         patch('src.jobs.execute.place_stop_loss_orders', new_callable=AsyncMock) as mock_sl, \
         patch('src.jobs.execute.close_position_market', new_callable=AsyncMock) as mock_close:
        
        # Setup mock client instance
        mock_client = mock_client_class.return_value
        mock_client.get_market = AsyncMock(return_value=mock_market_data)
        mock_client.close = AsyncMock()
        
        mock_profit.return_value = {'orders_placed': 0, 'positions_processed': 1}
        mock_sl.return_value = {'orders_placed': 0, 'positions_processed': 1}
        mock_close.return_value = (True, 0.40)
        
        # Run the tracking job
        await run_tracking(db_manager)
        
        # Verify sell was executed
        assert mock_close.called, "close_position_market was not called"
        db_manager.update_position_status.assert_called_with(1, 'closed')
        db_manager.add_trade_log.assert_called_once()
