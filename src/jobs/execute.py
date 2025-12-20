"""
Trade Execution Job

This job takes a position and executes it as a trade.
"""
import asyncio
import uuid
from datetime import datetime
from typing import Optional, Dict

from src.utils.database import DatabaseManager, Position
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.clients.kalshi_client import KalshiClient, KalshiAPIError

async def execute_position(
    position: Position, 
    live_mode: bool, 
    db_manager: DatabaseManager, 
    kalshi_client: KalshiClient,
    edge: float = 0.0  # Edge percentage for smart order type selection
) -> bool:
    """
    Executes a single trade position with smart order type selection.
    
    Order Type Logic:
    - IOC (Immediate or Cancel) when edge > 10% for urgent entries
    - Market orders for standard entries
    - 30-second timeout fallback: if IOC doesn't fill, retry with market
    
    Args:
        position: The position to execute.
        live_mode: Whether to execute a live or simulated trade.
        db_manager: The database manager instance.
        kalshi_client: The Kalshi client instance.
        edge: Edge percentage (0.0-1.0) for smart order type selection.
        
    Returns:
        True if execution was successful, False otherwise.
    """
    logger = get_trading_logger("trade_execution")
    logger.info(f"Executing position for market: {position.market_id}")

    # Validate ticker (KXMV is now standard for all sports markets)
    if not position.market_id or len(position.market_id) < 5:
        logger.error(f"❌ BLOCKED: Invalid market_id: {position.market_id}")
        return False

    if live_mode:
        try:
            client_order_id = str(uuid.uuid4())
            
            # 🎯 SMART ORDER TYPE SELECTION
            # IOC for high edge (>10%) - urgent entry needed
            # Market orders for standard entries
            use_ioc = edge > 0.10
            time_in_force = "immediate_or_cancel" if use_ioc else None  # API requires full name, not 'ioc'
            order_type = "market"
            
            if use_ioc:
                logger.info(f"📈 HIGH EDGE ({edge:.1%}) - Using IOC order for {position.market_id}")
            else:
                logger.info(f"📊 Standard edge ({edge:.1%}) - Using market order for {position.market_id}")
            
            order_args = {
                "ticker": position.market_id,
                "client_order_id": client_order_id,
                "side": position.side.lower(),
                "action": "buy",
                "count": position.quantity,
                "type_": order_type
            }
            
            # 🚨 FIX: Kalshi API requires exactly ONE price field, even for market orders
            # For market orders, use 99 cents (max willing to pay) for the side we're buying
            if position.side.lower() == "yes":
                order_args["yes_price"] = 99  # Max: 99 cents
            else:
                order_args["no_price"] = 99   # Max: 99 cents
            
            if time_in_force:
                order_args["time_in_force"] = time_in_force

            order_response = await kalshi_client.place_order(**order_args)
            order_info = order_response.get('order', {})
            order_id = order_info.get('order_id')
            order_status = order_info.get('status', 'unknown')
            
            logger.info(f"Order placed for {position.market_id}. Order ID: {order_id}, Status: {order_status}")
            
            # 🚨 FIX: Verify order actually filled before marking live
            if order_status in ['filled', 'executed']:  # Kalshi API may return 'executed' for immediate fills
                # Order filled immediately (market order behavior)
                fill_price = order_info.get('yes_price', order_info.get('no_price', position.entry_price * 100)) / 100
                await db_manager.update_position_to_live(position.id, fill_price)
                logger.info(f"✅ Order FILLED for {position.market_id} at ${fill_price:.2f}")
                return True
            elif order_status in ['pending', 'resting', 'canceled']:
                # IOC order may be canceled if not immediately filled
                if use_ioc and order_status == 'canceled':
                    logger.info(f"⚡ IOC order canceled (no immediate fill) - retrying with market order...")
                    # Retry with market order as fallback
                    retry_order_id = str(uuid.uuid4())
                    retry_args = {
                        "ticker": position.market_id,
                        "client_order_id": retry_order_id,
                        "side": position.side.lower(),
                        "action": "buy",
                        "count": position.quantity,
                        "type_": "market"
                    }
                    # Add required price for market order
                    if position.side.lower() == "yes":
                        retry_args["yes_price"] = 99
                    else:
                        retry_args["no_price"] = 99
                    retry_response = await kalshi_client.place_order(**retry_args)
                    retry_info = retry_response.get('order', {})
                    if retry_info.get('status') == 'filled':
                        fill_price = retry_info.get('yes_price', retry_info.get('no_price', position.entry_price * 100)) / 100
                        await db_manager.update_position_to_live(position.id, fill_price)
                        logger.info(f"✅ FALLBACK market order FILLED for {position.market_id}")
                        return True
                
                # Order didn't fill immediately - wait briefly then check again (30s max)
                max_wait_seconds = 30
                check_interval = 2.0
                checks = int(max_wait_seconds / check_interval)
                
                for attempt in range(checks):
                    await asyncio.sleep(check_interval)
                    try:
                        check_response = await kalshi_client.get_order(order_id)
                        check_status = check_response.get('order', {}).get('status', 'unknown')
                        if check_status == 'filled':
                            fill_price = check_response.get('order', {}).get('yes_price', position.entry_price * 100) / 100
                            await db_manager.update_position_to_live(position.id, fill_price)
                            logger.info(f"✅ Order FILLED (attempt {attempt+1}) for {position.market_id}")
                            return True
                        elif check_status == 'canceled':
                            break  # Exit waiting loop if order was canceled
                    except Exception as check_err:
                        logger.warning(f"Error checking order status: {check_err}")
                
                # 🔄 30-SECOND FALLBACK: If limit/IOC didn't fill, retry with market order
                logger.warning(f"⚠️ Order for {position.market_id} didn't fill after {max_wait_seconds}s, trying market order fallback...")
                try:
                    await kalshi_client.cancel_order(order_id)
                    logger.info(f"Cancelled unfilled order {order_id}")
                except Exception as cancel_err:
                    logger.warning(f"Could not cancel order: {cancel_err}")
                
                # Market order fallback
                fallback_order_id = str(uuid.uuid4())
                fallback_args = {
                    "ticker": position.market_id,
                    "client_order_id": fallback_order_id,
                    "side": position.side.lower(),
                    "action": "buy",
                    "count": position.quantity,
                    "type_": "market"
                }
                # Add required price for market order
                if position.side.lower() == "yes":
                    fallback_args["yes_price"] = 99
                else:
                    fallback_args["no_price"] = 99
                fallback_response = await kalshi_client.place_order(**fallback_args)
                fallback_info = fallback_response.get('order', {})
                if fallback_info.get('status') == 'filled':
                    fill_price = fallback_info.get('yes_price', fallback_info.get('no_price', position.entry_price * 100)) / 100
                    await db_manager.update_position_to_live(position.id, fill_price)
                    logger.info(f"✅ FALLBACK market order FILLED for {position.market_id} at ${fill_price:.2f}")
                    return True
                else:
                    logger.error(f"❌ Fallback market order also failed for {position.market_id}")
                    return False
            else:
                # Unexpected status - treat as potentially filled
                logger.warning(f"Unexpected order status: {order_status}, treating as success")
                await db_manager.update_position_to_live(position.id, position.entry_price)
                return True

        except KalshiAPIError as e:
            logger.error(f"Failed to place LIVE order for {position.market_id}: {e}")
            return False
    else:
        # Simulate the trade
        await db_manager.update_position_to_live(position.id, position.entry_price)
        logger.info(f"Successfully placed SIMULATED order for {position.market_id}")
        return True


async def place_sell_limit_order(
    position: Position,
    limit_price: float,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient
) -> bool:
    """
    Place a sell limit order to close an existing position.
    
    Args:
        position: The position to close
        limit_price: The limit price for the sell order (in dollars)
        db_manager: Database manager
        kalshi_client: Kalshi API client
    
    Returns:
        True if order placed successfully, False otherwise
    """
    logger = get_trading_logger("sell_limit_order")
    
    try:
        import uuid
        client_order_id = str(uuid.uuid4())
        
        # Convert price to cents for Kalshi API
        limit_price_cents = int(limit_price * 100)
        
        # For sell orders, we need to use the opposite side logic:
        # - If we have YES position, we sell YES shares (action="sell", side="yes")
        # - If we have NO position, we sell NO shares (action="sell", side="no")
        side = position.side.lower()  # "YES" -> "yes", "NO" -> "no"
        
        order_params = {
            "ticker": position.market_id,
            "client_order_id": client_order_id,
            "side": side,
            "action": "sell",  # We're selling our existing position
            "count": position.quantity,
            "type": "limit"
        }
        
        # Add the appropriate price parameter based on what we're selling
        if side == "yes":
            order_params["yes_price"] = limit_price_cents
        else:
            order_params["no_price"] = limit_price_cents
        
        logger.info(f"🎯 Placing SELL LIMIT order: {position.quantity} {side.upper()} at {limit_price_cents}¢ for {position.market_id}")
        
        # Place the sell limit order
        response = await kalshi_client.place_order(**order_params)
        
        if response and 'order' in response:
            order_id = response['order'].get('order_id', client_order_id)
            
            # Record the sell order in the database (we could add a sell_orders table if needed)
            logger.info(f"✅ SELL LIMIT ORDER placed successfully! Order ID: {order_id}")
            logger.info(f"   Market: {position.market_id}")
            logger.info(f"   Side: {side.upper()} (selling {position.quantity} shares)")
            logger.info(f"   Limit Price: {limit_price_cents}¢")
            logger.info(f"   Expected Proceeds: ${limit_price * position.quantity:.2f}")
            
            return True
        else:
            logger.error(f"❌ Failed to place sell limit order: {response}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error placing sell limit order for {position.market_id}: {e}")
        return False


async def place_profit_taking_orders(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    profit_threshold: float = 0.25  # 25% profit target
) -> Dict[str, int]:
    """
    Place sell limit orders for positions that have reached profit targets.
    
    Args:
        db_manager: Database manager
        kalshi_client: Kalshi API client
        profit_threshold: Minimum profit percentage to trigger sell order
    
    Returns:
        Dictionary with results: {'orders_placed': int, 'positions_processed': int}
    """
    logger = get_trading_logger("profit_taking")
    
    results = {'orders_placed': 0, 'positions_processed': 0}
    
    try:
        # Get all open live positions
        positions = await db_manager.get_open_live_positions()
        
        if not positions:
            logger.info("No open positions to process for profit taking")
            return results
        
        logger.info(f"📊 Checking {len(positions)} positions for profit-taking opportunities")
        
        for position in positions:
            try:
                results['positions_processed'] += 1
                
                # Get current market data
                market_response = await kalshi_client.get_market(position.market_id)
                market_data = market_response.get('market', {})
                
                if not market_data:
                    logger.warning(f"Could not get market data for {position.market_id}")
                    continue
                
                # Get current price based on position side
                if position.side == "YES":
                    current_price = market_data.get('yes_price', 0) / 100  # Convert cents to dollars
                else:
                    current_price = market_data.get('no_price', 0) / 100
                
                # Calculate current profit
                if current_price > 0:
                    profit_pct = (current_price - position.entry_price) / position.entry_price
                    unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    
                    logger.debug(f"Position {position.market_id}: Entry=${position.entry_price:.3f}, Current=${current_price:.3f}, Profit={profit_pct:.1%}, PnL=${unrealized_pnl:.2f}")
                    
                    # Check if we should place a profit-taking sell order
                    if profit_pct >= profit_threshold:
                        # Calculate sell limit price (slightly below current to ensure execution)
                        sell_price = current_price * 0.98  # 2% below current price for quick execution
                        
                        logger.info(f"💰 PROFIT TARGET HIT: {position.market_id} - {profit_pct:.1%} profit (${unrealized_pnl:.2f})")
                        
                        # Place sell limit order
                        success = await place_sell_limit_order(
                            position=position,
                            limit_price=sell_price,
                            db_manager=db_manager,
                            kalshi_client=kalshi_client
                        )
                        
                        if success:
                            results['orders_placed'] += 1
                            logger.info(f"✅ Profit-taking order placed for {position.market_id}")
                        else:
                            logger.error(f"❌ Failed to place profit-taking order for {position.market_id}")
                
            except Exception as e:
                logger.error(f"Error processing position {position.market_id} for profit taking: {e}")
                continue
        
        logger.info(f"🎯 Profit-taking summary: {results['orders_placed']} orders placed from {results['positions_processed']} positions")
        return results
        
    except Exception as e:
        logger.error(f"Error in profit-taking order placement: {e}")
        return results


async def place_stop_loss_orders(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    stop_loss_threshold: float = -0.10  # 10% stop loss
) -> Dict[str, int]:
    """
    Place sell limit orders for positions that need stop-loss protection.
    
    Args:
        db_manager: Database manager
        kalshi_client: Kalshi API client
        stop_loss_threshold: Maximum loss percentage before triggering stop loss
    
    Returns:
        Dictionary with results: {'orders_placed': int, 'positions_processed': int}
    """
    logger = get_trading_logger("stop_loss_orders")
    
    results = {'orders_placed': 0, 'positions_processed': 0}
    
    try:
        # Get all open live positions
        positions = await db_manager.get_open_live_positions()
        
        if not positions:
            logger.info("No open positions to process for stop-loss orders")
            return results
        
        logger.info(f"🛡️ Checking {len(positions)} positions for stop-loss protection")
        
        for position in positions:
            try:
                results['positions_processed'] += 1
                
                # Get current market data
                market_response = await kalshi_client.get_market(position.market_id)
                market_data = market_response.get('market', {})
                
                if not market_data:
                    logger.warning(f"Could not get market data for {position.market_id}")
                    continue
                
                # Get current price based on position side
                if position.side == "YES":
                    current_price = market_data.get('yes_price', 0) / 100
                else:
                    current_price = market_data.get('no_price', 0) / 100
                
                # Calculate current loss
                if current_price > 0:
                    loss_pct = (current_price - position.entry_price) / position.entry_price
                    unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    
                    # Check if we need stop-loss protection
                    if loss_pct <= stop_loss_threshold:  # Negative loss percentage
                        # Calculate stop-loss sell price
                        stop_price = position.entry_price * (1 + stop_loss_threshold * 1.1)  # Slightly more aggressive
                        stop_price = max(0.01, stop_price)  # Ensure price is at least 1¢
                        
                        logger.info(f"🛡️ STOP LOSS TRIGGERED: {position.market_id} - {loss_pct:.1%} loss (${unrealized_pnl:.2f})")
                        
                        # Place stop-loss sell order
                        success = await place_sell_limit_order(
                            position=position,
                            limit_price=stop_price,
                            db_manager=db_manager,
                            kalshi_client=kalshi_client
                        )
                        
                        if success:
                            results['orders_placed'] += 1
                            logger.info(f"✅ Stop-loss order placed for {position.market_id}")
                        else:
                            logger.error(f"❌ Failed to place stop-loss order for {position.market_id}")
                
            except Exception as e:
                logger.error(f"Error processing position {position.market_id} for stop loss: {e}")
                continue
        
        logger.info(f"🛡️ Stop-loss summary: {results['orders_placed']} orders placed from {results['positions_processed']} positions")
        return results
        
    except Exception as e:
        logger.error(f"Error in stop-loss order placement: {e}")
        return results
