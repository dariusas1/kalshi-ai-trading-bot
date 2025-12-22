"""
Trade Execution Job

This job takes a position and executes it as a trade.
"""
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List

from src.utils.database import DatabaseManager, Position
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger
from src.clients.kalshi_client import KalshiClient, KalshiAPIError
from src.utils.pricing_utils import PriceSelector, PriceConverter, create_order_price

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

    if settings.trading.paper_trading_mode and live_mode:
        logger.info("Paper trading mode enabled; forcing simulated execution")
        live_mode = False

    # 🚨 CRITICAL: Validate ticker before execution
    if position.market_id.startswith("KXMV"):
        logger.error(f"❌ BLOCKED: Cannot execute on KXMV combo market: {position.market_id}")
        return False
    
    if not position.market_id or len(position.market_id) < 5:
        logger.error(f"❌ BLOCKED: Invalid market_id: {position.market_id}")
        return False

    if live_mode:
        try:
            client_order_id = str(uuid.uuid4())
            
            # === PRODUCTION SAFETY: IDEMPOTENCY KEY TRACKING ===
            # Store key BEFORE API call to prevent duplicates on retry
            key_stored = await db_manager.store_idempotency_key(
                idempotency_key=client_order_id,
                market_id=position.market_id,
                side=position.side.lower(),
                action="buy",
                status="pending"
            )
            if not key_stored:
                # Key already exists - this is a duplicate request
                logger.warning(f"🚫 Duplicate order blocked for {position.market_id}")
                # Check if original order succeeded
                existing = await db_manager.check_idempotency_key(client_order_id)
                if existing and existing.get("status") == "filled":
                    logger.info(f"✅ Original order already filled for {position.market_id}")
                    return True
                return False
            
            # 🎯 SMART ORDER TYPE SELECTION
            # IOC for high edge (>10%) - urgent entry needed
            # Market orders for standard entries
            use_ioc = edge > 0.10 and settings.trading.algorithmic_execution
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
            
            # 🔧 FIX: Use proper bid/ask pricing logic for order execution
            buffer_cents = getattr(settings.trading, "market_order_price_buffer_cents", 2)
            market_data = await kalshi_client.get_market(position.market_id)
            market_info = market_data.get("market", {}) if market_data else {}

            # Create order price with proper bid/ask logic and safety buffer
            order_price = create_order_price(market_info, position, buffer_cents, max_price=99)

            if position.side.lower() == "yes":
                order_args["yes_price"] = order_price
            else:
                order_args["no_price"] = order_price
            
            if time_in_force:
                order_args["time_in_force"] = time_in_force

            order_response = await kalshi_client.place_order(**order_args)
            order_info = order_response.get('order', {})
            order_id = order_info.get('order_id')
            order_status = order_info.get('status', 'unknown')
            
            # === PRODUCTION SAFETY: UPDATE IDEMPOTENCY RESULT ===
            await db_manager.update_idempotency_result(
                idempotency_key=client_order_id,
                status=order_status,
                order_id=order_id,
                response_json=json.dumps(order_info)
            )
            
            logger.info(f"Order placed for {position.market_id}. Order ID: {order_id}, Status: {order_status}")
            
            # 🔧 FIX: Verify order actually filled before marking live
            if order_status in ['filled', 'executed']:  # Kalshi API may return 'executed' for immediate fills
                # Order filled immediately (market order behavior)
                # Use actual fill price from order, convert from cents to dollars
                fill_price_cents = order_info.get('yes_price', order_info.get('no_price', 0))
                if fill_price_cents > 0:
                    fill_price = PriceConverter.cents_to_dollars(fill_price_cents)
                else:
                    # Fallback to position entry price (stored in database as dollars)
                    fill_price = float(position.entry_price)
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
                        retry_args["yes_price"] = order_args.get("yes_price", 99)
                    else:
                        retry_args["no_price"] = order_args.get("no_price", 99)
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
                except KalshiAPIError as e:
                    # Check if error is 404 (Not Found) - order might be already gone/filled
                    error_str = str(e)
                    if "404" in error_str or "not_found" in error_str.lower():
                        logger.info(f"Order {order_id} not found during cancel (likely already filled/expired). Proceeding.")
                    else:
                        logger.warning(f"Could not cancel order: {e}")
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
                    fallback_args["yes_price"] = order_args.get("yes_price", 99)
                else:
                    fallback_args["no_price"] = order_args.get("no_price", 99)
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
                logger.warning(f"Unexpected order status: {order_status}, treating as not filled")
                return False

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
            "type_": "limit"
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
                    # 🎯 Use position-specific threshold if available, otherwise use default
                    current_threshold = profit_threshold
                    if position.take_profit_price:
                        # Target profit is entry + threshold, so threshold is (target - entry) / entry
                        current_threshold = (position.take_profit_price - position.entry_price) / position.entry_price
                    
                    profit_pct = (current_price - position.entry_price) / position.entry_price
                    unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    
                    logger.debug(f"Position {position.market_id}: Entry=${position.entry_price:.3f}, Current=${current_price:.3f}, Profit={profit_pct:.1%}, Target={current_threshold:.1%}, PnL=${unrealized_pnl:.2f}")
                    
                    # Check if we should place a profit-taking sell order
                    if profit_pct >= current_threshold:
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
                    # 🎯 Use position-specific stop loss if available
                    current_stop_threshold = stop_loss_threshold
                    if position.stop_loss_price:
                        # Distance from entry to stop price as percentage
                        current_stop_threshold = (position.stop_loss_price - position.entry_price) / position.entry_price
                    
                    loss_pct = (current_price - position.entry_price) / position.entry_price
                    unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    
                    # Check if we need stop-loss protection
                    if loss_pct <= current_stop_threshold:  # Negative loss percentage
                        
                        # 🚨 PANIC EXIT: If loss is > 1.5x expected, dump immediately with market order
                        # Example: Stop is -10%. Current is -16%. -0.16 < -0.10 * 1.5 (-0.15). Panic!
                        if loss_pct <= current_stop_threshold * 1.5:
                            logger.warning(f"🚨 PANIC EXIT TRIGGERED: {position.market_id} - Loss {loss_pct:.1%} exceeds 1.5x threshold {current_stop_threshold:.1%}!")
                            success, exit_price = await close_position_market(position, db_manager, kalshi_client)
                            
                            if success:
                                results['orders_placed'] += 1
                                logger.info(f"✅ EMERGENCY EXIT successful at ${exit_price:.2f}")
                            continue

                        # Standard Stop Loss: Use Limit Order
                        # Calculate stop-loss sell price
                        stop_price = position.entry_price * (1 + current_stop_threshold * 1.1)  # Slightly more aggressive
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
async def close_position_market(
    position: Position,
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient
) -> Tuple[bool, float]:
    """
    Immediately closes a position using a market order.
    Used for emergency exits, time-based exits, or resolution-based exits.
    
    Returns:
        Tuple of (success, exit_price)
    """
    logger = get_trading_logger("market_exit")
    
    try:
        import uuid
        client_order_id = str(uuid.uuid4())
        
        # Determine side for sell order
        # If we are long YES, we sell YES. If we are long NO, we sell NO.
        side = position.side.lower()
        
        order_params = {
            "ticker": position.market_id,
            "client_order_id": client_order_id,
            "side": side,
            "action": "sell",
            "count": position.quantity,
            "type_": "market"
        }
        
        # 🚨 FIX: Kalshi API requires exactly ONE price field
        # For market sell orders, use 1 cent as the minimum we'll accept
        if side == "yes":
            order_params["yes_price"] = 1
        else:
            order_params["no_price"] = 1
            
        logger.info(f"🛑 Executing MARKET EXIT: {position.quantity} {side.upper()} for {position.market_id}")
        
        live_mode = getattr(settings.trading, 'live_trading_enabled', False)
        
        if live_mode:
            response = await kalshi_client.place_order(**order_params)
            
            if response and 'order' in response:
                order_info = response.get('order', {})
                # Get fill price from response if available, otherwise estimate
                if side == "yes":
                    exit_price = order_info.get('yes_price', 0) / 100
                else:
                    exit_price = order_info.get('no_price', 0) / 100
                
                # If fill price is 0 from order response, try to fetch from market or fills
                if exit_price <= 0:
                    market_data = await kalshi_client.get_market(position.market_id)
                    market_info = market_data.get('market', {})
                    if side == "yes":
                        exit_price = market_info.get('yes_bid', 1) / 100
                    else:
                        exit_price = market_info.get('no_bid', 1) / 100
                
                logger.info(f"✅ Market exit successful for {position.market_id} at ${exit_price:.2f}")
                return True, exit_price
            else:
                logger.error(f"❌ Failed to place market exit order: {response}")
                return False, 0.0
        else:
            # Simulated mode
            market_data = await kalshi_client.get_market(position.market_id)
            market_info = market_data.get('market', {})
            if side == "yes":
                exit_price = market_info.get('yes_bid', 1) / 100
            else:
                exit_price = market_info.get('no_bid', 1) / 100
                
            logger.info(f"📝 SIMULATED market exit for {position.market_id} at ${exit_price:.2f}")
            return True, exit_price
            
    except Exception as e:
        logger.error(f"❌ Error in market exit: {e}")
        return False, 0.0


# =============================================================================
# === PRODUCTION SAFETY: EXPIRATION RISK AUTO-EXIT ===
# =============================================================================

async def auto_exit_expiring_positions(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    minutes_before_expiry: int = 30
) -> Dict[str, int]:
    """
    Automatically close positions that are within X minutes of market expiration.
    
    This prevents:
    - Force-settlement at unfavorable prices
    - Illiquidity as market approaches close
    - Surprise resolution outcomes
    
    Args:
        db_manager: Database manager
        kalshi_client: Kalshi API client
        minutes_before_expiry: Minutes before expiration to trigger auto-exit (default: 30)
    
    Returns:
        Dictionary with results: {'positions_closed': int, 'positions_checked': int, 'total_pnl': float}
    """
    logger = get_trading_logger("expiration_exit")
    
    results = {
        'positions_closed': 0,
        'positions_checked': 0,
        'total_pnl': 0.0,
        'positions_failed': 0
    }
    
    if not settings.trading.auto_exit_expiring_enabled:
        logger.debug("Expiration auto-exit is disabled")
        return results
    
    try:
        # Get all open live positions
        positions = await db_manager.get_open_live_positions()
        
        if not positions:
            logger.debug("No open positions to check for expiration risk")
            return results
        
        now = datetime.now()
        expiry_threshold = now + timedelta(minutes=minutes_before_expiry)
        expiry_threshold_ts = int(expiry_threshold.timestamp())
        
        logger.info(f"⏰ Checking {len(positions)} positions for expiration risk (threshold: {minutes_before_expiry} min)")
        
        for position in positions:
            try:
                results['positions_checked'] += 1
                
                # Get market data to check expiration
                market_response = await kalshi_client.get_market(position.market_id)
                market_data = market_response.get('market', {})
                
                if not market_data:
                    logger.warning(f"Could not get market data for {position.market_id}")
                    continue
                
                # Check expiration time
                expiration_ts = market_data.get('expiration_time') or market_data.get('close_time')
                
                if not expiration_ts:
                    # Try to parse from different formats
                    exp_str = market_data.get('expiration_time') or market_data.get('end_date')
                    if exp_str:
                        try:
                            expiration_ts = int(datetime.fromisoformat(exp_str.replace('Z', '+00:00')).timestamp())
                        except (ValueError, TypeError):
                            continue
                    else:
                        continue
                
                # Convert if it's a string timestamp
                if isinstance(expiration_ts, str):
                    try:
                        expiration_ts = int(datetime.fromisoformat(expiration_ts.replace('Z', '+00:00')).timestamp())
                    except (ValueError, TypeError):
                        expiration_ts = int(expiration_ts) if str(expiration_ts).isdigit() else 0
                
                # Check if market expires within threshold
                if expiration_ts <= expiry_threshold_ts:
                    minutes_remaining = (expiration_ts - int(now.timestamp())) / 60
                    
                    logger.warning(
                        f"⚠️ EXPIRATION RISK: {position.market_id} expires in {minutes_remaining:.0f} min "
                        f"(threshold: {minutes_before_expiry} min) - AUTO-EXITING"
                    )
                    
                    # Calculate current price for P&L
                    if position.side == "YES":
                        current_price = market_data.get('yes_bid', 0)
                        if isinstance(current_price, (int, float)):
                            current_price = current_price / 100 if current_price > 1 else current_price
                    else:
                        current_price = market_data.get('no_bid', 0)
                        if isinstance(current_price, (int, float)):
                            current_price = current_price / 100 if current_price > 1 else current_price
                    
                    # Execute market exit
                    success, exit_price = await close_position_market(
                        position=position,
                        db_manager=db_manager,
                        kalshi_client=kalshi_client
                    )
                    
                    if success:
                        results['positions_closed'] += 1
                        pnl = (exit_price - position.entry_price) * position.quantity
                        results['total_pnl'] += pnl
                        
                        # Close position in database
                        await db_manager.close_position(position.id)
                        
                        logger.info(
                            f"✅ AUTO-EXITED {position.market_id} before expiration: "
                            f"Entry=${position.entry_price:.2f}, Exit=${exit_price:.2f}, "
                            f"P&L=${pnl:.2f}"
                        )
                    else:
                        results['positions_failed'] += 1
                        logger.error(f"❌ Failed to auto-exit {position.market_id}")
                        
            except Exception as e:
                logger.error(f"Error checking expiration for {position.market_id}: {e}")
                continue
        
        # Summary
        if results['positions_closed'] > 0:
            logger.warning(
                f"⏰ Expiration auto-exit summary: "
                f"Closed {results['positions_closed']}/{results['positions_checked']} positions, "
                f"Total P&L: ${results['total_pnl']:.2f}"
            )
        else:
            logger.debug(f"No positions require expiration exit ({results['positions_checked']} checked)")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in expiration auto-exit: {e}")
        return results
