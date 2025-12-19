#!/usr/bin/env python3
"""
Sync positions and orders from Kalshi to local database.
Run this script to update your local database with actual Kalshi data.
"""
import asyncio
import sys
sys.path.insert(0, '.')

from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager


async def sync_positions():
    """Sync positions from Kalshi to local database."""
    print("🔄 Syncing positions from Kalshi...")
    
    kalshi = KalshiClient()
    db = DatabaseManager()
    
    try:
        # Fetch actual positions from Kalshi
        response = await kalshi.get_positions()
        positions = response.get('market_positions', [])
        
        print(f"\n📊 Found {len(positions)} positions on Kalshi:")
        print("-" * 80)
        
        for pos in positions:
            ticker = pos.get('ticker', 'N/A')
            position_qty = pos.get('position', 0)
            total_cost = pos.get('total_traded', 0) / 100  # cents to dollars
            market_value = pos.get('market_exposure', 0) / 100
            fees = pos.get('fees_paid', 0) / 100
            
            side = "YES" if position_qty > 0 else "NO"
            qty = abs(position_qty)
            
            if qty > 0:
                print(f"  • {ticker}")
                print(f"    Side: {side}, Quantity: {qty}, Cost: ${total_cost:.2f}, Value: ${market_value:.2f}")
        
        # Fetch resting orders
        print("\n📋 Fetching resting orders...")
        orders_response = await kalshi.get_orders(status='resting')
        orders = orders_response.get('orders', [])
        
        print(f"\n⏳ Found {len(orders)} resting orders:")
        print("-" * 80)
        
        for order in orders:
            ticker = order.get('ticker', 'N/A')
            side = order.get('side', 'N/A')
            action = order.get('action', 'N/A')
            remaining = order.get('remaining_count', 0)
            price = order.get('yes_price', order.get('no_price', 0)) / 100
            order_id = order.get('order_id', 'N/A')
            
            print(f"  • {ticker}")
            print(f"    {action.upper()} {remaining} {side.upper()} @ ${price:.2f}")
            print(f"    Order ID: {order_id}")
        
        # Fetch balance
        print("\n💰 Account Balance:")
        print("-" * 80)
        balance = await kalshi.get_balance()
        cash = balance.get('balance', 0) / 100  # cents to dollars
        bonus = balance.get('bonus_balance', 0) / 100
        portfolio = balance.get('portfolio_value', 0) / 100
        print(f"  Cash: ${cash:.2f}")
        print(f"  Bonus: ${bonus:.2f}")
        print(f"  Portfolio Value: ${portfolio:.2f}")
        print(f"  Total: ${cash + portfolio:.2f}")
        
        # Get local database positions
        print("\n📁 Local Database Positions:")
        print("-" * 80)
        local_positions = await db.get_open_positions()
        print(f"  Found {len(local_positions)} open positions in local DB")
        for pos in local_positions:
            print(f"  • {pos.market_id} - {pos.side} x{pos.quantity} @ ${pos.entry_price:.2f} (live={pos.live})")
        
        print("\n" + "=" * 80)
        print("⚠️  To cancel resting orders, run:")
        print("    python sync_kalshi.py --cancel-orders")
        print("⚠️  To mark local positions as closed, run:")
        print("    python sync_kalshi.py --close-stale")
        print("=" * 80)
        
    finally:
        await kalshi.close()


async def cancel_all_orders():
    """Cancel all resting orders."""
    print("🗑️ Cancelling all resting orders...")
    
    kalshi = KalshiClient()
    
    try:
        orders_response = await kalshi.get_orders(status='resting')
        orders = orders_response.get('orders', [])
        
        if not orders:
            print("No resting orders to cancel.")
            return
        
        for order in orders:
            order_id = order.get('order_id')
            ticker = order.get('ticker')
            try:
                await kalshi.cancel_order(order_id)
                print(f"  ✅ Cancelled order for {ticker}: {order_id}")
            except Exception as e:
                print(f"  ❌ Failed to cancel order {order_id}: {e}")
        
        print(f"\n✅ Cancelled {len(orders)} orders")
        
    finally:
        await kalshi.close()


async def close_stale_positions():
    """Close stale positions in local DB that don't match Kalshi."""
    print("🧹 Closing stale local positions...")
    
    kalshi = KalshiClient()
    db = DatabaseManager()
    
    try:
        # Get Kalshi positions
        response = await kalshi.get_positions()
        kalshi_positions = response.get('market_positions', [])
        kalshi_tickers = {p.get('ticker') for p in kalshi_positions if abs(p.get('position', 0)) > 0}
        
        # Get local positions
        local_positions = await db.get_open_positions()
        
        closed = 0
        for pos in local_positions:
            if pos.market_id not in kalshi_tickers:
                print(f"  Closing stale position: {pos.market_id}")
                await db.update_position_status(pos.id, 'closed')
                closed += 1
        
        print(f"\n✅ Closed {closed} stale positions")
        
    finally:
        await kalshi.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--cancel-orders":
            asyncio.run(cancel_all_orders())
        elif sys.argv[1] == "--close-stale":
            asyncio.run(close_stale_positions())
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Usage: python sync_kalshi.py [--cancel-orders | --close-stale]")
    else:
        asyncio.run(sync_positions())
