
import asyncio
import logging
import os
import sys

# Add project root to path for imports
project_root = "/Users/darius/Documents/1-Active-Projects/Personal-Apps/kalshi-ai-trading-bot"
sys.path.append(project_root)

from src.utils.database import DatabaseManager

async def reproduce():
    logging.basicConfig(level=logging.INFO)
    db = DatabaseManager('trading_system.db')
    await db.initialize()
    
    steps = [
        ("P&L by Period", lambda db: [db.get_pnl_by_period(p) for p in ['today', 'week', 'month', 'all']]),
        ("Category Performance", lambda db: [db.get_category_performance()]),
        ("Hourly Performance", lambda db: [db.get_hourly_performance()]),
        ("Expiry Performance", lambda db: [db.get_expiry_performance()]),
        ("Confidence Calibration", lambda db: [db.get_confidence_calibration()]),
        ("Trading Streaks", lambda db: [db.get_trading_streaks()]),
        ("AI Cost History", lambda db: [db.get_ai_cost_history(days=30)])
    ]

    for name, func in steps:
        print(f"Testing {name}...")
        try:
            await asyncio.gather(*func(db))
            print(f"Success! {name}")
        except Exception as e:
            print(f"FAILED {name}: {e}")
            import traceback
            traceback.print_exc()

    await db.close()

if __name__ == "__main__":
    asyncio.run(reproduce())
