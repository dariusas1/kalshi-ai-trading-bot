"""
Daily reconciliation job.

Syncs local positions with Kalshi and stores reconciliation summary in analytics_cache.
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional

from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.utils.pnl_tracker import PnLTracker
from src.utils.logging_setup import get_trading_logger


a_sync_lock = asyncio.Lock()


async def run_reconciliation(db_manager: Optional[DatabaseManager] = None) -> Dict[str, int]:
    """Run daily reconciliation and cache results."""
    logger = get_trading_logger("reconciliation")

    async with a_sync_lock:
        db_manager = db_manager or DatabaseManager()
        await db_manager.initialize()
        kalshi_client = KalshiClient()

        try:
            sync_result = await db_manager.sync_with_kalshi(kalshi_client)
            pnl_tracker = PnLTracker(db_manager, kalshi_client)
            fills_synced = await pnl_tracker.sync_fills_to_database()

            summary = {
                "timestamp": datetime.now().isoformat(),
                "kalshi_positions": sync_result.get("kalshi_positions", 0),
                "local_positions": sync_result.get("local_positions", 0),
                "stale_closed": sync_result.get("stale_closed", 0),
                "fills_synced": fills_synced,
                "error": sync_result.get("error")
            }

            await db_manager.set_cached_analytics("reconciliation_status", summary)
            logger.info("Reconciliation complete", **summary)
            return summary
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            summary = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
            await db_manager.set_cached_analytics("reconciliation_status", summary)
            return summary
        finally:
            await kalshi_client.close()


if __name__ == "__main__":
    asyncio.run(run_reconciliation())
