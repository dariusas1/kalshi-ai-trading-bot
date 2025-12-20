#!/usr/bin/env python3
"""
Background Analytics Processor 📊

Periodically computes and caches complex analytics for the trading dashboard to
ensure fast load times even with large trade histories.
"""

import asyncio
import json
import logging
from datetime import datetime
import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.database import DatabaseManager
from src.utils.logging_setup import setup_logging

logger = logging.getLogger("analytics_processor")

class AnalyticsProcessor:
    """Computes and caches analytics data for the system."""
    
    def __init__(self, db_path: str = 'trading_system.db'):
        self.db_manager = DatabaseManager(db_path)
        self.interval = 300  # 5 minutes
        
    async def run_once(self):
        """Run a single analytics computation cycle."""
        logger.info("📊 Starting analytics computation cycle...")
        start_time = datetime.now()
        
        try:
            await self.db_manager.initialize()
            
            # 1. P&L by Period
            logger.info("Computing P&L by period...")
            periods = ['today', 'week', 'month', 'all']
            pnl_data = {}
            for period in periods:
                pnl_data[period] = await self.db_manager.get_pnl_by_period(period)
            await self._cache_result("period_pnl", pnl_data)
            
            # 2. Category Performance
            logger.info("Computing category performance...")
            cat_perf = await self.db_manager.get_category_performance()
            await self._cache_result("category_performance", cat_perf)
            
            # 3. Time-of-Day Performance
            logger.info("Computing hourly performance...")
            hourly_perf = await self.db_manager.get_hourly_performance()
            await self._cache_result("hourly_performance", hourly_perf)
            
            # 4. Expiry Performance
            logger.info("Computing expiry performance...")
            expiry_perf = await self.db_manager.get_expiry_performance()
            await self._cache_result("expiry_performance", expiry_perf)
            
            # 5. Confidence Calibration
            logger.info("Computing confidence calibration...")
            calibration = await self.db_manager.get_confidence_calibration()
            await self._cache_result("confidence_calibration", calibration)
            
            # 6. Trading Streaks
            logger.info("Computing trading streaks...")
            streaks = await self.db_manager.get_trading_streaks()
            await self._cache_result("trading_streaks", streaks)
            
            # 7. AI Cost History
            logger.info("Computing AI cost history...")
            cost_hist = await self.db_manager.get_ai_cost_history(days=30)
            await self._cache_result("ai_cost_history", cost_hist)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"✅ Analytics computation complete in {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error in analytics cycle: {e}")
        finally:
            await self.db_manager.close()

    async def _cache_result(self, key: str, data: any):
        """Store computed data in the database cache."""
        try:
            import aiosqlite
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                await db.execute("""
                    INSERT OR REPLACE INTO analytics_cache (key, value, computed_at)
                    VALUES (?, ?, ?)
                """, (key, json.dumps(data), datetime.now().isoformat()))
                await db.commit()
        except Exception as e:
            logger.error(f"Error caching {key}: {e}")

    async def run_forever(self):
        """Run the processor in a loop."""
        logger.info(f"🚀 Analytics Processor started (Interval: {self.interval}s)")
        while True:
            await self.run_once()
            await asyncio.sleep(self.interval)

async def main():
    setup_logging(log_level="INFO")
    processor = AnalyticsProcessor()
    await processor.run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Analytics Processor stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

