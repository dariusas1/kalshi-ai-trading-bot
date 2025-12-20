#!/usr/bin/env python3
"""
Beast Mode Trading Bot 🚀

Main entry point for the Unified Advanced Trading System that orchestrates:
- Market Making Strategy (30% allocation)
- Directional Trading with Portfolio Optimization (40% allocation) 
- Quick Flip Scalping (30% allocation)
- Arbitrage Detection (0% by default)

Features:
- No time restrictions (trade any deadline)
- Dynamic exit strategies
- Kelly Criterion portfolio optimization
- Real-time risk management
- Market making for spread profits

Usage:
    python beast_mode_bot.py              # Paper trading mode
    python beast_mode_bot.py --live       # Live trading mode
    python beast_mode_bot.py --dashboard  # Live dashboard mode
"""

import asyncio
import argparse
import time
import signal
import os
from datetime import datetime, timedelta
from typing import Optional

from src.jobs.trade import run_trading_job
from src.jobs.ingest import run_ingestion
from src.jobs.track import run_tracking
from src.jobs.evaluate import run_evaluation
from src.jobs.performance_scheduler import start_performance_scheduler, stop_performance_scheduler
from src.utils.logging_setup import setup_logging, get_trading_logger
from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings

# Import Beast Mode components
from src.strategies.unified_trading_system import run_unified_trading_system, TradingSystemConfig
from beast_mode_dashboard import BeastModeDashboard


class BeastModeBot:
    """
    Beast Mode Trading Bot - Advanced Multi-Strategy Trading System 🚀
    
    This bot orchestrates all advanced strategies:
    1. Market Making (spread profits)
    2. Directional Trading (AI predictions with portfolio optimization)
    3. Arbitrage Detection (future feature)
    
    Features:
    - Unlimited market deadlines with dynamic exits
    - Cost controls and budget management
    - Real-time performance monitoring
    - Risk management and rebalancing
    """
    
    def __init__(self, live_mode: bool = False, dashboard_mode: bool = False):
        self.live_mode = live_mode
        self.dashboard_mode = dashboard_mode
        self.logger = get_trading_logger("beast_mode_bot")
        self.shutdown_event = asyncio.Event()
        
        # Set live trading in settings
        settings.trading.live_trading_enabled = live_mode
        settings.trading.paper_trading_mode = not live_mode
        
        self.logger.info(
            f"🚀 Beast Mode Bot initialized - "
            f"Mode: {'LIVE TRADING' if live_mode else 'PAPER TRADING'}"
        )

    async def run_dashboard_mode(self):
        """Run in live dashboard mode with real-time updates."""
        try:
            self.logger.info("🚀 Starting Beast Mode Dashboard Mode")
            dashboard = BeastModeDashboard()
            await dashboard.show_live_dashboard()
        except KeyboardInterrupt:
            self.logger.info("👋 Dashboard mode stopped")
        except Exception as e:
            self.logger.error(f"Error in dashboard mode: {e}")

    async def run_trading_mode(self):
        """Run the Beast Mode trading system with all strategies."""
        try:
            self.logger.info("🚀 BEAST MODE TRADING BOT STARTED")
            self.logger.info(f"📊 Trading Mode: {'LIVE' if self.live_mode else 'PAPER'}")
            self.logger.info(f"💰 Daily AI Budget: ${settings.trading.daily_ai_budget}")
            self.logger.info(f"🧮 Daily AI Spend Limit: ${settings.get_ai_daily_limit()}")
            self.logger.info(f"⚡ Features: Market Making + Portfolio Optimization + Dynamic Exits")
            
            # 🚨 CRITICAL FIX: Initialize database FIRST and wait for completion
            self.logger.info("🔧 Initializing database...")
            db_manager = DatabaseManager()
            await self._ensure_database_ready(db_manager)
            self.logger.info("✅ Database initialization complete!")
            
            # Initialize other components
            kalshi_client = KalshiClient()
            xai_client = XAIClient(db_manager=db_manager)  # Pass db_manager for LLM logging
            
            # 🔄 Sync with Kalshi on startup to ensure accurate position data
            self.logger.info("🔄 Syncing positions with Kalshi...")
            sync_result = await db_manager.sync_with_kalshi(kalshi_client)
            self.logger.info(f"✅ Kalshi sync complete: {sync_result}")
            
            # Small delay to ensure everything is ready
            await asyncio.sleep(1)
            
            # Start market ingestion first
            self.logger.info("🔄 Starting market ingestion...")
            ingestion_task = asyncio.create_task(self._run_market_ingestion(db_manager, kalshi_client))
            
            # Wait for initial market data ingestion
            await asyncio.sleep(10)
            
            performance_scheduler = None
            performance_manager_enabled = os.getenv("ENABLE_PERFORMANCE_SYSTEM_MANAGER", "false").strip().lower() in ("1", "true", "yes", "on")
            if settings.trading.performance_monitoring and not performance_manager_enabled:
                performance_scheduler = start_performance_scheduler()

            # Run remaining background tasks
            self.logger.info("🚀 Starting trading and monitoring tasks...")
            tasks = [
                ingestion_task,  # Already started
                asyncio.create_task(self._run_trading_cycles(db_manager, kalshi_client, xai_client)),
                asyncio.create_task(self._run_position_tracking(db_manager, kalshi_client)),
                asyncio.create_task(self._run_reconciliation(db_manager))
            ]
            if settings.trading.performance_monitoring:
                tasks.append(asyncio.create_task(self._run_performance_evaluation(db_manager)))
            else:
                self.logger.info("Performance monitoring disabled; skipping evaluation task")
            
            # Setup shutdown handler
            def signal_handler():
                self.logger.info("🛑 Shutdown signal received")
                self.shutdown_event.set()
                for task in tasks:
                    task.cancel()
                if performance_scheduler:
                    stop_performance_scheduler()
            
            # Handle Ctrl+C gracefully
            for sig in [signal.SIGINT, signal.SIGTERM]:
                signal.signal(sig, lambda s, f: signal_handler())
            
            # Wait for shutdown or completion
            await asyncio.gather(*tasks, return_exceptions=True)
            
            await xai_client.close()
            await kalshi_client.close()
            if performance_scheduler:
                stop_performance_scheduler()
            
            self.logger.info("🏁 Beast Mode Bot shut down gracefully")
            
        except Exception as e:
            self.logger.error(f"Error in Beast Mode Bot: {e}")
            raise

    async def _ensure_database_ready(self, db_manager: DatabaseManager):
        """Ensure database is fully initialized before starting any tasks."""
        try:
            # Initialize the database first to create all tables
            await db_manager.initialize()
            
            # Verify tables exist by checking one of them
            import aiosqlite
            async with aiosqlite.connect(db_manager.db_path) as db:
                await db.execute("SELECT COUNT(*) FROM positions LIMIT 1")
                await db.execute("SELECT COUNT(*) FROM markets LIMIT 1") 
                await db.execute("SELECT COUNT(*) FROM trade_logs LIMIT 1")
            
            self.logger.info("🎯 Database tables verified and ready")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise

    async def _run_market_ingestion(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        """Background task for market data ingestion."""
        while not self.shutdown_event.is_set():
            try:
                # Create a queue for market ingestion (though we're not using it in Beast Mode)
                market_queue = asyncio.Queue()
                # ✅ FIXED: Pass the shared database manager
                await run_ingestion(db_manager, market_queue)
                sleep_seconds = max(60, settings.trading.market_scan_interval)
                await asyncio.sleep(sleep_seconds)
            except Exception as e:
                self.logger.error(f"Error in market ingestion: {e}")
                await asyncio.sleep(60)

    async def _run_trading_cycles(self, db_manager: DatabaseManager, kalshi_client: KalshiClient, xai_client: XAIClient):
        """Main Beast Mode trading cycles."""
        cycle_count = 0
        
        while not self.shutdown_event.is_set():
            try:
                # Check daily AI cost limits before starting cycle
                if not await self._check_daily_ai_limits(xai_client):
                    # Sleep until next day if limits reached
                    await self._sleep_until_next_day()
                    continue
                
                cycle_count += 1
                
                # Check Kill Switch
                kill_switch = await db_manager.get_setting("kill_switch", "off")
                if kill_switch == "on":
                    self.logger.warning(f"🛑 Kill switch is ON. Skipping trading cycle #{cycle_count}")
                    await asyncio.sleep(60)
                    continue

                # Daily loss guardrail
                try:
                    pnl_today = await db_manager.get_pnl_by_period("today")
                    total_pnl = pnl_today.get("total_pnl", 0.0) if isinstance(pnl_today, dict) else 0.0
                    balance_response = await kalshi_client.get_balance()
                    total_capital = balance_response.get("balance", 0) / 100
                    loss_limit = settings.trading.max_daily_loss_pct / 100 * max(total_capital, 1)
                    if total_pnl < 0 and abs(total_pnl) >= loss_limit:
                        self.logger.warning(
                            f"🛑 Daily loss limit reached: ${total_pnl:.2f} (limit {settings.trading.max_daily_loss_pct:.1f}%)"
                        )
                        await asyncio.sleep(60)
                        continue
                except Exception as e:
                    self.logger.warning(f"Daily loss check failed: {e}")
                
                # Update settings from database dynamically
                await self._update_dynamic_settings(db_manager)

                # Intraday risk throttles
                try:
                    streaks = await db_manager.get_trading_streaks()
                    if (streaks.get("streak_type") == "loss" and
                            streaks.get("current_streak", 0) >= settings.trading.loss_streak_pause_threshold):
                        pause_minutes = settings.trading.loss_streak_pause_minutes
                        self.logger.warning(
                            f"🛑 Loss streak throttle: {streaks.get('current_streak')} losses. "
                            f"Pausing {pause_minutes} minutes."
                        )
                        await asyncio.sleep(pause_minutes * 60)
                        continue
                except Exception as e:
                    self.logger.warning(f"Loss streak check failed: {e}")

                # Trade frequency guardrail
                try:
                    if settings.trading.max_trades_per_hour:
                        recent_trades = await db_manager.get_recent_trades(limit=200)
                        one_hour_ago = datetime.now() - timedelta(hours=1)
                        trades_last_hour = 0
                        for trade in recent_trades:
                            ts = trade.get("exit_timestamp") or trade.get("timestamp")
                            if isinstance(ts, str):
                                try:
                                    ts_dt = datetime.fromisoformat(ts)
                                except ValueError:
                                    continue
                            else:
                                ts_dt = ts
                            if ts_dt and ts_dt >= one_hour_ago:
                                trades_last_hour += 1

                        if trades_last_hour >= settings.trading.max_trades_per_hour:
                            self.logger.warning(
                                f"⏳ Max trades/hour reached ({trades_last_hour}/{settings.trading.max_trades_per_hour}), pausing cycle"
                            )
                            await asyncio.sleep(60)
                            continue
                except Exception as e:
                    self.logger.warning(f"Trade frequency check failed: {e}")

                self.logger.info(f"🔄 Starting Beast Mode Trading Cycle #{cycle_count}")

                # Run the Beast Mode unified trading system
                # 🔒 CRITICAL FIX: Pass shared clients to ensure daily AI cost tracking works correctly
                # Previously, run_trading_job() created its own XAIClient, bypassing the cost limiter
                results = await run_trading_job(
                    xai_client=xai_client,
                    db_manager=db_manager,
                    kalshi_client=kalshi_client
                )
                
                if results and results.total_positions > 0:
                    self.logger.info(
                        f"✅ Cycle #{cycle_count} Complete - "
                        f"Positions: {results.total_positions}, "
                        f"Capital Used: ${results.total_capital_used:.0f} ({results.capital_efficiency:.1%}), "
                        f"Expected Return: {results.expected_annual_return:.1%}"
                    )
                else:
                    self.logger.info(f"📊 Cycle #{cycle_count} Complete - No new positions created")

                # Volatility throttle
                if results and results.portfolio_volatility > settings.trading.volatility_pause_threshold:
                    self.logger.warning(
                        f"🛑 Volatility throttle: {results.portfolio_volatility:.2%} > "
                        f"{settings.trading.volatility_pause_threshold:.2%}. Pausing 10 minutes."
                    )
                    await asyncio.sleep(600)

                # Wait for next cycle
                cycle_sleep = max(
                    settings.trading.scan_interval_seconds,
                    settings.trading.run_interval_minutes * 60
                )
                await asyncio.sleep(cycle_sleep)
                
            except Exception as e:
                self.logger.error(f"Error in trading cycle #{cycle_count}: {e}")
                await asyncio.sleep(60)

    async def _update_dynamic_settings(self, db_manager: DatabaseManager):
        """Update trading parameters dynamically from the database."""
        try:
            # 1. Confidence threshold
            min_conf = await db_manager.get_setting("min_confidence")
            if min_conf:
                settings.trading.min_confidence_to_trade = float(min_conf)
            
            # 2. Risk per trade
            max_risk = await db_manager.get_setting("max_risk_per_trade")
            if max_risk:
                settings.trading.max_position_size_pct = float(max_risk)
                
            # 3. Strategy allocations
            mm_w = await db_manager.get_setting("weight_market_making")
            dir_w = await db_manager.get_setting("weight_directional")
            qf_w = await db_manager.get_setting("weight_quick_flip")
            
            if mm_w and dir_w and qf_w:
                # Update settings (assumes these attributes exist in settings.trading)
                settings.trading.market_making_allocation = float(mm_w) / 100.0
                settings.trading.directional_allocation = float(dir_w) / 100.0
                settings.trading.quick_flip_allocation = float(qf_w) / 100.0
                
            # 4. Blacklist
            blacklist = await db_manager.get_setting("category_blacklist")
            if blacklist:
                settings.trading.category_blacklist = [c.strip().lower() for c in blacklist.split(",")]
            
        except Exception as e:
            self.logger.error(f"Error updating dynamic settings: {e}")

    async def _check_daily_ai_limits(self, xai_client: XAIClient) -> bool:
        """
        Check if we should continue trading based on daily AI cost limits.
        Returns True if we can continue, False if we should pause.
        """
        if not settings.trading.enable_daily_cost_limiting:
            return True
        
        # Check daily tracker in xAI client
        if hasattr(xai_client, 'daily_tracker') and xai_client.daily_tracker.is_exhausted:
            self.logger.warning(
                "🚫 Daily AI cost limit reached - trading paused",
                daily_cost=xai_client.daily_tracker.total_cost,
                daily_limit=xai_client.daily_tracker.daily_limit,
                requests_today=xai_client.daily_tracker.request_count
            )
            return False
        
        return True

    async def _sleep_until_next_day(self):
        """Sleep until the next day (midnight) when daily limits reset."""
        if not settings.trading.sleep_when_limit_reached:
            # Just sleep for a normal cycle if sleep is disabled
            await asyncio.sleep(60)
            return
        
        # Calculate time until next day
        now = datetime.now()
        next_day = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        seconds_until_next_day = (next_day - now).total_seconds()
        
        # Ensure we don't sleep for more than 24 hours (safety check)
        max_sleep = 24 * 60 * 60  # 24 hours
        sleep_time = min(seconds_until_next_day, max_sleep)
        
        if sleep_time > 0:
            hours_to_sleep = sleep_time / 3600
            self.logger.info(
                f"💤 Sleeping until next day to reset AI limits - {hours_to_sleep:.1f} hours"
            )
            
            # Sleep in chunks to allow for graceful shutdown
            chunk_size = 300  # 5 minutes per chunk
            while sleep_time > 0 and not self.shutdown_event.is_set():
                current_chunk = min(chunk_size, sleep_time)
                await asyncio.sleep(current_chunk)
                sleep_time -= current_chunk
            
            self.logger.info("🌅 Daily AI limits reset - resuming trading")
        else:
            # Safety fallback
            await asyncio.sleep(60)

    async def _run_position_tracking(self, db_manager: DatabaseManager, kalshi_client: KalshiClient):
        """Background task for position tracking and exit strategies."""
        sync_counter = 0
        while not self.shutdown_event.is_set():
            try:
                # Sync with Kalshi every 5 cycles (10 minutes) to keep positions accurate
                sync_counter += 1
                if sync_counter >= 5:
                    await db_manager.sync_with_kalshi(kalshi_client)
                    sync_counter = 0
                
                # ✅ FIXED: Pass the shared database manager
                await run_tracking(db_manager)
                await asyncio.sleep(max(10, settings.trading.position_check_interval))
            except Exception as e:
                self.logger.error(f"Error in position tracking: {e}")
                await asyncio.sleep(30)

    async def _run_performance_evaluation(self, db_manager: DatabaseManager):
        """Background task for performance evaluation."""
        while not self.shutdown_event.is_set():
            try:
                await run_evaluation()
                await asyncio.sleep(300)  # Run every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in performance evaluation: {e}")
                await asyncio.sleep(300)

    async def _run_reconciliation(self, db_manager: DatabaseManager):
        """Background task for daily reconciliation with Kalshi."""
        from src.jobs.reconcile import run_reconciliation

        while not self.shutdown_event.is_set():
            try:
                await run_reconciliation(db_manager)
                await asyncio.sleep(24 * 60 * 60)  # Run daily
            except Exception as e:
                self.logger.error(f"Error in reconciliation: {e}")
                await asyncio.sleep(60 * 60)

    async def run(self):
        """Main entry point for Beast Mode Bot."""
        if self.dashboard_mode:
            await self.run_dashboard_mode()
        else:
            await self.run_trading_mode()


async def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Beast Mode Trading Bot 🚀 - Advanced Multi-Strategy Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python beast_mode_bot.py              # Paper trading mode
  python beast_mode_bot.py --live       # Live trading mode  
  python beast_mode_bot.py --dashboard  # Live dashboard mode
  python beast_mode_bot.py --live --log-level DEBUG  # Live mode with debug logs
  python beast_mode_bot.py --live --no-market-making  # Live without market making

Beast Mode Features:
  • Market Making (30% allocation) - Profit from spreads
  • Directional Trading (40% allocation) - AI predictions with portfolio optimization
  • Quick Flip Scalping (30% allocation) - Rapid low-price contract trading
  • No time restrictions - Trade any deadline with dynamic exits
  • Kelly Criterion portfolio optimization
  • Real-time risk management and rebalancing
  • Cost controls and budget management
        """
    )
    
    parser.add_argument(
        "--live", 
        action="store_true", 
        help="Run in LIVE trading mode (default: paper trading)"
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run in live dashboard mode for monitoring"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO)"
    )
    
    # Strategy control arguments
    parser.add_argument(
        "--no-market-making",
        action="store_true",
        help="Disable market making strategy"
    )
    parser.add_argument(
        "--no-quick-flip",
        action="store_true",
        help="Disable quick flip scalping strategy"
    )
    parser.add_argument(
        "--directional-only",
        action="store_true",
        help="Run only directional trading strategy (disables market making and quick flip)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Apply strategy overrides from CLI
    from src.config import settings as settings_module
    if args.no_market_making or args.directional_only:
        settings_module.enable_market_making = False
        print("📊 Market making: DISABLED")
    else:
        print(f"📊 Market making: ENABLED ({settings.trading.market_making_allocation:.0%})")
    
    if args.no_quick_flip or args.directional_only:
        settings.trading.quick_flip_allocation = 0.0
        print("⚡ Quick flip: DISABLED")
    else:
        print(f"⚡ Quick flip: ENABLED ({settings.trading.quick_flip_allocation:.0%})")
    
    print(f"🎯 Directional trading: ENABLED ({settings.trading.directional_allocation:.0%})")
    
    # Warn about live mode
    if args.live and not args.dashboard:
        print("⚠️  WARNING: LIVE TRADING MODE ENABLED")
        print("💰 This will use real money and place actual trades!")
        print("🚀 LIVE TRADING MODE CONFIRMED")
    
    # Create and run Beast Mode Bot
    bot = BeastModeBot(live_mode=args.live, dashboard_mode=args.dashboard)
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Beast Mode Bot stopped by user")
    except Exception as e:
        print(f"❌ Beast Mode Bot error: {e}")
        raise 
