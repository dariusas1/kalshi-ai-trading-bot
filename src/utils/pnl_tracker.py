"""
P&L Tracker - Real-time Profit and Loss Tracking

Fetches actual trade history from Kalshi API and calculates real P&L metrics.
Updates dashboard with actual trading performance instead of placeholder values.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import aiosqlite

from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager
from src.utils.logging_setup import get_trading_logger


@dataclass
class PnLMetrics:
    """Comprehensive P&L metrics."""
    # Daily metrics
    trades_today: int = 0
    pnl_today: float = 0.0
    wins_today: int = 0
    losses_today: int = 0
    
    # Weekly metrics
    trades_7d: int = 0
    pnl_7d: float = 0.0
    wins_7d: int = 0
    losses_7d: int = 0
    win_rate_7d: float = 0.0
    
    # All-time metrics
    total_trades: int = 0
    total_pnl: float = 0.0
    total_wins: int = 0
    total_losses: int = 0
    win_rate: float = 0.0
    
    # Performance
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_holding_time_hours: float = 0.0


class PnLTracker:
    """
    Real-time P&L tracking using Kalshi API and local trade logs.
    
    Syncs trade history from Kalshi and calculates accurate metrics
    for the trading dashboard.
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.logger = get_trading_logger("pnl_tracker")
        
    async def get_comprehensive_pnl(self) -> PnLMetrics:
        """
        Get comprehensive P&L metrics from both Kalshi API and local database.
        """
        try:
            # Get fills from Kalshi API for real P&L
            kalshi_fills = await self._fetch_kalshi_fills()
            
            # Get trade logs from database
            db_trades = await self._fetch_db_trade_logs()
            
            # Calculate metrics
            metrics = await self._calculate_metrics(kalshi_fills, db_trades)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting P&L metrics: {e}")
            return PnLMetrics()
    
    async def _fetch_kalshi_fills(self) -> List[Dict]:
        """Fetch recent fills from Kalshi API."""
        try:
            fills_response = await self.kalshi_client.get_fills(limit=500)
            fills = fills_response.get('fills', [])
            
            self.logger.debug(f"Fetched {len(fills)} fills from Kalshi API")
            return fills
            
        except Exception as e:
            self.logger.error(f"Error fetching Kalshi fills: {e}")
            return []
    
    async def _fetch_db_trade_logs(self) -> List[Dict]:
        """Fetch trade logs from database."""
        try:
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute("""
                    SELECT * FROM trade_logs 
                    ORDER BY exit_timestamp DESC 
                    LIMIT 500
                """)
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            self.logger.debug(f"Error fetching DB trade logs: {e}")
            return []
    
    async def _calculate_metrics(
        self,
        kalshi_fills: List[Dict],
        db_trades: List[Dict]
    ) -> PnLMetrics:
        """Calculate P&L metrics from fills and trade logs."""
        metrics = PnLMetrics()
        
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = now - timedelta(days=7)
        
        # Process Kalshi fills to calculate realized P&L
        fills_by_market = {}
        for fill in kalshi_fills:
            ticker = fill.get('ticker', '')
            if ticker not in fills_by_market:
                fills_by_market[ticker] = []
            fills_by_market[ticker].append(fill)
        
        # Calculate P&L from paired buys/sells
        trade_pnls = []
        
        for ticker, market_fills in fills_by_market.items():
            try:
                # Separate buys and sells
                buys = [f for f in market_fills if f.get('action') == 'buy']
                sells = [f for f in market_fills if f.get('action') == 'sell']
                
                # Match buys with sells to calculate P&L
                for sell in sells:
                    sell_price = sell.get('yes_price', sell.get('no_price', 0)) / 100
                    sell_count = sell.get('count', 0)
                    sell_ts = self._parse_timestamp(sell.get('created_time', ''))
                    side = sell.get('side', '').lower()
                    
                    # Find matching buy
                    for buy in buys:
                        if buy.get('side', '').lower() == side and buy.get('count', 0) > 0:
                            buy_price = buy.get('yes_price', buy.get('no_price', 0)) / 100
                            matched_count = min(sell_count, buy.get('count', 0))
                            
                            if matched_count > 0:
                                # Calculate P&L for this match
                                pnl = (sell_price - buy_price) * matched_count
                                
                                trade_pnls.append({
                                    'pnl': pnl,
                                    'timestamp': sell_ts,
                                    'buy_price': buy_price,
                                    'sell_price': sell_price,
                                    'quantity': matched_count
                                })
                                
                                # Reduce remaining counts
                                sell_count -= matched_count
                                buy['count'] = buy.get('count', 0) - matched_count
                                
                                if sell_count <= 0:
                                    break
                                    
            except Exception as e:
                self.logger.debug(f"Error processing fills for {ticker}: {e}")
                continue
        
        # Also include P&L from database trade logs
        for trade in db_trades:
            try:
                pnl = float(trade.get('pnl', 0))
                exit_ts_str = trade.get('exit_timestamp', '')
                
                if isinstance(exit_ts_str, str):
                    try:
                        exit_ts = datetime.fromisoformat(exit_ts_str)
                    except:
                        exit_ts = now
                else:
                    exit_ts = exit_ts_str if exit_ts_str else now
                
                trade_pnls.append({
                    'pnl': pnl,
                    'timestamp': exit_ts,
                    'buy_price': float(trade.get('entry_price', 0)),
                    'sell_price': float(trade.get('exit_price', 0)),
                    'quantity': int(trade.get('quantity', 0))
                })
            except Exception as e:
                self.logger.debug(f"Error processing DB trade: {e}")
                continue
        
        # Calculate aggregate metrics
        if trade_pnls:
            # Remove duplicates by normalizing
            seen = set()
            unique_trades = []
            for t in trade_pnls:
                key = (round(t['pnl'], 2), t.get('timestamp'))
                if key not in seen:
                    seen.add(key)
                    unique_trades.append(t)
            trade_pnls = unique_trades
            
            # All-time metrics
            metrics.total_trades = len(trade_pnls)
            metrics.total_pnl = sum(t['pnl'] for t in trade_pnls)
            metrics.total_wins = sum(1 for t in trade_pnls if t['pnl'] > 0)
            metrics.total_losses = sum(1 for t in trade_pnls if t['pnl'] <= 0)
            metrics.win_rate = metrics.total_wins / metrics.total_trades if metrics.total_trades > 0 else 0
            
            pnl_values = [t['pnl'] for t in trade_pnls]
            metrics.best_trade = max(pnl_values) if pnl_values else 0
            metrics.worst_trade = min(pnl_values) if pnl_values else 0
            metrics.avg_trade_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0
            
            # Daily metrics
            today_trades = [t for t in trade_pnls if t.get('timestamp') and t['timestamp'] >= today_start]
            metrics.trades_today = len(today_trades)
            metrics.pnl_today = sum(t['pnl'] for t in today_trades)
            metrics.wins_today = sum(1 for t in today_trades if t['pnl'] > 0)
            metrics.losses_today = sum(1 for t in today_trades if t['pnl'] <= 0)
            
            # Weekly metrics
            week_trades = [t for t in trade_pnls if t.get('timestamp') and t['timestamp'] >= week_ago]
            metrics.trades_7d = len(week_trades)
            metrics.pnl_7d = sum(t['pnl'] for t in week_trades)
            metrics.wins_7d = sum(1 for t in week_trades if t['pnl'] > 0)
            metrics.losses_7d = sum(1 for t in week_trades if t['pnl'] <= 0)
            metrics.win_rate_7d = metrics.wins_7d / metrics.trades_7d if metrics.trades_7d > 0 else 0
        
        self.logger.info(
            f"📊 P&L Summary: Today ${metrics.pnl_today:+.2f} ({metrics.trades_today} trades), "
            f"7d ${metrics.pnl_7d:+.2f} ({metrics.trades_7d} trades), "
            f"Total ${metrics.total_pnl:+.2f} ({metrics.total_trades} trades)"
        )
        
        return metrics
    
    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """Parse timestamp string to datetime."""
        if not ts_str:
            return None
        try:
            # Handle ISO format
            return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except:
            try:
                # Try alternate formats
                return datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S.%fZ')
            except:
                return None
    
    async def sync_fills_to_database(self) -> int:
        """
        Sync Kalshi fills to local database for historical tracking.
        
        Returns:
            Number of new fills synced
        """
        try:
            fills = await self._fetch_kalshi_fills()
            synced = 0
            
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                # Ensure fills table exists
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS kalshi_fills (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fill_id TEXT UNIQUE,
                        ticker TEXT,
                        side TEXT,
                        action TEXT,
                        count INTEGER,
                        yes_price INTEGER,
                        no_price INTEGER,
                        created_time TEXT,
                        order_id TEXT,
                        synced_at TEXT
                    )
                """)
                
                for fill in fills:
                    fill_id = fill.get('fill_id', str(hash(str(fill))))
                    try:
                        await db.execute("""
                            INSERT OR IGNORE INTO kalshi_fills 
                            (fill_id, ticker, side, action, count, yes_price, no_price, created_time, order_id, synced_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            fill_id,
                            fill.get('ticker', ''),
                            fill.get('side', ''),
                            fill.get('action', ''),
                            fill.get('count', 0),
                            fill.get('yes_price', 0),
                            fill.get('no_price', 0),
                            fill.get('created_time', ''),
                            fill.get('order_id', ''),
                            datetime.now().isoformat()
                        ))
                        synced += 1
                    except Exception:
                        continue
                
                await db.commit()
            
            self.logger.info(f"📥 Synced {synced} fills to database")
            return synced
            
        except Exception as e:
            self.logger.error(f"Error syncing fills: {e}")
            return 0
