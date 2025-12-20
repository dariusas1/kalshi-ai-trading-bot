"""
Database manager for the Kalshi trading system.
"""

import aiosqlite
from dataclasses import dataclass, asdict
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from src.utils.logging_setup import TradingLoggerMixin


@dataclass
class Market:
    """Represents a market in the database."""
    market_id: str
    title: str
    yes_price: float
    no_price: float
    volume: int
    expiration_ts: int
    category: str
    status: str
    last_updated: datetime
    has_position: bool = False

@dataclass
class Position:
    """Represents a trading position."""
    market_id: str
    side: str  # "YES" or "NO"
    entry_price: float
    quantity: int
    timestamp: datetime
    rationale: Optional[str] = None
    confidence: Optional[float] = None
    live: bool = False
    status: str = "open"  # open, closed, pending
    id: Optional[int] = None
    strategy: Optional[str] = None  # Strategy that created this position
    
    # Enhanced exit strategy fields
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    max_hold_hours: Optional[int] = None  # Maximum hours to hold position
    target_confidence_change: Optional[float] = None  # Exit if confidence drops by this amount
    trailing_stop_price: Optional[float] = None  # Price for trailing stop loss

@dataclass
class TradeLog:
    """Represents a closed trade for logging and analysis."""
    market_id: str
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    entry_timestamp: datetime
    exit_timestamp: datetime
    rationale: str
    strategy: Optional[str] = None  # Strategy that created this trade
    id: Optional[int] = None

@dataclass
class LLMQuery:
    """Represents an LLM query and response for analysis."""
    timestamp: datetime
    strategy: str  # Which strategy made the query
    query_type: str  # Type of query (market_analysis, movement_prediction, etc.)
    market_id: Optional[str]  # Market being analyzed (if applicable)
    prompt: str  # The prompt sent to LLM
    response: str  # LLM response
    tokens_used: Optional[int] = None  # Tokens consumed
    cost_usd: Optional[float] = None  # Cost in USD
    confidence_extracted: Optional[float] = None  # Confidence if extracted
    decision_extracted: Optional[str] = None  # Decision if extracted
    id: Optional[int] = None


class DatabaseManager(TradingLoggerMixin):
    """Manages database operations for the trading system."""

    def __init__(self, db_path: str = "trading_system.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.logger.info("Initializing database manager", db_path=db_path)

    async def initialize(self) -> None:
        """Initialize database schema and run migrations."""
        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for concurrent access
            await db.execute("PRAGMA journal_mode=WAL;")
            await self._create_tables(db)
            await self._run_migrations(db)
            await db.commit()
        self.logger.info("Database initialized successfully")

    async def _run_migrations(self, db: aiosqlite.Connection) -> None:
        """Run database migrations for schema updates."""
        try:
            # Migration 1: Add strategy column to positions table
            cursor = await db.execute("PRAGMA table_info(positions)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if 'strategy' not in column_names:
                self.logger.info("Adding strategy column to positions table")
                await db.execute("ALTER TABLE positions ADD COLUMN strategy TEXT")
            
            # Migration 2: Add strategy column to trade_logs table
            cursor = await db.execute("PRAGMA table_info(trade_logs)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if 'strategy' not in column_names:
                self.logger.info("Adding strategy column to trade_logs table")
                await db.execute("ALTER TABLE trade_logs ADD COLUMN strategy TEXT")
            
            # Migration 3: Add LLM queries table if it doesn't exist
            cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_queries'")
            table_exists = await cursor.fetchone()
            
            if not table_exists:
                self.logger.info("Creating llm_queries table")
                await db.execute("""
                    CREATE TABLE llm_queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        query_type TEXT NOT NULL,
                        market_id TEXT,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        tokens_used INTEGER,
                        cost_usd REAL,
                        confidence_extracted REAL,
                        decision_extracted TEXT
                    )
                """)
                
            # Migration 4: Add missing columns for enhanced exit strategy
            if 'stop_loss_price' not in column_names:
                await db.execute("ALTER TABLE positions ADD COLUMN stop_loss_price REAL")
                self.logger.info("Added stop_loss_price column to positions table")
            if 'take_profit_price' not in column_names:
                await db.execute("ALTER TABLE positions ADD COLUMN take_profit_price REAL")
                self.logger.info("Added take_profit_price column to positions table")
            if 'max_hold_hours' not in column_names:
                await db.execute("ALTER TABLE positions ADD COLUMN max_hold_hours INTEGER")
                self.logger.info("Added max_hold_hours column to positions table")
            if 'target_confidence_change' not in column_names:
                await db.execute("ALTER TABLE positions ADD COLUMN target_confidence_change REAL")
                self.logger.info("Added target_confidence_change column to positions table")
            if 'trailing_stop_price' not in column_names:
                await db.execute("ALTER TABLE positions ADD COLUMN trailing_stop_price REAL")
                self.logger.info("Added trailing_stop_price column to positions table")

            # Migration 5: Update existing positions with strategy based on rationale
            await self._migrate_existing_strategy_data(db)
            
        except Exception as e:
            self.logger.error(f"Error running migrations: {e}")

    async def _migrate_existing_strategy_data(self, db: aiosqlite.Connection) -> None:
        """Migrate existing position data to include strategy information."""
        try:
            # Update positions based on rationale patterns
            await db.execute("""
                UPDATE positions 
                SET strategy = 'quick_flip_scalping' 
                WHERE strategy IS NULL AND rationale LIKE 'QUICK FLIP:%'
            """)
            
            await db.execute("""
                UPDATE positions 
                SET strategy = 'portfolio_optimization' 
                WHERE strategy IS NULL AND rationale LIKE 'Portfolio optimization allocation:%'
            """)
            
            await db.execute("""
                UPDATE positions 
                SET strategy = 'market_making' 
                WHERE strategy IS NULL AND (
                    rationale LIKE '%market making%' OR 
                    rationale LIKE '%spread profit%'
                )
            """)
            
            await db.execute("""
                UPDATE positions 
                SET strategy = 'directional_trading' 
                WHERE strategy IS NULL AND (
                    rationale LIKE 'High-confidence%' OR
                    rationale LIKE '%near-expiry%' OR
                    rationale LIKE '%decision%'
                )
            """)
            
            # Update trade_logs similarly
            await db.execute("""
                UPDATE trade_logs 
                SET strategy = 'quick_flip_scalping' 
                WHERE strategy IS NULL AND rationale LIKE 'QUICK FLIP:%'
            """)
            
            await db.execute("""
                UPDATE trade_logs 
                SET strategy = 'portfolio_optimization' 
                WHERE strategy IS NULL AND rationale LIKE 'Portfolio optimization allocation:%'
            """)
            
            await db.execute("""
                UPDATE trade_logs 
                SET strategy = 'market_making' 
                WHERE strategy IS NULL AND (
                    rationale LIKE '%market making%' OR 
                    rationale LIKE '%spread profit%'
                )
            """)
            
            await db.execute("""
                UPDATE trade_logs 
                SET strategy = 'directional_trading' 
                WHERE strategy IS NULL AND (
                    rationale LIKE 'High-confidence%' OR
                    rationale LIKE '%near-expiry%' OR
                    rationale LIKE '%decision%'
                )
            """)
            
            self.logger.info("Migrated existing position/trade data with strategy information")
            
        except Exception as e:
            self.logger.error(f"Error migrating existing strategy data: {e}")

    async def _create_tables(self, db: aiosqlite.Connection) -> None:
        """Create all database tables."""
        
        await db.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                market_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                yes_price REAL NOT NULL,
                no_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                expiration_ts INTEGER NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                has_position BOOLEAN NOT NULL DEFAULT 0
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                rationale TEXT,
                confidence REAL,
                live BOOLEAN NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'open',
                strategy TEXT,
                stop_loss_price REAL,
                take_profit_price REAL,
                max_hold_hours INTEGER,
                target_confidence_change REAL,
                trailing_stop_price REAL,
                UNIQUE(market_id, side)
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS trade_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                pnl REAL NOT NULL,
                entry_timestamp TEXT NOT NULL,
                exit_timestamp TEXT NOT NULL,
                rationale TEXT,
                strategy TEXT
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS market_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                analysis_timestamp TEXT NOT NULL,
                decision_action TEXT NOT NULL,
                confidence REAL,
                cost_usd REAL NOT NULL,
                analysis_type TEXT NOT NULL DEFAULT 'standard'
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS daily_cost_tracking (
                date TEXT PRIMARY KEY,
                total_ai_cost REAL NOT NULL DEFAULT 0.0,
                analysis_count INTEGER NOT NULL DEFAULT 0,
                decision_count INTEGER NOT NULL DEFAULT 0
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS llm_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                query_type TEXT NOT NULL,
                market_id TEXT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                tokens_used INTEGER,
                cost_usd REAL,
                confidence_extracted REAL,
                decision_extracted TEXT
            )
        """)

        # Add analysis_reports table for performance tracking
        await db.execute("""
            CREATE TABLE IF NOT EXISTS analysis_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                health_score REAL NOT NULL,
                critical_issues INTEGER DEFAULT 0,
                warnings INTEGER DEFAULT 0,
                action_items INTEGER DEFAULT 0,
                report_file TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Add kalshi_fills table for historical tracking
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

        # Create indices for performance
        await db.execute("CREATE INDEX IF NOT EXISTS idx_market_analyses_market_id ON market_analyses(market_id)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_market_analyses_timestamp ON market_analyses(analysis_timestamp)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_daily_cost_date ON daily_cost_tracking(date)")
        
        # Add trading_settings table for dashboard controls
        await db.execute("""
            CREATE TABLE IF NOT EXISTS trading_settings (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        """)
        
        # Add analytics_cache table for pre-computed metrics
        await db.execute("""
            CREATE TABLE IF NOT EXISTS analytics_cache (
                key TEXT PRIMARY KEY,
                value TEXT,
                computed_at TEXT
            )
        """)
        
        # Run migrations to ensure schema is up to date
        await self._run_migrations(db)
        
        self.logger.info("Tables created or already exist.")

    async def upsert_markets(self, markets: List[Market]):
        """
        Upsert a list of markets into the database.
        
        Args:
            markets: A list of Market dataclass objects.
        """
        async with aiosqlite.connect(self.db_path) as db:
            # SQLite STRFTIME arguments needs to be a string
            # and asdict converts datetime to datetime object
            # so we need to convert it to string manually
            market_dicts = []
            for m in markets:
                market_dict = asdict(m)
                market_dict['last_updated'] = m.last_updated.isoformat()
                market_dicts.append(market_dict)

            await db.executemany("""
                INSERT INTO markets (market_id, title, yes_price, no_price, volume, expiration_ts, category, status, last_updated, has_position)
                VALUES (:market_id, :title, :yes_price, :no_price, :volume, :expiration_ts, :category, :status, :last_updated, :has_position)
                ON CONFLICT(market_id) DO UPDATE SET
                    title=excluded.title,
                    yes_price=excluded.yes_price,
                    no_price=excluded.no_price,
                    volume=excluded.volume,
                    expiration_ts=excluded.expiration_ts,
                    category=excluded.category,
                    status=excluded.status,
                    last_updated=excluded.last_updated,
                    has_position=excluded.has_position
            """, market_dicts)
            await db.commit()
            self.logger.info(f"Upserted {len(markets)} markets.")

    async def get_eligible_markets(self, volume_min: int, max_days_to_expiry: int) -> List[Market]:
        """
        Get markets that are eligible for trading.

        Args:
            volume_min: Minimum trading volume.
            max_days_to_expiry: Maximum days to expiration.
        
        Returns:
            A list of eligible markets.
        """
        now_ts = int(datetime.now().timestamp())
        max_expiry_ts = now_ts + (max_days_to_expiry * 24 * 60 * 60)

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM markets
                WHERE
                    volume >= ? AND
                    expiration_ts > ? AND
                    expiration_ts <= ? AND
                    status = 'active' AND
                    has_position = 0
            """, (volume_min, now_ts, max_expiry_ts))
            rows = await cursor.fetchall()
            
            markets = []
            for row in rows:
                market_dict = dict(row)
                market_dict['last_updated'] = datetime.fromisoformat(market_dict['last_updated'])
                markets.append(Market(**market_dict))
            return markets

    async def get_markets_with_positions(self) -> set[str]:
        """
        Returns a set of market IDs that have associated open positions.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT DISTINCT market_id FROM positions WHERE status IN ('open', 'pending')
            """)
            rows = await cursor.fetchall()
            return {row[0] for row in rows}

    async def is_position_opening_for_market(self, market_id: str) -> bool:
        """
        Checks if a position is currently being opened for a given market.
        This is to prevent race conditions where multiple workers try to open a position for the same market.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT market_id FROM positions WHERE market_id = ? AND status = 'pending' LIMIT 1
            """, (market_id,))
            row = await cursor.fetchone()
            return row is not None

    async def get_open_non_live_positions(self) -> List[Position]:
        """
        Get all positions that are open and not live.
        
        Returns:
            A list of Position objects.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM positions WHERE status = 'open' AND live = 0")
            rows = await cursor.fetchall()
            
            positions = []
            for row in rows:
                position_dict = dict(row)
                position_dict['timestamp'] = datetime.fromisoformat(position_dict['timestamp'])
                positions.append(Position(**position_dict))
            return positions

    async def get_open_live_positions(self) -> List[Position]:
        """
        Get all positions that are open and live.
        
        Returns:
            A list of Position objects.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM positions WHERE status = 'open' AND live = 1")
            rows = await cursor.fetchall()
            
            positions = []
            for row in rows:
                position_dict = dict(row)
                position_dict['timestamp'] = datetime.fromisoformat(position_dict['timestamp'])
                positions.append(Position(**position_dict))
            return positions

    async def update_position_status(self, position_id: int, status: str):
        """
        Updates the status of a position.

        Args:
            position_id: The id of the position to update.
            status: The new status ('closed', 'voided').
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT market_id FROM positions WHERE id = ?",
                (position_id,)
            )
            row = await cursor.fetchone()
            market_id = row[0] if row else None

            await db.execute(
                "UPDATE positions SET status = ? WHERE id = ?",
                (status, position_id)
            )

            if market_id and status != "open":
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM positions WHERE market_id = ? AND status = 'open'",
                    (market_id,)
                )
                open_count = (await cursor.fetchone())[0]
                if open_count == 0:
                    await db.execute(
                        "UPDATE markets SET has_position = 0 WHERE market_id = ?",
                        (market_id,)
                    )
            await db.commit()
            self.logger.info(f"Updated position {position_id} status to {status}.")

    async def close_position(self, position_id: int):
        """
        Mark a position as closed.
        
        Args:
            position_id: The id of the position to close.
        """
        await self.update_position_status(position_id, 'closed')

    async def get_position_by_market_id(self, market_id: str) -> Optional[Position]:
        """
        Get a position by market ID.
        
        Args:
            market_id: The ID of the market.
            
        Returns:
            A Position object if found, otherwise None.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM positions WHERE market_id = ? AND status = 'open' LIMIT 1", (market_id,))
            row = await cursor.fetchone()
            if row:
                position_dict = dict(row)
                position_dict['timestamp'] = datetime.fromisoformat(position_dict['timestamp'])
                return Position(**position_dict)
            return None

    async def get_position_by_market_and_side(self, market_id: str, side: str) -> Optional[Position]:
        """
        Get a position by market ID and side.
        
        Args:
            market_id: The ID of the market.
            side: The side of the position ('YES' or 'NO').

        Returns:
            A Position object if found, otherwise None.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM positions WHERE market_id = ? AND side = ? AND status = 'open'", 
                (market_id, side)
            )
            row = await cursor.fetchone()
            if row:
                position_dict = dict(row)
                position_dict['timestamp'] = datetime.fromisoformat(position_dict['timestamp'])
                return Position(**position_dict)
            return None

    async def add_trade_log(self, trade_log: TradeLog) -> None:
        """
        Add a trade log entry.
        
        Args:
            trade_log: The trade log to add.
        """
        trade_dict = asdict(trade_log)
        trade_dict['entry_timestamp'] = trade_log.entry_timestamp.isoformat()
        trade_dict['exit_timestamp'] = trade_log.exit_timestamp.isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO trade_logs (market_id, side, entry_price, exit_price, quantity, pnl, entry_timestamp, exit_timestamp, rationale, strategy)
                VALUES (:market_id, :side, :entry_price, :exit_price, :quantity, :pnl, :entry_timestamp, :exit_timestamp, :rationale, :strategy)
            """, trade_dict)
            await db.commit()
            self.logger.info(f"Added trade log for market {trade_log.market_id}.")

    async def get_performance_by_strategy(self) -> Dict[str, Dict]:
        """
        Get performance metrics broken down by strategy.
        
        Returns:
            Dictionary with strategy names as keys and performance metrics as values.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Check if strategy column exists in trade_logs
            cursor = await db.execute("PRAGMA table_info(trade_logs)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            has_strategy_in_trades = 'strategy' in column_names
            
            completed_stats = []
            
            if has_strategy_in_trades:
                # Get stats from completed trades (trade_logs)
                cursor = await db.execute("""
                    SELECT 
                        strategy,
                        COUNT(*) as trade_count,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade
                    FROM trade_logs 
                    WHERE strategy IS NOT NULL
                    GROUP BY strategy
                """)
                completed_stats = await cursor.fetchall()
            else:
                # If no strategy column, create a generic entry
                cursor = await db.execute("""
                    SELECT 
                        'legacy_trades' as strategy,
                        COUNT(*) as trade_count,
                        SUM(pnl) as total_pnl,
                        AVG(pnl) as avg_pnl,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade
                    FROM trade_logs
                """)
                result = await cursor.fetchone()
                if result and result['trade_count'] > 0:
                    completed_stats = [result]
            
            # Check if strategy column exists in positions
            cursor = await db.execute("PRAGMA table_info(positions)")
            columns = await cursor.fetchall()
            column_names = [col[1] for col in columns]
            has_strategy_in_positions = 'strategy' in column_names
            
            open_stats = []
            
            if has_strategy_in_positions:
                # Get current open positions by strategy
                cursor = await db.execute("""
                    SELECT 
                        strategy,
                        COUNT(*) as open_positions,
                        SUM(quantity * entry_price) as capital_deployed
                    FROM positions 
                    WHERE status = 'open' AND strategy IS NOT NULL
                    GROUP BY strategy
                """)
                open_stats = await cursor.fetchall()
            else:
                # If no strategy column, create a generic entry
                cursor = await db.execute("""
                    SELECT 
                        'legacy_positions' as strategy,
                        COUNT(*) as open_positions,
                        SUM(quantity * entry_price) as capital_deployed
                    FROM positions 
                    WHERE status = 'open'
                """)
                result = await cursor.fetchone()
                if result and result['open_positions'] > 0:
                    open_stats = [result]
            
            # Combine the results
            performance = {}
            
            # Add completed trade stats
            for row in completed_stats:
                strategy = row['strategy'] or 'unknown'
                win_rate = (row['winning_trades'] / row['trade_count']) * 100 if row['trade_count'] > 0 else 0
                
                performance[strategy] = {
                    'completed_trades': row['trade_count'],
                    'total_pnl': row['total_pnl'],
                    'avg_pnl_per_trade': row['avg_pnl'],
                    'win_rate_pct': win_rate,
                    'winning_trades': row['winning_trades'],
                    'losing_trades': row['losing_trades'],
                    'best_trade': row['best_trade'],
                    'worst_trade': row['worst_trade'],
                    'open_positions': 0,
                    'capital_deployed': 0.0
                }
            
            # Add open position stats
            for row in open_stats:
                strategy = row['strategy'] or 'unknown'
                if strategy not in performance:
                    performance[strategy] = {
                        'completed_trades': 0,
                        'total_pnl': 0.0,
                        'avg_pnl_per_trade': 0.0,
                        'win_rate_pct': 0.0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'best_trade': 0.0,
                        'worst_trade': 0.0,
                        'open_positions': 0,
                        'capital_deployed': 0.0
                    }
                
                performance[strategy]['open_positions'] = row['open_positions']
                performance[strategy]['capital_deployed'] = row['capital_deployed']
            
            return performance

    async def log_llm_query(self, llm_query: LLMQuery) -> None:
        """Log an LLM query and response for analysis."""
        try:
            query_dict = asdict(llm_query)
            query_dict['timestamp'] = llm_query.timestamp.isoformat()
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO llm_queries (
                        timestamp, strategy, query_type, market_id, prompt, response,
                        tokens_used, cost_usd, confidence_extracted, decision_extracted
                    ) VALUES (
                        :timestamp, :strategy, :query_type, :market_id, :prompt, :response,
                        :tokens_used, :cost_usd, :confidence_extracted, :decision_extracted
                    )
                """, query_dict)
                await db.commit()
                
        except Exception as e:
            self.logger.error(f"Error logging LLM query: {e}")

    async def get_llm_queries(
        self, 
        strategy: Optional[str] = None,
        hours_back: int = 24,
        limit: int = 100
    ) -> List[LLMQuery]:
        """Get recent LLM queries, optionally filtered by strategy."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Check if llm_queries table exists
                cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_queries'")
                table_exists = await cursor.fetchone()
                
                if not table_exists:
                    self.logger.info("LLM queries table doesn't exist yet - will be created on first query")
                    return []
                
                if strategy:
                    cursor = await db.execute("""
                        SELECT * FROM llm_queries 
                        WHERE strategy = ? AND timestamp >= ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (strategy, cutoff_time.isoformat(), limit))
                else:
                    cursor = await db.execute("""
                        SELECT * FROM llm_queries 
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC LIMIT ?
                    """, (cutoff_time.isoformat(), limit))
                
                rows = await cursor.fetchall()
                
                queries = []
                for row in rows:
                    query_dict = dict(row)
                    query_dict['timestamp'] = datetime.fromisoformat(query_dict['timestamp'])
                    queries.append(LLMQuery(**query_dict))
                
                return queries
                
        except Exception as e:
            self.logger.error(f"Error getting LLM queries: {e}")
            return []

    async def get_llm_stats_by_strategy(self) -> Dict[str, Dict]:
        """Get LLM usage statistics by strategy."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Check if llm_queries table exists
                cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='llm_queries'")
                table_exists = await cursor.fetchone()
                
                if not table_exists:
                    self.logger.info("LLM queries table doesn't exist yet - will be created on first query")
                    return {}
                
                cursor = await db.execute("""
                    SELECT 
                        strategy,
                        COUNT(*) as query_count,
                        SUM(tokens_used) as total_tokens,
                        SUM(cost_usd) as total_cost,
                        AVG(confidence_extracted) as avg_confidence,
                        MIN(timestamp) as first_query,
                        MAX(timestamp) as last_query
                    FROM llm_queries 
                    WHERE timestamp >= datetime('now', '-7 days')
                    GROUP BY strategy
                """)
                
                rows = await cursor.fetchall()
                
                stats = {}
                for row in rows:
                    stats[row['strategy']] = {
                        'query_count': row['query_count'],
                        'total_tokens': row['total_tokens'] or 0,
                        'total_cost': row['total_cost'] or 0.0,
                        'avg_confidence': row['avg_confidence'] or 0.0,
                        'first_query': row['first_query'],
                        'last_query': row['last_query']
                    }
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting LLM stats: {e}")
            return {}

    async def close(self):
        """Close database connections (no-op for aiosqlite)."""
        # aiosqlite doesn't require explicit closing of connections
        # since we use context managers, but we provide this method
        # for compatibility with other code that expects it
        pass

    async def record_market_analysis(
        self, 
        market_id: str, 
        decision_action: str, 
        confidence: float, 
        cost_usd: float,
        analysis_type: str = 'standard'
    ) -> None:
        """Record that a market was analyzed to prevent duplicate analysis."""
        now = datetime.now().isoformat()
        today = datetime.now().strftime('%Y-%m-%d')
        
        async with aiosqlite.connect(self.db_path) as db:
            # Record the analysis
            await db.execute("""
                INSERT INTO market_analyses (market_id, analysis_timestamp, decision_action, confidence, cost_usd, analysis_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (market_id, now, decision_action, confidence, cost_usd, analysis_type))
            
            # Update daily cost tracking
            await db.execute("""
                INSERT INTO daily_cost_tracking (date, total_ai_cost, analysis_count, decision_count)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(date) DO UPDATE SET
                    total_ai_cost = total_ai_cost + excluded.total_ai_cost,
                    analysis_count = analysis_count + 1,
                    decision_count = decision_count + excluded.decision_count
            """, (today, cost_usd, 1 if decision_action != 'SKIP' else 0))
            
            await db.commit()

    async def was_recently_analyzed(self, market_id: str, hours: int = 6) -> bool:
        """Check if market was analyzed within the specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        cutoff_str = cutoff_time.isoformat()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT COUNT(*) FROM market_analyses 
                WHERE market_id = ? AND analysis_timestamp > ?
            """, (market_id, cutoff_str))
            count = (await cursor.fetchone())[0]
            return count > 0

    async def get_daily_ai_cost(self, date: str = None) -> float:
        """Get total AI cost for a specific date (defaults to today)."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT total_ai_cost FROM daily_cost_tracking WHERE date = ?
            """, (date,))
            row = await cursor.fetchone()
            return row[0] if row else 0.0

    async def get_daily_ai_cost_breakdown(self, date: str = None) -> Dict[str, Any]:
        """Get detailed AI cost breakdown for a specific date."""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT total_ai_cost, analysis_count 
                FROM daily_cost_tracking 
                WHERE date = ?
            """, (date,))
            row = await cursor.fetchone()
            
            if row:
                total_cost = row['total_ai_cost']
                count = row['analysis_count']
                return {
                    'total_cost': total_cost,
                    'request_count': count,
                    'avg_cost': total_cost / count if count > 0 else 0.0
                }
            return {
                'total_cost': 0.0,
                'request_count': 0,
                'avg_cost': 0.0
            }

    async def get_market_analysis_count_today(self, market_id: str) -> int:
        """Get number of times market was analyzed today."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT COUNT(*) FROM market_analyses 
                WHERE market_id = ? AND DATE(analysis_timestamp) = ?
            """, (market_id, today))
            count = (await cursor.fetchone())[0]
            return count

    async def get_all_trade_logs(self) -> List[TradeLog]:
        """
        Get all trade logs from the database.
        
        Returns:
            A list of TradeLog objects.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM trade_logs")
            rows = await cursor.fetchall()
            
            logs = []
            for row in rows:
                log_dict = dict(row)
                log_dict['entry_timestamp'] = datetime.fromisoformat(log_dict['entry_timestamp'])
                log_dict['exit_timestamp'] = datetime.fromisoformat(log_dict['exit_timestamp'])
                logs.append(TradeLog(**log_dict))
            return logs

    async def update_position_to_live(self, position_id: int, entry_price: float):
        """
        Updates the status and entry price of a position after it has been executed.

        Args:
            position_id: The ID of the position to update.
            entry_price: The actual entry price from the exchange.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE positions 
                SET live = 1, entry_price = ?
                WHERE id = ?
            """, (entry_price, position_id))
            await db.commit()
        self.logger.info(f"Updated position {position_id} to live.")

    async def update_trailing_stop_price(self, position_id: int, trailing_stop_price: float):
        """
        Updates the trailing stop price for a position.

        Args:
            position_id: The ID of the position to update.
            trailing_stop_price: The new trailing stop price.
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE positions 
                SET trailing_stop_price = ?
                WHERE id = ?
            """, (trailing_stop_price, position_id))
            await db.commit()
        self.logger.info(f"Updated position {position_id} trailing stop to {trailing_stop_price:.4f}.")

    async def add_position(self, position: Position) -> Optional[int]:
        """
        Adds a new position to the database, if one doesn't already exist for the same market and side.
        
        Args:
            position: The position to add.
        
        Returns:
            The ID of the newly inserted position, or None if a position already exists.
        """
        existing_position = await self.get_position_by_market_and_side(position.market_id, position.side)
        if existing_position:
            self.logger.warning(f"Position already exists for market {position.market_id} and side {position.side}.")
            return None

        async with aiosqlite.connect(self.db_path) as db:
            position_dict = asdict(position)
            # aiosqlite does not support dataclasses with datetime objects
            position_dict['timestamp'] = position.timestamp.isoformat()

            cursor = await db.execute("""
                INSERT INTO positions (market_id, side, entry_price, quantity, timestamp, rationale, confidence, live, status, strategy, stop_loss_price, take_profit_price, max_hold_hours, target_confidence_change, trailing_stop_price)
                VALUES (:market_id, :side, :entry_price, :quantity, :timestamp, :rationale, :confidence, :live, :status, :strategy, :stop_loss_price, :take_profit_price, :max_hold_hours, :target_confidence_change, :trailing_stop_price)
            """, position_dict)
            await db.commit()
            
            # Set has_position to True for the market
            await db.execute("UPDATE markets SET has_position = 1 WHERE market_id = ?", (position.market_id,))
            await db.commit()

            self.logger.info(f"Added position for market {position.market_id}", position_id=cursor.lastrowid)
            return cursor.lastrowid

    async def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM positions WHERE status = 'open'"
            )
            rows = await cursor.fetchall()
            
            positions = []
            for row in rows:
                # Convert database row to Position object
                position = Position(
                    market_id=row[1],
                    side=row[2],
                    entry_price=row[3],
                    quantity=row[4],
                    timestamp=datetime.fromisoformat(row[5]),
                    rationale=row[6],
                    confidence=row[7],
                    live=bool(row[8]),
                    status=row[9],
                    id=row[0],
                    strategy=row[10],
                    stop_loss_price=row[11],
                    take_profit_price=row[12],
                    max_hold_hours=row[13],
                    target_confidence_change=row[14],
                    trailing_stop_price=row[15]
                )
                positions.append(position)
            
            return positions
    
    async def sync_with_kalshi(self, kalshi_client) -> Dict[str, int]:
        """
        Sync local database with actual Kalshi positions.
        
        - Closes stale local positions that don't exist on Kalshi
        - Logs discrepancies for debugging
        
        Args:
            kalshi_client: KalshiClient instance for API calls
            
        Returns:
            Dict with sync statistics
        """
        try:
            # Get positions from Kalshi
            response = await kalshi_client.get_positions()
            kalshi_positions = (
                response.get('market_positions')
                or response.get('positions')
                or []
            )
            
            # Build set of active Kalshi positions (tickers with non-zero quantity)
            kalshi_active = set()
            for pos in kalshi_positions:
                ticker = pos.get('ticker') or pos.get('market_id', '')
                position_qty = pos.get('position', pos.get('quantity', 0))
                if abs(position_qty) > 0:
                    kalshi_active.add(ticker)
            
            # Get local open positions
            local_positions = await self.get_open_positions()
            
            stale_closed = 0
            for pos in local_positions:
                if pos.market_id not in kalshi_active:
                    # This local position doesn't exist on Kalshi - close it
                    await self.update_position_status(pos.id, 'closed')
                    self.logger.info(f"🧹 Closed stale position: {pos.market_id}")
                    stale_closed += 1
            
            # Log sync summary
            self.logger.info(
                f"🔄 Kalshi sync complete: {len(kalshi_active)} active Kalshi positions, "
                f"{len(local_positions)} local positions, {stale_closed} stale closed"
            )
            
            return {
                'kalshi_positions': len(kalshi_active),
                'local_positions': len(local_positions),
                'stale_closed': stale_closed
            }
            
        except Exception as e:
            self.logger.error(f"Error syncing with Kalshi: {e}")
            return {'error': str(e)}

    # ==================== DASHBOARD ANALYTICS METHODS ====================

    async def get_pnl_by_period(self, period: str = 'today') -> Dict:
        """
        Get P&L metrics for a specific time period.
        
        Args:
            period: 'today', 'week', 'month', or 'all'
            
        Returns:
            Dict with pnl, trades, wins, losses, win_rate
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Build date filter based on period
            if period == 'today':
                date_filter = "DATE(exit_timestamp) = DATE('now')"
            elif period == 'week':
                date_filter = "DATE(exit_timestamp) >= DATE('now', '-7 days')"
            elif period == 'month':
                date_filter = "DATE(exit_timestamp) >= DATE('now', '-30 days')"
            else:  # 'all'
                date_filter = "1=1"
            
            cursor = await db.execute(f"""
                SELECT 
                    COUNT(*) as total_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 0) as wins,
                    COALESCE(SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END), 0) as losses,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    COALESCE(MAX(pnl), 0) as best_trade,
                    COALESCE(MIN(pnl), 0) as worst_trade
                FROM trade_logs
                WHERE {date_filter}
            """)
            row = await cursor.fetchone()
            
            if row and row['total_trades'] > 0:
                win_rate = (row['wins'] / row['total_trades']) * 100
            else:
                win_rate = 0.0
            
            return {
                'period': period,
                'total_pnl': row['total_pnl'] if row else 0.0,
                'total_trades': row['total_trades'] if row else 0,
                'wins': row['wins'] if row else 0,
                'losses': row['losses'] if row else 0,
                'win_rate': win_rate,
                'avg_pnl': row['avg_pnl'] if row else 0.0,
                'best_trade': row['best_trade'] if row else 0.0,
                'worst_trade': row['worst_trade'] if row else 0.0
            }

    async def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trade executions for the trade feed."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT 
                    market_id, side, entry_price, exit_price, quantity, 
                    pnl, exit_timestamp, strategy
                FROM trade_logs 
                ORDER BY exit_timestamp DESC 
                LIMIT ?
            """, (limit,))
            rows = await cursor.fetchall()
            
            return [dict(row) for row in rows]

    async def get_setting(self, key: str, default: str = None) -> Optional[str]:
        """Get a trading setting value."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT value FROM trading_settings WHERE key = ?", (key,)
            )
            row = await cursor.fetchone()
            return row[0] if row else default

    async def set_setting(self, key: str, value: str) -> None:
        """Set a trading setting value."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO trading_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = ?
            """, (key, value, datetime.now().isoformat(), value, datetime.now().isoformat()))
            await db.commit()

    async def get_category_performance(self) -> Dict[str, Dict]:
        """Get win rate and P&L broken down by market category."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Join trade_logs with markets to get category
            cursor = await db.execute("""
                SELECT 
                    m.category,
                    COUNT(*) as trades,
                    SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(t.pnl) as total_pnl,
                    AVG(t.pnl) as avg_pnl
                FROM trade_logs t
                LEFT JOIN markets m ON t.market_id = m.market_id
                WHERE m.category IS NOT NULL
                GROUP BY m.category
                ORDER BY SUM(t.pnl) DESC
            """)
            rows = await cursor.fetchall()
            
            result = {}
            for row in rows:
                category = row['category'] or 'unknown'
                trades = row['trades']
                wins = row['wins'] or 0
                result[category] = {
                    'trades': trades,
                    'wins': wins,
                    'losses': trades - wins,
                    'win_rate': (wins / trades * 100) if trades > 0 else 0,
                    'total_pnl': row['total_pnl'] or 0,
                    'avg_pnl': row['avg_pnl'] or 0
                }
            return result

    async def get_hourly_performance(self) -> Dict[int, Dict]:
        """Get win rate and P&L broken down by hour of day."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            cursor = await db.execute("""
                SELECT 
                    CAST(strftime('%H', exit_timestamp) AS INTEGER) as hour,
                    COUNT(*) as trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl
                FROM trade_logs
                GROUP BY hour
                ORDER BY hour
            """)
            rows = await cursor.fetchall()
            
            result = {}
            for row in rows:
                hour = row['hour']
                trades = row['trades']
                wins = row['wins'] or 0
                result[hour] = {
                    'trades': trades,
                    'wins': wins,
                    'win_rate': (wins / trades * 100) if trades > 0 else 0,
                    'total_pnl': row['total_pnl'] or 0,
                    'avg_pnl': row['avg_pnl'] or 0
                }
            return result

    async def get_expiry_performance(self) -> Dict[str, Dict]:
        """Get win rate and P&L broken down by time-to-expiry bucket."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            # Calculate hours to expiry at entry time
            cursor = await db.execute("""
                SELECT 
                    CASE 
                        WHEN (m.expiration_ts - strftime('%s', t.entry_timestamp)) / 3600.0 < 1 THEN '0-1h'
                        WHEN (m.expiration_ts - strftime('%s', t.entry_timestamp)) / 3600.0 < 6 THEN '1-6h'
                        WHEN (m.expiration_ts - strftime('%s', t.entry_timestamp)) / 3600.0 < 24 THEN '6-24h'
                        WHEN (m.expiration_ts - strftime('%s', t.entry_timestamp)) / 3600.0 < 168 THEN '1-7d'
                        ELSE '7d+'
                    END as expiry_bucket,
                    COUNT(*) as trades,
                    SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(t.pnl) as total_pnl,
                    AVG(t.pnl) as avg_pnl
                FROM trade_logs t
                LEFT JOIN markets m ON t.market_id = m.market_id
                WHERE m.expiration_ts IS NOT NULL
                GROUP BY expiry_bucket
            """)
            rows = await cursor.fetchall()
            
            result = {}
            for row in rows:
                bucket = row['expiry_bucket'] or 'unknown'
                trades = row['trades']
                wins = row['wins'] or 0
                result[bucket] = {
                    'trades': trades,
                    'wins': wins,
                    'win_rate': (wins / trades * 100) if trades > 0 else 0,
                    'total_pnl': row['total_pnl'] or 0,
                    'avg_pnl': row['avg_pnl'] or 0
                }
            return result

    async def get_confidence_calibration(self) -> List[Dict]:
        """
        Get confidence calibration data showing predicted vs actual win rates.
        
        Returns list of buckets: {range, avg_confidence, actual_win_rate, trades}
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            cursor = await db.execute("""
                SELECT 
                    CASE 
                        WHEN p.confidence < 0.6 THEN '50-60%'
                        WHEN p.confidence < 0.7 THEN '60-70%'
                        WHEN p.confidence < 0.8 THEN '70-80%'
                        WHEN p.confidence < 0.9 THEN '80-90%'
                        ELSE '90-100%'
                    END as bucket,
                    AVG(p.confidence) as avg_confidence,
                    COUNT(*) as trades,
                    SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as wins
                FROM positions p
                JOIN trade_logs t ON p.market_id = t.market_id AND p.side = t.side
                WHERE p.confidence IS NOT NULL
                GROUP BY bucket
                ORDER BY avg_confidence
            """)
            rows = await cursor.fetchall()
            
            result = []
            for row in rows:
                trades = row['trades']
                wins = row['wins'] or 0
                result.append({
                    'bucket': row['bucket'],
                    'avg_confidence': (row['avg_confidence'] or 0) * 100,
                    'actual_win_rate': (wins / trades * 100) if trades > 0 else 0,
                    'trades': trades
                })
            return result

    async def get_daily_pnl_history(self, days: int = 30) -> List[Dict]:
        """Get daily P&L for calendar view."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            
            cursor = await db.execute("""
                SELECT 
                    DATE(exit_timestamp) as date,
                    SUM(pnl) as pnl,
                    COUNT(*) as trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins
                FROM trade_logs
                WHERE DATE(exit_timestamp) >= DATE('now', ?)
                GROUP BY DATE(exit_timestamp)
                ORDER BY date DESC
            """, (f'-{days} days',))
            rows = await cursor.fetchall()
            
            return [dict(row) for row in rows]

    async def get_trading_streaks(self) -> Dict:
        """Get current and max win/loss streaks."""
        trades = await self.get_recent_trades(limit=500)
        
        if not trades:
            return {'current_streak': 0, 'streak_type': 'none', 
                    'max_win_streak': 0, 'max_loss_streak': 0}
        
        current_streak = 0
        streak_type = 'none'
        max_win_streak = 0
        max_loss_streak = 0
        temp_win = 0
        temp_loss = 0
        
        for trade in trades:
            pnl = trade.get('pnl', 0)
            
            if pnl > 0:
                temp_win += 1
                if temp_loss > max_loss_streak:
                    max_loss_streak = temp_loss
                temp_loss = 0
            else:
                temp_loss += 1
                if temp_win > max_win_streak:
                    max_win_streak = temp_win
                temp_win = 0
        
        # Check final streaks
        if temp_win > max_win_streak:
            max_win_streak = temp_win
        if temp_loss > max_loss_streak:
            max_loss_streak = temp_loss
        
        # Current streak from most recent trades
        if trades:
            first_pnl = trades[0].get('pnl', 0)
            streak_type = 'win' if first_pnl > 0 else 'loss'
            current_streak = 1
            for trade in trades[1:]:
                if (trade.get('pnl', 0) > 0) == (first_pnl > 0):
                    current_streak += 1
                else:
                    break
        
        return {
            'current_streak': current_streak,
            'streak_type': streak_type,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak
        }

    async def get_ai_cost_history(self, days: int = 30) -> List[Dict]:
        """
        Get daily AI cost history for the last N days.
        
        Returns:
            List of dicts with 'date' and 'cost'.
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT date, total_ai_cost as cost 
                FROM daily_cost_tracking 
                ORDER BY date DESC 
                LIMIT ?
            """, (days,))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_cached_analytics(self, key: str) -> Optional[Dict]:
        """
        Get cached analytics data by key.
        
        Returns:
            Dict of data or None if not found or expired (> 15 mins).
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT value, computed_at FROM analytics_cache WHERE key = ?
            """, (key,))
            row = await cursor.fetchone()
            
            if row:
                # Check expiry (optional, processor runs every 5 mins)
                computed_at = datetime.fromisoformat(row['computed_at'])
                if (datetime.now() - computed_at).total_seconds() < 900:  # 15 mins
                    return json.loads(row['value'])
            return None

    async def set_cached_analytics(self, key: str, value: Dict) -> None:
        """Set cached analytics data by key."""
        try:
            payload = json.dumps(value)
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO analytics_cache (key, value, computed_at) VALUES (?, ?, ?)",
                    (key, payload, datetime.now().isoformat())
                )
                await db.commit()
        except Exception as e:
            self.logger.error(f"Error setting analytics cache for {key}: {e}")
