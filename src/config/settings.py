"""
Configuration settings for the Kalshi trading system.
Manages trading parameters, API configurations, and risk management settings.
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")

# Load environment variables
load_dotenv()


@dataclass
class APIConfig:
    """API configuration settings."""
    kalshi_api_key: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    kalshi_base_url: str = field(default_factory=lambda: os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    xai_api_key: str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.cometapi.com/v1"))


# Trading strategy configuration - INCREASED AGGRESSIVENESS
@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    # Position sizing and risk management - SNIPER MODE (Concentrated High Conviction)
    max_position_size_pct: float = 5.0  # INCREASED: Max 5% for high conviction
    max_daily_loss_pct: float = 8.0    # Max 8% daily loss limit
    max_positions: int = 4              # REDUCED: Max 4 concurrent positions (Concentrated)
    min_balance: float = 25.0           # REDUCED: Lower minimum to trade more (was 100)
    live_trading_enabled: bool = field(default_factory=lambda: _env_flag("LIVE_TRADING_ENABLED", "false"))
    
    # Market filtering criteria - STRICT TIME LIMITS
    min_volume: float = 750.0            # Minimum volume to consider market
    max_time_to_expiry_days: int = 1     # STRICT: Only trade markets expiring within 24 hours
    max_market_expiry_hours: int = 24    # NEW: Maximum market expiry in hours for clarity
    
    # AI decision making - SNIPER MODE (High Conviction)
    min_confidence_to_trade: float = 0.65   # INCREASED: 65% minimum AI confidence to trade (Sniper Mode)
    scan_interval_seconds: int = 300      # INCREASED: Scan markets every 5 minutes (300s) to avoid rate limits
    
    # AI model configuration
    primary_model: str = "grok-4-1-fast-reasoning" # xAI fast reasoning model
    fallback_model: str = "grok-3"  # Fallback to available model
    ai_temperature: float = 0  # Lower temperature for more consistent JSON output
    ai_max_tokens: int = 8000    # Reasonable limit for reasoning models (grok-4 works better with 8000)
    
    # Enhanced AI Client settings
    ai_timeout: float = 45.0
    ai_max_retries: int = 3
    xai_models: List[str] = field(default_factory=lambda: ["grok-4.1", "grok-3"])
    openai_models: List[str] = field(default_factory=lambda: ["gpt-5.2", "gpt-5"])
    
    # Position sizing (LEGACY - now using Kelly-primary approach)
    default_position_size: float = 3.0  # REDUCED: Now using Kelly Criterion as primary method (was 5%, now 3%)
    position_size_multiplier: float = 1.0  # Multiplier for AI confidence
    
    # Kelly Criterion settings (PRIMARY position sizing method) - MORE AGGRESSIVE
    use_kelly_criterion: bool = True        # Use Kelly Criterion for position sizing (PRIMARY METHOD)
    kelly_fraction: float = 0.55           # 55% Kelly fraction (balanced aggressiveness)
    max_single_position: float = 0.04       # Max 4% in single position
    kalshi_fee_rate: float = 0.01          # Estimated fee rate per contract
    expected_slippage: float = 0.005       # Slippage haircut on edge/returns
    high_volatility_kelly_cap: float = 0.5  # Cap Kelly fraction in high-volatility regimes

    # Trading frequency - SNIPER MODE (Slower, More Deliberate)
    market_scan_interval: int = 300      # 300 second scan interval
    position_check_interval: int = 60       # Check positions every 60 seconds
    max_trades_per_hour: int = 4           # Max 4 trades per hour
    run_interval_minutes: int = 30          # INCREASED: Run less frequently (30m) to save costs/rates
    num_processor_workers: int = 5      # Number of concurrent market processor workers
    
    # Strategy allocations (should sum to 1.0)
    market_making_allocation: float = 0.30    # 30% for market making
    directional_allocation: float = 0.40     # 40% for directional trading  
    quick_flip_allocation: float = 0.30      # 30% for quick flip scalping
    arbitrage_allocation: float = 0.00       # Reserved for future
    
    # Market selection preferences
    preferred_categories: List[str] = field(default_factory=lambda: [])
    excluded_categories: List[str] = field(default_factory=lambda: [])
    category_blacklist: List[str] = field(default_factory=lambda: [])
    
    # High-confidence, near-expiry strategy
    enable_high_confidence_strategy: bool = True
    high_confidence_threshold: float = 0.90  # LLM confidence needed
    high_confidence_market_odds: float = 0.85 # Market price to look for
    high_confidence_expiry_hours: int = 18   # Max hours until expiry
    
    # Trailing stop loss settings
    trailing_stop_enabled: bool = True
    trailing_stop_distance_pct: float = 0.05  # 5% trailing distance
    trailing_stop_activation_pct: float = 0.03 # Activate trailing stop after 3% profit

    # AI trading criteria - SNIPER MODE
    max_analysis_cost_per_decision: float = 0.05  # REDUCED: $0.05 max per-decision cost (Efficient usage)
    min_confidence_threshold: float = 0.65  # INCREASED: Higher confidence threshold (sync with min_confidence_to_trade)

    # Cost control and market analysis frequency - SNIPER MODE
    daily_ai_budget: float = 3.50  # REDUCED: $3.50 daily AI budget (Target <$5)
    max_ai_cost_per_decision: float = 0.05  # REDUCED: Sync with above
    analysis_cooldown_hours: int = 12  # INCREASED: 12 hour cooldown (Analyze once per day)
    max_analyses_per_market_per_day: int = 1  # REDUCED: Max 1 analysis per market per day (One Shot)
    
    # Daily AI spending limits - SAFETY CONTROLS
    daily_ai_cost_limit: float = 4.50  # REDUCED: Hard cap at $4.50/day
    enable_daily_cost_limiting: bool = True  # Enable daily cost limits
    sleep_when_limit_reached: bool = True  # Sleep until next day when limit reached

    # Enhanced market filtering to reduce analyses - SNIPER MODE
    min_volume_for_analysis: float = 1000.0  # INCREASED: Only analyze liquid markets ($1000+)
    skip_news_for_low_volume: bool = True
    news_search_volume_threshold: float = 2500.0 # INCREASED: Only search news for very active markets
    exclude_low_liquidity_categories: List[str] = field(default_factory=lambda: [
        # REMOVED weather and entertainment - trade all categories
    ])

    # === VOLATILITY-ADJUSTED POSITION SIZING ===
    enable_volatility_sizing: bool = True
    volatility_baseline: float = 0.15
    volatility_multiplier_min: float = 0.5
    volatility_multiplier_max: float = 2.0

    # === THETA DECAY STRATEGY ===
    enable_theta_decay: bool = True
    theta_min_yes_price: float = 0.75
    theta_max_expiry_hours: int = 24
    theta_allocation: float = 0.10  # 10% of directional capital

    # === ML PRICE PREDICTIONS ===
    enable_ml_predictions: bool = True
    ml_lookback_hours: int = 168  # 7 days
    ml_confidence_threshold: float = 0.60

    # === EXECUTION QUALITY ===
    market_order_price_buffer_cents: int = 2  # Cap market orders at ask + buffer

    # === INTRADAY RISK THROTTLES ===
    loss_streak_pause_threshold: int = 8  # INCREASED: Allow more losses before pausing (was 3)
    loss_streak_pause_minutes: int = 30    # REDUCED: Shorter pause duration (was 60)
    volatility_pause_threshold: float = 0.35

    # === PERFORMANCE GATING ===
    min_strategy_win_rate: float = 0.45
    min_strategy_pnl: float = 0.0

    # === MARKET MAKING SETTINGS ===
    enable_market_making: bool = True
    min_spread_for_making: float = 0.01
    max_inventory_risk: float = 0.15
    max_inventory_skew: float = 500.0
    order_refresh_minutes: int = 15
    max_orders_per_market: int = 4
    max_bid_ask_spread: float = 0.15
    max_concurrent_markets: int = 10

    # === MARKET SELECTION (ADVANCED) ===
    min_volume_for_analysis: float = 1000.0 # Sync with above
    min_volume_for_market_making: float = 2000.0 # INCREASED: Safer market making
    min_price_movement: float = 0.0     # DISABLED: Allow 50/50 markets (was 0.02)
    min_confidence_long_term: float = 0.45

    # === PORTFOLIO OPTIMIZATION SETTINGS ===
    use_risk_parity: bool = True
    rebalance_hours: int = 6
    min_position_size: float = 5.0
    max_opportunities_per_batch: int = 50

    # === RISK MANAGEMENT LIMITS ===
    max_volatility: float = 0.80
    max_correlation: float = 0.95
    max_drawdown: float = 0.50
    max_sector_exposure: float = 0.90
    min_trade_edge: float = 0.08 # INCREASED: Require 8% edge
    min_confidence_for_large_size: float = 0.65 # INCREASED: Sync with min_confidence
    max_reduction_per_cycle: float = 0.30
    min_position_value: float = 5.0
    rebalance_threshold: float = 0.10

    # === PERFORMANCE TARGETS ===
    target_sharpe: float = 0.3
    target_return: float = 0.15

    # === DYNAMIC EXIT STRATEGIES ===
    use_dynamic_exits: bool = True
    profit_threshold: float = 0.15
    loss_threshold: float = 0.08
    confidence_decay_threshold: float = 0.25
    max_hold_time_hours: int = 24         # STRICT: Max 24 hours hold time
    volatility_adjustment: bool = True

    # === POSITION LIMITS ===
    warning_positions_threshold: int = 12
    emergency_position_limit: int = 20
    min_cash_reserve_pct: float = 0.5
    max_position_size_pct_override: Optional[float] = None

    # === CASH RESERVES ===
    minimum_reserve_pct: float = 0.5
    optimal_reserve_pct: float = 1.0
    emergency_threshold_pct: float = 0.2
    critical_threshold_pct: float = 0.05
    max_single_trade_impact: float = 5.0
    buffer_for_opportunities: float = 0.5

    # === SYSTEM BEHAVIOR ===
    beast_mode_enabled: bool = True
    fallback_to_legacy: bool = True
    paper_trading_mode: bool = False
    log_level: str = "INFO"
    performance_monitoring: bool = field(default_factory=lambda: _env_flag("ENABLE_PERFORMANCE_MONITORING", "true"))

    # === ADVANCED FEATURES ===
    cross_market_arbitrage: bool = False
    sentiment_analysis: bool = True
    options_strategies: bool = False
    algorithmic_execution: bool = False

    # === ENSEMBLE CONFIGURATION ===
    # Advanced ensemble settings for multi-model AI integration
    enable_advanced_ensemble: bool = True               # Enable advanced ensemble engine
    ensemble_consensus_threshold: float = 0.7           # Minimum consensus for ensemble agreement
    ensemble_disagreement_threshold: float = 0.4         # Threshold for detecting model disagreement
    ensemble_uncertainty_threshold: float = 0.6         # Uncertainty threshold for risk adjustment
    ensemble_enable_weighted_voting: bool = True        # Use performance-based weighted voting
    ensemble_enable_confidence_calibration: bool = True # Calibrate model confidence levels
    ensemble_performance_weight_factor: float = 2.0     # Weight factor for performance in voting
    ensemble_max_models_per_decision: int = 3           # Maximum models to consult per decision
    ensemble_timeout_seconds: int = 30                 # Timeout for ensemble decision making

    # Cascading ensemble thresholds based on trade value
    ensemble_cascading_low_value_threshold: float = 10.0  # Trade value for single model
    ensemble_cascading_medium_value_threshold: float = 50.0  # Trade value for full ensemble
    ensemble_cascading_high_value_threshold: float = 100.0   # Trade value for enhanced consensus

    # Ensemble model configuration
    ensemble_primary_models: List[str] = field(default_factory=lambda: ["grok-4"])
    ensemble_fallback_models: List[str] = field(default_factory=lambda: ["grok-3"])
    ensemble_backup_providers: List[str] = field(default_factory=lambda: ["openai-gpt-4"])
    ensemble_enable_local_models: bool = False          # Enable local model fallbacks

    # Performance and cost optimization
    ensemble_cache_decisions: bool = True               # Cache ensemble decisions for similar markets
    ensemble_cache_similarity_threshold: float = 0.8     # Similarity threshold for cache reuse
    ensemble_cache_ttl_hours: int = 2                   # Cache time-to-live in hours
    ensemble_cost_optimization_enabled: bool = True     # Enable cost-aware model selection
    ensemble_budget_aware_selection: bool = True        # Use budget constraints in model selection

    # Health monitoring and failover
    ensemble_health_check_interval_seconds: int = 300    # Health check frequency (5 minutes)
    ensemble_max_consecutive_failures: int = 3           # Max failures before model deselection
    ensemble_auto_recovery_enabled: bool = True          # Enable automatic model recovery
    ensemble_emergency_mode_enabled: bool = True         # Enable emergency trading mode

    # Monitoring and analytics
    ensemble_enable_performance_tracking: bool = True   # Track ensemble performance metrics
    ensemble_enable_disagreement_logging: bool = True    # Log model disagreements for analysis
    ensemble_enable_uncertainty_quantification: bool = True # Quantify decision uncertainty
    ensemble_enable_contribution_analysis: bool = True   # Track individual model contributions

    # === CIRCUIT BREAKER SETTINGS (Production Safety) ===
    circuit_breaker_enabled: bool = True  # Master switch for circuit breaker
    circuit_breaker_hourly_loss_pct: float = 0.05  # 5% hourly loss triggers global pause
    circuit_breaker_require_manual_reset: bool = True  # Require operator to reset after trip
    circuit_breaker_cooldown_minutes: int = 60  # Hourly window duration

    # === EXPIRATION RISK MANAGEMENT (Production Safety) ===
    auto_exit_expiring_enabled: bool = True  # Auto-close positions near expiry
    auto_exit_minutes_before_expiry: int = 30  # Close positions 30 minutes before market close

    # === DUAL-AI CONFIGURATION (Grok Forecaster + GPT Critic) ===
    # Two-stage AI decision system: Grok researches/predicts, GPT validates before trading
    enable_dual_ai_mode: bool = True                    # Enable Grok forecaster + GPT critic system
    dual_ai_min_trade_value: float = 5.0                # Minimum trade value to use dual-AI ($5)
    dual_ai_critic_must_approve: bool = True            # Critic must approve for trade to execute
    dual_ai_max_cost_per_decision: float = 0.12         # Max combined cost for dual-AI analysis
    dual_ai_min_agreement_score: float = 0.6            # Minimum critic agreement score to approve
    dual_ai_skip_critic_for_skip: bool = True           # Skip GPT critic if Grok says SKIP (saves cost)
    dual_ai_use_for_high_stakes_only: bool = False      # Only use dual-AI for high-value trades


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "DEBUG"))
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "logs/trading_system.log"))
    enable_file_logging: bool = field(default_factory=lambda: _env_flag("ENABLE_FILE_LOGGING", "true"))
    enable_console_logging: bool = field(default_factory=lambda: _env_flag("ENABLE_CONSOLE_LOGGING", "true"))
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


# BEAST MODE UNIFIED TRADING SYSTEM CONFIGURATION 🚀
# These settings control the advanced multi-strategy trading system

# === CAPITAL ALLOCATION ACROSS STRATEGIES ===
# Allocate capital across different trading approaches
market_making_allocation: float = 0.30  # 30% for market making (spread profits)
directional_allocation: float = 0.40    # 40% for directional trading (AI predictions) 
arbitrage_allocation: float = 0.00      # 0% for arbitrage opportunities (disabled by default)

  # === PORTFOLIO OPTIMIZATION SETTINGS ===
# Kelly Criterion is now the PRIMARY position sizing method (moved to TradingConfig)
# total_capital: DYNAMICALLY FETCHED from Kalshi balance - never hardcoded!
use_risk_parity: bool = True            # Equal risk allocation vs equal capital
rebalance_hours: int = 6                # Rebalance portfolio every 6 hours
min_position_size: float = 5.0          # Minimum position size ($5 vs $10)
max_opportunities_per_batch: int = 50   # Limit opportunities to prevent optimization issues

# === RISK MANAGEMENT LIMITS ===
# Portfolio-level risk constraints (EXTREMELY RELAXED FOR TESTING)
max_volatility: float = 0.80            # Very high volatility allowed (80%)
max_correlation: float = 0.95           # Very high correlation allowed (95%)
max_drawdown: float = 0.50              # High drawdown tolerance (50%)
max_sector_exposure: float = 0.90       # Very high sector concentration (90%)

# === PERFORMANCE TARGETS ===
# System performance objectives - MORE AGGRESSIVE FOR MORE TRADES
target_sharpe: float = 0.3              # DECREASED: Lower Sharpe requirement (was 0.5, now 0.3)
target_return: float = 0.15             # INCREASED: Higher return target (was 0.10, now 0.15)
min_trade_edge: float = 0.05           # DECREASED: Lower edge requirement (was 0.15, now 5%)
min_confidence_for_large_size: float = 0.65  # DECREASED: Lower confidence requirement (was 0.65, now 50%)

# === DYNAMIC EXIT STRATEGIES ===
# Enhanced exit strategy settings - MORE AGGRESSIVE
use_dynamic_exits: bool = True
profit_threshold: float = 0.15          # DECREASED: Take profits sooner (was 0.25, now 0.20)
loss_threshold: float = 0.08            # INCREASED: Allow larger losses (was 0.10, now 0.15)
confidence_decay_threshold: float = 0.25  # INCREASED: Allow more confidence decay (was 0.20, now 0.25)
max_hold_time_hours: int = 24          # STRICT: Max 24 hours hold time to limit exposure
volatility_adjustment: bool = True      # Adjust exits based on volatility

# === MARKET MAKING STRATEGY ===
# Settings for limit order market making - MORE AGGRESSIVE
enable_market_making: bool = True        # Enable market making strategy
min_spread_for_making: float = 0.01     # DECREASED: Accept smaller spreads (was 0.02, now 1¢)
max_inventory_risk: float = 0.15        # INCREASED: Allow higher inventory risk (was 0.10, now 15%)
order_refresh_minutes: int = 15         # Refresh orders every 15 minutes
max_orders_per_market: int = 4          # Maximum orders per market (2 each side)

# === MARKET SELECTION (ENHANCED FOR MORE OPPORTUNITIES) ===
# Removed time restrictions - trade ANY deadline with dynamic exits!
# max_time_to_expiry_days: REMOVED      # No longer used - trade any timeline!
min_volume_for_analysis: float = 200.0  # DECREASED: Much lower minimum volume (was 1000, now 200)
min_volume_for_market_making: float = 500.0  # DECREASED: Lower volume for market making (was 2000, now 500)
min_price_movement: float = 0.0        # DISABLED: Allow 50/50 markets (was 2%)
max_bid_ask_spread: float = 0.15        # INCREASED: Allow wider spreads (was 0.10, now 15¢)
min_confidence_long_term: float = 0.45  # DECREASED: Lower confidence for distant expiries (was 0.65, now 45%)

# === COST OPTIMIZATION (SNIPER MODE) ===
# Enhanced cost controls for the beast mode system
daily_ai_budget: float = 3.50           # REDUCED: Sniper mode budget
max_ai_cost_per_decision: float = 0.05  # REDUCED: Efficient usage
analysis_cooldown_hours: int = 12        # INCREASED: Analyze once per day
max_analyses_per_market_per_day: int = 1 # REDUCED: One shot per day
skip_news_for_low_volume: bool = True   # Skip expensive searches for low volume
news_search_volume_threshold: float = 2500.0  # News threshold

# === SYSTEM BEHAVIOR ===
# Overall system behavior settings
beast_mode_enabled: bool = True          # Enable the unified advanced system
fallback_to_legacy: bool = True         # Fallback to legacy system if needed
live_trading_enabled: bool = True       # Set to True for live trading
paper_trading_mode: bool = False        # Paper trading for testing
log_level: str = "INFO"                 # Logging level
performance_monitoring: bool = True     # Enable performance monitoring

# === ADVANCED FEATURES ===
# Cutting-edge features for maximum performance
cross_market_arbitrage: bool = False    # Enable when arbitrage module ready
multi_model_ensemble: bool = True       # Use Grok-4 + Grok-3 consensus for high-stakes trades
sentiment_analysis: bool = True         # News sentiment analysis enabled
options_strategies: bool = False        # Complex options strategies (future)
algorithmic_execution: bool = False     # Smart order execution (future)


@dataclass
class Settings:
    """Main settings class combining all configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Advanced feature flags (referenced from module-level settings above)
    multi_model_ensemble: bool = True    # Use Grok-4 + Grok-3 consensus for high-stakes trades
    sentiment_analysis: bool = True      # News sentiment analysis enabled

    def __post_init__(self) -> None:
        """Sync module-level settings with TradingConfig to avoid drift."""
        module_to_trading = [
            "market_making_allocation",
            "directional_allocation",
            "arbitrage_allocation",
            "use_risk_parity",
            "rebalance_hours",
            "min_position_size",
            "max_opportunities_per_batch",
            "max_volatility",
            "max_correlation",
            "max_drawdown",
            "max_sector_exposure",
            "target_sharpe",
            "target_return",
            "min_trade_edge",
            "min_confidence_for_large_size",
            "use_dynamic_exits",
            "profit_threshold",
            "loss_threshold",
            "confidence_decay_threshold",
            "max_hold_time_hours",
            "volatility_adjustment",
            "enable_market_making",
            "min_spread_for_making",
            "max_inventory_risk",
            "order_refresh_minutes",
            "max_orders_per_market",
            "min_volume_for_analysis",
            "min_volume_for_market_making",
            "min_price_movement",
            "max_bid_ask_spread",
            "min_confidence_long_term",
            "daily_ai_budget",
            "max_ai_cost_per_decision",
            "analysis_cooldown_hours",
            "max_analyses_per_market_per_day",
            "skip_news_for_low_volume",
            "news_search_volume_threshold",
            "beast_mode_enabled",
            "fallback_to_legacy",
            "live_trading_enabled",
            "paper_trading_mode",
            "log_level",
            "performance_monitoring",
            "cross_market_arbitrage",
            "multi_model_ensemble",
            "sentiment_analysis",
            "options_strategies",
            "algorithmic_execution",
        ]

        for name in module_to_trading:
            if hasattr(self.trading, name):
                globals()[name] = getattr(self.trading, name)
                if hasattr(self, name):
                    setattr(self, name, getattr(self.trading, name))
            elif hasattr(self, name):
                globals()[name] = getattr(self, name)
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.api.kalshi_api_key:
            raise ValueError("KALSHI_API_KEY environment variable is required")
        
        if not self.api.xai_api_key:
            raise ValueError("XAI_API_KEY environment variable is required")
        
        if self.trading.max_position_size_pct <= 0 or self.trading.max_position_size_pct > 100:
            raise ValueError("max_position_size_pct must be between 0 and 100")
        
        if self.trading.min_confidence_to_trade <= 0 or self.trading.min_confidence_to_trade > 1:
            raise ValueError("min_confidence_to_trade must be between 0 and 1")
        
        return True

    def get_ai_daily_limit(self) -> float:
        """Unified daily AI spend limit."""
        if not self.trading.enable_daily_cost_limiting:
            return self.trading.daily_ai_budget
        return min(self.trading.daily_ai_budget, self.trading.daily_ai_cost_limit)


# Global settings instance
settings = Settings()

# Validate settings on import
try:
    settings.validate()
except ValueError as e:
    print(f"Configuration validation error: {e}")
    print("Please check your environment variables and configuration.") 
