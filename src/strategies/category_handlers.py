"""
Category-Specific Strategy Handlers

Specialized trading parameters for high-volume categories:
- Elections: Higher confidence threshold, longer holds
- Sports: Quick entries/exits, event-driven timing
- Crypto: Volatility-adjusted sizing, tighter stops
"""

from dataclasses import dataclass
from typing import Optional
from src.utils.logging_setup import get_trading_logger


@dataclass
class CategoryConfig:
    """Trading parameters for a specific category."""
    category: str
    confidence_threshold: float  # Minimum confidence to trade
    max_hold_hours: int  # Maximum hours to hold position
    stop_loss_pct: float  # Stop loss percentage
    take_profit_pct: float  # Take profit percentage
    position_size_multiplier: float  # Multiplier for position sizing
    urgency_factor: float  # 1.0 = normal, >1 = faster entry needed


# Category-specific configurations
CATEGORY_CONFIGS = {
    # Elections: High confidence needed, longer holds, predictable patterns
    "elections": CategoryConfig(
        category="elections",
        confidence_threshold=0.75,  # Higher threshold (default 0.65)
        max_hold_hours=48,  # Longer holds for election markets
        stop_loss_pct=0.12,  # Wider stops
        take_profit_pct=0.30,  # Higher profit target
        position_size_multiplier=1.2,  # Slightly larger positions
        urgency_factor=0.8  # Less urgency, wait for good entries
    ),
    
    # Sports: Event-driven, time-sensitive, quick exits
    "sports": CategoryConfig(
        category="sports",
        confidence_threshold=0.65,  # Standard threshold
        max_hold_hours=4,  # Quick exits (game duration)
        stop_loss_pct=0.08,  # Tighter stops
        take_profit_pct=0.20,  # Quick profits
        position_size_multiplier=0.8,  # Smaller positions (higher variance)
        urgency_factor=1.5  # High urgency for game-time entries
    ),
    
    # Crypto: High volatility, need tight risk management
    "crypto": CategoryConfig(
        category="crypto",
        confidence_threshold=0.70,  # Slightly higher for volatility
        max_hold_hours=12,  # Medium holds
        stop_loss_pct=0.05,  # Very tight stops (high volatility)
        take_profit_pct=0.25,  # Good-sized targets
        position_size_multiplier=0.6,  # Smaller positions (volatility)
        urgency_factor=1.3  # Fast-moving markets
    ),
    
    # Financial/Econ: Similar to elections but shorter timeframes
    "financial": CategoryConfig(
        category="financial",
        confidence_threshold=0.72,
        max_hold_hours=24,
        stop_loss_pct=0.10,
        take_profit_pct=0.25,
        position_size_multiplier=1.0,
        urgency_factor=1.0
    ),
    
    # Weather: Binary outcomes, specific timing
    "weather": CategoryConfig(
        category="weather",
        confidence_threshold=0.70,
        max_hold_hours=36,
        stop_loss_pct=0.10,
        take_profit_pct=0.25,
        position_size_multiplier=0.9,
        urgency_factor=0.9
    )
}

# Default config for unknown categories
DEFAULT_CONFIG = CategoryConfig(
    category="default",
    confidence_threshold=0.65,
    max_hold_hours=24,
    stop_loss_pct=0.10,
    take_profit_pct=0.25,
    position_size_multiplier=1.0,
    urgency_factor=1.0
)


class CategoryStrategyHandler:
    """
    Handler for category-specific trading parameters.
    
    Provides adjusted trading parameters based on market category
    to optimize for different market behaviors.
    """
    
    def __init__(self):
        self.logger = get_trading_logger("category_handler")
        self.configs = CATEGORY_CONFIGS
    
    def get_category_config(self, category: str) -> CategoryConfig:
        """
        Get trading configuration for a specific category.
        
        Args:
            category: Market category (e.g., "elections", "sports", "crypto")
            
        Returns:
            CategoryConfig with optimized parameters for the category
        """
        # Normalize category name
        category_lower = category.lower().strip()
        
        # Try to find matching config
        for key, config in self.configs.items():
            if key in category_lower or category_lower in key:
                self.logger.debug(f"Using {key} config for category: {category}")
                return config
        
        # Check for partial matches
        if "election" in category_lower or "politic" in category_lower or "vote" in category_lower:
            return self.configs["elections"]
        if "sport" in category_lower or "nfl" in category_lower or "nba" in category_lower or "game" in category_lower:
            return self.configs["sports"]
        if "crypto" in category_lower or "bitcoin" in category_lower or "ethereum" in category_lower:
            return self.configs["crypto"]
        if "econ" in category_lower or "fed" in category_lower or "rate" in category_lower:
            return self.configs["financial"]
        
        self.logger.debug(f"Using default config for category: {category}")
        return DEFAULT_CONFIG
    
    def should_use_ioc(self, category: str, edge: float) -> bool:
        """
        Determine if IOC order should be used based on category urgency.
        
        Args:
            category: Market category
            edge: Trading edge percentage
            
        Returns:
            True if IOC order should be used
        """
        config = self.get_category_config(category)
        
        # IOC if edge is high relative to urgency factor
        adjusted_threshold = 0.10 / config.urgency_factor
        return edge > adjusted_threshold
    
    def get_adjusted_position_size(self, base_size: float, category: str) -> float:
        """
        Get position size adjusted for category risk profile.
        
        Args:
            base_size: Base position size in dollars
            category: Market category
            
        Returns:
            Adjusted position size
        """
        config = self.get_category_config(category)
        return base_size * config.position_size_multiplier


# Singleton instance for easy access
_handler_instance: Optional[CategoryStrategyHandler] = None


def get_category_handler() -> CategoryStrategyHandler:
    """Get or create the singleton category handler instance."""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = CategoryStrategyHandler()
    return _handler_instance


def get_category_config(category: str) -> CategoryConfig:
    """Convenience function to get category config."""
    return get_category_handler().get_category_config(category)
