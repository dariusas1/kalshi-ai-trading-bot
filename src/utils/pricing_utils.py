"""
Price utilities for consistent pricing across the trading system.

Handles price unit conversions, bid/ask logic, and price validation.
"""

import logging
from typing import Any, Dict, Optional, Tuple
from src.exceptions.trading import ValidationError

logger = logging.getLogger(__name__)


class PriceConverter:
    """Utility class for price conversions and operations."""

    @staticmethod
    def dollars_to_cents(dollars: float) -> int:
        """
        Convert dollars to cents (multiply by 100).

        Args:
            dollars: Price in dollars

        Returns:
            Price in cents (integer)
        """
        if not isinstance(dollars, (int, float)):
            raise ValueError(f"Price must be numeric, got {type(dollars)}")
        return int(dollars * 100)

    @staticmethod
    def cents_to_dollars(cents: int) -> float:
        """
        Convert cents to dollars (divide by 100).

        Args:
            cents: Price in cents

        Returns:
            Price in dollars (float)
        """
        if not isinstance(cents, (int, float)):
            raise ValueError(f"Price must be numeric, got {type(cents)}")
        return float(cents) / 100.0

    @staticmethod
    def normalize_price(price: float, assume_cents: bool = False) -> float:
        """
        Normalize price to 0-1 dollar range.

        Args:
            price: Price value
            assume_cents: If True, assume input is in cents and divide by 100

        Returns:
            Normalized price in dollars (0-1 range)
        """
        if assume_cents:
            return price / 100.0

        # If price is > 2, it's likely in cents, convert to dollars
        if price > 2:
            return price / 100.0

        # Otherwise assume it's already in dollars
        return price

    @staticmethod
    def validate_price_range(price_cents: int, market_id: str = "") -> bool:
        """
        Validate that price is reasonable for binary options.

        Binary options typically trade between 1-99 cents.

        Args:
            price_cents: Price in cents
            market_id: Market identifier for logging

        Returns:
            True if price is valid, False otherwise
        """
        if not (1 <= price_cents <= 99):
            if market_id:
                logger.warning(f"Unusual price {price_cents} cents for {market_id}")
            return False
        return True


class PriceSelector:
    """Utility class for selecting appropriate prices based on order type."""

    @staticmethod
    def get_execution_price(market_info: Dict[str, Any], side: str, action: str) -> int:
        """
        Get the appropriate price for order execution.

        For IMMEDIATE execution (crossing the spread):
        - To BUY: Use ASK price (the price sellers are offering)
        - To SELL: Use BID price (the price buyers are offering)
        
        This is the "taker" logic - you cross the spread to get filled immediately.

        Args:
            market_info: Market information dictionary
            side: "yes" or "no"
            action: "buy" or "sell"

        Returns:
            Price in cents for order execution
        """
        price_key = f"{side}_price"
        bid_key = f"{side}_bid"
        ask_key = f"{side}_ask"

        # Get base price as fallback
        base_price = market_info.get(price_key, 50)  # API returns in cents
        if base_price < 1:  # If it's in dollars (0.xx), convert to cents
            base_price = int(base_price * 100)

        if action == "buy":
            # To BUY immediately, pay the ASK price (what sellers are asking)
            ask_price = market_info.get(ask_key, 0)
            if ask_price > 0:
                return ask_price  # API returns ask in cents
            else:
                logger.info(f"No {ask_key} found, using base price {base_price} for buy order")
                return base_price

        else:  # sell
            # To SELL immediately, accept the BID price (what buyers are offering)
            bid_price = market_info.get(bid_key, 0)
            if bid_price > 0:
                return bid_price  # API returns bid in cents
            else:
                logger.info(f"No {bid_key} found, using base price {base_price} for sell order")
                return base_price

    @staticmethod
    def get_mid_price(market_info: Dict[str, Any], side: str) -> int:
        """
        Get mid price (average of bid and ask) for market analysis.

        Args:
            market_info: Market information dictionary
            side: "yes" or "no"

        Returns:
            Mid price in cents
        """
        bid_key = f"{side}_bid"
        ask_key = f"{side}_ask"
        price_key = f"{side}_price"

        bid_price = market_info.get(bid_key, 0)
        ask_price = market_info.get(ask_key, 0)

        if bid_price > 0 and ask_price > 0:
            return (bid_price + ask_price) // 2
        elif bid_price > 0:
            return bid_price
        elif ask_price > 0:
            return ask_price
        else:
            # Fall back to stored base price
            base_price_dollars = market_info.get(price_key, 0.50)
            return PriceConverter.dollars_to_cents(base_price_dollars)

    @staticmethod
    def get_spread(market_info: Dict[str, Any], side: str) -> int:
        """
        Calculate bid-ask spread in cents.

        Args:
            market_info: Market information dictionary
            side: "yes" or "no"

        Returns:
            Spread in cents (0 if no spread available)
        """
        bid_key = f"{side}_bid"
        ask_key = f"{side}_ask"

        bid_price = market_info.get(bid_key, 0)
        ask_price = market_info.get(ask_key, 0)

        if bid_price > 0 and ask_price > 0:
            return ask_price - bid_price
        else:
            return 0


def create_order_price(
    market_info: Dict[str, Any],
    position,
    buffer_cents: int = 1,
    max_price: int = 99
) -> int:
    """
    Create appropriate order price with buffer for execution.

    Args:
        market_info: Market information from API
        position: Position object with side information
        buffer_cents: Buffer to add to price for execution safety
        max_price: Maximum price limit

    Returns:
        Price in cents for order placement
    """
    action = "buy" if position.quantity > 0 else "sell"
    side = "yes" if position.side.upper() == "YES" else "no"

    # Get appropriate execution price
    execution_price = PriceSelector.get_execution_price(market_info, side, action)

    # Add buffer for execution safety
    if action == "buy":
        # For buying, we want slightly HIGHER price to ensure fill (cross spread)
        buffered_price = execution_price + buffer_cents
    else:
        # For selling, we want slightly LOWER price to ensure fill (cross spread)
        buffered_price = execution_price - buffer_cents

    # Apply maximum price constraints
    if side == "yes":
        return min(max_price, max(1, buffered_price))
    else:
        return min(max_price, max(1, buffered_price))