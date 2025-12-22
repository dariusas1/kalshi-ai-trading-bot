"""
Input validation utilities for trading system.

Provides comprehensive validation for market data, trading decisions, and API responses.
"""

from typing import Dict, Any, Optional
import logging
from datetime import datetime, timezone

from src.exceptions.trading import ValidationError

logger = logging.getLogger(__name__)


class MarketDataValidator:
    """Validates market data before processing."""

    @staticmethod
    def validate_market_data(market_data: Dict[str, Any]) -> None:
        """
        Validate market data structure and values.

        Args:
            market_data: Dictionary containing market information

        Raises:
            ValidationError: If market data is invalid
        """
        if not isinstance(market_data, dict):
            raise ValidationError("Market data must be a dictionary")

        # Check required fields
        required_fields = ['market_id', 'title', 'yes_price', 'no_price']
        for field in required_fields:
            if field not in market_data:
                raise ValidationError(f"Missing required field: {field}", field=field)

        # Validate market_id
        if not market_data['market_id'] or not isinstance(market_data['market_id'], str):
            raise ValidationError("Invalid market_id", field='market_id', value=market_data['market_id'])

        # Validate title
        if not market_data['title'] or len(market_data['title'].strip()) == 0:
            raise ValidationError("Invalid title", field='title', value=market_data['title'])

        # Validate price ranges (0-1 for stored dollar prices)
        for price_field in ['yes_price', 'no_price']:
            price = market_data[price_field]
            if not isinstance(price, (int, float)):
                raise ValidationError(f"Invalid {price_field} type", field=price_field, value=price)
            if not (0 <= price <= 1):
                raise ValidationError(
                    f"Invalid {price_field}: must be between 0 and 1 (dollars)",
                    field=price_field,
                    value=price
                )

        # Validate optional fields
        optional_fields = {
            'volume': (int, float, type(None)),
            'close_time': (str, type(None)),
            'settlement_time': (str, type(None)),
            'active': (bool, type(None)),
            'tick_size': (int, float, type(None))
        }

        for field, valid_types in optional_fields.items():
            if field in market_data and not isinstance(market_data[field], valid_types):
                raise ValidationError(
                    f"Invalid {field} type",
                    field=field,
                    value=market_data[field]
                )

    @staticmethod
    def validate_trading_decision(decision: Dict[str, Any]) -> None:
        """
        Validate trading decision structure and values.

        Args:
            decision: Dictionary containing trading decision

        Raises:
            ValidationError: If decision is invalid
        """
        if not isinstance(decision, dict):
            raise ValidationError("Trading decision must be a dictionary")

        required_fields = ['action', 'confidence', 'reasoning']
        for field in required_fields:
            if field not in decision:
                raise ValidationError(f"Missing required field: {field}", field=field)

        # Validate action
        valid_actions = ['BUY', 'SELL', 'HOLD', 'CANCEL', 'SKIP']
        if decision['action'] not in valid_actions:
            raise ValidationError(
                f"Invalid action: {decision['action']}",
                field='action',
                value=decision['action']
            )

        # Validate confidence (0-1)
        confidence = decision['confidence']
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            raise ValidationError(
                f"Invalid confidence: {confidence}",
                field='confidence',
                value=confidence
            )

        # Validate reasoning
        if not isinstance(decision['reasoning'], str) or len(decision['reasoning'].strip()) == 0:
            raise ValidationError(
                "Invalid reasoning: must be non-empty string",
                field='reasoning',
                value=decision['reasoning']
            )

    @staticmethod
    def validate_position_limits(position_size: float, account_balance: float, max_position_pct: float = 0.05) -> None:
        """
        Validate position size against account balance and limits.

        Args:
            position_size: Proposed position size in dollars
            account_balance: Current account balance
            max_position_pct: Maximum position as percentage of balance (default 5%)

        Raises:
            ValidationError: If position violates limits
        """
        if not isinstance(position_size, (int, float)):
            raise ValidationError("Position size must be numeric", field='position_size', value=position_size)

        if position_size <= 0:
            raise ValidationError("Position size must be positive", field='position_size', value=position_size)

        if not isinstance(account_balance, (int, float)) or account_balance <= 0:
            raise ValidationError("Account balance must be positive", field='account_balance', value=account_balance)

        max_allowed = account_balance * max_position_pct
        if position_size > max_allowed:
            raise ValidationError(
                f"Position size {position_size:.2f} exceeds maximum {max_allowed:.2f} ({max_position_pct:.1%} of balance)",
                field='position_size',
                value=position_size
            )

    @staticmethod
    def validate_timestamp(timestamp: Optional[str]) -> None:
        """
        Validate timestamp format and values.

        Args:
            timestamp: ISO timestamp string or None

        Raises:
            ValidationError: If timestamp is invalid
        """
        if timestamp is None:
            return  # Optional field

        try:
            # Try to parse as ISO format
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            # Validate it's a reasonable time (not in distant past or future)
            now = datetime.now(timezone.utc)
            if dt < now.replace(year=now.year - 1) or dt > now.replace(year=now.year + 2):
                raise ValidationError("Timestamp is outside reasonable range")
        except ValueError as e:
            raise ValidationError(f"Invalid timestamp format: {timestamp}") from e


class RiskValidator:
    """Validates risk management parameters."""

    @staticmethod
    def validate_risk_parameters(
        max_daily_loss_pct: float,
        max_position_pct: float,
        max_concurrent_positions: int
    ) -> None:
        """
        Validate risk management parameters.

        Args:
            max_daily_loss_pct: Maximum daily loss as percentage
            max_position_pct: Maximum position size as percentage
            max_concurrent_positions: Maximum number of concurrent positions
            account_balance: Current account balance

        Raises:
            ValidationError: If any risk parameter is invalid
        """
        if not (0 <= max_daily_loss_pct <= 1):
            raise ValidationError(
                f"Invalid max_daily_loss_pct: {max_daily_loss_pct}",
                field='max_daily_loss_pct'
            )

        if not (0 <= max_position_pct <= 1):
            raise ValidationError(
                f"Invalid max_position_pct: {max_position_pct}",
                field='max_position_pct'
            )

        # FIXED: Removed incorrect validation that max_daily_loss_pct <= max_position_pct
        # It is valid to have cumulative daily loss limit > single position limit.

        if not isinstance(max_concurrent_positions, int) or max_concurrent_positions <= 0:
            raise ValidationError(
                f"Invalid max_concurrent_positions: {max_concurrent_positions}",
                field='max_concurrent_positions'
            )

        if max_concurrent_positions > 50:  # Reasonable limit
            raise ValidationError(
                f"max_concurrent_positions too high: {max_concurrent_positions}",
                field='max_concurrent_positions'
            )