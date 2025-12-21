"""
Custom exception hierarchy for the AI trading system.

Provides structured error handling and categorization for different types of failures.
"""

from typing import Optional, Dict, Any


class TradingError(Exception):
    """Base exception for all trading system errors."""

    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.message = message


class APIError(TradingError):
    """Base class for external API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response = response


class XAIError(APIError):
    """Errors from xAI/Grok API calls."""
    pass


class XAIRateLimitError(XAIError):
    """Rate limit exceeded for xAI API."""

    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class XAITimeoutError(XAIError):
    """Request timeout for xAI API."""
    pass


class XAIServerError(XAIError):
    """Server error (5xx) from xAI API."""
    pass


class KalshiError(APIError):
    """Errors from Kalshi trading API."""
    pass


class ValidationError(TradingError):
    """Input validation errors."""

    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value


class DatabaseError(TradingError):
    """Database operation errors."""
    pass


class DatabaseTransactionError(DatabaseError):
    """Database transaction rollback errors."""
    pass


class ConfigurationError(TradingError):
    """Configuration or settings errors."""
    pass


class CircuitBreakerError(TradingError):
    """Circuit breaker tripped errors."""
    pass


class EnsembleError(TradingError):
    """AI ensemble coordination errors."""
    pass


class ModelHealthError(EnsembleError):
    """AI model health check failures."""
    pass


class RiskManagementError(TradingError):
    """Risk management and position limit errors."""
    pass