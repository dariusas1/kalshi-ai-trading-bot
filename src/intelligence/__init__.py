"""
Intelligence module for AI model selection and ensemble management.

Provides intelligent model selection, health monitoring, cost optimization,
and ensemble coordination for the trading system.
"""

from .model_selector import (
    ModelSelector,
    SelectionCriteria,
    ModelSelectionResult,
    ModelHealthStatus,
    DisagreementResult
)

__all__ = [
    'ModelSelector',
    'SelectionCriteria',
    'ModelSelectionResult',
    'ModelHealthStatus',
    'DisagreementResult'
]