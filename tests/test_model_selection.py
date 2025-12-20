"""
Tests for Model Selection Algorithms.

Tests cover model selection based on recent performance, context-aware routing,
cost-benefit optimization, automatic model deselection during outages,
and ensemble disagreement resolution.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from src.utils.database import DatabaseManager, ModelPerformance, ModelHealth
from src.utils.performance_tracker import PerformanceTracker
from src.intelligence.model_selector import (
    ModelSelector,
    SelectionCriteria,
    ModelSelectionResult,
    ModelHealthStatus
)

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager for testing."""
    manager = Mock(spec=DatabaseManager)

    # Mock performance data
    manager.get_model_performance_by_model = AsyncMock(return_value=[])
    manager.get_model_performance_aggregation = AsyncMock(return_value=None)
    manager.get_available_models = AsyncMock(return_value=["grok-4", "grok-3", "gpt-4"])
    manager.get_model_health = AsyncMock(return_value=None)
    manager.save_model_health = AsyncMock(return_value=1)
    manager.save_ensemble_decision = AsyncMock(return_value=1)

    return manager


@pytest.fixture
def mock_performance_tracker(mock_db_manager):
    """Create a mock performance tracker for testing."""
    tracker = Mock(spec=PerformanceTracker)
    tracker.db_manager = mock_db_manager

    # Mock performance metrics
    tracker.get_model_ranking = AsyncMock(return_value=[])
    tracker.get_rolling_window_metrics = AsyncMock()
    tracker.get_model_strengths = AsyncMock()
    tracker.get_cost_performance_metrics = AsyncMock()
    tracker.get_confidence_calibration = AsyncMock()

    return tracker


@pytest.fixture
def model_selector(mock_performance_tracker):
    """Create a ModelSelector instance for testing."""
    return ModelSelector(mock_performance_tracker)


class TestModelSelectionAlgorithms:
    """Test suite for model selection algorithms."""

    async def test_model_selection_based_on_recent_performance(
        self, model_selector, mock_performance_tracker, mock_db_manager
    ):
        """
        Test that model selection prefers models with better recent performance.

        Verifies that the selection algorithm correctly weighs recent accuracy
        and confidence calibration when choosing between available models.
        """
        # Mock healthy models
        healthy_models_data = [
            ModelHealth(
                model_name="grok-4",
                is_available=True,
                last_check_time=datetime.now(),
                consecutive_failures=0,
                avg_response_time=1200.0
            ),
            ModelHealth(
                model_name="grok-3",
                is_available=True,
                last_check_time=datetime.now(),
                consecutive_failures=0,
                avg_response_time=800.0
            )
        ]
        mock_db_manager.get_model_health.return_value = healthy_models_data

        # Mock performance data showing grok-4 performing better than grok-3
        from src.utils.performance_tracker import ModelPerformanceMetrics

        mock_performance_data = [
            ModelPerformanceMetrics(
                model_name="grok-4",
                total_predictions=50,
                correct_predictions=42,
                accuracy=0.84,
                avg_confidence=0.78,
                avg_response_time_ms=1200,
                total_cost=5.25,
                cost_per_correct_prediction=0.125,
                cost_performance_ratio=0.16,
                avg_decision_quality=0.82,
                timestamp_window=(datetime.now() - timedelta(hours=24), datetime.now())
            ),
            ModelPerformanceMetrics(
                model_name="grok-3",
                total_predictions=50,
                correct_predictions=38,
                accuracy=0.76,
                avg_confidence=0.71,
                avg_response_time_ms=800,
                total_cost=4.00,
                cost_per_correct_prediction=0.105,
                cost_performance_ratio=0.19,
                avg_decision_quality=0.74,
                timestamp_window=(datetime.now() - timedelta(hours=24), datetime.now())
            )
        ]

        mock_performance_tracker.get_model_ranking.return_value = mock_performance_data

        # Test selection with performance weighting
        selection_result = await model_selector.select_optimal_model(
            market_category="technology",
            trade_value=25.0,
            selection_criteria=SelectionCriteria(
                performance_weight=0.6,
                cost_weight=0.2,
                speed_weight=0.2
            )
        )

        # Should select grok-4 due to higher accuracy and decision quality
        assert selection_result.selected_model == "grok-4"
        assert selection_result.confidence > 0.7
        assert selection_result.reasoning is not None
        assert "higher recent performance" in selection_result.reasoning.lower()

        # Verify ranking was called with correct parameters
        mock_performance_tracker.get_model_ranking.assert_called_once()

    async def test_context_aware_routing_for_market_types(
        self, model_selector, mock_performance_tracker, mock_db_manager
    ):
        """
        Test that models are routed based on their strengths in specific market categories.

        Verifies that the selector considers model expertise in different
        market types when making selection decisions.
        """
        # Mock model strengths showing different models excel in different categories
        from src.utils.performance_tracker import ModelStrengths

        mock_strengths = {
            "grok-4": ModelStrengths(
                strong_categories={"technology": 0.82, "finance": 0.79},
                weak_categories={"politics": 0.65, "sports": 0.68},
                preferred_conditions=["market_category:technology", "market_category:finance"],
                avoided_conditions=["market_category:politics"],
                overall_reliability=0.80
            ),
            "grok-3": ModelStrengths(
                strong_categories={"politics": 0.85, "sports": 0.80},
                weak_categories={"technology": 0.70, "finance": 0.68},
                preferred_conditions=["market_category:politics", "market_category:sports"],
                avoided_conditions=["market_category:technology"],
                overall_reliability=0.75
            )
        }

        def get_strengths_side_effect(model_name, window_hours):
            return mock_strengths.get(model_name, ModelStrengths({}, [], [], [], 0.0))

        mock_performance_tracker.get_model_strengths.side_effect = get_strengths_side_effect

        # Test selection for technology market (should prefer grok-4)
        tech_selection = await model_selector.select_optimal_model(
            market_category="technology",
            trade_value=15.0,
            selection_criteria=SelectionCriteria(
                context_weight=0.7,
                performance_weight=0.3
            )
        )

        assert tech_selection.selected_model == "grok-4"
        assert "technology" in tech_selection.reasoning.lower()
        assert "strong" in tech_selection.reasoning.lower()

        # Test selection for politics market (should prefer grok-3)
        politics_selection = await model_selector.select_optimal_model(
            market_category="politics",
            trade_value=15.0,
            selection_criteria=SelectionCriteria(
                context_weight=0.7,
                performance_weight=0.3
            )
        )

        assert politics_selection.selected_model == "grok-3"
        assert "politics" in politics_selection.reasoning.lower()
        assert "strong" in politics_selection.reasoning.lower()

    async def test_cost_benefit_optimization_logic(
        self, model_selector, mock_performance_tracker, mock_db_manager
    ):
        """
        Test that cost-benefit optimization considers budget constraints and value.

        Verifies that the selector balances performance against cost,
        especially for different trade values and budget scenarios.
        """
        # Mock cost performance metrics
        from src.utils.performance_tracker import CostPerformanceMetrics

        mock_cost_metrics = {
            "grok-4": CostPerformanceMetrics(
                total_cost=10.0,
                cost_per_prediction=0.20,
                cost_per_correct_prediction=0.24,
                cost_performance_ratio=0.35,
                roi_score=8.4,
                budget_efficiency=0.42
            ),
            "grok-3": CostPerformanceMetrics(
                total_cost=6.0,
                cost_per_prediction=0.12,
                cost_per_correct_prediction=0.16,
                cost_performance_ratio=0.47,
                roi_score=12.7,
                budget_efficiency=0.63
            )
        }

        def get_cost_metrics_side_effect(model_name, window_hours):
            return mock_cost_metrics.get(model_name, CostPerformanceMetrics(0, 0, 0, 0, 0, 0))

        mock_performance_tracker.get_cost_performance_metrics.side_effect = get_cost_metrics_side_effect

        # Test high-value trade (should prioritize performance over cost)
        high_value_selection = await model_selector.select_optimal_model(
            market_category="finance",
            trade_value=100.0,
            remaining_budget=50.0,
            selection_criteria=SelectionCriteria(
                performance_weight=0.7,
                cost_weight=0.3
            )
        )

        # For high-value trades, prioritize accuracy even if more expensive
        # Should select grok-4 if it has better performance metrics
        assert high_value_selection.selected_model in ["grok-4", "grok-3"]

        # Test low-value trade (should prioritize cost efficiency)
        low_value_selection = await model_selector.select_optimal_model(
            market_category="general",
            trade_value=5.0,
            remaining_budget=10.0,
            selection_criteria=SelectionCriteria(
                performance_weight=0.4,
                cost_weight=0.6
            )
        )

        # For low-value trades, should prefer more cost-efficient model
        assert "cost" in low_value_selection.reasoning.lower() or "efficiency" in low_value_selection.reasoning.lower()

    async def test_automatic_model_deselection_during_outages(
        self, model_selector, mock_performance_tracker, mock_db_manager
    ):
        """
        Test that models with health issues are automatically deselected.

        Verifies that the selector checks model health and excludes
        unavailable or poorly performing models from selection.
        """
        # Mock health data showing grok-4 is unavailable
        mock_health_data = [
            ModelHealth(
                model_name="grok-4",
                is_available=False,
                last_check_time=datetime.now() - timedelta(minutes=5),
                consecutive_failures=3,
                avg_response_time=5000.0
            ),
            ModelHealth(
                model_name="grok-3",
                is_available=True,
                last_check_time=datetime.now() - timedelta(minutes=5),
                consecutive_failures=0,
                avg_response_time=800.0
            )
        ]

        mock_db_manager.get_model_health.return_value = mock_health_data

        # Mock some performance data for grok-3 (only the healthy model)
        from src.utils.performance_tracker import ModelPerformanceMetrics
        mock_performance_data = [
            ModelPerformanceMetrics(
                model_name="grok-3",
                total_predictions=10,
                correct_predictions=7,
                accuracy=0.70,
                avg_confidence=0.65,
                avg_response_time_ms=800,
                total_cost=1.00,
                cost_per_correct_prediction=0.14,
                cost_performance_ratio=0.50,
                avg_decision_quality=0.68,
                timestamp_window=(datetime.now() - timedelta(hours=24), datetime.now())
            )
        ]

        mock_performance_tracker.get_model_ranking.return_value = mock_performance_data

        # Test selection during outage
        selection_result = await model_selector.select_optimal_model(
            market_category="technology",
            trade_value=20.0
        )

        # Should select grok-3 since grok-4 is unavailable
        assert selection_result.selected_model == "grok-3"
        assert selection_result.disqualified_models == ["grok-4"]

        # Verify health check was performed
        mock_db_manager.get_model_health.assert_called_once()

    async def test_ensemble_disagreement_resolution(
        self, model_selector, mock_performance_tracker
    ):
        """
        Test handling of ensemble disagreements and confidence conflicts.

        Verifies that the selector can resolve conflicts when models
        disagree or when there are significant confidence differences.
        """
        # Mock scenario where models would likely disagree
        mock_performance_tracker.get_model_ranking.return_value = [
            Mock(model_name="grok-4", accuracy=0.75, avg_confidence=0.80),
            Mock(model_name="grok-3", accuracy=0.72, avg_confidence=0.65)
        ]

        # Test selection with high disagreement threshold
        consensus_result = await model_selector.resolve_model_disagreement(
            model_predictions={
                "grok-4": {"action": "buy", "confidence": 0.80, "reasoning": "Strong technical indicators"},
                "grok-3": {"action": "sell", "confidence": 0.65, "reasoning": "Market sentiment negative"}
            },
            market_category="technology",
            disagreement_threshold=0.3
        )

        # Should detect disagreement and provide resolution
        assert consensus_result.disagreement_detected == True
        assert consensus_result.disagreement_level > 0.3
        assert consensus_result.resolution_method is not None
        assert consensus_result.final_model is not None
        assert consensus_result.confidence_adjustment is not None

        # Test when models agree
        agreement_result = await model_selector.resolve_model_disagreement(
            model_predictions={
                "grok-4": {"action": "buy", "confidence": 0.75, "reasoning": "Positive momentum"},
                "grok-3": {"action": "buy", "confidence": 0.70, "reasoning": "Bullish indicators"}
            },
            market_category="technology",
            disagreement_threshold=0.3
        )

        # Should detect agreement and use consensus
        assert agreement_result.disagreement_detected == False
        assert agreement_result.disagreement_level < 0.3
        assert agreement_result.final_model in ["grok-4", "grok-3"]
        assert "consensus" in agreement_result.reasoning.lower() or "agreement" in agreement_result.reasoning.lower()


class TestModelSelectionEdgeCases:
    """Test edge cases and error handling for model selection."""

    async def test_no_available_models(self, model_selector, mock_db_manager):
        """Test behavior when no models are available."""
        # Mock no available models
        mock_db_manager.get_model_health.return_value = [
            ModelHealth(
                model_name="grok-4",
                is_available=False,
                last_check_time=datetime.now(),
                consecutive_failures=5,
                avg_response_time=0.0
            ),
            ModelHealth(
                model_name="grok-3",
                is_available=False,
                last_check_time=datetime.now(),
                consecutive_failures=3,
                avg_response_time=0.0
            )
        ]

        selection_result = await model_selector.select_optimal_model(
            market_category="technology",
            trade_value=25.0
        )

        # Should return no selection with appropriate reasoning
        assert selection_result.selected_model is None
        assert selection_result.confidence == 0.0
        assert "no available" in selection_result.reasoning.lower() or "unavailable" in selection_result.reasoning.lower()

    async def test_budget_exceeded_handling(self, model_selector, mock_performance_tracker):
        """Test behavior when budget constraints prevent model usage."""
        # Mock high-cost model
        mock_performance_tracker.get_cost_performance_metrics.return_value = Mock(
            cost_per_prediction=5.0,
            cost_performance_ratio=0.1
        )

        selection_result = await model_selector.select_optimal_model(
            market_category="technology",
            trade_value=5.0,
            remaining_budget=2.0,
            selection_criteria=SelectionCriteria(cost_weight=1.0)
        )

        # Should indicate budget constraint issue
        assert "budget" in selection_result.reasoning.lower() or "cost" in selection_result.reasoning.lower()
        assert selection_result.confidence < 0.5

    async def test_insufficient_performance_data(self, model_selector, mock_performance_tracker):
        """Test behavior when models have insufficient performance data."""
        # Mock empty performance data
        mock_performance_tracker.get_model_ranking.return_value = []

        selection_result = await model_selector.select_optimal_model(
            market_category="new_category",
            trade_value=20.0
        )

        # Should fall back to default model selection
        assert selection_result.selected_model is not None
        assert "insufficient data" in selection_result.reasoning.lower() or "fallback" in selection_result.reasoning.lower()


# Performance and stress tests
class TestModelSelectionPerformance:
    """Test performance characteristics of model selection algorithms."""

    async def test_selection_speed_under_load(self, model_selector, mock_performance_tracker):
        """Test that model selection completes quickly even with many models."""
        # Mock many models
        many_models = [f"model-{i}" for i in range(20)]
        mock_performance_tracker.get_model_ranking.return_value = [
            Mock(model_name=model, accuracy=0.7 + (i * 0.01), avg_confidence=0.75)
            for i, model in enumerate(many_models)
        ]

        # Time the selection process
        start_time = datetime.now()

        for _ in range(10):  # Run 10 selections
            await model_selector.select_optimal_model(
                market_category="test",
                trade_value=25.0
            )

        end_time = datetime.now()
        selection_time = (end_time - start_time).total_seconds()

        # Should complete 10 selections in under 1 second
        assert selection_time < 1.0, f"Selection took too long: {selection_time}s"

    async def test_concurrent_selection_requests(self, model_selector, mock_performance_tracker):
        """Test handling of concurrent selection requests."""
        # Mock basic performance data
        mock_performance_tracker.get_model_ranking.return_value = [
            Mock(model_name="grok-4", accuracy=0.8, avg_confidence=0.75)
        ]

        # Run multiple concurrent selections
        tasks = [
            model_selector.select_optimal_model(
                market_category="test",
                trade_value=25.0
            )
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete successfully
        assert all(isinstance(result, ModelSelectionResult) for result in results)
        assert all(result.selected_model == "grok-4" for result in results)