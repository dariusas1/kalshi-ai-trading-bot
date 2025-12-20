"""
Tests for Cost Optimization Framework.

Tests the CostOptimizer class and its cost-efficiency calculations,
dynamic cost-per-performance modeling, budget-aware selection,
intelligent caching, and real-time cost monitoring.
"""

import asyncio
import unittest
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from src.intelligence.cost_optimizer import (
    CostOptimizer,
    CostEfficiencyMetrics,
    BudgetStatus,
    CacheEntry,
    SpendingControl,
    CostOptimizationConfig,
    DynamicCostModel
)
from src.utils.performance_tracker import PerformanceTracker, ModelPerformanceMetrics, CostPerformanceMetrics
from src.utils.database import DatabaseManager


class TestDynamicCostPerformanceModeling(unittest.TestCase):
    """Test dynamic cost-per-performance modeling functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test fixtures."""
        self.loop.close()

    async def _get_cost_optimizer(self):
        """Create cost optimizer instance for testing."""
        db_manager = AsyncMock(spec=DatabaseManager)
        config = CostOptimizationConfig(
            enable_dynamic_modeling=True,
            cost_performance_window_hours=24,
            min_predictions_for_modeling=5
        )
        performance_tracker = MagicMock()
        optimizer = CostOptimizer(db_manager, performance_tracker, config)
        return optimizer

    def test_calculate_dynamic_cost_efficiency_ratio(self):
        """Test dynamic cost-efficiency ratio calculation for models."""

        async def run_test():
            cost_optimizer = await self._get_cost_optimizer()

            # Mock performance data
            mock_metrics = ModelPerformanceMetrics(
                model_name="grok-4",
                total_predictions=100,
                correct_predictions=75,
                accuracy=0.75,
                avg_confidence=0.8,
                avg_response_time_ms=1200,
                total_cost=5.0,
                cost_per_correct_prediction=0.067,
                cost_performance_ratio=0.15,  # accuracy / cost
                avg_decision_quality=0.7,
                timestamp_window=(datetime.now() - timedelta(hours=24), datetime.now())
            )

            with patch.object(cost_optimizer, '_get_model_cost_metrics', return_value=mock_metrics):
                efficiency_ratio = await cost_optimizer.calculate_cost_efficiency(
                    "grok-4",
                    market_category="technology",
                    time_window_hours=24
                )

            assert efficiency_ratio > 0.0
            assert efficiency_ratio <= 1.0
            # Higher accuracy with lower cost should result in higher efficiency
            assert efficiency_ratio == mock_metrics.cost_performance_ratio / 10.0  # Normalized to 0-1

        self.loop.run_until_complete(run_test())

    @pytest.mark.asyncio
    async def test_update_cost_performance_model(self, cost_optimizer):
        """Test updating cost-per-performance model with new data."""
        model_name = "grok-4"
        market_category = "finance"
        accuracy = 0.8
        cost_usd = 0.05
        response_time_ms = 1000

        with patch.object(cost_optimizer.db_manager, 'save_cost_model_data') as mock_save:
            await cost_optimizer.update_cost_performance_model(
                model_name, market_category, accuracy, cost_usd, response_time_ms
            )

        mock_save.assert_called_once()
        call_args = mock_save.call_args[0][0]
        assert call_args.model_name == model_name
        assert call_args.market_category == market_category
        assert call_args.accuracy_score == accuracy
        assert call_args.cost_usd == cost_usd

    @pytest.mark.asyncio
    async def test_predict_model_cost_efficiency(self, cost_optimizer):
        """Test cost efficiency prediction for models in specific scenarios."""
        scenario = {
            "market_category": "technology",
            "volatility_regime": "high",
            "time_to_expiry": 6,
            "trade_value": 100.0
        }

        # Mock model predictions
        with patch.object(cost_optimizer, '_get_scenario_cost_predictions') as mock_predict:
            mock_predict.return_value = {
                "grok-4": 0.85,
                "grok-3": 0.72,
                "gpt-4": 0.78
            }

            predictions = await cost_optimizer.predict_model_cost_efficiency(scenario)

        assert len(predictions) == 3
        assert "grok-4" in predictions
        assert "grok-3" in predictions
        assert "gpt-4" in predictions
        assert predictions["grok-4"] > predictions["grok-3"]  # grok-4 should be most efficient

    @pytest.mark.asyncio
    async def test_adapt_cost_model_based_on_budget(self, cost_optimizer):
        """Test cost model adaptation based on current budget constraints."""
        initial_budget = 50.0
        remaining_budget = 10.0  # Low budget scenario

        with patch.object(cost_optimizer, '_get_adapted_cost_model') as mock_adapt:
            mock_adapt.return_value = {
                "cost_sensitivity": 0.8,  # High cost sensitivity
                "preferred_models": ["grok-3"],  # Cheaper models preferred
                "accuracy_weight": 0.6,  # Reduced accuracy weight
                "cost_weight": 0.4    # Increased cost weight
            }

            adapted_model = await cost_optimizer.adapt_cost_model_based_on_budget(
                initial_budget, remaining_budget
            )

        assert adapted_model["cost_sensitivity"] > 0.5
        assert "grok-3" in adapted_model["preferred_models"]
        assert adapted_model["cost_weight"] > adapted_model["accuracy_weight"]


class TestBudgetAwareSelection:
    """Test budget-aware model selection logic."""

    @pytest.fixture
    async def cost_optimizer(self):
        """Create cost optimizer with budget constraints."""
        db_manager = AsyncMock(spec=DatabaseManager)
        config = CostOptimizationConfig(
            daily_budget_limit=20.0,
            enable_budget_controls=True,
            budget_alert_threshold=0.8
        )
        performance_tracker = MagicMock()
        optimizer = CostOptimizer(db_manager, performance_tracker, config)
        return optimizer

    @pytest.mark.asyncio
    async def test_select_models_with_budget_constraints(self, cost_optimizer):
        """Test model selection considering remaining budget."""
        available_models = ["grok-4", "grok-3", "gpt-4"]
        remaining_budget = 5.0
        trade_value = 50.0

        # Mock model costs and performance
        model_costs = {
            "grok-4": 0.05,
            "grok-3": 0.03,
            "gpt-4": 0.04
        }

        with patch.object(cost_optimizer, '_get_model_costs', return_value=model_costs):
            with patch.object(cost_optimizer, '_calculate_model_roi') as mock_roi:
                mock_roi.side_effect = lambda model, budget: {
                    "grok-4": 0.8,
                    "grok-3": 0.7,
                    "gpt-4": 0.75
                }[model]

                selected_models = await cost_optimizer.select_models_with_budget(
                    available_models, remaining_budget, trade_value
                )

        # Should prefer cost-effective models within budget
        assert len(selected_models) >= 1
        assert "grok-3" in selected_models  # Most cost-effective should be included

    @pytest.mark.asyncio
    async def test_enforce_daily_budget_limits(self, cost_optimizer):
        """Test enforcement of daily budget limits."""
        # Simulate spending close to limit
        current_spend = 18.0
        daily_limit = cost_optimizer.config.daily_budget_limit

        with patch.object(cost_optimizer, '_get_current_daily_spend', return_value=current_spend):
            budget_status = await cost_optimizer.enforce_budget_limits()

        assert budget_status.status == "warning"
        assert budget_status.remaining_budget == daily_limit - current_spend
        assert budget_status.percentage_used == (current_spend / daily_limit) * 100

    @pytest.mark.asyncio
    async def test_get_spending_recommendations(self, cost_optimizer):
        """Test spending recommendations based on budget status."""
        remaining_budget = 15.0
        trade_frequency = 10  # trades per hour
        avg_cost_per_trade = 0.05

        with patch.object(cost_optimizer, '_analyze_spending_pattern') as mock_analyze:
            mock_analyze.return_value = {
                "recommended_max_trades_per_hour": 8,
                "suggested_cost_reduction": 0.01,
                "budget_optimization_tips": [
                    "Use ensemble only for high-value trades",
                    "Prefer grok-3 for low-value decisions"
                ]
            }

            recommendations = await cost_optimizer.get_spending_recommendations(
                remaining_budget, trade_frequency, avg_cost_per_trade
            )

        assert "recommended_max_trades_per_hour" in recommendations
        assert recommendations["recommended_max_trades_per_hour"] < trade_frequency
        assert len(recommendations["budget_optimization_tips"]) > 0


class TestIntelligentCaching:
    """Test intelligent caching system for cost optimization."""

    @pytest.fixture
    async def cost_optimizer(self):
        """Create cost optimizer with caching enabled."""
        db_manager = AsyncMock(spec=DatabaseManager)
        config = CostOptimizationConfig(
            enable_intelligent_caching=True,
            cache_ttl_minutes=30,
            max_cache_size=1000
        )
        performance_tracker = MagicMock()
        optimizer = CostOptimizer(db_manager, performance_tracker, config)
        return optimizer

    @pytest.mark.asyncio
    async def test_cache_and_reuse_results(self, cost_optimizer):
        """Test caching and reuse of model results."""
        market_data = {
            "market_id": "TECH-2024-01",
            "category": "technology",
            "volume": 5000.0,
            "price": 0.75
        }
        model_result = {
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Strong technology trend"
        }

        # First call should cache result
        with patch.object(cost_optimizer, '_should_cache_result', return_value=True):
            cache_key_1 = await cost_optimizer.cache_model_result(
                "grok-4", market_data, model_result
            )

        # Second call with similar data should return cached result
        similar_market_data = market_data.copy()
        similar_market_data["price"] = 0.76  # Small price change

        cached_result = await cost_optimizer.get_cached_result(
            "grok-4", similar_market_data, similarity_threshold=0.9
        )

        assert cached_result is not None
        assert cached_result["action"] == model_result["action"]
        assert cached_result["confidence"] == model_result["confidence"]

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_market_changes(self, cost_optimizer):
        """Test cache invalidation when market conditions change significantly."""
        original_market_data = {
            "market_id": "FINANCE-2024-01",
            "category": "finance",
            "volatility": "low",
            "volume": 2000.0
        }

        changed_market_data = {
            "market_id": "FINANCE-2024-01",
            "category": "finance",
            "volatility": "high",  # Significant change
            "volume": 5000.0
        }

        # Cache original result
        with patch.object(cost_optimizer, '_should_cache_result', return_value=True):
            await cost_optimizer.cache_model_result(
                "grok-4", original_market_data, {"action": "BUY"}
            )

        # Check if cache should be invalidated due to market changes
        should_invalidate = await cost_optimizer.should_invalidate_cache(
            original_market_data, changed_market_data
        )

        assert should_invalidate is True

    @pytest.mark.asyncio
    async def test_cache_efficiency_metrics(self, cost_optimizer):
        """Test cache efficiency monitoring and metrics."""
        # Simulate cache operations
        cache_operations = {
            "cache_hits": 150,
            "cache_misses": 50,
            "cache_size": 200,
            "memory_usage_mb": 50.0
        }

        with patch.object(cost_optimizer, '_get_cache_stats', return_value=cache_operations):
            efficiency_metrics = await cost_optimizer.get_cache_efficiency_metrics()

        hit_rate = efficiency_metrics["hit_rate"]
        assert hit_rate == 150 / (150 + 50)  # 75% hit rate
        assert efficiency_metrics["total_requests"] == 200
        assert efficiency_metrics["hit_rate"] > 0.5  # Should have decent hit rate


class TestRealTimeCostMonitoring:
    """Test real-time cost monitoring and automated controls."""

    @pytest.fixture
    async def cost_optimizer(self):
        """Create cost optimizer with real-time monitoring."""
        db_manager = AsyncMock(spec=DatabaseManager)
        config = CostOptimizationConfig(
            enable_real_time_monitoring=True,
            monitoring_interval_seconds=60,
            auto_spending_controls=True
        )
        performance_tracker = MagicMock()
        optimizer = CostOptimizer(db_manager, performance_tracker, config)
        return optimizer

    @pytest.mark.asyncio
    async def test_track_real_time_costs(self, cost_optimizer):
        """Test real-time cost tracking functionality."""
        model_costs = [
            {"model": "grok-4", "cost": 0.05, "timestamp": datetime.now()},
            {"model": "grok-3", "cost": 0.03, "timestamp": datetime.now()},
            {"model": "gpt-4", "cost": 0.04, "timestamp": datetime.now()}
        ]

        with patch.object(cost_optimizer.db_manager, 'save_cost_tracking') as mock_save:
            await cost_optimizer.track_real_time_costs(model_costs)

        assert mock_save.call_count == len(model_costs)

        # Check real-time metrics
        current_costs = await cost_optimizer.get_current_costs()
        assert current_costs["total_cost"] == sum(item["cost"] for item in model_costs)
        assert current_costs["transaction_count"] == len(model_costs)

    @pytest.mark.asyncio
    async def test_automated_spending_controls(self, cost_optimizer):
        """Test automated spending control mechanisms."""
        # Simulate approaching budget limit
        current_spend = 16.0
        daily_limit = 20.0
        hourly_rate = 2.0

        with patch.object(cost_optimizer, '_get_current_spending_rate', return_value=hourly_rate):
            with patch.object(cost_optimizer, '_get_remaining_budget', return_value=daily_limit - current_spend):
                control_actions = await cost_optimizer.evaluate_spending_controls(current_spend, daily_limit)

        assert len(control_actions) > 0

        # Should recommend cost-saving measures when approaching limit
        actions = [action["type"] for action in control_actions]
        assert "reduce_model_usage" in actions or "switch_to_cheaper_models" in actions

    @pytest.mark.asyncio
    async def test_generate_cost_alerts(self, cost_optimizer):
        """Test generation of cost alerts and notifications."""
        alert_scenarios = [
            {"budget_used": 0.5, "expected": False},   # 50% used - no alert
            {"budget_used": 0.85, "expected": True},  # 85% used - alert
            {"budget_used": 0.95, "expected": True}   # 95% used - critical alert
        ]

        for scenario in alert_scenarios:
            budget_used = scenario["budget_used"]
            expected_alert = scenario["expected"]

            with patch.object(cost_optimizer.db_manager, 'save_cost_alert') as mock_alert:
                alert = await cost_optimizer.generate_cost_alert_if_needed(budget_used)

                if expected_alert:
                    assert alert is not None
                    assert alert["severity"] in ["warning", "critical"]
                    assert alert["budget_used_percentage"] == budget_used * 100
                    mock_alert.assert_called_once()
                else:
                    assert alert is None
                    mock_alert.assert_not_called()

    @pytest.mark.asyncio
    async def test_cost_optimization_recommendations(self, cost_optimizer):
        """Test cost optimization recommendations based on spending patterns."""
        spending_data = {
            "model_costs": {
                "grok-4": 8.0,
                "grok-3": 3.0,
                "gpt-4": 5.0
            },
            "accuracy_by_model": {
                "grok-4": 0.82,
                "grok-3": 0.75,
                "gpt-4": 0.78
            },
            "total_spent": 16.0,
            "daily_budget": 20.0
        }

        with patch.object(cost_optimizer, '_analyze_spending_efficiency') as mock_analyze:
            mock_analyze.return_value = {
                "most_cost_effective": "grok-3",
                "least_cost_effective": "grok-4",
                "potential_savings": 4.0,
                "recommendations": [
                    "Use grok-3 for 70% of decisions",
                    "Reserve grok-4 for high-value trades only",
                    "Consider batch processing to reduce costs"
                ]
            }

            recommendations = await cost_optimizer.get_cost_optimization_recommendations(spending_data)

        assert "most_cost_effective" in recommendations
        assert "potential_savings" in recommendations
        assert recommendations["potential_savings"] > 0
        assert len(recommendations["recommendations"]) > 0


class TestIntegratedCostOptimization:
    """Test integrated cost optimization functionality."""

    @pytest.fixture
    async def cost_optimizer(self):
        """Create fully configured cost optimizer."""
        db_manager = AsyncMock(spec=DatabaseManager)
        config = CostOptimizationConfig(
            enable_dynamic_modeling=True,
            enable_budget_controls=True,
            enable_intelligent_caching=True,
            enable_real_time_monitoring=True,
            daily_budget_limit=25.0
        )
        performance_tracker = MagicMock()
        optimizer = CostOptimizer(db_manager, performance_tracker, config)
        return optimizer

    @pytest.mark.asyncio
    async def test_comprehensive_cost_optimization_workflow(self, cost_optimizer):
        """Test complete cost optimization workflow."""
        trade_request = {
            "market_data": {
                "category": "technology",
                "volume": 3000.0,
                "volatility": "medium"
            },
            "trade_value": 75.0,
            "available_models": ["grok-4", "grok-3", "gpt-4"],
            "current_budget_status": {
                "spent": 15.0,
                "remaining": 10.0,
                "daily_limit": 25.0
            }
        }

        # Mock all components
        with patch.object(cost_optimizer, 'select_models_with_budget') as mock_select:
            with patch.object(cost_optimizer, 'get_cached_result') as mock_cache:
                with patch.object(cost_optimizer, 'track_real_time_costs') as mock_track:

                    mock_select.return_value = ["grok-3"]  # Budget-conscious selection
                    mock_cache.return_value = None  # No cached result
                    mock_track.return_value = None

                    optimization_result = await cost_optimizer.optimize_decision_cost(
                        trade_request
                    )

        assert optimization_result["selected_models"] == ["grok-3"]
        assert optimization_result["estimated_cost"] > 0
        assert optimization_result["within_budget"] is True
        assert "optimization_reasoning" in optimization_result

    @pytest.mark.asyncio
    async def test_cost_optimization_performance_metrics(self, cost_optimizer):
        """Test cost optimization performance and efficiency metrics."""
        # Simulate optimization operations over time
        with patch.object(cost_optimizer, '_get_optimization_stats') as mock_stats:
            mock_stats.return_value = {
                "total_optimizations": 1000,
                "cost_savings": 150.0,
                "avg_cost_reduction": 0.15,  # 15% reduction
                "cache_hit_rate": 0.75,
                "budget_utilization": 0.82
            }

            metrics = await cost_optimizer.get_optimization_performance_metrics()

        assert metrics["total_optimizations"] == 1000
        assert metrics["cost_savings"] > 0
        assert 0 < metrics["avg_cost_reduction"] < 1
        assert metrics["cache_hit_rate"] > 0.5
        assert metrics["budget_utilization"] < 1.0