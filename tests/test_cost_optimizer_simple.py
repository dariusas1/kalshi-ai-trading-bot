"""
Tests for Cost Optimization Framework (Simplified).

Tests the CostOptimizer class core functionality using unittest pattern.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from src.intelligence.cost_optimizer import (
    CostOptimizer,
    BudgetStatus,
    CacheEntry,
    CostOptimizationConfig,
    DynamicCostModel
)
from src.utils.performance_tracker import PerformanceTracker, ModelPerformanceMetrics, CostPerformanceMetrics
from src.utils.database import DatabaseManager


class TestCostOptimizer(unittest.TestCase):
    """Test CostOptimizer core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        """Clean up test fixtures."""
        self.loop.close()

    def _get_cost_optimizer(self):
        """Create cost optimizer instance for testing."""
        db_manager = AsyncMock(spec=DatabaseManager)
        performance_tracker = MagicMock()
        config = CostOptimizationConfig(
            enable_dynamic_modeling=True,
            daily_budget_limit=50.0,
            enable_budget_controls=True,
            enable_intelligent_caching=True,
            enable_real_time_monitoring=True
        )
        return CostOptimizer(db_manager, performance_tracker, config)

    def test_initialization(self):
        """Test CostOptimizer initialization."""
        optimizer = self._get_cost_optimizer()

        assert optimizer.config.enable_dynamic_modeling is True
        assert optimizer.config.daily_budget_limit == 50.0
        assert optimizer.config.enable_budget_controls is True
        assert len(optimizer.result_cache) == 0
        assert optimizer.cache_hits == 0
        assert optimizer.cache_misses == 0

    def test_cost_efficiency_calculation(self):
        """Test cost efficiency ratio calculation."""

        async def run_test():
            optimizer = self._get_cost_optimizer()

            # Mock performance data
            mock_cost_metrics = CostPerformanceMetrics(
                total_cost=5.0,
                cost_per_prediction=0.05,
                cost_per_correct_prediction=0.067,
                cost_performance_ratio=0.15,
                roi_score=8.0,
                budget_efficiency=7.5
            )

            mock_model_metrics = ModelPerformanceMetrics(
                model_name="grok-4",
                total_predictions=100,
                correct_predictions=75,
                accuracy=0.75,
                avg_confidence=0.8,
                avg_response_time_ms=1200,
                total_cost=5.0,
                cost_per_correct_prediction=0.067,
                cost_performance_ratio=0.15,
                avg_decision_quality=0.7,
                timestamp_window=(datetime.now() - timedelta(hours=24), datetime.now())
            )

            # Test the efficiency score calculation
            efficiency_score = optimizer._calculate_efficiency_score(mock_cost_metrics, mock_model_metrics)

            assert efficiency_score > 0.0
            assert efficiency_score <= 1.0
            assert isinstance(efficiency_score, float)

        self.loop.run_until_complete(run_test())

    def test_cache_key_generation(self):
        """Test cache key generation for consistent caching."""
        optimizer = self._get_cost_optimizer()

        market_data1 = {
            "category": "technology",
            "volume": 3000.0,
            "price": 0.65,
            "volatility": "medium"
        }

        market_data2 = {
            "category": "technology",
            "volume": 3200.0,  # Small difference in volume
            "price": 0.66,    # Small difference in price
            "volatility": "medium"
        }

        # Should generate same key for similar data
        key1 = optimizer._generate_cache_key("grok-4", market_data1)
        key2 = optimizer._generate_cache_key("grok-4", market_data2)

        assert key1 == key2  # Same category and volume range should generate same key

        # Different model should generate different key
        key3 = optimizer._generate_cache_key("grok-3", market_data1)
        assert key1 != key3

    def test_volume_range_classification(self):
        """Test volume range classification for cache keys."""
        optimizer = self._get_cost_optimizer()

        assert optimizer._get_volume_range(500) == "low"
        assert optimizer._get_volume_range(2500) == "medium"
        assert optimizer._get_volume_range(6000) == "high"

    def test_price_range_classification(self):
        """Test price range classification for cache keys."""
        optimizer = self._get_cost_optimizer()

        assert optimizer._get_price_range(0.2) == "low"
        assert optimizer._get_price_range(0.5) == "medium"
        assert optimizer._get_price_range(0.8) == "high"

    def test_cache_entry_validation(self):
        """Test cache entry validity checking."""
        optimizer = self._get_cost_optimizer()

        # Create valid cache entry
        valid_entry = CacheEntry(
            cache_key="test_key",
            model_name="grok-4",
            market_data={},
            result={},
            timestamp=datetime.now(),
            cost=0.05,
            accuracy=0.8,
            ttl_minutes=30
        )

        assert optimizer._is_cache_entry_valid(valid_entry) is True

        # Create expired cache entry
        expired_entry = CacheEntry(
            cache_key="test_key",
            model_name="grok-4",
            market_data={},
            result={},
            timestamp=datetime.now() - timedelta(minutes=31),
            cost=0.05,
            accuracy=0.8,
            ttl_minutes=30
        )

        assert optimizer._is_cache_entry_valid(expired_entry) is False

    def test_budget_status_calculation(self):
        """Test budget status calculation."""

        async def run_test():
            optimizer = self._get_cost_optimizer()

            # Mock daily spend
            with patch.object(optimizer, '_get_daily_spend', return_value=30.0):
                with patch.object(optimizer, '_generate_budget_recommendations') as mock_recs:
                    mock_recs.return_value = ["Test recommendation"]

                    budget_status = await optimizer._calculate_budget_status()

            assert budget_status.daily_limit == 50.0
            assert budget_status.spent == 30.0
            assert budget_status.remaining == 20.0
            assert budget_status.percentage_used == 60.0
            assert budget_status.status == "healthy"  # 60% should be healthy
            assert len(budget_status.recommended_actions) > 0

        self.loop.run_until_complete(run_test())

    def test_model_cost_estimation(self):
        """Test model cost estimation based on trade value."""

        async def run_test():
            optimizer = self._get_cost_optimizer()

            # Test different trade values
            low_cost = await optimizer._estimate_model_cost("grok-4", 5.0)
            normal_cost = await optimizer._estimate_model_cost("grok-4", 50.0)
            high_cost = await optimizer._estimate_model_cost("grok-4", 150.0)

            assert low_cost < normal_cost
            assert normal_cost < high_cost
            assert isinstance(low_cost, float)
            assert isinstance(normal_cost, float)
            assert isinstance(high_cost, float)

        self.loop.run_until_complete(run_test())

    def test_budget_aware_model_scoring(self):
        """Test budget-aware model scoring."""

        async def run_test():
            optimizer = self._get_cost_optimizer()

            # Test scoring with different budget scenarios
            high_budget_score = await optimizer._calculate_budget_aware_score(
                "grok-4", remaining_budget=40.0, trade_value=75.0, market_category="technology"
            )

            low_budget_score = await optimizer._calculate_budget_aware_score(
                "grok-4", remaining_budget=2.0, trade_value=75.0, market_category="technology"
            )

            assert 0.0 <= high_budget_score <= 1.0
            assert 0.0 <= low_budget_score <= 1.0
            # High budget should allow better scores
            assert high_budget_score >= low_budget_score

        self.loop.run_until_complete(run_test())

    def test_cache_efficiency_metrics(self):
        """Test cache efficiency metrics calculation."""

        async def run_test():
            optimizer = self._get_cost_optimizer()

            # Simulate cache activity
            optimizer.cache_hits = 150
            optimizer.cache_misses = 50

            # Add some cache entries
            for i in range(5):
                entry = CacheEntry(
                    cache_key=f"key_{i}",
                    model_name="grok-4",
                    market_data={},
                    result={},
                    timestamp=datetime.now(),
                    cost=0.05,
                    accuracy=0.8,
                    ttl_minutes=30,
                    hit_count=i + 1
                )
                optimizer.result_cache[f"key_{i}"] = entry

            metrics = await optimizer.get_cache_efficiency_metrics()

            assert metrics["hit_rate"] == 150 / (150 + 50)  # 75% hit rate
            assert metrics["total_requests"] == 200
            assert metrics["cache_hits"] == 150
            assert metrics["cache_misses"] == 50
            assert metrics["cache_size"] == 5
            assert metrics["avg_hit_count"] == 3.0  # (1+2+3+4+5)/5
            assert metrics["max_hit_count"] == 5

        self.loop.run_until_complete(run_test())

    def test_cost_model_update(self):
        """Test cost performance model updating."""

        async def run_test():
            optimizer = self._get_cost_optimizer()

            # Update cost model for grok-4 in technology category
            await optimizer.update_cost_performance_model(
                model_name="grok-4",
                market_category="technology",
                accuracy_score=0.85,
                cost_usd=0.045,
                response_time_ms=1100
            )

            # Check if model was created
            model_key = "grok-4_technology"
            assert model_key in optimizer.cost_models

            model = optimizer.cost_models[model_key]
            assert model.model_name == "grok-4"
            assert model.market_category == "technology"
            assert model.cost_efficiency_factor > 0.0

        self.loop.run_until_complete(run_test())

    def test_spending_monitoring(self):
        """Test real-time spending monitoring."""

        async def run_test():
            optimizer = self._get_cost_optimizer()

            # Simulate spending activity
            model_costs = [
                {"model": "grok-4", "cost": 0.05},
                {"model": "grok-3", "cost": 0.03},
                {"model": "gpt-4", "cost": 0.04}
            ]

            # Mock budget status calculation
            with patch.object(optimizer, '_calculate_budget_status') as mock_budget:
                mock_budget.return_value = BudgetStatus(
                    daily_limit=50.0,
                    spent=12.0,
                    remaining=38.0,
                    percentage_used=24.0,
                    status="healthy",
                    projected_daily_spend=15.0,
                    recommended_actions=[]
                )

                spending_status = await optimizer.monitor_spend(model_costs)

            assert spending_status["current_spend"] == 0.12  # 0.05 + 0.03 + 0.04
            assert spending_status["transaction_count"] == 3
            assert "model_breakdown" in spending_status
            assert spending_status["model_breakdown"]["grok-4"] == 0.05
            assert spending_status["model_breakdown"]["grok-3"] == 0.03
            assert spending_status["model_breakdown"]["gpt-4"] == 0.04

        self.loop.run_until_complete(run_test())


if __name__ == '__main__':
    unittest.main()