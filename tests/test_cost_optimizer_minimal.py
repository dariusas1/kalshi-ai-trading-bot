"""
Minimal test for CostOptimizer class structure.
Tests basic functionality without full database integration.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.intelligence.cost_optimizer import (
    CostOptimizer,
    BudgetStatus,
    CacheEntry,
    CostOptimizationConfig,
    DynamicCostModel
)


class TestCostOptimizerMinimal(unittest.TestCase):
    """Minimal test for CostOptimizer basic functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.loop = None

    def test_config_initialization(self):
        """Test that CostOptimizationConfig initializes correctly."""
        config = CostOptimizationConfig()

        assert config.enable_dynamic_modeling is True
        assert config.daily_budget_limit == 50.0
        assert config.enable_budget_controls is True
        assert config.enable_intelligent_caching is True
        assert config.enable_real_time_monitoring is True
        assert config.cache_ttl_minutes == 30
        assert config.max_cache_size == 1000

    def test_config_custom_values(self):
        """Test CostOptimizationConfig with custom values."""
        config = CostOptimizationConfig(
            daily_budget_limit=100.0,
            cache_ttl_minutes=60,
            budget_alert_threshold=0.9
        )

        assert config.daily_budget_limit == 100.0
        assert config.cache_ttl_minutes == 60
        assert config.budget_alert_threshold == 0.9

    def test_cache_entry_creation(self):
        """Test CacheEntry dataclass creation."""
        entry = CacheEntry(
            cache_key="test_key",
            model_name="grok-4",
            market_data={"category": "technology"},
            result={"action": "BUY"},
            timestamp=datetime.now(),
            cost=0.05,
            accuracy=0.8,
            ttl_minutes=30,
            hit_count=5
        )

        assert entry.cache_key == "test_key"
        assert entry.model_name == "grok-4"
        assert entry.cost == 0.05
        assert entry.accuracy == 0.8
        assert entry.hit_count == 5

    def test_dynamic_cost_model_creation(self):
        """Test DynamicCostModel dataclass creation."""
        model = DynamicCostModel(
            model_name="grok-4",
            market_category="technology",
            base_cost_per_request=0.05,
            cost_efficiency_factor=15.0,
            accuracy_cost_tradeoff=0.5,
            budget_sensitivity=0.3,
            performance_window_hours=24
        )

        assert model.model_name == "grok-4"
        assert model.market_category == "technology"
        assert model.base_cost_per_request == 0.05
        assert model.cost_efficiency_factor == 15.0

    def test_budget_status_creation(self):
        """Test BudgetStatus dataclass creation."""
        status = BudgetStatus(
            daily_limit=50.0,
            spent=20.0,
            remaining=30.0,
            percentage_used=40.0,
            status="healthy",
            projected_daily_spend=25.0,
            recommended_actions=["Continue monitoring"]
        )

        assert status.daily_limit == 50.0
        assert status.spent == 20.0
        assert status.remaining == 30.0
        assert status.percentage_used == 40.0
        assert status.status == "healthy"
        assert len(status.recommended_actions) == 1

    def test_volume_range_classification(self):
        """Test volume range classification logic."""
        # Create mock optimizer to test helper methods
        db_manager = MagicMock()
        config = CostOptimizationConfig()
        optimizer = CostOptimizer(db_manager, config)

        assert optimizer._get_volume_range(500) == "low"
        assert optimizer._get_volume_range(2500) == "medium"
        assert optimizer._get_volume_range(6000) == "high"

    def test_price_range_classification(self):
        """Test price range classification logic."""
        db_manager = MagicMock()
        config = CostOptimizationConfig()
        optimizer = CostOptimizer(db_manager, config)

        assert optimizer._get_price_range(0.2) == "low"
        assert optimizer._get_price_range(0.5) == "medium"
        assert optimizer._get_price_range(0.8) == "high"

    def test_cache_key_generation(self):
        """Test cache key generation for consistent caching."""
        db_manager = MagicMock()
        config = CostOptimizationConfig()
        optimizer = CostOptimizer(db_manager, config)

        market_data1 = {
            "category": "technology",
            "volume": 3000.0,
            "price": 0.65,
            "volatility": "medium"
        }

        market_data2 = {
            "category": "technology",
            "volume": 3200.0,
            "price": 0.66,
            "volatility": "medium"
        }

        key1 = optimizer._generate_cache_key("grok-4", market_data1)
        key2 = optimizer._generate_cache_key("grok-4", market_data2)
        key3 = optimizer._generate_cache_key("grok-3", market_data1)

        assert key1 == key2  # Similar data should generate same key
        assert key1 != key3  # Different model should generate different key
        assert len(key1) == 32  # MD5 hash length

    def test_cache_entry_validity(self):
        """Test cache entry validity checking."""
        db_manager = MagicMock()
        config = CostOptimizationConfig()
        optimizer = CostOptimizer(db_manager, config)

        # Valid entry
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

        # Expired entry
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

        assert optimizer._is_cache_entry_valid(valid_entry) is True
        assert optimizer._is_cache_entry_valid(expired_entry) is False

    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation logic."""
        db_manager = MagicMock()
        config = CostOptimizationConfig()
        optimizer = CostOptimizer(db_manager, config)

        # Mock cost metrics
        cost_metrics = MagicMock()
        cost_metrics.cost_performance_ratio = 0.15
        cost_metrics.roi_score = 8.0
        cost_metrics.budget_efficiency = 7.5

        # Mock model metrics
        model_metrics = MagicMock()
        model_metrics.avg_decision_quality = 0.7
        model_metrics.avg_response_time_ms = 1200

        efficiency_score = optimizer._calculate_efficiency_score(cost_metrics, model_metrics)

        assert 0.0 <= efficiency_score <= 1.0
        assert isinstance(efficiency_score, float)

    def test_cost_estimation(self):
        """Test model cost estimation."""
        db_manager = MagicMock()
        config = CostOptimizationConfig()
        optimizer = CostOptimizer(db_manager, config)

        # Test different trade values
        low_cost = optimizer._estimate_model_cost("grok-4", 5.0)
        normal_cost = optimizer._estimate_model_cost("grok-4", 50.0)
        high_cost = optimizer._estimate_model_cost("grok-4", 150.0)

        assert low_cost < normal_cost
        assert normal_cost < high_cost
        assert all(isinstance(cost, float) for cost in [low_cost, normal_cost, high_cost])
        assert all(cost > 0 for cost in [low_cost, normal_cost, high_cost])


if __name__ == '__main__':
    unittest.main()