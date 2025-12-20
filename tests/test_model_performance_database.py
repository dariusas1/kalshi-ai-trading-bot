"""
Tests for model performance tracking database functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from src.utils.database import DatabaseManager
from src.utils.database import ModelPerformance, ModelHealth, EnsembleDecision


@pytest.fixture
async def db_manager():
    """Create a fresh database for testing."""
    # Use in-memory database for tests
    db_manager = DatabaseManager(":memory:")
    await db_manager.initialize()
    yield db_manager
    # Cleanup handled automatically for in-memory database


# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio


class TestModelPerformanceTracking:
    """Test suite for model performance tracking database functionality."""
    async def test_model_performance_record_creation_and_retrieval(self, db_manager):
        """Test creating and retrieving model performance records."""
        # Create a test performance record
        performance = ModelPerformance(
            model_name="grok-4",
            timestamp=datetime.now(),
            market_category="technology",
            accuracy_score=0.85,
            confidence_calibration=0.78,
            response_time_ms=1250,
            cost_usd=0.0042,
            decision_quality=0.92
        )

        # Save the record
        record_id = await db_manager.save_model_performance(performance)
        assert record_id is not None
        assert record_id > 0

        # Retrieve the record
        retrieved = await db_manager.get_model_performance(record_id)
        assert retrieved is not None
        assert retrieved.model_name == "grok-4"
        assert retrieved.accuracy_score == 0.85
        assert retrieved.market_category == "technology"
        assert retrieved.decision_quality == 0.92

        # Test multiple records for same model
        performance2 = ModelPerformance(
            model_name="grok-4",
            timestamp=datetime.now() + timedelta(minutes=5),
            market_category="finance",
            accuracy_score=0.79,
            confidence_calibration=0.82,
            response_time_ms=980,
            cost_usd=0.0038,
            decision_quality=0.88
        )

        record_id2 = await db_manager.save_model_performance(performance2)
        assert record_id2 is not None
        assert record_id2 != record_id

        # Get all records for the model
        model_records = await db_manager.get_model_performance_by_model("grok-4")
        assert len(model_records) == 2
        assert all(r.model_name == "grok-4" for r in model_records)

    async def test_performance_window_aggregation(self, db_manager):
        """Test performance aggregation across different time windows."""
        model_name = "gpt-4"
        base_time = datetime.now()

        # Create performance records over different time periods
        records = []
        for hours_ago in range(72, 0, -4):  # Every 4 hours for 3 days
            timestamp = base_time - timedelta(hours=hours_ago)

            # Vary performance to test aggregation
            accuracy = 0.75 + (hours_ago % 10) * 0.02  # Vary between 0.75-0.93
            cost = 0.003 + (hours_ago % 5) * 0.001  # Vary between 0.003-0.007

            record = ModelPerformance(
                model_name=model_name,
                timestamp=timestamp,
                market_category="mixed",
                accuracy_score=accuracy,
                confidence_calibration=accuracy - 0.05,  # Slightly lower
                response_time_ms=1000 + hours_ago * 10,
                cost_usd=cost,
                decision_quality=accuracy + 0.05
            )
            records.append(record)

        # Save all records
        for record in records:
            await db_manager.save_model_performance(record)

        # Test 24-hour window aggregation
        agg_24h = await db_manager.get_model_performance_aggregation(
            model_name, window_hours=24
        )
        assert agg_24h is not None
        assert agg_24h['record_count'] > 0
        assert 'avg_accuracy' in agg_24h
        assert 'total_cost' in agg_24h
        assert 'avg_response_time' in agg_24h
        assert 0 <= agg_24h['avg_accuracy'] <= 1
        assert agg_24h['total_cost'] > 0

        # Test 7-day window aggregation (should include all records)
        agg_7d = await db_manager.get_model_performance_aggregation(
            model_name, window_hours=168
        )
        assert agg_7d['record_count'] == len(records)
        assert agg_7d['total_cost'] > agg_24h['total_cost']

        # Test 30-day window (same as 7-day in our test data)
        agg_30d = await db_manager.get_model_performance_aggregation(
            model_name, window_hours=720
        )
        assert agg_30d['record_count'] == agg_7d['record_count']

    async def test_cost_tracking_and_budget_enforcement(self, db_manager):
        """Test cost tracking and budget enforcement functionality."""
        model_name = "claude-3-sonnet"

        # Create records with varying costs
        costs = [0.002, 0.003, 0.0025, 0.004, 0.0035]
        total_expected_cost = sum(costs)

        for i, cost in enumerate(costs):
            performance = ModelPerformance(
                model_name=model_name,
                timestamp=datetime.now() + timedelta(minutes=i*10),
                market_category="test",
                accuracy_score=0.80 + i * 0.02,
                confidence_calibration=0.75 + i * 0.01,
                response_time_ms=900 + i * 50,
                cost_usd=cost,
                decision_quality=0.85 + i * 0.01
            )
            await db_manager.save_model_performance(performance)

        # Test cost tracking for today
        today_cost = await db_manager.get_model_cost_today(model_name)
        assert abs(today_cost - total_expected_cost) < 0.0001  # Allow for floating point precision

        # Test budget enforcement
        daily_budget = 0.01  # Lower than actual cost

        # This should return False because we exceeded budget
        can_execute = await db_manager.check_model_budget(model_name, daily_budget)
        assert not can_execute

        # Test with higher budget
        higher_budget = 0.05  # Higher than actual cost
        can_execute = await db_manager.check_model_budget(model_name, higher_budget)
        assert can_execute

        # Test budget tracking for all models
        all_costs = await db_manager.get_all_model_costs_today()
        assert model_name in all_costs
        assert all_costs[model_name] == total_expected_cost

    async def test_model_availability_and_health_status_tracking(self, db_manager):
        """Test model health monitoring and availability tracking."""
        model_name = "grok-3"

        # Initialize model health as available
        health = ModelHealth(
            model_name=model_name,
            is_available=True,
            last_check_time=datetime.now(),
            consecutive_failures=0,
            avg_response_time=850
        )

        await db_manager.update_model_health(health)

        # Retrieve health status
        retrieved_health = await db_manager.get_model_health(model_name)
        assert retrieved_health is not None
        assert retrieved_health.model_name == model_name
        assert retrieved_health.is_available is True
        assert retrieved_health.consecutive_failures == 0
        assert retrieved_health.avg_response_time == 850

        # Simulate a failure
        failed_health = ModelHealth(
            model_name=model_name,
            is_available=False,
            last_check_time=datetime.now(),
            consecutive_failures=1,
            avg_response_time=5000  # Slow response indicates issues
        )

        await db_manager.update_model_health(failed_health)

        updated_health = await db_manager.get_model_health(model_name)
        assert updated_health.is_available is False
        assert updated_health.consecutive_failures == 1
        assert updated_health.avg_response_time == 5000

        # Simulate recovery
        recovered_health = ModelHealth(
            model_name=model_name,
            is_available=True,
            last_check_time=datetime.now(),
            consecutive_failures=0,  # Reset on recovery
            avg_response_time=900
        )

        await db_manager.update_model_health(recovered_health)

        final_health = await db_manager.get_model_health(model_name)
        assert final_health.is_available is True
        assert final_health.consecutive_failures == 0

        # Test getting all available models
        # Add another model
        other_model = "gpt-4-turbo"
        other_health = ModelHealth(
            model_name=other_model,
            is_available=True,
            last_check_time=datetime.now(),
            consecutive_failures=0,
            avg_response_time=1200
        )
        await db_manager.update_model_health(other_health)

        available_models = await db_manager.get_available_models()
        assert len(available_models) == 2
        assert model_name in available_models
        assert other_model in available_models

        # Make one model unavailable
        unavailable_health = ModelHealth(
            model_name=other_model,
            is_available=False,
            last_check_time=datetime.now(),
            consecutive_failures=3,
            avg_response_time=10000
        )
        await db_manager.update_model_health(unavailable_health)

        available_models = await db_manager.get_available_models()
        assert len(available_models) == 1
        assert model_name in available_models
        assert other_model not in available_models

    async def test_ensemble_decision_logging_and_analysis(self, db_manager):
        """Test ensemble decision logging and analysis functionality."""
        market_id = "TEST_MARKET_001"

        # Create an ensemble decision record
        decision = EnsembleDecision(
            market_id=market_id,
            models_consulted=["grok-4", "gpt-4", "claude-3-sonnet"],
            final_decision="YES",
            disagreement_level=0.3,  # Medium disagreement
            selected_model="grok-4",
            reasoning="Primary model grok-4 showed highest confidence. gpt-4 agreed but with lower confidence. claude-3-sonnet disagreed due to different interpretation of market signals."
        )

        decision_id = await db_manager.save_ensemble_decision(decision)
        assert decision_id is not None
        assert decision_id > 0

        # Retrieve the decision
        retrieved = await db_manager.get_ensemble_decision(decision_id)
        assert retrieved is not None
        assert retrieved.market_id == market_id
        assert retrieved.final_decision == "YES"
        assert len(retrieved.models_consulted) == 3
        assert "grok-4" in retrieved.models_consulted
        assert retrieved.selected_model == "grok-4"
        assert retrieved.disagreement_level == 0.3
        assert len(retrieved.reasoning) > 0

        # Test getting decisions by market
        market_decisions = await db_manager.get_ensemble_decisions_by_market(market_id)
        assert len(market_decisions) == 1
        assert market_decisions[0].market_id == market_id

        # Create another decision for the same market
        decision2 = EnsembleDecision(
            market_id=market_id,
            models_consulted=["gpt-4", "claude-3-sonnet"],
            final_decision="NO",
            disagreement_level=0.1,  # Low disagreement
            selected_model="gpt-4",
            reasoning="Both models agreed on negative outlook with high confidence."
        )

        await db_manager.save_ensemble_decision(decision2)

        # Should now have 2 decisions for this market
        market_decisions = await db_manager.get_ensemble_decisions_by_market(market_id)
        assert len(market_decisions) == 2

        # Test ensemble analysis
        analysis = await db_manager.get_ensemble_analysis(hours_back=24)
        assert analysis is not None
        assert 'total_decisions' in analysis
        assert 'avg_disagreement' in analysis
        assert 'most_selected_model' in analysis
        assert analysis['total_decisions'] >= 2
        assert 0 <= analysis['avg_disagreement'] <= 1

        # Test model usage statistics
        usage_stats = await db_manager.get_model_usage_stats(hours_back=24)
        assert usage_stats is not None
        assert 'grok-4' in usage_stats
        assert 'gpt-4' in usage_stats
        assert 'claude-3-sonnet' in usage_stats
        assert usage_stats['grok-4']['times_selected'] >= 1
        assert usage_stats['gpt-4']['times_selected'] >= 1

        # Test disagreement analysis
        disagreement_analysis = await db_manager.get_disagreement_analysis(hours_back=24)
        assert disagreement_analysis is not None
        assert 'high_disagreement_decisions' in disagreement_analysis
        assert 'low_disagreement_decisions' in disagreement_analysis
        assert 'avg_disagreement_by_decision_count' in disagreement_analysis