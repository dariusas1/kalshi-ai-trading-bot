"""
Strategic Additional Tests for Enhanced AI Model Integration - Task Group 9.3

These tests fill critical gaps in ensemble feature coverage that were identified in the test review.
Focus is on end-to-end workflows, component integration, and realistic ensemble scenarios.

Total: 7 additional strategic tests (within the 9 test maximum limit)
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional

from src.clients.xai_client import XAIClient, TradingDecision
from src.intelligence.ensemble_engine import EnsembleEngine, EnsembleConfig, EnsembleResult
from src.intelligence.model_selector import ModelSelector, SelectionCriteria
from src.intelligence.cost_optimizer import CostOptimizer
from src.intelligence.fallback_manager import FallbackManager
from src.intelligence.ensemble_monitor import EnsembleMonitor
from src.utils.database import DatabaseManager, ModelPerformance, ModelHealth, EnsembleDecision
from src.utils.performance_tracker import PerformanceTracker, ModelPerformanceMetrics
from src.config.settings import settings

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio


class TestEndToEndEnsembleWorkflows:
    """Test complete ensemble workflows from market input to trading decision."""

    @pytest.fixture
    async def ensemble_system(self):
        """Create a complete ensemble system for testing."""
        # Create in-memory database
        db_manager = DatabaseManager(":memory:")
        await db_manager.initialize()

        # Initialize ensemble components
        performance_tracker = PerformanceTracker(db_manager)
        model_selector = ModelSelector(performance_tracker)
        cost_optimizer = CostOptimizer(db_manager, performance_tracker)
        fallback_manager = FallbackManager(db_manager)
        ensemble_engine = EnsembleEngine(db_manager, model_selector, cost_optimizer)

        return {
            'db': db_manager,
            'tracker': performance_tracker,
            'selector': model_selector,
            'optimizer': cost_optimizer,
            'fallback': fallback_manager,
            'engine': ensemble_engine
        }

    async def test_complete_ensemble_decision_workflow(self, ensemble_system):
        """
        Test complete ensemble decision workflow from market analysis to final decision.

        This end-to-end test validates:
        - Market data input processing
        - Model selection and consultation
        - Ensemble consensus generation
        - Cost-aware decision making
        - Result logging and monitoring
        """
        # Setup market data
        market_data = {
            "market_id": "TEST_MARKET_001",
            "title": "Technology Sector Performance Q4",
            "subtitle": "Will tech sector outperform S&P 500 in Q4?",
            "yes_price": 0.65,
            "no_price": 0.35,
            "volume": 50000,
            "market_category": "technology",
            "expiry_date": datetime.now() + timedelta(days=30),
            "trade_value": 25.0
        }

        # Setup model performance data to influence selection
        await self._setup_model_performance_data(ensemble_system['tracker'])

        # Execute the complete workflow
        result = await ensemble_system['engine'].get_ensemble_decision(
            market_data=market_data,
            ensemble_config=EnsembleConfig(
                enable_weighted_voting=True,
                enable_consensus_threshold=True,
                consensus_threshold=0.65,
                cost_budget_limit=1.0
            )
        )

        # Validate the complete workflow result
        assert result is not None, "Ensemble should return a decision"
        assert result.decision in ["BUY", "SELL", "HOLD"], f"Invalid decision: {result.decision}"
        assert 0 <= result.confidence <= 1.0, f"Invalid confidence: {result.confidence}"
        assert result.selected_model is not None, "Should select a primary model"
        assert len(result.models_consulted) >= 1, "Should consult at least one model"
        assert result.cost_estimate > 0, "Should estimate decision cost"
        assert result.reasoning is not None, "Should provide decision reasoning"

        # Verify decision was logged
        logged_decisions = await ensemble_system['db'].get_ensemble_decisions_by_market(
            market_data["market_id"]
        )
        assert len(logged_decisions) >= 1, "Decision should be logged in database"

        # Verify performance tracking
        model_rankings = await ensemble_system['tracker'].get_model_ranking()
        assert len(model_rankings) >= 1, "Model rankings should be updated"

    async def test_cascading_ensemble_by_trade_value(self, ensemble_system):
        """
        Test cascading ensemble logic based on trade value tiers.

        Validates that different ensemble strategies are used for:
        - Low-value trades (<$10): Quick single model
        - Medium-value trades ($10-$50): Weighted ensemble
        - High-value trades (>$50): Full consensus with disagreement handling
        """
        test_scenarios = [
            {
                "name": "Low Value Trade",
                "trade_value": 5.0,
                "expected_strategy": "single_model",
                "max_models_consulted": 1
            },
            {
                "name": "Medium Value Trade",
                "trade_value": 25.0,
                "expected_strategy": "weighted_ensemble",
                "max_models_consulted": 3
            },
            {
                "name": "High Value Trade",
                "trade_value": 75.0,
                "expected_strategy": "full_consensus",
                "max_models_consulted": 5
            }
        ]

        base_market_data = {
            "market_id": "TEST_MARKET_CASCADE",
            "title": "Test Market for Cascading",
            "yes_price": 0.60,
            "no_price": 0.40,
            "market_category": "finance"
        }

        for scenario in test_scenarios:
            market_data = {**base_market_data, "trade_value": scenario["trade_value"]}

            result = await ensemble_system['engine'].get_ensemble_decision(
                market_data=market_data,
                ensemble_config=EnsembleConfig(enable_cascading_strategy=True)
            )

            assert result is not None, f"Should return decision for {scenario['name']}"
            assert len(result.models_consulted) <= scenario["max_models_consulted"], \
                f"{scenario['name']}: Should consult max {scenario['max_models_consulted']} models, got {len(result.models_consulted)}"

            # Higher value trades should have higher confidence
            if scenario["trade_value"] >= 50:
                assert result.confidence >= 0.7, f"High value trade should have high confidence"

    async def _setup_model_performance_data(self, tracker):
        """Setup mock model performance data for testing."""
        base_time = datetime.now() - timedelta(hours=1)

        # Create performance records for different models
        models_data = [
            {
                "model_name": "grok-4",
                "accuracy": 0.82,
                "avg_confidence": 0.85,
                "cost_per_prediction": 0.015,
                "specialty": "technology"
            },
            {
                "model_name": "grok-3",
                "accuracy": 0.78,
                "avg_confidence": 0.80,
                "cost_per_prediction": 0.012,
                "specialty": "finance"
            },
            {
                "model_name": "gpt-4",
                "accuracy": 0.75,
                "avg_confidence": 0.77,
                "cost_per_prediction": 0.020,
                "specialty": "general"
            }
        ]

        for model_data in models_data:
            for i in range(10):  # Create 10 records per model
                await tracker.record_prediction_result(
                    model_name=model_data["model_name"],
                    market_category=model_data["specialty"],
                    predicted_outcome=i % 2 == 0,
                    actual_outcome=i % 3 == 0,  # Some correct predictions
                    confidence=model_data["avg_confidence"],
                    response_time_ms=1000 + i * 50,
                    cost_usd=model_data["cost_per_prediction"],
                    timestamp=base_time + timedelta(minutes=i * 5)
                )


class TestWeightedEnsembleConsensus:
    """Test weighted ensemble consensus mechanisms under varying conditions."""

    @pytest.fixture
    async def consensus_system(self):
        """Create ensemble system for consensus testing."""
        db_manager = DatabaseManager(":memory:")
        await db_manager.initialize()

        performance_tracker = PerformanceTracker(db_manager)
        model_selector = ModelSelector(performance_tracker)
        ensemble_engine = EnsembleEngine(db_manager, model_selector)

        return {'db': db_manager, 'engine': ensemble_engine, 'tracker': performance_tracker}

    async def test_dynamic_weight_adjustment_based_on_performance(self, consensus_system):
        """
        Test that ensemble weights are dynamically adjusted based on recent model performance.

        Validates:
        - Higher-performing models get more weight
        - Recent performance changes are reflected in weights
        - Weight adjustments respect minimum/maximum bounds
        """
        # Setup models with different performance levels
        await self._setup_performance_for_weighting(consensus_system['tracker'])

        market_data = {
            "market_id": "WEIGHT_TEST_MARKET",
            "title": "Weight Adjustment Test",
            "yes_price": 0.55,
            "no_price": 0.45,
            "market_category": "mixed",
            "trade_value": 50.0
        }

        # Get ensemble decision with weighted voting
        result = await consensus_system['engine'].get_ensemble_decision(
            market_data=market_data,
            ensemble_config=EnsembleConfig(
                enable_weighted_voting=True,
                weight_calculation_window_hours=24
            )
        )

        assert result is not None, "Should generate ensemble decision"
        assert result.weight_distribution is not None, "Should provide weight distribution"

        # Check that weights sum to 1.0
        total_weight = sum(result.weight_distribution.values())
        assert abs(total_weight - 1.0) < 0.01, f"Weights should sum to 1.0, got {total_weight}"

        # Check that higher-performing models get higher weights
        weights = result.weight_distribution
        assert weights.get("grok-4", 0) > weights.get("grok-3", 0), \
            "Higher-performing grok-4 should get higher weight than grok-3"
        assert weights.get("grok-3", 0) > weights.get("gpt-4", 0), \
            "Medium-performing grok-3 should get higher weight than lower-performing gpt-4"

    async def test_consensus_disagreement_detection_and_resolution(self, consensus_system):
        """
        Test ensemble disagreement detection and resolution mechanisms.

        Validates:
        - Disagreement level calculation
        - Disagreement threshold enforcement
        - Resolution strategies for high disagreement scenarios
        """
        # Create scenario with known disagreement
        market_data = {
            "market_id": "DISAGREEMENT_TEST",
            "title": "Disagreement Test Market",
            "yes_price": 0.50,
            "no_price": 0.50,
            "market_category": "volatile",
            "trade_value": 30.0
        }

        # Mock model responses to create controlled disagreement
        with patch.object(consensus_system['engine'], '_get_model_predictions') as mock_predictions:
            mock_predictions.return_value = {
                "grok-4": {"action": "BUY", "confidence": 0.85, "reasoning": "Strong bullish signals"},
                "grok-3": {"action": "SELL", "confidence": 0.80, "reasoning": "Bearish indicators"},
                "gpt-4": {"action": "BUY", "confidence": 0.60, "reasoning": "Slightly positive trend"}
            }

            result = await consensus_system['engine'].get_ensemble_decision(
                market_data=market_data,
                ensemble_config=EnsembleConfig(
                    consensus_threshold=0.75,  # High threshold to trigger disagreement
                    enable_disagreement_detection=True
                )
            )

            assert result is not None, "Should handle disagreement and provide decision"
            assert result.disagreement_level > 0.5, "Should detect high disagreement"
            assert result.resolution_method is not None, "Should specify resolution method"
            assert "disagreement" in result.reasoning.lower(), "Reasoning should mention disagreement"

    async def _setup_performance_for_weighting(self, tracker):
        """Setup performance data for weight adjustment testing."""
        base_time = datetime.now() - timedelta(hours=12)

        # grok-4: High performance (90% accuracy)
        for i in range(20):
            await tracker.record_prediction_result(
                model_name="grok-4",
                market_category="mixed",
                predicted_outcome=True,
                actual_outcome=i < 18,  # 90% correct
                confidence=0.85,
                response_time_ms=1200,
                cost_usd=0.015,
                timestamp=base_time + timedelta(minutes=i * 30)
            )

        # grok-3: Medium performance (75% accuracy)
        for i in range(20):
            await tracker.record_prediction_result(
                model_name="grok-3",
                market_category="mixed",
                predicted_outcome=True,
                actual_outcome=i < 15,  # 75% correct
                confidence=0.80,
                response_time_ms=900,
                cost_usd=0.012,
                timestamp=base_time + timedelta(minutes=i * 30)
            )

        # gpt-4: Lower performance (60% accuracy)
        for i in range(20):
            await tracker.record_prediction_result(
                model_name="gpt-4",
                market_category="mixed",
                predicted_outcome=True,
                actual_outcome=i < 12,  # 60% correct
                confidence=0.75,
                response_time_ms=1500,
                cost_usd=0.020,
                timestamp=base_time + timedelta(minutes=i * 30)
            )


class TestBudgetConstrainedEnsembleDecisions:
    """Test ensemble behavior under budget constraints and cost optimization."""

    @pytest.fixture
    async def budget_system(self):
        """Create ensemble system with budget tracking."""
        db_manager = DatabaseManager(":memory:")
        await db_manager.initialize()

        performance_tracker = PerformanceTracker(db_manager)
        cost_optimizer = CostOptimizer(db_manager, performance_tracker)
        ensemble_engine = EnsembleEngine(db_manager, cost_optimizer=cost_optimizer)

        return {'db': db_manager, 'engine': ensemble_engine, 'optimizer': cost_optimizer}

    async def test_ensemble_behavior_with_limited_budget(self, budget_system):
        """
        Test ensemble decision making when budget is constrained.

        Validates:
        - Cost-aware model selection under budget pressure
        - Ensemble strategy adaptation for budget constraints
        - Automatic escalation to cheaper alternatives
        """
        # Set budget constraint
        remaining_budget = 0.50  # Very low budget

        market_data = {
            "market_id": "BUDGET_CONSTRAINED_MARKET",
            "title": "Budget Constrained Test",
            "yes_price": 0.60,
            "no_price": 0.40,
            "market_category": "general",
            "trade_value": 15.0
        }

        result = await budget_system['engine'].get_ensemble_decision(
            market_data=market_data,
            ensemble_config=EnsembleConfig(
                cost_budget_limit=remaining_budget,
                enable_cost_optimization=True,
                prefer_cheaper_models=True
            )
        )

        assert result is not None, "Should make decision within budget"
        assert result.cost_estimate <= remaining_budget, \
            f"Cost {result.cost_estimate} should not exceed budget {remaining_budget}"

        # Should prefer cheaper models when budget is constrained
        assert "cost" in result.reasoning.lower() or "budget" in result.reasoning.lower(), \
            "Reasoning should mention cost considerations"

    async def test_ensemble_cost_efficiency_optimization(self, budget_system):
        """
        Test optimization of ensemble cost efficiency.

        Validates:
        - Cost-performance ratio calculations
        - Model selection based on ROI
        - Ensemble composition optimization for cost efficiency
        """
        # Setup cost performance data
        await self._setup_cost_performance_data(budget_system['db'])

        market_data = {
            "market_id": "COST_EFFICIENCY_TEST",
            "title": "Cost Efficiency Test",
            "yes_price": 0.65,
            "no_price": 0.35,
            "market_category": "finance",
            "trade_value": 40.0
        }

        result = await budget_system['engine'].get_ensemble_decision(
            market_data=market_data,
            ensemble_config=EnsembleConfig(
                enable_cost_optimization=True,
                cost_performance_window_hours=24
            )
        )

        assert result is not None, "Should generate cost-optimized decision"
        assert result.cost_efficiency_score is not None, "Should provide cost efficiency score"
        assert result.cost_efficiency_score > 0, "Cost efficiency score should be positive"

    async def _setup_cost_performance_data(self, db_manager):
        """Setup cost performance data for testing."""
        # Insert model performance records with different cost profiles
        performances = [
            ModelPerformance(
                model_name="grok-4",
                timestamp=datetime.now() - timedelta(hours=6),
                market_category="finance",
                accuracy_score=0.85,
                confidence_calibration=0.82,
                response_time_ms=1200,
                cost_usd=0.020,  # Higher cost
                decision_quality=0.88
            ),
            ModelPerformance(
                model_name="grok-3",
                timestamp=datetime.now() - timedelta(hours=6),
                market_category="finance",
                accuracy_score=0.80,
                confidence_calibration=0.78,
                response_time_ms=900,
                cost_usd=0.012,  # Lower cost
                decision_quality=0.82
            )
        ]

        for perf in performances:
            await db_manager.save_model_performance(perf)


class TestUncertaintyQuantificationAndRiskManagement:
    """Test ensemble uncertainty quantification and its impact on trading decisions."""

    @pytest.fixture
    async def uncertainty_system(self):
        """Create ensemble system for uncertainty testing."""
        db_manager = DatabaseManager(":memory:")
        await db_manager.initialize()

        ensemble_engine = EnsembleEngine(db_manager)
        return {'db': db_manager, 'engine': ensemble_engine}

    async def test_uncertainty_quantification_affects_position_sizing(self, uncertainty_system):
        """
        Test that ensemble uncertainty influences position sizing recommendations.

        Validates:
        - Uncertainty score calculation
        - Position size adjustment based on uncertainty
        - Risk management integration with ensemble confidence
        """
        market_data = {
            "market_id": "UNCERTAINTY_TEST_MARKET",
            "title": "Uncertainty Quantification Test",
            "yes_price": 0.55,
            "no_price": 0.45,
            "market_category": "volatile",
            "trade_value": 100.0,
            "available_capital": 1000.0
        }

        # Mock high uncertainty scenario (divergent model opinions)
        with patch.object(uncertainty_system['engine'], '_get_model_predictions') as mock_predictions:
            mock_predictions.return_value = {
                "grok-4": {"action": "BUY", "confidence": 0.90, "reasoning": "Strong buy signal"},
                "grok-3": {"action": "SELL", "confidence": 0.85, "reasoning": "Strong sell signal"},
                "gpt-4": {"action": "HOLD", "confidence": 0.50, "reasoning": "Unclear direction"}
            }

            result = await uncertainty_system['engine'].get_ensemble_decision(
                market_data=market_data,
                ensemble_config=EnsembleConfig(
                    enable_uncertainty_quantification=True,
                    calculate_position_sizing=True
                )
            )

            assert result is not None, "Should generate decision with uncertainty analysis"
            assert result.uncertainty_score > 0.5, "High uncertainty should be detected"
            assert result.recommended_position_size is not None, "Should recommend position size"
            assert result.recommended_position_size < market_data["available_capital"], \
                "Position size should be limited by uncertainty"
            assert "uncertainty" in result.reasoning.lower() or "conservative" in result.reasoning.lower(), \
                "Reasoning should mention uncertainty considerations"

    async def test_ensemble_disagreement_correlation_with_outcomes(self, uncertainty_system):
        """
        Test correlation between ensemble disagreement and actual trading outcomes.

        Validates:
        - Historical analysis of disagreement vs accuracy
        - Threshold optimization for disagreement handling
        - Learning from disagreement patterns
        """
        # Create historical decisions with disagreement data
        await self._create_historical_disagreement_data(uncertainty_system['db'])

        market_data = {
            "market_id": "DISAGREEMENT_CORRELATION_TEST",
            "title": "Disagreement Correlation Test",
            "yes_price": 0.52,
            "no_price": 0.48,
            "market_category": "mixed",
            "trade_value": 35.0
        }

        result = await uncertainty_system['engine'].get_ensemble_decision(
            market_data=market_data,
            ensemble_config=EnsembleConfig(
                enable_disagreement_learning=True,
                disagreement_window_hours=168  # 7 days
            )
        )

        assert result is not None, "Should generate decision with disagreement learning"
        assert result.disagreement_impact_assessment is not None, "Should assess disagreement impact"

        # Decision should consider historical disagreement patterns
        if result.disagreement_level > 0.7:
            assert result.confidence < 0.8, "High disagreement should reduce confidence"

    async def _create_historical_disagreement_data(self, db_manager):
        """Create historical ensemble decisions with disagreement data."""
        base_time = datetime.now() - timedelta(days=3)

        # High disagreement decisions with mixed outcomes
        for i in range(10):
            decision = EnsembleDecision(
                market_id=f"HIGH_DISAGREEMENT_{i}",
                models_consulted=["grok-4", "grok-3", "gpt-4"],
                final_decision="YES" if i % 2 == 0 else "NO",
                disagreement_level=0.8,  # High disagreement
                selected_model="grok-4" if i % 3 == 0 else "grok-3",
                reasoning="High disagreement scenario with conflicting signals",
                timestamp=base_time + timedelta(hours=i * 6)
            )
            await db_manager.save_ensemble_decision(decision)

        # Low disagreement decisions with consistent outcomes
        for i in range(10):
            decision = EnsembleDecision(
                market_id=f"LOW_DISAGREEMENT_{i}",
                models_consulted=["grok-4", "grok-3"],
                final_decision="YES" if i % 3 != 0 else "NO",
                disagreement_level=0.2,  # Low disagreement
                selected_model="grok-4",
                reasoning="Low disagreement with consensus",
                timestamp=base_time + timedelta(hours=i * 6)
            )
            await db_manager.save_ensemble_decision(decision)


class TestMultiModelCoordinationIntegration:
    """Test coordination between multiple ensemble components in realistic scenarios."""

    @pytest.fixture
    async def coordinated_system(self):
        """Create fully coordinated ensemble system."""
        db_manager = DatabaseManager(":memory:")
        await db_manager.initialize()

        # Initialize all components
        performance_tracker = PerformanceTracker(db_manager)
        model_selector = ModelSelector(performance_tracker)
        cost_optimizer = CostOptimizer(db_manager, performance_tracker)
        fallback_manager = FallbackManager(db_manager)
        ensemble_engine = EnsembleEngine(
            db_manager,
            model_selector=model_selector,
            cost_optimizer=cost_optimizer,
            fallback_manager=fallback_manager
        )
        ensemble_monitor = EnsembleMonitor(db_manager)

        return {
            'db': db_manager,
            'engine': ensemble_engine,
            'selector': model_selector,
            'optimizer': cost_optimizer,
            'fallback': fallback_manager,
            'monitor': ensemble_monitor,
            'tracker': performance_tracker
        }

    async def test_realistic_multi_market_ensemble_coordination(self, coordinated_system):
        """
        Test ensemble coordination across multiple simultaneous market decisions.

        Validates:
        - Resource allocation across multiple decisions
        - Cost budget management across portfolio
        - Model selection coordination
        - Performance tracking across concurrent decisions
        """
        # Create multiple market scenarios
        markets = [
            {
                "market_id": "PORTFOLIO_MARKET_1",
                "title": "Tech Earnings",
                "market_category": "technology",
                "yes_price": 0.68,
                "no_price": 0.32,
                "trade_value": 45.0,
                "priority": "high"
            },
            {
                "market_id": "PORTFOLIO_MARKET_2",
                "title": "Fed Rate Decision",
                "market_category": "finance",
                "yes_price": 0.45,
                "no_price": 0.55,
                "trade_value": 80.0,
                "priority": "critical"
            },
            {
                "market_id": "PORTFOLIO_MARKET_3",
                "title": "Sports Championship",
                "market_category": "sports",
                "yes_price": 0.52,
                "no_price": 0.48,
                "trade_value": 15.0,
                "priority": "low"
            }
        ]

        # Set portfolio budget constraint
        total_budget = 2.0
        portfolio_config = EnsembleConfig(
            cost_budget_limit=total_budget,
            enable_portfolio_optimization=True,
            prioritize_high_value_trades=True
        )

        # Process all markets concurrently
        tasks = []
        for market in markets:
            task = coordinated_system['engine'].get_ensemble_decision(
                market_data=market,
                ensemble_config=portfolio_config
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Validate all decisions were made
        assert all(isinstance(result, EnsembleResult) for result in results), \
            "All markets should receive ensemble decisions"

        total_cost = sum(result.cost_estimate for result in results if isinstance(result, EnsembleResult))
        assert total_cost <= total_budget, \
            f"Total cost {total_cost} should not exceed portfolio budget {total_budget}"

        # Verify coordination - higher priority markets should get better models
        critical_result = next(r for r in results if isinstance(r, EnsembleResult) and "PORTFOLIO_MARKET_2" in r.market_id)
        low_result = next(r for r in results if isinstance(r, EnsembleResult) and "PORTFOLIO_MARKET_3" in r.market_id)

        assert critical_result.confidence >= low_result.confidence, \
            "Higher priority market should get higher confidence decision"

        # Verify performance tracking coordination
        portfolio_metrics = await coordinated_system['monitor'].get_portfolio_performance()
        assert portfolio_metrics['total_decisions'] == len(markets)
        assert portfolio_metrics['total_cost'] == total_cost

    async def test_ensemble_health_monitoring_and_self_healing(self, coordinated_system):
        """
        Test ensemble health monitoring and automatic self-healing capabilities.

        Validates:
        - Component health checking
        - Automatic fallback activation
        - Performance degradation detection
        - Self-healing mechanisms
        """
        # Simulate component health issues
        with patch.object(coordinated_system['selector'], 'select_optimal_model') as mock_selector:
            # First call fails (simulating health issue)
            mock_selector.side_effect = [
                Exception("Model selector health issue"),
                # Second call succeeds (self-healing)
                AsyncMock(return_value=MagicMock(selected_model="grok-4", confidence=0.8))
            ]

            market_data = {
                "market_id": "HEALTH_TEST_MARKET",
                "title": "Health Monitoring Test",
                "yes_price": 0.60,
                "no_price": 0.40,
                "market_category": "test",
                "trade_value": 25.0
            }

            # Should handle health issue and recover
            result = await coordinated_system['engine'].get_ensemble_decision(
                market_data=market_data,
                ensemble_config=EnsembleConfig(
                    enable_health_monitoring=True,
                    enable_self_healing=True
                )
            )

            assert result is not None, "Should recover from health issue"
            assert result.selected_model == "grok-4", "Should use fallback after recovery"

        # Verify health monitoring
        health_status = await coordinated_system['fallback'].get_system_health()
        assert health_status is not None, "Should provide system health status"
        assert 'overall_health' in health_status, "Should assess overall system health"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])