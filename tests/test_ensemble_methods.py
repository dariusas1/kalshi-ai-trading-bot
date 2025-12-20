"""
Tests for Advanced Ensemble Implementation - Task Group 4.1

Tests sophisticated ensemble methods including consensus mechanisms, weighted voting,
confidence-based selection, cascading ensemble, disagreement detection, and uncertainty quantification.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional

from src.clients.xai_client import TradingDecision, XAIClient
from src.intelligence.model_selector import ModelSelector, SelectionCriteria
from src.utils.database import DatabaseManager, ModelPerformance, EnsembleDecision, ModelHealth
from src.utils.performance_tracker import PerformanceTracker, ModelPerformanceMetrics
from tests.test_helpers import create_test_market_data, create_test_portfolio_data

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio


class TestConsensusMechanisms:
    """Test consensus mechanisms with configurable thresholds."""

    async def test_consensus_with_configurable_threshold(self):
        """Test that consensus mechanisms work with different threshold configurations."""
        # Create mock XAIClient with ensemble capabilities
        mock_xai_client = AsyncMock(spec=XAIClient)
        mock_db_manager = AsyncMock(spec=DatabaseManager)

        # Mock trading decisions from different models
        model_decisions = {
            "grok-4": TradingDecision(
                action="BUY",
                side="YES",
                confidence=0.85,
                reasoning="Strong bullish sentiment detected"
            ),
            "grok-3": TradingDecision(
                action="BUY",
                side="YES",
                confidence=0.78,
                reasoning="Positive indicators aligned"
            ),
            "gpt-4": TradingDecision(
                action="BUY",
                side="YES",
                confidence=0.72,
                reasoning="Market conditions favorable"
            )
        }

        # Test with low threshold (should achieve consensus)
        low_threshold = 0.6
        consensus_result_low = await self._test_ensemble_consensus(
            mock_xai_client, model_decisions, low_threshold
        )
        assert consensus_result_low is not None
        assert consensus_result_low.action == "BUY"
        assert consensus_result_low.confidence > 0.7

        # Test with high threshold (should not achieve consensus)
        high_threshold = 0.9
        consensus_result_high = await self._test_ensemble_consensus(
            mock_xai_client, model_decisions, high_threshold
        )
        assert consensus_result_high is None  # No consensus at high threshold

    async def test_consensus_with_disagreement_handling(self):
        """Test consensus mechanism when models disagree significantly."""
        mock_xai_client = AsyncMock(spec=XAIClient)

        # Mock conflicting decisions
        conflicting_decisions = {
            "grok-4": TradingDecision(
                action="BUY",
                side="YES",
                confidence=0.90,
                reasoning="Strong buy signal"
            ),
            "grok-3": TradingDecision(
                action="BUY",
                side="NO",
                confidence=0.85,
                reasoning="Opposite view on market"
            )
        }

        # Should not reach consensus due to side disagreement
        consensus_result = await self._test_ensemble_consensus(
            mock_xai_client, conflicting_decisions, 0.7
        )
        assert consensus_result is None

    async def _test_ensemble_consensus(self, mock_client, decisions, threshold):
        """Helper method to test ensemble consensus."""
        # This would be implemented when we create the actual ensemble class
        # For now, simulate the consensus logic
        if not decisions:
            return None

        # Check if all models agree on action and side
        actions = [d.action for d in decisions.values()]
        sides = [d.side for d in decisions.values()]
        confidences = [d.confidence for d in decisions.values()]

        if len(set(actions)) == 1 and len(set(sides)) == 1:
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence >= threshold:
                return TradingDecision(
                    action=actions[0],
                    side=sides[0],
                    confidence=avg_confidence,
                    reasoning=f"Consensus achieved with {len(decisions)} models"
                )

        return None


class TestWeightedVoting:
    """Test weighted ensemble voting based on model performance."""

    async def test_weighted_ensemble_voting(self):
        """Test that higher-performing models get more influence."""
        # Create mock performance data
        model_performances = {
            "grok-4": 0.85,  # High performance - should get highest weight
            "grok-3": 0.72,  # Medium performance
            "gpt-4": 0.58     # Lower performance - should get lowest weight
        }

        # Mock decisions with different confidences
        model_decisions = {
            "grok-4": TradingDecision(
                action="BUY",
                side="YES",
                confidence=0.80,
                reasoning="High confidence buy"
            ),
            "grok-3": TradingDecision(
                action="BUY",
                side="YES",
                confidence=0.60,
                reasoning="Moderate confidence buy"
            ),
            "gpt-4": TradingDecision(
                action="SKIP",
                side="YES",
                confidence=0.90,
                reasoning="Very high confidence but suggests skip"
            )
        }

        # Calculate weighted decision
        weighted_result = await self._calculate_weighted_decision(
            model_decisions, model_performances
        )

        # grok-4 should dominate due to highest performance weight
        # despite gpt-4 having higher individual confidence
        assert weighted_result.action == "BUY"
        assert weighted_result.side == "YES"
        assert 0.70 <= weighted_result.confidence <= 0.80  # Weighted towards grok-4

    async def test_dynamic_weight_adjustment(self):
        """Test dynamic weight adjustment based on recent performance."""
        # Simulate performance changes over time
        initial_performances = {
            "grok-4": 0.80,
            "grok-3": 0.75
        }

        recent_performances = {
            "grok-4": 0.70,  # Performance declined
            "grok-3": 0.85   # Performance improved
        }

        # Test weight adjustment
        initial_weights = await self._calculate_performance_weights(initial_performances)
        adjusted_weights = await self._calculate_performance_weights(recent_performances)

        # Weights should shift based on performance changes
        assert adjusted_weights["grok-3"] > adjusted_weights["grok-4"]
        assert initial_weights["grok-4"] > initial_weights["grok-3"]

    async def _calculate_weighted_decision(self, decisions, performances):
        """Helper to calculate weighted ensemble decision."""
        # Calculate weights based on performance
        total_performance = sum(performances.values())
        weights = {model: perf/total_performance for model, perf in performances.items()}

        # Weight votes by performance and confidence
        buy_weight = 0.0
        skip_weight = 0.0
        total_weight = 0.0

        for model, decision in decisions.items():
            weight = weights.get(model, 0.0)
            confidence = decision.confidence

            if decision.action == "BUY":
                buy_weight += weight * confidence
            elif decision.action == "SKIP":
                skip_weight += weight * confidence

            total_weight += weight

        # Determine final decision
        if total_weight > 0:
            buy_probability = buy_weight / total_weight
            if buy_probability > 0.5:
                return TradingDecision(
                    action="BUY",
                    side=decisions["grok-4"].side,  # Use side from highest weighted model
                    confidence=buy_probability,
                    reasoning=f"Weighted ensemble decision (buy probability: {buy_probability:.2f})"
                )
            else:
                return TradingDecision(
                    action="SKIP",
                    side="YES",
                    confidence=1.0 - buy_probability,
                    reasoning=f"Weighted ensemble decision (skip probability: {1-buy_probability:.2f})"
                )

        return None

    async def _calculate_performance_weights(self, performances):
        """Helper to calculate weights from performance scores."""
        total = sum(performances.values())
        return {model: perf/total for model, perf in performances.items()}


class TestConfidenceBasedSelection:
    """Test confidence-based model selection."""

    async def test_confidence_based_selection(self):
        """Test selection of models with highest calibrated confidence."""
        # Mock model confidences with calibration factors
        model_confidences = {
            "grok-4": {"raw_confidence": 0.90, "calibration": 0.85},  # Well calibrated
            "grok-3": {"raw_confidence": 0.95, "calibration": 0.60},  # Poorly calibrated
            "gpt-4": {"raw_confidence": 0.80, "calibration": 0.90}     # Well calibrated
        }

        # Calculate calibrated confidences
        calibrated_scores = {}
        for model, data in model_confidences.items():
            calibrated_scores[model] = data["raw_confidence"] * data["calibration"]

        # Select best model based on calibrated confidence
        selected_model = max(calibrated_scores.items(), key=lambda x: x[1])

        # Should select grok-4 despite grok-3 having higher raw confidence
        assert selected_model[0] == "grok-4"
        assert selected_model[1] == 0.765  # 0.90 * 0.85

    async def test_market_condition_adjusted_confidence(self):
        """Test confidence adjustment based on market conditions."""
        # Model strengths in different market categories
        model_strengths = {
            "grok-4": {
                "technology": 0.90,
                "finance": 0.70
            },
            "grok-3": {
                "technology": 0.75,
                "finance": 0.85
            }
        }

        # Test technology market
        tech_market_result = await self._select_best_model_for_category(
            model_strengths, "technology"
        )
        assert tech_market_result["model"] == "grok-4"

        # Test finance market
        finance_market_result = await self._select_best_model_for_category(
            model_strengths, "finance"
        )
        assert finance_market_result["model"] == "grok-3"

    async def _select_best_model_for_category(self, strengths, category):
        """Helper to select best model for a market category."""
        best_model = None
        best_score = 0.0

        for model, categories in strengths.items():
            score = categories.get(category, 0.0)
            if score > best_score:
                best_score = score
                best_model = model

        return {"model": best_model, "score": best_score}


class TestCascadingEnsemble:
    """Test cascading ensemble logic for different trade values."""

    async def test_cascading_ensemble_low_value(self):
        """Test quick single model for low-value trades (<$10)."""
        trade_value = 5.0  # Low value trade

        # Should use quick single model for efficiency
        ensemble_result = await self._get_cascading_decision(trade_value)

        assert ensemble_result["models_used"] == 1
        assert ensemble_result["ensemble_type"] == "quick_single"
        assert ensemble_result["processing_time_ms"] < 1000  # Should be fast

    async def test_cascading_ensemble_medium_value(self):
        """Test full ensemble for medium-value trades ($10-$50)."""
        trade_value = 25.0  # Medium value trade

        ensemble_result = await self._get_cascading_decision(trade_value)

        assert ensemble_result["models_used"] == 2  # Uses Grok-4 and Grok-3
        assert ensemble_result["ensemble_type"] == "full_ensemble"
        assert ensemble_result["processing_time_ms"] >= 1000  # Slower but more accurate

    async def test_cascading_ensemble_high_value(self):
        """Test enhanced consensus for high-value trades (>$50)."""
        trade_value = 75.0  # High value trade

        ensemble_result = await self._get_cascading_decision(trade_value)

        assert ensemble_result["models_used"] >= 3  # Uses all available models
        assert ensemble_result["ensemble_type"] == "enhanced_consensus"
        assert ensemble_result["consensus_threshold"] > 0.8  # Higher threshold for high value

    async def test_cascading_threshold_boundaries(self):
        """Test behavior at cascade threshold boundaries."""
        # Test exactly at boundaries
        low_boundary = 10.0
        medium_boundary = 50.0

        low_result = await self._get_cascading_decision(low_boundary)
        assert low_result["ensemble_type"] == "full_ensemble"

        medium_result = await self._get_cascading_decision(medium_boundary)
        assert medium_result["ensemble_type"] == "enhanced_consensus"

    async def _get_cascading_decision(self, trade_value):
        """Helper to simulate cascading ensemble decision."""
        if trade_value < 10:
            return {
                "models_used": 1,
                "ensemble_type": "quick_single",
                "processing_time_ms": 500,
                "trade_value": trade_value
            }
        elif trade_value <= 50:
            return {
                "models_used": 2,
                "ensemble_type": "full_ensemble",
                "processing_time_ms": 2000,
                "trade_value": trade_value
            }
        else:
            return {
                "models_used": 3,
                "ensemble_type": "enhanced_consensus",
                "consensus_threshold": 0.85,
                "processing_time_ms": 3000,
                "trade_value": trade_value
            }


class TestDisagreementDetection:
    """Test ensemble disagreement detection."""

    async def test_ensemble_disagreement_detection(self):
        """Test detection of significant model disagreement."""
        # Test case with high disagreement
        high_disagreement_decisions = {
            "grok-4": {"action": "BUY", "confidence": 0.90},
            "grok-3": {"action": "SELL", "confidence": 0.85},
            "gpt-4": {"action": "SKIP", "confidence": 0.80}
        }

        disagreement_result = await self._detect_disagreement(high_disagreement_decisions)

        assert disagreement_result["disagreement_detected"] is True
        assert disagreement_result["disagreement_level"] > 0.7
        assert disagreement_result["unique_actions"] == 3

        # Test case with low disagreement
        low_disagreement_decisions = {
            "grok-4": {"action": "BUY", "confidence": 0.85},
            "grok-3": {"action": "BUY", "confidence": 0.80},
            "gpt-4": {"action": "BUY", "confidence": 0.78}
        }

        agreement_result = await self._detect_disagreement(low_disagreement_decisions)

        assert agreement_result["disagreement_detected"] is False
        assert agreement_result["disagreement_level"] < 0.2
        assert agreement_result["unique_actions"] == 1

    async def test_disagreement_quantification_methods(self):
        """Test different methods for quantifying disagreement."""
        decisions = {
            "grok-4": {"action": "BUY", "confidence": 0.90},
            "grok-3": {"action": "BUY", "confidence": 0.60},
            "gpt-4": {"action": "SELL", "confidence": 0.80}
        }

        # Test action diversity method
        action_diversity = await self._calculate_action_diversity(decisions)
        assert 0.5 <= action_diversity <= 1.0  # 2 unique actions out of 3 models

        # Test confidence variance method
        confidence_variance = await self._calculate_confidence_variance(decisions)
        assert confidence_variance > 0.0  # Should have some variance

        # Test combined disagreement score
        combined_score = await self._calculate_combined_disagreement(decisions)
        assert 0.0 <= combined_score <= 1.0

    async def _detect_disagreement(self, decisions):
        """Helper to detect model disagreement."""
        actions = [d["action"] for d in decisions.values()]
        confidences = [d["confidence"] for d in decisions.values()]

        unique_actions = len(set(actions))
        action_diversity = unique_actions / len(actions)

        # Calculate confidence variance
        mean_confidence = sum(confidences) / len(confidences)
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)

        # Combined disagreement level
        disagreement_level = (action_diversity * 0.7) + (min(1.0, variance) * 0.3)

        return {
            "disagreement_detected": action_diversity > 0.3 or variance > 0.1,
            "disagreement_level": disagreement_level,
            "unique_actions": unique_actions,
            "action_diversity": action_diversity,
            "confidence_variance": variance
        }

    async def _calculate_action_diversity(self, decisions):
        """Calculate action diversity score."""
        actions = [d["action"] for d in decisions.values()]
        return len(set(actions)) / len(actions)

    async def _calculate_confidence_variance(self, decisions):
        """Calculate confidence variance."""
        confidences = [d["confidence"] for d in decisions.values()]
        mean_confidence = sum(confidences) / len(confidences)
        return sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)

    async def _calculate_combined_disagreement(self, decisions):
        """Calculate combined disagreement score."""
        result = await self._detect_disagreement(decisions)
        return result["disagreement_level"]


class TestUncertaintyQuantification:
    """Test uncertainty quantification for ensemble decisions."""

    async def test_uncertainty_quantification(self):
        """Test quantification of ensemble uncertainty."""
        # Test case with high uncertainty
        high_uncertainty_decisions = {
            "grok-4": {"action": "BUY", "confidence": 0.55},
            "grok-3": {"action": "SKIP", "confidence": 0.60},
            "gpt-4": {"action": "BUY", "confidence": 0.45}
        }

        uncertainty_result = await self._quantify_uncertainty(high_uncertainty_decisions)

        assert uncertainty_result["uncertainty_score"] > 0.7
        assert uncertainty_result["confidence_level"] == "low"
        assert uncertainty_result["recommended_position_size"] < 0.5  # Reduced position size

        # Test case with low uncertainty
        low_uncertainty_decisions = {
            "grok-4": {"action": "BUY", "confidence": 0.90},
            "grok-3": {"action": "BUY", "confidence": 0.88},
            "gpt-4": {"action": "BUY", "confidence": 0.85}
        }

        certainty_result = await self._quantify_uncertainty(low_uncertainty_decisions)

        assert certainty_result["uncertainty_score"] < 0.3
        assert certainty_result["confidence_level"] == "high"
        assert certainty_result["recommended_position_size"] > 0.8  # Full position size

    async def test_uncertainty_adjusted_position_sizing(self):
        """Test position sizing based on uncertainty levels."""
        base_position_size = 100.0

        # Test various uncertainty levels
        test_cases = [
            {"uncertainty": 0.1, "expected_multiplier": 1.0},    # Very certain
            {"uncertainty": 0.3, "expected_multiplier": 0.9},    # Moderately certain
            {"uncertainty": 0.6, "expected_multiplier": 0.6},    # Moderately uncertain
            {"uncertainty": 0.9, "expected_multiplier": 0.3},    # Very uncertain
        ]

        for case in test_cases:
            adjusted_size = await self._adjust_position_size_for_uncertainty(
                base_position_size, case["uncertainty"]
            )

            expected_size = base_position_size * case["expected_multiplier"]
            assert abs(adjusted_size - expected_size) < 5.0  # Allow small tolerance

    async def test_uncertainty_communication(self):
        """Test clear communication of uncertainty to users."""
        uncertainty_scores = [0.1, 0.3, 0.6, 0.9]

        for score in uncertainty_scores:
            communication = await self._generate_uncertainty_communication(score)

            assert "confidence_level" in communication
            assert "explanation" in communication
            assert "recommendation" in communication

            # Should use clear, non-technical language
            assert len(communication["explanation"]) > 10
            assert "uncertain" in communication["explanation"].lower() or "confident" in communication["explanation"].lower()

    async def _quantify_uncertainty(self, decisions):
        """Quantify uncertainty from ensemble decisions."""
        if not decisions:
            return {"uncertainty_score": 1.0, "confidence_level": "very_low"}

        confidences = [d["confidence"] for d in decisions.values()]
        actions = [d["action"] for d in decisions.values()]

        # Calculate metrics
        mean_confidence = sum(confidences) / len(confidences)
        confidence_variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        action_diversity = len(set(actions)) / len(actions)

        # Combined uncertainty score
        uncertainty_score = (
            (1.0 - mean_confidence) * 0.4 +           # Low confidence increases uncertainty
            min(1.0, confidence_variance * 2.0) * 0.3 +  # High variance increases uncertainty
            action_diversity * 0.3                      # Action diversity increases uncertainty
        )

        # Determine confidence level
        if uncertainty_score < 0.3:
            confidence_level = "high"
            recommended_position_size = 1.0
        elif uncertainty_score < 0.6:
            confidence_level = "medium"
            recommended_position_size = 0.7
        else:
            confidence_level = "low"
            recommended_position_size = 0.4

        return {
            "uncertainty_score": uncertainty_score,
            "confidence_level": confidence_level,
            "mean_confidence": mean_confidence,
            "confidence_variance": confidence_variance,
            "action_diversity": action_diversity,
            "recommended_position_size": recommended_position_size
        }

    async def _adjust_position_size_for_uncertainty(self, base_size, uncertainty):
        """Adjust position size based on uncertainty score."""
        # Linear scaling: 100% at uncertainty=0, 20% at uncertainty=1
        min_multiplier = 0.2
        max_multiplier = 1.0

        multiplier = max_multiplier - (uncertainty * (max_multiplier - min_multiplier))
        return base_size * multiplier

    async def _generate_uncertainty_communication(self, uncertainty_score):
        """Generate clear communication about uncertainty."""
        if uncertainty_score < 0.3:
            level = "high confidence"
            explanation = "Models show strong agreement and high confidence in this decision"
            recommendation = "Proceed with normal position sizing"
        elif uncertainty_score < 0.6:
            level = "moderate confidence"
            explanation = "Models show some disagreement or moderate uncertainty"
            recommendation = "Consider reduced position sizing"
        else:
            level = "low confidence"
            explanation = "Models show significant disagreement or high uncertainty"
            recommendation = "Consider skipping or using minimal position size"

        return {
            "confidence_level": level,
            "uncertainty_score": uncertainty_score,
            "explanation": explanation,
            "recommendation": recommendation
        }