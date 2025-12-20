"""
Tests for Multi-Model Performance Tracking System.

These tests validate the core performance tracking functionality including
accuracy calculations, confidence calibration, rolling window aggregation,
cost-per-performance ratios, pattern recognition, and data persistence.
"""

import asyncio
import os
import json
import unittest
from datetime import datetime, timedelta
from typing import Dict, Any, List
from src.utils.database import DatabaseManager, ModelPerformance, ModelHealth
from src.utils.performance_tracker import PerformanceTracker, ModelPerformanceMetrics

TEST_DB = "test_performance_tracking.db"


class TestAccuracyCalculation(unittest.IsolatedAsyncioTestCase):
    """Test accuracy calculation across different market conditions."""

    async def test_accuracy_by_market_category(self):
        """Test accuracy calculation segmented by market category."""
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

        db_manager = DatabaseManager(db_path=TEST_DB)
        await db_manager.initialize()

        try:
            tracker = PerformanceTracker(db_manager)

            # Simulate predictions for different market categories
            predictions = [
                # Technology markets - good performance
                {"model_name": "grok-4", "market_category": "technology", "predicted_outcome": True, "actual_outcome": True, "confidence": 0.8},
                {"model_name": "grok-4", "market_category": "technology", "predicted_outcome": False, "actual_outcome": False, "confidence": 0.7},
                {"model_name": "grok-4", "market_category": "technology", "predicted_outcome": True, "actual_outcome": True, "confidence": 0.9},

                # Finance markets - mixed performance
                {"model_name": "grok-4", "market_category": "finance", "predicted_outcome": True, "actual_outcome": False, "confidence": 0.6},
                {"model_name": "grok-4", "market_category": "finance", "predicted_outcome": True, "actual_outcome": True, "confidence": 0.8},
                {"model_name": "grok-4", "market_category": "finance", "predicted_outcome": False, "actual_outcome": True, "confidence": 0.7},
            ]

            # Record predictions
            for pred in predictions:
                await tracker.record_prediction_result(
                    model_name=pred["model_name"],
                    market_category=pred["market_category"],
                    predicted_outcome=pred["predicted_outcome"],
                    actual_outcome=pred["actual_outcome"],
                    confidence=pred["confidence"],
                    response_time_ms=150,
                    cost_usd=0.01
                )

            # Calculate accuracy by category
            tech_accuracy = await tracker.get_model_accuracy("grok-4", market_category="technology")
            finance_accuracy = await tracker.get_model_accuracy("grok-4", market_category="finance")

            # Technology should have 100% accuracy (3/3 correct)
            assert abs(tech_accuracy - 1.0) < 1e-6, f"Expected 1.0, got {tech_accuracy}"

            # Finance should have 33.3% accuracy (1/3 correct)
            assert abs(finance_accuracy - 0.333) < 1e-2, f"Expected 0.333, got {finance_accuracy}"

            # Overall accuracy should be 66.7% (4/6 correct)
            overall_accuracy = await tracker.get_model_accuracy("grok-4")
            assert abs(overall_accuracy - 0.667) < 1e-2, f"Expected 0.667, got {overall_accuracy}"

        finally:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)

    async def test_accuracy_by_volatility_regime(self):
        """Test accuracy calculation considering market volatility."""
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

        db_manager = DatabaseManager(db_path=TEST_DB)
        await db_manager.initialize()

        try:
            tracker = PerformanceTracker(db_manager)

            # Simulate predictions in different volatility regimes
            predictions = [
                # Low volatility - better accuracy
                {"model_name": "grok-3", "volatility_regime": "low", "predicted_outcome": True, "actual_outcome": True},
                {"model_name": "grok-3", "volatility_regime": "low", "predicted_outcome": False, "actual_outcome": False},
                {"model_name": "grok-3", "volatility_regime": "low", "predicted_outcome": True, "actual_outcome": True},

                # High volatility - lower accuracy
                {"model_name": "grok-3", "volatility_regime": "high", "predicted_outcome": True, "actual_outcome": False},
                {"model_name": "grok-3", "volatility_regime": "high", "predicted_outcome": False, "actual_outcome": True},
                {"model_name": "grok-3", "volatility_regime": "high", "predicted_outcome": True, "actual_outcome": False},
            ]

            # Record predictions with metadata
            for pred in predictions:
                await tracker.record_prediction_result(
                    model_name=pred["model_name"],
                    market_category="test",
                    predicted_outcome=pred["predicted_outcome"],
                    actual_outcome=pred["actual_outcome"],
                    confidence=0.75,
                    response_time_ms=120,
                    cost_usd=0.008,
                    volatility_regime=pred["volatility_regime"]
                )

            # Get accuracy by volatility regime
            low_vol_accuracy = await tracker.get_model_accuracy("grok-3", volatility_regime="low")
            high_vol_accuracy = await tracker.get_model_accuracy("grok-3", volatility_regime="high")

            assert abs(low_vol_accuracy - 1.0) < 1e-6, f"Low volatility accuracy should be 1.0, got {low_vol_accuracy}"
            assert abs(high_vol_accuracy - 0.0) < 1e-6, f"High volatility accuracy should be 0.0, got {high_vol_accuracy}"

        finally:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)


class TestConfidenceCalibration(unittest.IsolatedAsyncioTestCase):
    """Test confidence calibration assessment."""

    async def test_calibration_curve_generation(self):
        """Test generation of calibration curves comparing confidence vs accuracy."""
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

        db_manager = DatabaseManager(db_path=TEST_DB)
        await db_manager.initialize()

        try:
            tracker = PerformanceTracker(db_manager)

            # Create predictions with varying confidence levels
            predictions = [
                # High confidence (0.8-1.0) - should be accurate
                {"confidence": 0.95, "predicted": True, "actual": True},
                {"confidence": 0.90, "predicted": True, "actual": True},
                {"confidence": 0.85, "predicted": False, "actual": False},

                # Medium confidence (0.6-0.8) - mixed results
                {"confidence": 0.75, "predicted": True, "actual": True},
                {"confidence": 0.70, "predicted": True, "actual": False},
                {"confidence": 0.65, "predicted": False, "actual": True},

                # Low confidence (0.4-0.6) - poor accuracy
                {"confidence": 0.55, "predicted": True, "actual": False},
                {"confidence": 0.50, "predicted": False, "actual": True},
                {"confidence": 0.45, "predicted": True, "actual": False},
            ]

            for pred in predictions:
                await tracker.record_prediction_result(
                    model_name="grok-4",
                    market_category="test",
                    predicted_outcome=pred["predicted"],
                    actual_outcome=pred["actual"],
                    confidence=pred["confidence"],
                    response_time_ms=100,
                    cost_usd=0.01
                )

            # Get calibration metrics
            calibration = await tracker.get_confidence_calibration("grok-4")

            # Check calibration buckets exist and have reasonable values
            high_conf_accuracy = calibration.high_confidence_accuracy
            med_conf_accuracy = calibration.medium_confidence_accuracy
            low_conf_accuracy = calibration.low_confidence_accuracy

            print(f"High confidence accuracy: {high_conf_accuracy}")
            print(f"Medium confidence accuracy: {med_conf_accuracy}")
            print(f"Low confidence accuracy: {low_conf_accuracy}")

            # At least one bucket should have data
            assert (high_conf_accuracy > 0 or med_conf_accuracy > 0 or low_conf_accuracy > 0), "At least one confidence bucket should have data"

            # Overall calibration score (closer to 0 is better)
            calibration_score = calibration.calibration_score
            assert 0 <= calibration_score <= 1, "Calibration score should be between 0 and 1"

            # Check that confidence buckets are populated correctly
            assert 'high' in calibration.confidence_buckets, "High confidence bucket should exist"
            assert 'medium' in calibration.confidence_buckets, "Medium confidence bucket should exist"
            assert 'low' in calibration.confidence_buckets, "Low confidence bucket should exist"

        finally:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)

    async def test_confidence_adjustment_factors(self):
        """Test calculation of confidence adjustment factors."""
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

        db_manager = DatabaseManager(db_path=TEST_DB)
        await db_manager.initialize()

        try:
            tracker = PerformanceTracker(db_manager)

            # Model that is overconfident (predicts high confidence but has low accuracy)
            overconfident_predictions = [
                {"confidence": 0.9, "predicted": True, "actual": False},
                {"confidence": 0.85, "predicted": True, "actual": False},
                {"confidence": 0.95, "predicted": False, "actual": True},
            ]

            for pred in overconfident_predictions:
                await tracker.record_prediction_result(
                    model_name="overconfident_model",
                    market_category="test",
                    predicted_outcome=pred["predicted"],
                    actual_outcome=pred["actual"],
                    confidence=pred["confidence"],
                    response_time_ms=100,
                    cost_usd=0.01
                )

            # Get adjustment factors
            adjustment_factors = await tracker.get_confidence_adjustment_factors("overconfident_model")

            # Should recommend reducing confidence
            high_conf_adjustment = adjustment_factors.get("high_confidence_adjustment", 1.0)
            assert high_conf_adjustment < 1.0, "Overconfident model should have confidence reduced"
            assert high_conf_adjustment > 0.1, "Adjustment should not be extreme"

        finally:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)


class TestRollingWindowAggregation(unittest.IsolatedAsyncioTestCase):
    """Test rolling window performance aggregation."""

    async def test_24h_rolling_window(self):
        """Test 24-hour rolling window performance calculation."""
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

        db_manager = DatabaseManager(db_path=TEST_DB)
        await db_manager.initialize()

        try:
            tracker = PerformanceTracker(db_manager)
            now = datetime.now()

            # Create predictions at different times
            predictions_times = [
                # 23 hours ago - should be included
                now - timedelta(hours=23),
                # 12 hours ago - should be included
                now - timedelta(hours=12),
                # 1 hour ago - should be included
                now - timedelta(hours=1),
                # 25 hours ago - should be excluded
                now - timedelta(hours=25),
            ]

            for i, pred_time in enumerate(predictions_times):
                await tracker.record_prediction_result(
                    model_name="grok-4",
                    market_category="test",
                    predicted_outcome=True,
                    actual_outcome=i < 3,  # First 3 are correct, last one is wrong
                    confidence=0.8,
                    response_time_ms=100,
                    cost_usd=0.01,
                    timestamp=pred_time
                )

            # Get 24h rolling metrics
            metrics_24h = await tracker.get_rolling_window_metrics("grok-4", window_hours=24)

            # Should include 3 predictions (exclude the 25-hour-old one)
            assert metrics_24h.total_predictions == 3, f"Should have 3 predictions in 24h window, got {metrics_24h.total_predictions}"

            # Should have 100% accuracy (3/3 correct)
            assert abs(metrics_24h.accuracy - 1.0) < 1e-6, "24h accuracy should be 100%"

            # Test that the 25h window includes all 4 predictions
            metrics_25h = await tracker.get_rolling_window_metrics("grok-4", window_hours=25)
            assert metrics_25h.total_predictions == 4, "Should have 4 predictions in 25h window"
            assert abs(metrics_25h.accuracy - 0.75) < 1e-6, "25h accuracy should be 75%"

        finally:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)

    async def test_7d_30d_windows(self):
        """Test 7-day and 30-day rolling windows."""
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

        db_manager = DatabaseManager(db_path=TEST_DB)
        await db_manager.initialize()

        try:
            tracker = PerformanceTracker(db_manager)
            now = datetime.now()

            # Create predictions spanning different time periods
            for days_ago in range(1, 35):  # Last 34 days
                pred_time = now - timedelta(days=days_ago)
                await tracker.record_prediction_result(
                    model_name="grok-3",
                    market_category="test",
                    predicted_outcome=days_ago % 2 == 0,  # Alternating correct/incorrect
                    actual_outcome=days_ago % 3 == 0,    # Some pattern
                    confidence=0.75,
                    response_time_ms=120,
                    cost_usd=0.008,
                    timestamp=pred_time
                )

            # Get 7-day metrics
            metrics_7d = await tracker.get_rolling_window_metrics("grok-3", window_hours=168)  # 7 days
            assert metrics_7d.total_predictions == 7, "Should have 7 predictions in 7-day window"

            # Get 30-day metrics
            metrics_30d = await tracker.get_rolling_window_metrics("grok-3", window_hours=720)  # 30 days
            assert metrics_30d.total_predictions == 30, "Should have 30 predictions in 30-day window"

            # Verify time windows are different
            assert metrics_7d.accuracy != metrics_30d.accuracy, "7d and 30d accuracies should differ"

        finally:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)


class TestCostPerformanceRatio(unittest.IsolatedAsyncioTestCase):
    """Test cost-per-performance ratio calculations."""

    async def test_cost_efficiency_calculation(self):
        """Test calculation of cost-per-performance ratios."""
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

        db_manager = DatabaseManager(db_path=TEST_DB)
        await db_manager.initialize()

        try:
            tracker = PerformanceTracker(db_manager)

            # Model A - high cost, high performance
            for i in range(10):
                await tracker.record_prediction_result(
                    model_name="expensive_model",
                    market_category="test",
                    predicted_outcome=True,
                    actual_outcome=True,  # 100% accuracy
                    confidence=0.9,
                    response_time_ms=80,
                    cost_usd=0.05  # $0.05 per prediction
                )

            # Model B - low cost, lower performance
            for i in range(10):
                await tracker.record_prediction_result(
                    model_name="cheap_model",
                    market_category="test",
                    predicted_outcome=True,
                    actual_outcome=i < 7,  # 70% accuracy
                    confidence=0.6,
                    response_time_ms=150,
                    cost_usd=0.01  # $0.01 per prediction
                )

            # Calculate cost-per-performance ratios
            expensive_metrics = await tracker.get_cost_performance_metrics("expensive_model")
            cheap_metrics = await tracker.get_cost_performance_metrics("cheap_model")

            # Expensive model: $0.50 total cost, 100% accuracy
            assert abs(expensive_metrics.total_cost - 0.50) < 1e-6, "Expensive model cost should be $0.50"
            assert abs(expensive_metrics.cost_per_correct_prediction - 0.05) < 1e-6, "Cost per correct should be $0.05"
            assert abs(expensive_metrics.cost_performance_ratio - 20.0) < 1e-6, "Cost-performance ratio should be 20"

            # Cheap model: $0.10 total cost, 70% accuracy, 7 correct predictions
            assert abs(cheap_metrics.total_cost - 0.10) < 1e-6, "Cheap model cost should be $0.10"
            assert abs(cheap_metrics.cost_per_correct_prediction - 0.0143) < 1e-2, "Cost per correct should be ~$0.014"
            assert abs(cheap_metrics.cost_performance_ratio - 70.0) < 1e-2, "Cost-performance ratio should be ~70"

            # Cheap model should have better cost-performance ratio
            assert cheap_metrics.cost_performance_ratio > expensive_metrics.cost_performance_ratio, \
                "Cheap model should have better cost-performance ratio"

        finally:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)


class TestPerformancePatternIdentification(unittest.IsolatedAsyncioTestCase):
    """Test model-specific performance pattern identification."""

    async def test_market_category_strengths(self):
        """Test identification of model strengths in specific market categories."""
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

        db_manager = DatabaseManager(db_path=TEST_DB)
        await db_manager.initialize()

        try:
            tracker = PerformanceTracker(db_manager)

            # Model A - excels in technology, struggles in finance
            for i in range(20):
                if i < 10:  # Technology markets
                    await tracker.record_prediction_result(
                        model_name="model_A",
                        market_category="technology",
                        predicted_outcome=True,
                        actual_outcome=i < 8,  # 80% accuracy
                        confidence=0.8,
                        response_time_ms=100,
                        cost_usd=0.01
                    )
                else:  # Finance markets
                    await tracker.record_prediction_result(
                        model_name="model_A",
                        market_category="finance",
                        predicted_outcome=True,
                        actual_outcome=i < 13,  # 30% accuracy
                        confidence=0.7,
                        response_time_ms=120,
                        cost_usd=0.012
                    )

            # Get model strengths
            strengths = await tracker.get_model_strengths("model_A")

            # Should identify technology as strength
            assert "technology" in strengths.strong_categories, "Technology should be identified as strong category"
            assert strengths.strong_categories["technology"] >= 0.7, "Technology accuracy should be >= 70%"

            # Should identify finance as weakness
            assert "finance" in strengths.weak_categories, "Finance should be identified as weak category"
            assert strengths.weak_categories["finance"] <= 0.5, "Finance accuracy should be <= 50%"

        finally:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)

    async def test_temporal_performance_patterns(self):
        """Test identification of temporal performance patterns."""
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

        db_manager = DatabaseManager(db_path=TEST_DB)
        await db_manager.initialize()

        try:
            tracker = PerformanceTracker(db_manager)
            now = datetime.now()

            # Simulate performance that degrades over time
            for days_ago in range(1, 31):
                pred_time = now - timedelta(days=days_ago)
                # More recent predictions are more accurate
                accuracy = 0.9 - (days_ago * 0.02)  # Degradation over time
                actual_outcome = days_ago <= 15  # First 15 days are correct

                await tracker.record_prediction_result(
                    model_name="degrading_model",
                    market_category="test",
                    predicted_outcome=True,
                    actual_outcome=actual_outcome,
                    confidence=0.75,
                    response_time_ms=100,
                    cost_usd=0.01,
                    timestamp=pred_time
                )

            # Analyze temporal patterns
            patterns = await tracker.get_temporal_performance_patterns("degrading_model")

            # Should detect performance trend
            assert "performance_trend" in patterns, "Should detect performance trend"
            assert patterns["performance_trend"]["direction"] == "declining", "Should detect declining performance"
            assert "recent_accuracy" in patterns, "Should have recent accuracy"
            assert "historical_accuracy" in patterns, "Should have historical accuracy"

        finally:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)


class TestDataPersistenceRetrieval(unittest.IsolatedAsyncioTestCase):
    """Test performance data persistence and retrieval."""

    async def test_data_persistence_accuracy(self):
        """Test that performance data is accurately persisted and retrieved."""
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

        db_manager = DatabaseManager(db_path=TEST_DB)
        await db_manager.initialize()

        try:
            tracker = PerformanceTracker(db_manager)

            # Record a variety of performance data
            test_data = [
                {
                    "model_name": "grok-4",
                    "market_category": "technology",
                    "predicted_outcome": True,
                    "actual_outcome": True,
                    "confidence": 0.85,
                    "response_time_ms": 145,
                    "cost_usd": 0.012,
                    "decision_quality": 0.9
                },
                {
                    "model_name": "grok-3",
                    "market_category": "finance",
                    "predicted_outcome": False,
                    "actual_outcome": True,
                    "confidence": 0.72,
                    "response_time_ms": 89,
                    "cost_usd": 0.008,
                    "decision_quality": 0.4
                }
            ]

            # Record the data
            recorded_ids = []
            for data in test_data:
                result_id = await tracker.record_prediction_result(**data)
                recorded_ids.append(result_id)

            # Verify data was recorded
            assert all(recorded_ids), "All records should be successfully created with IDs"

            # Retrieve the data
            for i, data in enumerate(test_data):
                # Query by model and time range
                metrics = await tracker.get_model_ranking(
                    model_name=data["model_name"],
                    start_time=datetime.now() - timedelta(hours=1),
                    end_time=datetime.now()
                )

                assert len(metrics) >= 1, f"Should retrieve at least one record for {data['model_name']}"

                # Verify the data integrity
                retrieved = metrics[0]
                assert retrieved.model_name == data["model_name"], "Model name should match"
                assert retrieved.market_category == data["market_category"], "Market category should match"

            # Test aggregated data retrieval
            all_metrics = await tracker.get_model_ranking()
            assert len(all_metrics) >= 2, "Should retrieve records for all models"

        finally:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)

    async def test_database_integrity(self):
        """Test database integrity and constraint enforcement."""
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)

        db_manager = DatabaseManager(db_path=TEST_DB)
        await db_manager.initialize()

        try:
            tracker = PerformanceTracker(db_manager)

            # Test valid data insertion
            valid_data = {
                "model_name": "test_model",
                "market_category": "test_category",
                "predicted_outcome": True,
                "actual_outcome": False,
                "confidence": 0.75,
                "response_time_ms": 100,
                "cost_usd": 0.01
            }

            result_id = await tracker.record_prediction_result(**valid_data)
            assert result_id, "Valid data should be successfully inserted"

            # Test data retrieval consistency
            retrieved = await db_manager.get_model_performance(
                model_name="test_model",
                start_time=datetime.now() - timedelta(hours=1)
            )

            assert len(retrieved) == 1, "Should retrieve exactly one record"
            assert retrieved[0].model_name == "test_model", "Retrieved model name should match"
            assert abs(retrieved[0].accuracy_score - 0.0) < 1e-6, "Accuracy should be 0.0 (wrong prediction)"

            # Test indexing by querying performance
            start_time = datetime.now() - timedelta(minutes=5)
            performance_records = await db_manager.get_model_performance(
                model_name="test_model",
                start_time=start_time
            )

            assert len(performance_records) >= 1, "Should find records by time range"

        finally:
            if os.path.exists(TEST_DB):
                os.remove(TEST_DB)


if __name__ == '__main__':
    unittest.main()