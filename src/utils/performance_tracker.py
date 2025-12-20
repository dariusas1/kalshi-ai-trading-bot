"""
Performance Tracker for Multi-Model AI System.

This module provides comprehensive performance tracking for AI models including
accuracy calculations, confidence calibration, rolling window analysis,
cost-per-performance metrics, and pattern recognition.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import statistics

from src.utils.database import DatabaseManager, ModelPerformance
from src.utils.logging_setup import TradingLoggerMixin


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive performance metrics for a model."""
    model_name: str
    total_predictions: int
    correct_predictions: int
    accuracy: float
    avg_confidence: float
    avg_response_time_ms: float
    total_cost: float
    cost_per_correct_prediction: float
    cost_performance_ratio: float
    avg_decision_quality: float
    timestamp_window: Tuple[datetime, datetime]


@dataclass
class RollingWindowMetrics:
    """Metrics for a specific rolling time window."""
    window_hours: int
    total_predictions: int
    accuracy: float
    avg_confidence: float
    avg_response_time_ms: float
    total_cost: float
    start_time: datetime
    end_time: datetime


@dataclass
class ConfidenceCalibrationMetrics:
    """Metrics for confidence calibration analysis."""
    calibration_score: float
    high_confidence_accuracy: float
    medium_confidence_accuracy: float
    low_confidence_accuracy: float
    confidence_buckets: Dict[str, List[Dict[str, Any]]]
    adjustment_factors: Dict[str, float]


@dataclass
class ModelStrengths:
    """Identified model strengths and weaknesses."""
    strong_categories: Dict[str, float]
    weak_categories: Dict[str, float]
    preferred_conditions: List[str]
    avoided_conditions: List[str]
    overall_reliability: float


@dataclass
class CostPerformanceMetrics:
    """Cost efficiency metrics for a model."""
    total_cost: float
    cost_per_prediction: float
    cost_per_correct_prediction: float
    cost_performance_ratio: float
    roi_score: float
    budget_efficiency: float


class PerformanceTracker(TradingLoggerMixin):
    """
    Advanced performance tracking system for AI models.

    Tracks accuracy, confidence calibration, cost efficiency, and patterns
    to enable intelligent model selection and ensemble optimization.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize performance tracker.

        Args:
            db_manager: DatabaseManager instance for persistence
        """
        self.db_manager = db_manager
        self.logger.info("Performance tracker initialized")

    async def record_prediction_result(
        self,
        model_name: str,
        market_category: str,
        predicted_outcome: bool,
        actual_outcome: bool,
        confidence: float,
        response_time_ms: int,
        cost_usd: float,
        timestamp: Optional[datetime] = None,
        volatility_regime: Optional[str] = None,
        time_to_expiry: Optional[int] = None,
        decision_quality: Optional[float] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        Record a prediction result for performance tracking.

        Args:
            model_name: Name of the AI model
            market_category: Category of the market
            predicted_outcome: What the model predicted
            actual_outcome: What actually happened
            confidence: Model's confidence level (0-1)
            response_time_ms: Response time in milliseconds
            cost_usd: Cost of this prediction in USD
            timestamp: When the prediction was made (defaults to now)
            volatility_regime: Market volatility regime (low/medium/high)
            time_to_expiry: Hours until market expiry
            decision_quality: Quality of the decision made (0-1)
            additional_metadata: Additional context information

        Returns:
            Database record ID if successful, None otherwise
        """
        try:
            # Calculate accuracy score
            accuracy_score = 1.0 if predicted_outcome == actual_outcome else 0.0

            # Calculate confidence calibration (how well confidence matches accuracy)
            confidence_calibration = await self._calculate_confidence_calibration(
                confidence, accuracy_score
            )

            # Default decision quality if not provided
            if decision_quality is None:
                decision_quality = accuracy_score * confidence

            # Create performance record
            performance = ModelPerformance(
                model_name=model_name,
                timestamp=timestamp or datetime.now(),
                market_category=market_category,
                accuracy_score=accuracy_score,
                confidence_calibration=confidence_calibration,
                response_time_ms=response_time_ms,
                cost_usd=cost_usd,
                decision_quality=decision_quality
            )

            # Save to database
            record_id = await self.db_manager.save_model_performance(performance)

            if record_id:
                # Store original confidence in metadata for calibration analysis
                metadata = additional_metadata or {}
                metadata['original_confidence'] = confidence

                # Store additional metadata in a separate table for complex queries
                if metadata or volatility_regime or time_to_expiry:
                    await self._store_prediction_metadata(
                        record_id, volatility_regime, time_to_expiry, metadata
                    )

                self.logger.debug(
                    f"Recorded prediction result for {model_name}",
                    accuracy=accuracy_score,
                    confidence=confidence,
                    cost_usd=cost_usd
                )

            return record_id

        except Exception as e:
            self.logger.error(f"Error recording prediction result for {model_name}: {e}")
            return None

    async def _calculate_confidence_calibration(
        self, confidence: float, actual_accuracy: float
    ) -> float:
        """Calculate how well confidence matches actual accuracy."""
        # Perfect calibration when confidence == actual_accuracy
        calibration_error = abs(confidence - actual_accuracy)

        # Convert to calibration score (0-1, where 1 is perfect)
        calibration_score = 1.0 - calibration_error
        return max(0.0, min(1.0, calibration_score))

    async def _store_prediction_metadata(
        self,
        record_id: int,
        volatility_regime: Optional[str],
        time_to_expiry: Optional[int],
        additional_metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Store additional metadata for complex analysis."""
        try:
            metadata = {}
            if volatility_regime:
                metadata['volatility_regime'] = volatility_regime
            if time_to_expiry is not None:
                metadata['time_to_expiry'] = time_to_expiry
            if additional_metadata:
                metadata.update(additional_metadata)

            if metadata:
                # Store as JSON in a notes column or separate metadata table
                # For now, we'll use the existing database structure
                pass

        except Exception as e:
            self.logger.warning(f"Error storing prediction metadata: {e}")

    async def get_model_accuracy(
        self,
        model_name: str,
        market_category: Optional[str] = None,
        volatility_regime: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> float:
        """
        Get accuracy for a model with optional filters.

        Args:
            model_name: Name of the model
            market_category: Filter by market category
            volatility_regime: Filter by volatility regime
            start_time: Start of time window
            end_time: End of time window

        Returns:
            Accuracy score (0-1)
        """
        try:
            records = await self.db_manager.get_model_performance_by_model(model_name, limit=1000)

            if not records:
                return 0.0

            # Apply filters
            filtered_records = []
            for record in records:
                # Time filter
                if start_time and record.timestamp < start_time:
                    continue
                if end_time and record.timestamp > end_time:
                    continue

                # Category filter
                if market_category and record.market_category != market_category:
                    continue

                filtered_records.append(record)

            if not filtered_records:
                return 0.0

            # Calculate accuracy
            correct_predictions = sum(1 for r in filtered_records if r.accuracy_score == 1.0)
            accuracy = correct_predictions / len(filtered_records)

            return accuracy

        except Exception as e:
            self.logger.error(f"Error getting accuracy for {model_name}: {e}")
            return 0.0

    async def get_model_ranking(
        self,
        model_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_predictions: int = 10
    ) -> List[ModelPerformanceMetrics]:
        """
        Get ranking of models by performance metrics.

        Args:
            model_name: Filter by specific model (None for all models)
            start_time: Start of analysis window
            end_time: End of analysis window
            min_predictions: Minimum predictions required for ranking

        Returns:
            List of performance metrics ranked by accuracy
        """
        try:
            # Get all models or specific model
            if model_name:
                models = [model_name]
            else:
                models = await self.db_manager.get_available_models()
                if not models:
                    return []

            rankings = []

            for model in models:
                # Get performance records
                records = await self.db_manager.get_model_performance_by_model(model, limit=1000)

                if not records:
                    continue

                # Apply time filters
                if start_time or end_time:
                    filtered_records = []
                    for record in records:
                        if start_time and record.timestamp < start_time:
                            continue
                        if end_time and record.timestamp > end_time:
                            continue
                        filtered_records.append(record)
                    records = filtered_records

                # Skip if not enough predictions
                if len(records) < min_predictions:
                    continue

                # Calculate metrics
                total_predictions = len(records)
                correct_predictions = sum(1 for r in records if r.accuracy_score == 1.0)
                accuracy = correct_predictions / total_predictions

                confidence_values = [self._extract_confidence(r) for r in records]
                avg_confidence = statistics.mean(confidence_values)
                avg_response_time = statistics.mean([r.response_time_ms for r in records])
                total_cost = sum(r.cost_usd for r in records)
                avg_decision_quality = statistics.mean([r.decision_quality for r in records])

                # Calculate cost metrics
                cost_per_correct = total_cost / correct_predictions if correct_predictions > 0 else float('inf')
                cost_performance_ratio = accuracy / (total_cost + 0.001)  # Avoid division by zero

                # Create metrics object
                metrics = ModelPerformanceMetrics(
                    model_name=model,
                    total_predictions=total_predictions,
                    correct_predictions=correct_predictions,
                    accuracy=accuracy,
                    avg_confidence=avg_confidence,
                    avg_response_time_ms=avg_response_time,
                    total_cost=total_cost,
                    cost_per_correct_prediction=cost_per_correct,
                    cost_performance_ratio=cost_performance_ratio,
                    avg_decision_quality=avg_decision_quality,
                    timestamp_window=(
                        min(r.timestamp for r in records),
                        max(r.timestamp for r in records)
                    )
                )

                rankings.append(metrics)

            # Sort by accuracy (descending)
            rankings.sort(key=lambda x: x.accuracy, reverse=True)

            return rankings

        except Exception as e:
            self.logger.error(f"Error getting model ranking: {e}")
            return []

    def _extract_confidence(self, record: ModelPerformance) -> float:
        """Extract confidence from performance record."""
        # For now, use estimation based on confidence calibration and accuracy
        # In a real implementation, this would be stored separately
        return min(1.0, record.confidence_calibration + record.accuracy_score) / 2

    async def get_rolling_window_metrics(
        self,
        model_name: str,
        window_hours: int = 24,
        volatility_regime: Optional[str] = None
    ) -> RollingWindowMetrics:
        """
        Get performance metrics for a rolling time window.

        Args:
            model_name: Name of the model
            window_hours: Size of the rolling window in hours
            volatility_regime: Filter by volatility regime

        Returns:
            Rolling window metrics
        """
        try:
            # Get aggregation from database
            aggregation = await self.db_manager.get_model_performance_aggregation(
                model_name, window_hours
            )

            if not aggregation:
                return RollingWindowMetrics(
                    window_hours=window_hours,
                    total_predictions=0,
                    accuracy=0.0,
                    avg_confidence=0.0,
                    avg_response_time_ms=0.0,
                    total_cost=0.0,
                    start_time=datetime.now() - timedelta(hours=window_hours),
                    end_time=datetime.now()
                )

            return RollingWindowMetrics(
                window_hours=window_hours,
                total_predictions=aggregation['record_count'],
                accuracy=aggregation['avg_accuracy'],
                avg_confidence=aggregation.get('avg_confidence_calibration', 0.5),
                avg_response_time_ms=aggregation['avg_response_time'],
                total_cost=aggregation['total_cost'],
                start_time=datetime.now() - timedelta(hours=window_hours),
                end_time=datetime.now()
            )

        except Exception as e:
            self.logger.error(f"Error getting rolling window metrics for {model_name}: {e}")
            return RollingWindowMetrics(
                window_hours=window_hours,
                total_predictions=0,
                accuracy=0.0,
                avg_confidence=0.0,
                avg_response_time_ms=0.0,
                total_cost=0.0,
                start_time=datetime.now() - timedelta(hours=window_hours),
                end_time=datetime.now()
            )

    async def get_confidence_calibration(
        self,
        model_name: str,
        window_hours: int = 168  # 7 days
    ) -> ConfidenceCalibrationMetrics:
        """
        Analyze confidence calibration for a model.

        Args:
            model_name: Name of the model to analyze
            window_hours: Time window for analysis

        Returns:
            Confidence calibration metrics
        """
        try:
            records = await self.db_manager.get_model_performance_by_model(model_name, limit=1000)

            # Filter by time window
            cutoff_time = datetime.now() - timedelta(hours=window_hours)
            records = [r for r in records if r.timestamp >= cutoff_time]

            if not records:
                return ConfidenceCalibrationMetrics(
                    calibration_score=0.0,
                    high_confidence_accuracy=0.0,
                    medium_confidence_accuracy=0.0,
                    low_confidence_accuracy=0.0,
                    confidence_buckets={},
                    adjustment_factors={}
                )

            # Create confidence buckets
            high_confidence = []  # > 0.8
            medium_confidence = []  # 0.6-0.8
            low_confidence = []    # < 0.6

            for record in records:
                confidence = self._extract_confidence(record)

                if confidence > 0.8:
                    high_confidence.append(record)
                elif confidence >= 0.6:
                    medium_confidence.append(record)
                else:
                    low_confidence.append(record)

            # Calculate accuracy by bucket
            high_acc = sum(r.accuracy_score for r in high_confidence) / len(high_confidence) if high_confidence else 0.0
            med_acc = sum(r.accuracy_score for r in medium_confidence) / len(medium_confidence) if medium_confidence else 0.0
            low_acc = sum(r.accuracy_score for r in low_confidence) / len(low_confidence) if low_confidence else 0.0

            # Calculate overall calibration score
            expected_high_acc = 0.9  # High confidence should be ~90% accurate
            expected_med_acc = 0.7  # Medium confidence should be ~70% accurate
            expected_low_acc = 0.5  # Low confidence should be ~50% accurate

            calibration_error = (
                abs(high_acc - expected_high_acc) +
                abs(med_acc - expected_med_acc) +
                abs(low_acc - expected_low_acc)
            ) / 3

            calibration_score = 1.0 - calibration_error

            # Pre-calculate confidence values for buckets
            high_bucket = []
            medium_bucket = []
            low_bucket = []

            for r in high_confidence:
                high_bucket.append({'accuracy': r.accuracy_score, 'confidence': self._extract_confidence(r)})
            for r in medium_confidence:
                medium_bucket.append({'accuracy': r.accuracy_score, 'confidence': self._extract_confidence(r)})
            for r in low_confidence:
                low_bucket.append({'accuracy': r.accuracy_score, 'confidence': self._extract_confidence(r)})

            # Calculate adjustment factors
            adjustment_factors = {}
            if high_confidence:
                adjustment_factors['high_confidence_adjustment'] = expected_high_acc / (high_acc + 0.001)
            if medium_confidence:
                adjustment_factors['medium_confidence_adjustment'] = expected_med_acc / (med_acc + 0.001)
            if low_confidence:
                adjustment_factors['low_confidence_adjustment'] = expected_low_acc / (low_acc + 0.001)

            return ConfidenceCalibrationMetrics(
                calibration_score=calibration_score,
                high_confidence_accuracy=high_acc,
                medium_confidence_accuracy=med_acc,
                low_confidence_accuracy=low_acc,
                confidence_buckets={
                    'high': high_bucket,
                    'medium': medium_bucket,
                    'low': low_bucket
                },
                adjustment_factors=adjustment_factors
            )

        except Exception as e:
            self.logger.error(f"Error getting confidence calibration for {model_name}: {e}")
            return ConfidenceCalibrationMetrics(
                calibration_score=0.0,
                high_confidence_accuracy=0.0,
                medium_confidence_accuracy=0.0,
                low_confidence_accuracy=0.0,
                confidence_buckets={},
                adjustment_factors={}
            )

    async def get_confidence_adjustment_factors(
        self,
        model_name: str,
        window_hours: int = 168
    ) -> Dict[str, float]:
        """Get confidence adjustment factors for a model."""
        calibration = await self.get_confidence_calibration(model_name, window_hours)
        return calibration.adjustment_factors

    async def get_model_strengths(
        self,
        model_name: str,
        window_hours: int = 720  # 30 days
    ) -> ModelStrengths:
        """
        Identify model strengths and weaknesses by analyzing performance patterns.

        Args:
            model_name: Name of the model to analyze
            window_hours: Time window for analysis

        Returns:
            Model strengths analysis
        """
        try:
            records = await self.db_manager.get_model_performance_by_model(model_name, limit=2000)

            # Filter by time window
            cutoff_time = datetime.now() - timedelta(hours=window_hours)
            records = [r for r in records if r.timestamp >= cutoff_time]

            if not records:
                return ModelStrengths(
                    strong_categories={},
                    weak_categories={},
                    preferred_conditions=[],
                    avoided_conditions=[],
                    overall_reliability=0.0
                )

            # Group by market category
            category_performance = {}
            for record in records:
                if record.market_category not in category_performance:
                    category_performance[record.market_category] = []
                category_performance[record.market_category].append(record)

            # Calculate accuracy by category
            category_accuracies = {}
            for category, cat_records in category_performance.items():
                if len(cat_records) >= 5:  # Minimum sample size
                    accuracy = sum(r.accuracy_score for r in cat_records) / len(cat_records)
                    category_accuracies[category] = accuracy

            # Identify strong and weak categories
            strong_categories = {
                cat: acc for cat, acc in category_accuracies.items()
                if acc >= 0.7 and len(category_performance[cat]) >= 10
            }

            weak_categories = {
                cat: acc for cat, acc in category_accuracies.items()
                if acc <= 0.5 and len(category_performance[cat]) >= 10
            }

            # Calculate overall reliability
            overall_accuracy = sum(r.accuracy_score for r in records) / len(records)
            overall_reliability = overall_accuracy

            # Determine preferred conditions (simplified)
            preferred_conditions = []
            avoided_conditions = []

            if strong_categories:
                preferred_conditions.extend([f"market_category:{cat}" for cat in strong_categories.keys()])
            if weak_categories:
                avoided_conditions.extend([f"market_category:{cat}" for cat in weak_categories.keys()])

            return ModelStrengths(
                strong_categories=strong_categories,
                weak_categories=weak_categories,
                preferred_conditions=preferred_conditions,
                avoided_conditions=avoided_conditions,
                overall_reliability=overall_reliability
            )

        except Exception as e:
            self.logger.error(f"Error getting model strengths for {model_name}: {e}")
            return ModelStrengths(
                strong_categories={},
                weak_categories={},
                preferred_conditions=[],
                avoided_conditions=[],
                overall_reliability=0.0
            )

    async def get_temporal_performance_patterns(
        self,
        model_name: str,
        window_hours: int = 168  # 7 days
    ) -> Dict[str, Any]:
        """
        Analyze temporal performance patterns and trends.

        Args:
            model_name: Name of the model to analyze
            window_hours: Time window for analysis

        Returns:
            Temporal performance patterns
        """
        try:
            records = await self.db_manager.get_model_performance_by_model(model_name, limit=1000)

            # Filter by time window
            cutoff_time = datetime.now() - timedelta(hours=window_hours)
            records = [r for r in records if r.timestamp >= cutoff_time]

            if not records:
                return {
                    'performance_trend': {'direction': 'stable', 'slope': 0.0},
                    'recent_accuracy': 0.0,
                    'historical_accuracy': 0.0,
                    'peak_performance_time': None,
                    'low_performance_time': None
                }

            # Sort by timestamp
            records.sort(key=lambda x: x.timestamp)

            # Split into recent and historical halves
            mid_point = len(records) // 2
            recent_records = records[mid_point:]
            historical_records = records[:mid_point]

            # Calculate accuracies
            recent_accuracy = sum(r.accuracy_score for r in recent_records) / len(recent_records)
            historical_accuracy = sum(r.accuracy_score for r in historical_records) / len(historical_records)

            # Determine trend
            if recent_accuracy > historical_accuracy + 0.05:
                trend_direction = 'improving'
                slope = (recent_accuracy - historical_accuracy) / (len(recent_records) + 1)
            elif recent_accuracy < historical_accuracy - 0.05:
                trend_direction = 'declining'
                slope = (recent_accuracy - historical_accuracy) / (len(recent_records) + 1)
            else:
                trend_direction = 'stable'
                slope = 0.0

            # Find peak and low performance times
            accuracies_by_time = [(r.timestamp, r.accuracy_score) for r in records]
            peak_time = max(accuracies_by_time, key=lambda x: x[1])[0] if accuracies_by_time else None
            low_time = min(accuracies_by_time, key=lambda x: x[1])[0] if accuracies_by_time else None

            return {
                'performance_trend': {
                    'direction': trend_direction,
                    'slope': slope
                },
                'recent_accuracy': recent_accuracy,
                'historical_accuracy': historical_accuracy,
                'peak_performance_time': peak_time,
                'low_performance_time': low_time
            }

        except Exception as e:
            self.logger.error(f"Error getting temporal performance patterns for {model_name}: {e}")
            return {
                'performance_trend': {'direction': 'unknown', 'slope': 0.0},
                'recent_accuracy': 0.0,
                'historical_accuracy': 0.0,
                'peak_performance_time': None,
                'low_performance_time': None
            }

    async def get_cost_performance_metrics(
        self,
        model_name: str,
        window_hours: int = 168  # 7 days
    ) -> CostPerformanceMetrics:
        """
        Calculate cost efficiency metrics for a model.

        Args:
            model_name: Name of the model to analyze
            window_hours: Time window for analysis

        Returns:
            Cost performance metrics
        """
        try:
            aggregation = await self.db_manager.get_model_performance_aggregation(model_name, window_hours)

            if not aggregation:
                return CostPerformanceMetrics(
                    total_cost=0.0,
                    cost_per_prediction=0.0,
                    cost_per_correct_prediction=0.0,
                    cost_performance_ratio=0.0,
                    roi_score=0.0,
                    budget_efficiency=0.0
                )

            total_predictions = aggregation['record_count']
            accuracy = aggregation['avg_accuracy']
            total_cost = aggregation['total_cost']

            # Calculate metrics
            cost_per_prediction = total_cost / total_predictions if total_predictions > 0 else 0.0
            correct_predictions = total_predictions * accuracy
            cost_per_correct = total_cost / correct_predictions if correct_predictions > 0 else float('inf')

            # Cost-performance ratio: accuracy per dollar spent
            cost_performance_ratio = accuracy / (total_cost + 0.001)

            # ROI score: value relative to cost
            roi_score = (accuracy * 100) / (total_cost + 0.001)

            # Budget efficiency: how well we're using our budget
            budget_efficiency = accuracy / (cost_per_prediction + 0.001)

            return CostPerformanceMetrics(
                total_cost=total_cost,
                cost_per_prediction=cost_per_prediction,
                cost_per_correct_prediction=cost_per_correct,
                cost_performance_ratio=cost_performance_ratio,
                roi_score=roi_score,
                budget_efficiency=budget_efficiency
            )

        except Exception as e:
            self.logger.error(f"Error getting cost performance metrics for {model_name}: {e}")
            return CostPerformanceMetrics(
                total_cost=0.0,
                cost_per_prediction=0.0,
                cost_per_correct_prediction=0.0,
                cost_performance_ratio=0.0,
                roi_score=0.0,
                budget_efficiency=0.0
            )

    async def update_rolling_windows(self, model_name: str) -> None:
        """
        Update rolling window performance data for a model.

        Args:
            model_name: Name of the model to update
        """
        try:
            # Update common rolling windows
            windows = [1, 6, 24, 168, 720]  # 1h, 6h, 24h, 7d, 30d

            for hours in windows:
                await self.get_rolling_window_metrics(model_name, hours)

            self.logger.info(f"Updated rolling windows for {model_name}")

        except Exception as e:
            self.logger.error(f"Error updating rolling windows for {model_name}: {e}")

    async def cleanup_old_records(self, days_to_keep: int = 90) -> int:
        """
        Clean up old performance records to manage database size.

        Args:
            days_to_keep: Number of days to keep records

        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            # This would be implemented in the database manager
            # For now, return 0 as a placeholder
            return 0

        except Exception as e:
            self.logger.error(f"Error cleaning up old records: {e}")
            return 0