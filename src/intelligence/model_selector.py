"""
Intelligent Model Selection Engine.

Advanced model selection system that considers performance, cost, market conditions,
and health status to optimize AI model selection for trading decisions.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import statistics
from enum import Enum

from src.utils.performance_tracker import (
    PerformanceTracker,
    ModelPerformanceMetrics,
    ModelStrengths,
    CostPerformanceMetrics,
    RollingWindowMetrics
)
from src.utils.database import DatabaseManager, ModelHealth
from src.utils.logging_setup import TradingLoggerMixin


class SelectionStrategy(Enum):
    """Model selection strategies."""
    PERFORMANCE_FIRST = "performance_first"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"
    CONTEXT_AWARE = "context_aware"
    HEALTH_AWARE = "health_aware"


class HealthStatus(Enum):
    """Model health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SelectionCriteria:
    """Criteria for model selection decision."""
    performance_weight: float = 0.4
    cost_weight: float = 0.2
    speed_weight: float = 0.1
    context_weight: float = 0.2
    health_weight: float = 0.1

    # Thresholds
    min_performance_threshold: float = 0.6
    max_cost_threshold: float = 0.5
    max_response_time_ms: float = 3000.0
    min_health_score: float = 0.7

    # Budget constraints
    budget_conscious: bool = False
    remaining_budget: float = 0.0
    daily_budget_limit: float = 50.0

    def validate(self) -> bool:
        """Validate selection criteria weights sum to 1.0."""
        total = (self.performance_weight + self.cost_weight +
                self.speed_weight + self.context_weight + self.health_weight)
        return abs(total - 1.0) < 0.01


@dataclass
class ModelScore:
    """Individual model scoring result."""
    model_name: str
    total_score: float
    performance_score: float
    cost_score: float
    speed_score: float
    context_score: float
    health_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSelectionResult:
    """Result of model selection process."""
    selected_model: Optional[str]
    confidence: float
    reasoning: str
    selection_strategy: SelectionStrategy
    alternative_models: List[str] = field(default_factory=list)
    selection_metadata: Dict[str, Any] = field(default_factory=dict)
    disqualified_models: List[str] = field(default_factory=list)


@dataclass
class ModelHealthStatus:
    """Health status summary for a model."""
    model_name: str
    is_available: bool
    consecutive_failures: int
    avg_response_time: float
    last_check_time: datetime
    health_score: float
    status: HealthStatus


@dataclass
class DisagreementResult:
    """Result of resolving model disagreements."""
    disagreement_detected: bool
    disagreement_level: float
    final_model: Optional[str]
    resolution_method: str
    confidence_adjustment: Optional[float]
    reasoning: str
    consensus_models: List[str] = field(default_factory=list)


class ModelSelector(TradingLoggerMixin):
    """
    Intelligent model selection engine.

    Selects optimal AI models based on performance metrics, cost efficiency,
    market context, and health status. Implements sophisticated selection
    algorithms with automatic failover and budget awareness.
    """

    def __init__(self, performance_tracker: PerformanceTracker):
        """
        Initialize model selector.

        Args:
            performance_tracker: PerformanceTracker instance for metrics
        """
        self.performance_tracker = performance_tracker
        self.db_manager = performance_tracker.db_manager
        self.model_health_cache: Dict[str, ModelHealthStatus] = {}
        self.last_health_check: datetime = datetime.now() - timedelta(hours=1)
        self.health_check_interval_minutes = 5

        # Default selection criteria
        self.default_criteria = SelectionCriteria()

        # Model availability cache
        self.available_models_cache: Set[str] = set()
        self.last_availability_check = datetime.now() - timedelta(hours=1)

        self.logger.info("Model selector initialized")

    async def select_optimal_model(
        self,
        market_category: str,
        trade_value: float,
        volatility_regime: Optional[str] = None,
        time_to_expiry: Optional[int] = None,
        remaining_budget: Optional[float] = None,
        selection_criteria: Optional[SelectionCriteria] = None,
        force_health_check: bool = False
    ) -> ModelSelectionResult:
        """
        Select the optimal model for the given context.

        Args:
            market_category: Category of the market
            trade_value: Value of the potential trade
            volatility_regime: Market volatility regime
            time_to_expiry: Hours until market expiry
            remaining_budget: Remaining budget for AI calls
            selection_criteria: Custom selection criteria
            force_health_check: Force fresh health check

        Returns:
            ModelSelectionResult with selected model and reasoning
        """
        try:
            # Use default criteria if none provided
            criteria = selection_criteria or self.default_criteria
            criteria.validate()

            # Update remaining budget if provided
            if remaining_budget is not None:
                criteria.remaining_budget = remaining_budget

            self.logger.info(
                f"Selecting optimal model for {market_category} market",
                trade_value=trade_value,
                volatility_regime=volatility_regime
            )

            # Check model health and availability
            await self._update_model_health(force_health_check)
            healthy_models = await self._get_healthy_models(criteria.min_health_score)

            if not healthy_models:
                self.logger.warning("No healthy models available")
                return ModelSelectionResult(
                    selected_model=None,
                    confidence=0.0,
                    reasoning="No healthy models available for selection",
                    selection_strategy=SelectionStrategy.HEALTH_AWARE,
                    disqualified_models=list(self.model_health_cache.keys())
                )

            # Get model performance data
            model_rankings = await self._get_model_performance_data(healthy_models)

            if not model_rankings:
                self.logger.warning("No performance data available, using fallback")
                return await self._fallback_selection(
                    healthy_models, market_category, criteria
                )

            # Score models based on criteria
            model_scores = await self._score_models(
                model_rankings, market_category, trade_value, criteria,
                volatility_regime, time_to_expiry
            )

            # Select best model
            selected_model = await self._select_best_model(model_scores, criteria)

            # Generate reasoning
            reasoning = await self._generate_selection_reasoning(
                selected_model, model_scores, criteria, market_category
            )

            # Determine strategy used
            strategy = self._determine_selection_strategy(criteria)

            result = ModelSelectionResult(
                selected_model=selected_model.model_name if selected_model else None,
                confidence=selected_model.total_score if selected_model else 0.0,
                reasoning=reasoning,
                selection_strategy=strategy,
                alternative_models=[m.model_name for m in model_scores[:3]
                                 if m.model_name != (selected_model.model_name if selected_model else None)],
                selection_metadata={
                    "total_models_evaluated": len(model_scores),
                    "healthy_models": len(healthy_models),
                    "selection_criteria": criteria.__dict__,
                    "evaluation_time": datetime.now().isoformat()
                },
                disqualified_models=[
                    m for m in self.model_health_cache.keys()
                    if m not in healthy_models
                ]
            )

            self.logger.info(
                f"Model selected: {result.selected_model}",
                confidence=result.confidence,
                strategy=strategy.value
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in model selection: {e}")
            # Return safe fallback
            return ModelSelectionResult(
                selected_model=None,
                confidence=0.0,
                reasoning=f"Error in model selection: {str(e)}",
                selection_strategy=SelectionStrategy.HEALTH_AWARE
            )

    async def get_model_health(self, model_name: str, force_check: bool = False) -> ModelHealthStatus:
        """
        Get health status for a specific model.

        Args:
            model_name: Name of the model to check
            force_check: Force fresh health check

        Returns:
            ModelHealthStatus for the model
        """
        await self._update_model_health(force_check)
        return self.model_health_cache.get(model_name, self._create_unknown_health_status(model_name))

    async def calculate_performance_cost_ratio(
        self, model_name: str, window_hours: int = 24
    ) -> Optional[float]:
        """
        Calculate performance-to-cost ratio for a model.

        Args:
            model_name: Name of the model
            window_hours: Time window for analysis

        Returns:
            Performance-to-cost ratio or None if data unavailable
        """
        try:
            cost_metrics = await self.performance_tracker.get_cost_performance_metrics(
                model_name, window_hours
            )
            return cost_metrics.cost_performance_ratio
        except Exception as e:
            self.logger.error(f"Error calculating performance-cost ratio for {model_name}: {e}")
            return None

    async def resolve_model_disagreement(
        self,
        model_predictions: Dict[str, Dict[str, Any]],
        market_category: str,
        disagreement_threshold: float = 0.3,
        fallback_strategy: str = "highest_confidence"
    ) -> DisagreementResult:
        """
        Resolve disagreements between model predictions.

        Args:
            model_predictions: Dictionary of model predictions
            market_category: Market category for context
            disagreement_threshold: Threshold for detecting disagreement
            fallback_strategy: Strategy for resolving conflicts

        Returns:
            DisagreementResult with resolution
        """
        try:
            if not model_predictions:
                return DisagreementResult(
                    disagreement_detected=False,
                    disagreement_level=0.0,
                    final_model=None,
                    resolution_method="no_predictions",
                    confidence_adjustment=None,
                    reasoning="No model predictions provided"
                )

            # Extract actions and confidences
            actions = {}
            confidences = {}

            for model, prediction in model_predictions.items():
                actions[model] = prediction.get("action", "unknown")
                confidences[model] = prediction.get("confidence", 0.0)

            # Check for disagreement
            unique_actions = set(actions.values())
            disagreement_detected = len(unique_actions) > 1
            disagreement_level = self._calculate_disagreement_level(actions, confidences)

            result = DisagreementResult(
                disagreement_detected=disagreement_detected,
                disagreement_level=disagreement_level,
                final_model=None,
                resolution_method="",
                confidence_adjustment=None,
                reasoning=""
            )

            if not disagreement_detected or disagreement_level < disagreement_threshold:
                # Models agree or disagreement is minor
                result.resolution_method = "consensus"
                result.consensus_models = list(model_predictions.keys())

                # Select model with highest confidence
                best_model = max(confidences.items(), key=lambda x: x[1])[0]
                result.final_model = best_model
                result.reasoning = f"Models agree on {actions[best_model]} action"

            else:
                # Significant disagreement detected
                result.resolution_method = fallback_strategy
                result.reasoning = f"Significant disagreement detected (level: {disagreement_level:.2f})"

                if fallback_strategy == "highest_confidence":
                    best_model = max(confidences.items(), key=lambda x: x[1])[0]
                    result.final_model = best_model
                    result.reasoning += f". Selected {best_model} with highest confidence ({confidences[best_model]:.2f})"

                elif fallback_strategy == "context_aware":
                    # Use model strongest in this category
                    selection = await self.select_optimal_model(
                        market_category=market_category,
                        trade_value=25.0,
                        selection_criteria=SelectionCriteria(context_weight=0.8)
                    )
                    result.final_model = selection.selected_model
                    result.reasoning += f". Selected {selection.selected_model} based on category expertise"

                elif fallback_strategy == "ensemble_weighted":
                    # Could implement weighted ensemble here
                    best_model = max(confidences.items(), key=lambda x: x[1])[0]
                    result.final_model = best_model
                    result.reasoning += f". Weighted ensemble favored {best_model}"

            return result

        except Exception as e:
            self.logger.error(f"Error resolving model disagreement: {e}")
            return DisagreementResult(
                disagreement_detected=True,
                disagreement_level=1.0,
                final_model=None,
                resolution_method="error",
                confidence_adjustment=None,
                reasoning=f"Error resolving disagreement: {str(e)}"
            )

    async def _update_model_health(self, force_check: bool = False) -> None:
        """Update cached model health information."""
        now = datetime.now()

        # Check if we need to update health info
        time_since_check = (now - self.last_health_check).total_seconds() / 60
        if not force_check and time_since_check < self.health_check_interval_minutes:
            return

        try:
            health_records = await self.db_manager.get_model_health()

            for record in health_records:
                health_score = self._calculate_health_score(record)

                self.model_health_cache[record.model_name] = ModelHealthStatus(
                    model_name=record.model_name,
                    is_available=record.is_available,
                    consecutive_failures=record.consecutive_failures,
                    avg_response_time=record.avg_response_time,
                    last_check_time=record.last_check_time,
                    health_score=health_score,
                    status=self._determine_health_status(health_score, record.consecutive_failures)
                )

            self.last_health_check = now
            self.logger.debug(f"Updated health status for {len(health_records)} models")

        except Exception as e:
            self.logger.error(f"Error updating model health: {e}")

    async def _get_healthy_models(self, min_health_score: float) -> Set[str]:
        """Get set of models that meet minimum health requirements."""
        healthy_models = set()

        for model_name, health_status in self.model_health_cache.items():
            if (health_status.is_available and
                health_status.health_score >= min_health_score and
                health_status.consecutive_failures <= 3):
                healthy_models.add(model_name)

        return healthy_models

    async def _get_model_performance_data(self, models: Set[str]) -> List[ModelPerformanceMetrics]:
        """Get performance data for specified models."""
        try:
            rankings = await self.performance_tracker.get_model_ranking()

            # Filter to only requested models
            return [r for r in rankings if r.model_name in models]

        except Exception as e:
            self.logger.error(f"Error getting model performance data: {e}")
            return []

    async def _score_models(
        self,
        model_rankings: List[ModelPerformanceMetrics],
        market_category: str,
        trade_value: float,
        criteria: SelectionCriteria,
        volatility_regime: Optional[str],
        time_to_expiry: Optional[int]
    ) -> List[ModelScore]:
        """Score models based on selection criteria."""
        scores = []

        for model_metrics in model_rankings:
            model_name = model_metrics.model_name

            # Performance score
            performance_score = min(1.0, model_metrics.accuracy * model_metrics.avg_decision_quality)

            # Cost score (higher is better)
            cost_score = self._calculate_cost_score(model_metrics, criteria, trade_value)

            # Speed score (higher is better)
            speed_score = self._calculate_speed_score(model_metrics.avg_response_time_ms, criteria)

            # Context score (market category expertise)
            context_score = await self._calculate_context_score(
                model_name, market_category, volatility_regime, time_to_expiry
            )

            # Health score
            health_status = self.model_health_cache.get(model_name)
            health_score = health_status.health_score if health_status else 0.5

            # Calculate total weighted score
            total_score = (
                performance_score * criteria.performance_weight +
                cost_score * criteria.cost_weight +
                speed_score * criteria.speed_weight +
                context_score * criteria.context_weight +
                health_score * criteria.health_weight
            )

            scores.append(ModelScore(
                model_name=model_name,
                total_score=total_score,
                performance_score=performance_score,
                cost_score=cost_score,
                speed_score=speed_score,
                context_score=context_score,
                health_score=health_score,
                metadata={
                    "accuracy": model_metrics.accuracy,
                    "avg_response_time": model_metrics.avg_response_time_ms,
                    "cost_performance_ratio": model_metrics.cost_performance_ratio
                }
            ))

        # Sort by total score (descending)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores

    def _calculate_cost_score(
        self, model_metrics: ModelPerformanceMetrics, criteria: SelectionCriteria, trade_value: float
    ) -> float:
        """Calculate cost score for a model."""
        cost_performance_ratio = model_metrics.cost_performance_ratio

        # Apply budget constraints if enabled
        if criteria.budget_conscious and criteria.remaining_budget > 0:
            # Check if model cost fits within budget
            avg_cost_per_prediction = model_metrics.total_cost / model_metrics.total_predictions

            if avg_cost_per_prediction > criteria.remaining_budget:
                return 0.0  # Disqualify if exceeds budget

            # Bonus for cost efficiency under budget constraints
            budget_utilization = avg_cost_per_prediction / criteria.remaining_budget
            cost_score = cost_performance_ratio * (1.0 - budget_utilization * 0.3)
        else:
            cost_score = min(1.0, cost_performance_ratio * 2.0)  # Normalize to 0-1

        return max(0.0, min(1.0, cost_score))

    def _calculate_speed_score(self, avg_response_time_ms: float, criteria: SelectionCriteria) -> float:
        """Calculate speed score for a model."""
        if avg_response_time_ms <= criteria.max_response_time_ms:
            # Faster responses get higher scores
            return 1.0 - (avg_response_time_ms / criteria.max_response_time_ms) * 0.3
        else:
            # Slow responses get lower scores
            return max(0.0, 1.0 - (avg_response_time_ms - criteria.max_response_time_ms) / criteria.max_response_time_ms)

    async def _calculate_context_score(
        self, model_name: str, market_category: str, volatility_regime: Optional[str], time_to_expiry: Optional[int]
    ) -> float:
        """Calculate context score based on model expertise in market conditions."""
        try:
            # Get model strengths
            strengths = await self.performance_tracker.get_model_strengths(model_name)

            # Check if model is strong in this market category
            category_strength = strengths.strong_categories.get(market_category, 0.5)

            # Check weak categories
            if market_category in strengths.weak_categories:
                category_strength = min(category_strength, strengths.weak_categories[market_category])

            # Factor in volatility regime if known
            if volatility_regime:
                # Simplified: assume models have some regime preferences
                regime_modifier = 1.0
                if volatility_regime == "high" and "high_volatility" in strengths.preferred_conditions:
                    regime_modifier = 1.2
                elif volatility_regime == "high" and "high_volatility" in strengths.avoided_conditions:
                    regime_modifier = 0.8

                category_strength *= regime_modifier

            return max(0.0, min(1.0, category_strength))

        except Exception as e:
            self.logger.error(f"Error calculating context score for {model_name}: {e}")
            return 0.5  # Neutral score if error occurs

    async def _select_best_model(self, model_scores: List[ModelScore], criteria: SelectionCriteria) -> Optional[ModelScore]:
        """Select the best model from scored candidates."""
        if not model_scores:
            return None

        # Filter models below minimum performance threshold
        qualified_models = [
            score for score in model_scores
            if score.performance_score >= criteria.min_performance_threshold
        ]

        if not qualified_models:
            # If no models meet threshold, take the best available
            qualified_models = model_scores[:1]

        return qualified_models[0]

    async def _generate_selection_reasoning(
        self, selected_model: Optional[ModelScore], model_scores: List[ModelScore],
        criteria: SelectionCriteria, market_category: str
    ) -> str:
        """Generate human-readable reasoning for model selection."""
        if not selected_model:
            return "No suitable model found for selection"

        reasoning_parts = []

        # Performance reasoning
        if selected_model.performance_score >= 0.8:
            reasoning_parts.append("excellent recent performance")
        elif selected_model.performance_score >= 0.7:
            reasoning_parts.append("good recent performance")
        else:
            reasoning_parts.append("acceptable performance")

        # Cost reasoning
        if selected_model.cost_score >= 0.7:
            reasoning_parts.append("cost-effective")
        elif selected_model.cost_score <= 0.3:
            reasoning_parts.append("higher cost")

        # Context reasoning
        if selected_model.context_score >= 0.7:
            reasoning_parts.append(f"strong expertise in {market_category} markets")
        elif selected_model.context_score <= 0.3:
            reasoning_parts.append("limited expertise in this market type")

        # Health reasoning
        if selected_model.health_score >= 0.9:
            reasoning_parts.append("excellent health status")
        elif selected_model.health_score <= 0.5:
            reasoning_parts.append("some health concerns")

        return f"Selected {selected_model.model_name} based on: {', '.join(reasoning_parts)}"

    def _determine_selection_strategy(self, criteria: SelectionCriteria) -> SelectionStrategy:
        """Determine which selection strategy was used."""
        if criteria.context_weight >= 0.5:
            return SelectionStrategy.CONTEXT_AWARE
        elif criteria.cost_weight >= 0.5:
            return SelectionStrategy.COST_OPTIMIZED
        elif criteria.performance_weight >= 0.6:
            return SelectionStrategy.PERFORMANCE_FIRST
        else:
            return SelectionStrategy.BALANCED

    async def _fallback_selection(
        self, healthy_models: Set[str], market_category: str, criteria: SelectionCriteria
    ) -> ModelSelectionResult:
        """Fallback selection when performance data is unavailable."""
        if not healthy_models:
            return ModelSelectionResult(
                selected_model=None,
                confidence=0.0,
                reasoning="No healthy models available",
                selection_strategy=SelectionStrategy.HEALTH_AWARE
            )

        # Simple alphabetical fallback
        selected_model = sorted(healthy_models)[0]

        return ModelSelectionResult(
            selected_model=selected_model,
            confidence=0.5,  # Low confidence due to lack of data
            reasoning=f"Selected {selected_model} due to lack of performance data (alphabetical fallback)",
            selection_strategy=SelectionStrategy.BALANCED
        )

    def _calculate_health_score(self, health_record: ModelHealth) -> float:
        """Calculate overall health score for a model."""
        base_score = 1.0

        # Penalize consecutive failures
        failure_penalty = min(0.5, health_record.consecutive_failures * 0.1)

        # Penalize slow response times
        response_penalty = min(0.3, health_record.avg_response_time / 10000.0)  # Penalty for >10s

        # Check availability
        availability_score = 1.0 if health_record.is_available else 0.0

        # Time since last check (more recent is better)
        time_factor = max(0.5, 1.0 - (datetime.now() - health_record.last_check_time).total_seconds() / 3600.0)

        health_score = base_score - failure_penalty - response_penalty
        health_score *= availability_score * time_factor

        return max(0.0, min(1.0, health_score))

    def _determine_health_status(self, health_score: float, consecutive_failures: int) -> HealthStatus:
        """Determine health status based on score and failures."""
        if consecutive_failures >= 5 or health_score < 0.3:
            return HealthStatus.UNHEALTHY
        elif consecutive_failures >= 2 or health_score < 0.6:
            return HealthStatus.DEGRADED
        elif health_score >= 0.8:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.DEGRADED

    def _create_unknown_health_status(self, model_name: str) -> ModelHealthStatus:
        """Create unknown health status for models not in cache."""
        return ModelHealthStatus(
            model_name=model_name,
            is_available=False,
            consecutive_failures=0,
            avg_response_time=0.0,
            last_check_time=datetime.now(),
            health_score=0.0,
            status=HealthStatus.UNKNOWN
        )

    def _calculate_disagreement_level(self, actions: Dict[str, str], confidences: Dict[str, float]) -> float:
        """Calculate level of disagreement between models."""
        if len(actions) <= 1:
            return 0.0

        unique_actions = set(actions.values())
        action_diversity = len(unique_actions) / len(actions)

        # Confidence variance (higher variance indicates more uncertainty)
        if len(confidences) > 1:
            conf_values = list(confidences.values())
            confidence_variance = statistics.variance(conf_values) if len(conf_values) > 1 else 0.0
            normalized_variance = min(1.0, confidence_variance * 4.0)  # Normalize to 0-1
        else:
            normalized_variance = 0.0

        # Combine factors
        disagreement_level = (action_diversity * 0.6) + (normalized_variance * 0.4)

        return min(1.0, disagreement_level)