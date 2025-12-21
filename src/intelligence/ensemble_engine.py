"""
Advanced Ensemble Engine for Enhanced AI Model Integration.

Implements sophisticated ensemble methods including weighted voting, consensus mechanisms,
confidence-based selection, cascading ensemble, disagreement detection, and uncertainty quantification.
"""

import asyncio
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from src.clients.xai_client import TradingDecision
from src.intelligence.model_selector import ModelSelector, SelectionCriteria
from src.utils.database import DatabaseManager, EnsembleDecision, ModelPerformance
from src.utils.performance_tracker import PerformanceTracker
from src.utils.logging_setup import TradingLoggerMixin


class EnsembleStrategy(Enum):
    """Ensemble strategy types."""
    CONSENSUS = "consensus"
    WEIGHTED_VOTING = "weighted_voting"
    CONFIDENCE_BASED = "confidence_based"
    CASCADING = "cascading"
    UNCERTAINTY_AWARE = "uncertainty_aware"


class ConfidenceLevel(Enum):
    """Confidence level classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ModelPrediction:
    """Individual model prediction with metadata."""
    model_name: str
    decision: TradingDecision
    performance_score: float = 0.0
    confidence_calibration: float = 1.0
    response_time_ms: float = 0.0
    cost_usd: float = 0.0
    weight: float = 1.0


@dataclass
class EnsembleResult:
    """Result of ensemble decision making."""
    final_decision: Optional[TradingDecision]
    ensemble_strategy: EnsembleStrategy
    models_consulted: List[str]
    consensus_level: float
    disagreement_detected: bool
    disagreement_level: float
    uncertainty_score: float
    confidence_level: ConfidenceLevel
    reasoning: str
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble decision making."""
    min_consensus_threshold: float = 0.7
    disagreement_threshold: float = 0.4
    uncertainty_threshold: float = 0.6
    enable_weighted_voting: bool = True
    enable_confidence_calibration: bool = True
    performance_weight_factor: float = 2.0
    cascading_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "low_value": 10.0,
        "medium_value": 50.0
    })
    max_models_per_decision: int = 3
    timeout_seconds: int = 30


class EnsembleEngine(TradingLoggerMixin):
    """
    Advanced ensemble engine for sophisticated AI model decision making.

    Implements multiple ensemble strategies with dynamic weight adjustment,
    disagreement detection, and uncertainty quantification.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        performance_tracker: PerformanceTracker,
        model_selector: ModelSelector,
        config: Optional[EnsembleConfig] = None,
        clients: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ensemble engine.

        Args:
            db_manager: Database manager for logging
            performance_tracker: Performance tracking system
            model_selector: Model selection engine
            config: Ensemble configuration
            clients: Dictionary of client instances for interacting with models
        """
        self.db_manager = db_manager
        self.performance_tracker = performance_tracker
        self.model_selector = model_selector
        self.config = config or EnsembleConfig()
        self.clients = clients or {}

        # Cache for model performance data
        self.performance_cache: Dict[str, float] = {}
        self.last_performance_update = datetime.now() - timedelta(hours=1)

        self.logger.info("Ensemble engine initialized", config=self.config.__dict__)

    async def get_ensemble_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        trade_value: float,
        market_category: str,
        strategy: Optional[str] = None
    ) -> EnsembleResult:
        """
        Get ensemble trading decision using sophisticated methods.

        Args:
            market_data: Market information
            portfolio_data: Portfolio state
            trade_value: Value of potential trade
            market_category: Category of the market
            strategy: Preferred ensemble strategy

        Returns:
            EnsembleResult with detailed decision information
        """
        start_time = datetime.now()

        try:
            self.logger.info(
                "Starting ensemble decision process",
                trade_value=trade_value,
                market_category=market_category
            )

            # Determine ensemble strategy based on trade value
            ensemble_strategy = await self._determine_ensemble_strategy(trade_value, strategy)

            # Get available models
            available_models = await self._get_available_models()

            if not available_models:
                return self._create_no_models_result()

            # Get predictions from selected models
            predictions = await self._get_model_predictions(
                market_data, portfolio_data, available_models, ensemble_strategy
            )

            if not predictions:
                return self._create_no_predictions_result()

            # Apply ensemble strategy
            ensemble_result = await self._apply_ensemble_strategy(
                predictions, ensemble_strategy, market_data, trade_value
            )

            # Calculate metrics
            ensemble_result.processing_time_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            # Log ensemble decision
            await self._log_ensemble_decision(ensemble_result, market_data)

            return ensemble_result

        except Exception as e:
            self.logger.error(f"Error in ensemble decision: {e}")
            return self._create_error_result(e, start_time)

    async def weighted_consensus(
        self,
        predictions: List[ModelPrediction],
        market_category: str
    ) -> EnsembleResult:
        """
        Calculate weighted consensus from model predictions.

        Args:
            predictions: List of model predictions
            market_category: Market category for context

        Returns:
            EnsembleResult with weighted consensus
        """
        try:
            self.logger.debug(
                "Calculating weighted consensus",
                num_predictions=len(predictions),
                market_category=market_category
            )

            # Update model weights based on performance
            await self._update_model_weights(predictions, market_category)

            # Calculate weighted votes
            weighted_votes = await self._calculate_weighted_votes(predictions)

            # Detect disagreement
            disagreement_result = await self.detect_disagreement(predictions)

            # Quantify uncertainty
            uncertainty_result = await self.quantify_uncertainty(predictions)

            # Make final decision
            final_decision = await self._make_weighted_decision(
                weighted_votes, disagreement_result, uncertainty_result
            )

            return EnsembleResult(
                final_decision=final_decision,
                ensemble_strategy=EnsembleStrategy.WEIGHTED_VOTING,
                models_consulted=[p.model_name for p in predictions],
                consensus_level=1.0 - disagreement_result["disagreement_level"],
                disagreement_detected=disagreement_result["disagreement_detected"],
                disagreement_level=disagreement_result["disagreement_level"],
                uncertainty_score=uncertainty_result["uncertainty_score"],
                confidence_level=self._determine_confidence_level(uncertainty_result["uncertainty_score"]),
                reasoning=await self._generate_weighted_reasoning(
                    weighted_votes, disagreement_result, uncertainty_result
                ),
                processing_time_ms=0.0,  # Will be set by caller
                metadata={
                    "weighted_votes": weighted_votes,
                    "model_weights": {p.model_name: p.weight for p in predictions}
                }
            )

        except Exception as e:
            self.logger.error(f"Error in weighted consensus: {e}")
            return self._create_error_result(e, datetime.now())

    async def detect_disagreement(
        self,
        predictions: List[ModelPrediction]
    ) -> Dict[str, Any]:
        """
        Detect and quantify disagreement among model predictions.

        Args:
            predictions: List of model predictions

        Returns:
            Disagreement analysis result
        """
        try:
            if len(predictions) < 2:
                return {
                    "disagreement_detected": False,
                    "disagreement_level": 0.0,
                    "unique_actions": 1,
                    "action_diversity": 0.0,
                    "confidence_variance": 0.0,
                    "weighted_disagreement": 0.0
                }

            # Extract actions and confidences
            actions = [p.decision.action for p in predictions]
            confidences = [p.decision.confidence for p in predictions]
            weights = [p.weight for p in predictions]

            # Calculate action diversity
            unique_actions = len(set(actions))
            action_diversity = unique_actions / len(actions)

            # Calculate confidence variance
            mean_confidence = sum(confidences) / len(confidences)
            confidence_variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)

            # Calculate weighted disagreement
            weighted_disagreement = await self._calculate_weighted_disagreement(predictions)

            # Combined disagreement level
            disagreement_level = (
                action_diversity * 0.4 +
                min(1.0, confidence_variance * 2.0) * 0.3 +
                weighted_disagreement * 0.3
            )

            disagreement_detected = (
                action_diversity > 0.3 or
                confidence_variance > 0.1 or
                disagreement_level > self.config.disagreement_threshold
            )

            result = {
                "disagreement_detected": disagreement_detected,
                "disagreement_level": disagreement_level,
                "unique_actions": unique_actions,
                "action_diversity": action_diversity,
                "confidence_variance": confidence_variance,
                "weighted_disagreement": weighted_disagreement,
                "actions_distribution": self._get_action_distribution(actions, weights)
            }

            self.logger.debug(
                "Disagreement analysis complete",
                disagreement_level=disagreement_level,
                detected=disagreement_detected
            )

            return result

        except Exception as e:
            self.logger.error(f"Error detecting disagreement: {e}")
            return {
                "disagreement_detected": True,
                "disagreement_level": 1.0,
                "error": str(e)
            }

    async def quantify_uncertainty(
        self,
        predictions: List[ModelPrediction]
    ) -> Dict[str, Any]:
        """
        Quantify uncertainty in ensemble predictions.

        Args:
            predictions: List of model predictions

        Returns:
            Uncertainty analysis result
        """
        try:
            if not predictions:
                return {
                    "uncertainty_score": 1.0,
                    "confidence_level": ConfidenceLevel.VERY_LOW.value,
                    "sources": ["no_predictions"]
                }

            # Extract confidences and actions
            confidences = [p.decision.confidence for p in predictions]
            actions = [p.decision.action for p in predictions]
            calibrations = [p.confidence_calibration for p in predictions]

            # Calculate calibrated confidences
            calibrated_confidences = [c * cal for c, cal in zip(confidences, calibrations)]

            # Calculate uncertainty components
            mean_confidence = sum(calibrated_confidences) / len(calibrated_confidences)
            confidence_variance = statistics.variance(calibrated_confidences) if len(calibrated_confidences) > 1 else 0.0
            action_diversity = len(set(actions)) / len(actions)

            # Calibration uncertainty (how well calibrated are the confidences)
            calibration_uncertainty = statistics.stdev(calibrations) if len(calibrations) > 1 else 0.0

            # Combined uncertainty score
            uncertainty_score = (
                (1.0 - mean_confidence) * 0.3 +  # Low confidence increases uncertainty
                min(1.0, confidence_variance * 3.0) * 0.25 +  # High variance increases uncertainty
                action_diversity * 0.25 +  # Action diversity increases uncertainty
                min(1.0, calibration_uncertainty * 2.0) * 0.2  # Poor calibration increases uncertainty
            )

            # Determine confidence level
            confidence_level = self._determine_confidence_level(uncertainty_score)

            # Identify uncertainty sources
            uncertainty_sources = []
            if mean_confidence < 0.6:
                uncertainty_sources.append("low_confidence")
            if confidence_variance > 0.1:
                uncertainty_sources.append("high_variance")
            if action_diversity > 0.3:
                uncertainty_sources.append("action_diversity")
            if calibration_uncertainty > 0.2:
                uncertainty_sources.append("poor_calibration")

            result = {
                "uncertainty_score": uncertainty_score,
                "confidence_level": confidence_level.value,
                "mean_confidence": mean_confidence,
                "confidence_variance": confidence_variance,
                "action_diversity": action_diversity,
                "calibration_uncertainty": calibration_uncertainty,
                "sources": uncertainty_sources,
                "recommendations": await self._generate_uncertainty_recommendations(uncertainty_score)
            }

            self.logger.debug(
                "Uncertainty quantification complete",
                uncertainty_score=uncertainty_score,
                confidence_level=confidence_level.value
            )

            return result

        except Exception as e:
            self.logger.error(f"Error quantifying uncertainty: {e}")
            return {
                "uncertainty_score": 1.0,
                "confidence_level": ConfidenceLevel.VERY_LOW.value,
                "error": str(e),
                "sources": ["error"]
            }

    async def _determine_ensemble_strategy(
        self,
        trade_value: float,
        preferred_strategy: Optional[str]
    ) -> EnsembleStrategy:
        """Determine which ensemble strategy to use based on trade value."""
        if preferred_strategy:
            try:
                return EnsembleStrategy(preferred_strategy)
            except ValueError:
                self.logger.warning(f"Invalid preferred strategy: {preferred_strategy}")

        # Cascading strategy based on trade value
        if trade_value < self.config.cascading_thresholds["low_value"]:
            return EnsembleStrategy.CONFIDENCE_BASED  # Quick single model
        elif trade_value < self.config.cascading_thresholds["medium_value"]:
            return EnsembleStrategy.WEIGHTED_VOTING  # Standard ensemble
        else:
            return EnsembleStrategy.UNCERTAINTY_AWARE  # Enhanced consensus

    async def _get_available_models(self) -> List[str]:
        """Get list of available models."""
        # This would integrate with the model selector to get healthy models
        # For now, return default models
        return ["grok-4", "grok-3", "gpt-4"]

    async def _get_model_predictions(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        models: List[str],
        strategy: EnsembleStrategy
    ) -> List[ModelPrediction]:
        """Get predictions from specified models."""
        predictions = []

        for model_name in models:
            try:
                # This would call the actual model client
                # For now, create mock predictions
                prediction = await self._get_model_prediction(
                    model_name, market_data, portfolio_data
                )
                if prediction:
                    predictions.append(prediction)
            except Exception as e:
                self.logger.warning(f"Failed to get prediction from {model_name}: {e}")

        return predictions

    async def _get_model_prediction(
        self,
        model_name: str,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any]
    ) -> Optional[ModelPrediction]:
        """Get prediction from a specific model."""
        import time
        from src.clients.xai_client import TradingDecision

        start_time = time.time()
        
        try:
            # Determine which client to use
            client = None
            if "grok" in model_name.lower():
                client = self.clients.get("xai")
            elif "gpt" in model_name.lower() or "o1" in model_name.lower():
                client = self.clients.get("openai")
            
            # Fallback to xai (primary) if specific client not found but xai is available
            if not client and "xai" in self.clients:
                client = self.clients["xai"]
                
            if not client:
                self.logger.warning(f"No client available for model {model_name}")
                return None

            # Get decision from client
            # Note: We rely on the client's get_trading_decision interface
            # For specific models, we might need to set a context var or pass a param
            # But XAIClient manages its own models, so we trust it for now for Grok
            
            decision = None
            if "grok" in model_name.lower() and hasattr(client, 'get_trading_decision'):
                # XAI Client
                # TODO: Pass specific model requirement if XAIClient supports it per-call
                decision = await client.get_trading_decision(
                    market_data, portfolio_data, 
                    news_summary=market_data.get("news_summary", "")
                )
            elif "gpt" in model_name.lower() and hasattr(client, 'get_trading_decision'):
                # OpenAI Client
                decision = await client.get_trading_decision(
                    market_data, portfolio_data,
                    news_summary=market_data.get("news_summary", "")
                )
                
            if not decision:
                return None

            duration_ms = (time.time() - start_time) * 1000
            
            # Calculate mock cost if not provided (approximate)
            cost = 0.01  # Placeholder, should get from client response if possible

            return ModelPrediction(
                model_name=model_name,
                decision=decision,
                performance_score=0.8, # Placeholder until we query tracker
                confidence_calibration=1.0, # Placeholder
                response_time_ms=duration_ms,
                cost_usd=cost,
                weight=1.0
            )

        except Exception as e:
            self.logger.error(f"Error getting prediction from {model_name}: {e}")
            return None

    async def _apply_ensemble_strategy(
        self,
        predictions: List[ModelPrediction],
        strategy: EnsembleStrategy,
        market_data: Dict[str, Any],
        trade_value: float
    ) -> EnsembleResult:
        """Apply the specified ensemble strategy."""
        if strategy == EnsembleStrategy.CONSENSUS:
            return await self._apply_consensus_strategy(predictions)
        elif strategy == EnsembleStrategy.WEIGHTED_VOTING:
            return await self.weighted_consensus(predictions, market_data.get("category", "unknown"))
        elif strategy == EnsembleStrategy.CONFIDENCE_BASED:
            return await self._apply_confidence_based_strategy(predictions)
        elif strategy == EnsembleStrategy.CASCADING:
            return await self._apply_cascading_strategy(predictions, trade_value)
        elif strategy == EnsembleStrategy.UNCERTAINTY_AWARE:
            return await self._apply_uncertainty_aware_strategy(predictions)
        else:
            return await self.weighted_consensus(predictions, market_data.get("category", "unknown"))

    async def _apply_consensus_strategy(self, predictions: List[ModelPrediction]) -> EnsembleResult:
        """Apply consensus strategy."""
        if not predictions:
            return self._create_no_predictions_result()

        # Check if all models agree on action and side
        actions = [p.decision.action for p in predictions]
        sides = [p.decision.side for p in predictions]
        confidences = [p.decision.confidence for p in predictions]

        action_consensus = len(set(actions)) == 1
        side_consensus = len(set(sides)) == 1
        avg_confidence = sum(confidences) / len(confidences)

        if action_consensus and side_consensus and avg_confidence >= self.config.min_consensus_threshold:
            final_decision = TradingDecision(
                action=actions[0],
                side=sides[0],
                confidence=avg_confidence,
                reasoning=f"Consensus achieved with {len(predictions)} models"
            )
        else:
            final_decision = None

        return EnsembleResult(
            final_decision=final_decision,
            ensemble_strategy=EnsembleStrategy.CONSENSUS,
            models_consulted=[p.model_name for p in predictions],
            consensus_level=1.0 if final_decision else 0.0,
            disagreement_detected=not (action_consensus and side_consensus),
            disagreement_level=0.0 if final_decision else 1.0,
            uncertainty_score=1.0 - avg_confidence if final_decision else 1.0,
            confidence_level=self._determine_confidence_level(1.0 - avg_confidence if final_decision else 1.0),
            reasoning=final_decision.reasoning if final_decision else "No consensus reached",
            processing_time_ms=0.0
        )

    async def _apply_confidence_based_strategy(self, predictions: List[ModelPrediction]) -> EnsembleResult:
        """Apply confidence-based strategy."""
        if not predictions:
            return self._create_no_predictions_result()

        # Select model with highest calibrated confidence
        best_prediction = max(
            predictions,
            key=lambda p: p.decision.confidence * p.confidence_calibration
        )

        return EnsembleResult(
            final_decision=best_prediction.decision,
            ensemble_strategy=EnsembleStrategy.CONFIDENCE_BASED,
            models_consulted=[best_prediction.model_name],
            consensus_level=best_prediction.decision.confidence,
            disagreement_detected=False,
            disagreement_level=0.0,
            uncertainty_score=1.0 - best_prediction.decision.confidence,
            confidence_level=self._determine_confidence_level(1.0 - best_prediction.decision.confidence),
            reasoning=f"Selected {best_prediction.model_name} with highest calibrated confidence",
            processing_time_ms=0.0
        )

    async def _apply_cascading_strategy(self, predictions: List[ModelPrediction], trade_value: float) -> EnsembleResult:
        """Apply cascading strategy based on trade value."""
        if not predictions:
            return self._create_no_predictions_result()

        if trade_value < self.config.cascading_thresholds["low_value"]:
            # Quick single model for low value
            return await self._apply_confidence_based_strategy(predictions)
        elif trade_value < self.config.cascading_thresholds["medium_value"]:
            # Standard ensemble for medium value
            return await self.weighted_consensus(predictions, "unknown")
        else:
            # Enhanced consensus for high value
            return await self._apply_uncertainty_aware_strategy(predictions)

    async def _apply_uncertainty_aware_strategy(self, predictions: List[ModelPrediction]) -> EnsembleResult:
        """Apply uncertainty-aware strategy."""
        if not predictions:
            return self._create_no_predictions_result()

        # Quantify uncertainty first
        uncertainty_result = await self.quantify_uncertainty(predictions)

        # If uncertainty is too high, consider skipping
        if uncertainty_result["uncertainty_score"] > self.config.uncertainty_threshold:
            return EnsembleResult(
                final_decision=None,
                ensemble_strategy=EnsembleStrategy.UNCERTAINTY_AWARE,
                models_consulted=[p.model_name for p in predictions],
                consensus_level=0.0,
                disagreement_detected=True,
                disagreement_level=uncertainty_result["uncertainty_score"],
                uncertainty_score=uncertainty_result["uncertainty_score"],
                confidence_level=uncertainty_result["confidence_level"],
                reasoning=f"Skipping due to high uncertainty: {uncertainty_result['uncertainty_score']:.2f}",
                processing_time_ms=0.0
            )

        # Otherwise, proceed with weighted voting
        result = await self.weighted_consensus(predictions, "unknown")
        result.ensemble_strategy = EnsembleStrategy.UNCERTAINTY_AWARE
        result.uncertainty_score = uncertainty_result["uncertainty_score"]

        return result

    async def _update_model_weights(
        self,
        predictions: List[ModelPrediction],
        market_category: str
    ) -> None:
        """Update model weights based on recent performance."""
        for prediction in predictions:
            try:
                # Get recent performance for this model
                performance_score = await self._get_model_performance_score(
                    prediction.model_name, market_category
                )

                # Update weight (higher performance = higher weight)
                prediction.weight = max(0.1, performance_score * self.config.performance_weight_factor)
                prediction.performance_score = performance_score

            except Exception as e:
                self.logger.warning(f"Error updating weight for {prediction.model_name}: {e}")
                prediction.weight = 1.0  # Default weight

    async def _get_model_performance_score(self, model_name: str, market_category: str) -> float:
        """Get performance score for a model in a specific category."""
        try:
            # Update cache if needed
            if (datetime.now() - self.last_performance_update).total_seconds() > 3600:
                await self._update_performance_cache()

            # Get from cache
            cache_key = f"{model_name}_{market_category}"
            return self.performance_cache.get(cache_key, 0.5)  # Default score if not found

        except Exception as e:
            self.logger.error(f"Error getting performance score for {model_name}: {e}")
            return 0.5

    async def _update_performance_cache(self) -> None:
        """Update performance score cache."""
        try:
            # This would query the performance tracker for recent data
            # For now, use mock data
            mock_scores = {
                "grok-4_technology": 0.85,
                "grok-4_finance": 0.78,
                "grok-3_technology": 0.72,
                "grok-3_finance": 0.82,
                "gpt-4_technology": 0.80,
                "gpt-4_finance": 0.75
            }

            self.performance_cache.update(mock_scores)
            self.last_performance_update = datetime.now()

        except Exception as e:
            self.logger.error(f"Error updating performance cache: {e}")

    async def _calculate_weighted_votes(
        self,
        predictions: List[ModelPrediction]
    ) -> Dict[str, float]:
        """Calculate weighted votes for each action."""
        votes = {"BUY": 0.0, "SKIP": 0.0, "SELL": 0.0}
        total_weight = 0.0

        for prediction in predictions:
            action = prediction.decision.action
            confidence = prediction.decision.confidence
            weight = prediction.weight

            votes[action] += confidence * weight
            total_weight += weight

        # Normalize votes
        if total_weight > 0:
            for action in votes:
                votes[action] /= total_weight

        return votes

    async def _calculate_weighted_disagreement(self, predictions: List[ModelPrediction]) -> float:
        """Calculate weighted disagreement score."""
        if len(predictions) < 2:
            return 0.0

        # Compare each prediction to the weighted average
        weighted_actions = {}
        total_weight = 0.0

        for prediction in predictions:
            action = prediction.decision.action
            weight = prediction.weight

            if action not in weighted_actions:
                weighted_actions[action] = 0.0
            weighted_actions[action] += weight
            total_weight += weight

        # Calculate disagreement as 1 - max vote share
        if total_weight > 0:
            max_vote_share = max(weighted_actions.values()) / total_weight
            return 1.0 - max_vote_share

        return 0.0

    async def _make_weighted_decision(
        self,
        weighted_votes: Dict[str, float],
        disagreement_result: Dict[str, Any],
        uncertainty_result: Dict[str, Any]
    ) -> Optional[TradingDecision]:
        """Make final decision based on weighted votes and uncertainty."""
        if not weighted_votes:
            return None

        # Get action with highest weighted vote
        best_action = max(weighted_votes.items(), key=lambda x: x[1])

        # Skip if vote share is too low or disagreement is too high
        if (best_action[1] < 0.5 or
            disagreement_result.get("disagreement_level", 0) > self.config.disagreement_threshold or
            uncertainty_result.get("uncertainty_score", 0) > self.config.uncertainty_threshold):
            return None

        # Determine side based on market context and action
        # For now, use YES for BUY actions and NO for SKIP/SELL actions
        # TODO: Implement more sophisticated side determination based on market data
        determined_side = "YES" if best_action[0] == "BUY" else "NO"

        return TradingDecision(
            action=best_action[0],
            side=determined_side,
            confidence=best_action[1],
            reasoning=f"Weighted ensemble decision: {best_action[0]} ({best_action[1]:.2f} vote share)"
        )

    async def _generate_weighted_reasoning(
        self,
        weighted_votes: Dict[str, float],
        disagreement_result: Dict[str, Any],
        uncertainty_result: Dict[str, Any]
    ) -> str:
        """Generate reasoning for weighted decision."""
        reasoning_parts = []

        # Weighted vote explanation
        best_action = max(weighted_votes.items(), key=lambda x: x[1])
        reasoning_parts.append(
            f"Weighted voting favored {best_action[0]} with {best_action[1]:.2f} vote share"
        )

        # Disagreement explanation
        if disagreement_result["disagreement_detected"]:
            reasoning_parts.append(
                f"Models show disagreement (level: {disagreement_result['disagreement_level']:.2f})"
            )
        else:
            reasoning_parts.append("Models show strong agreement")

        # Uncertainty explanation
        confidence_level = uncertainty_result["confidence_level"]
        reasoning_parts.append(f"Decision made with {confidence_level} confidence")

        return " | ".join(reasoning_parts)

    def _determine_confidence_level(self, uncertainty_score: float) -> ConfidenceLevel:
        """Determine confidence level from uncertainty score."""
        if uncertainty_score < 0.2:
            return ConfidenceLevel.VERY_HIGH
        elif uncertainty_score < 0.4:
            return ConfidenceLevel.HIGH
        elif uncertainty_score < 0.6:
            return ConfidenceLevel.MEDIUM
        elif uncertainty_score < 0.8:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _get_action_distribution(self, actions: List[str], weights: List[float]) -> Dict[str, float]:
        """Get distribution of actions with weights."""
        distribution = {}
        total_weight = sum(weights)

        for action, weight in zip(actions, weights):
            if action not in distribution:
                distribution[action] = 0.0
            distribution[action] += weight

        if total_weight > 0:
            for action in distribution:
                distribution[action] /= total_weight

        return distribution

    async def _generate_uncertainty_recommendations(self, uncertainty_score: float) -> List[str]:
        """Generate recommendations based on uncertainty level."""
        recommendations = []

        if uncertainty_score < 0.3:
            recommendations.append("Proceed with normal position sizing")
            recommendations.append("High confidence in ensemble decision")
        elif uncertainty_score < 0.6:
            recommendations.append("Consider reduced position sizing")
            recommendations.append("Monitor market conditions closely")
        else:
            recommendations.append("Consider skipping this trade")
            recommendations.append("Wait for more favorable conditions")
            recommendations.append("Gather additional market information")

        return recommendations

    def _create_no_models_result(self) -> EnsembleResult:
        """Create result for when no models are available."""
        return EnsembleResult(
            final_decision=None,
            ensemble_strategy=EnsembleStrategy.CONSENSUS,
            models_consulted=[],
            consensus_level=0.0,
            disagreement_detected=True,
            disagreement_level=1.0,
            uncertainty_score=1.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            reasoning="No models available for ensemble decision",
            processing_time_ms=0.0
        )

    def _create_no_predictions_result(self) -> EnsembleResult:
        """Create result for when no predictions are available."""
        return EnsembleResult(
            final_decision=None,
            ensemble_strategy=EnsembleStrategy.CONSENSUS,
            models_consulted=[],
            consensus_level=0.0,
            disagreement_detected=True,
            disagreement_level=1.0,
            uncertainty_score=1.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            reasoning="No valid predictions from models",
            processing_time_ms=0.0
        )

    def _create_error_result(self, error: Exception, start_time: datetime) -> EnsembleResult:
        """Create result for when an error occurs."""
        return EnsembleResult(
            final_decision=None,
            ensemble_strategy=EnsembleStrategy.CONSENSUS,
            models_consulted=[],
            consensus_level=0.0,
            disagreement_detected=True,
            disagreement_level=1.0,
            uncertainty_score=1.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            reasoning=f"Error in ensemble processing: {str(error)}",
            processing_time_ms=(datetime.now() - start_time).total_seconds() * 1000
        )

    async def _log_ensemble_decision(
        self,
        result: EnsembleResult,
        market_data: Dict[str, Any]
    ) -> None:
        """Log ensemble decision to database."""
        try:
            ensemble_decision = EnsembleDecision(
                market_id=market_data.get("market_id", "unknown"),
                models_consulted=result.models_consulted,
                final_decision=result.final_decision.action if result.final_decision else "NONE",
                disagreement_level=result.disagreement_level,
                selected_model=result.models_consulted[0] if result.models_consulted else "NONE",
                reasoning=result.reasoning,
                timestamp=datetime.now()
            )

            await self.db_manager.save_ensemble_decision(ensemble_decision)

        except Exception as e:
            self.logger.error(f"Error logging ensemble decision: {e}")