"""
Ensemble Coordination Layer - Manages coordination between all ensemble components.

This layer provides a unified interface for the ensemble system, coordinating between:
- Performance tracking
- Model selection
- Cost optimization
- Ensemble engine
- Fallback management
- Provider management

Ensures consistent state management and configuration across system restarts.
"""

import asyncio
import json
import pickle
import os
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path

from src.utils.logging_setup import TradingLoggerMixin
from src.intelligence.ensemble_engine import EnsembleEngine, EnsembleConfig, EnsembleResult
from src.intelligence.model_selector import ModelSelector
from src.intelligence.cost_optimizer import CostOptimizer
from src.intelligence.fallback_manager import FallbackManager
from src.intelligence.provider_manager import ProviderManager
from src.utils.performance_tracker import PerformanceTracker
from src.utils.database import DatabaseManager
from src.config.settings import settings


@dataclass
class EnsembleState:
    """Persistent ensemble state management."""
    last_cleanup_time: datetime = field(default_factory=datetime.now)
    model_health_status: Dict[str, bool] = field(default_factory=dict)
    performance_cache: Dict[str, float] = field(default_factory=dict)
    recent_decisions: List[Dict[str, Any]] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    configuration_version: str = "1.0"


@dataclass
class EnsembleMetrics:
    """Real-time ensemble performance metrics."""
    total_decisions: int = 0
    successful_decisions: int = 0
    ensemble_agreements: int = 0
    ensemble_disagreements: int = 0
    average_confidence: float = 0.0
    average_uncertainty: float = 0.0
    cost_savings: float = 0.0
    response_time_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class EnsembleCoordinator(TradingLoggerMixin):
    """
    Coordinates all ensemble components and manages system state.

    Provides:
    - Unified interface for ensemble operations
    - State persistence across system restarts
    - Component health monitoring
    - Performance metrics collection
    - Configuration management
    - Error recovery coordination
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize ensemble coordinator.

        Args:
            db_manager: Database manager for persistence
        """
        self.db_manager = db_manager

        # State management
        self.state = EnsembleState()
        self.metrics = EnsembleMetrics()

        # State file paths
        self.state_file = Path("logs/ensemble_state.pkl")
        self.metrics_file = Path("logs/ensemble_metrics.json")

        # Ensure logs directory exists
        self.state_file.parent.mkdir(exist_ok=True)

        # Component references (initialized on demand)
        self._performance_tracker: Optional[PerformanceTracker] = None
        self._model_selector: Optional[ModelSelector] = None
        self._cost_optimizer: Optional[CostOptimizer] = None
        self._ensemble_engine: Optional[EnsembleEngine] = None
        self._fallback_manager: Optional[FallbackManager] = None
        self._provider_manager: Optional[ProviderManager] = None

        # Coordination state
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
        self._last_health_check = datetime.now()

        self.logger.info("Ensemble coordinator initialized")

    async def initialize(self) -> bool:
        """
        Initialize all ensemble components.

        Returns:
            True if initialization successful, False otherwise
        """
        async with self._initialization_lock:
            if self._initialized:
                return True

            try:
                self.logger.info("Initializing ensemble components...")

                # Load persistent state
                await self._load_state()

                # Initialize core components
                await self._initialize_core_components()

                # Initialize ensemble engine
                await self._initialize_ensemble_engine()

                # Initialize management components
                await self._initialize_management_components()

                # Perform health check
                await self._perform_health_check()

                # Start background tasks
                await self._start_background_tasks()

                self._initialized = True
                self.logger.info("Ensemble coordinator initialized successfully")
                return True

            except Exception as e:
                self.logger.error(f"Failed to initialize ensemble coordinator: {e}")
                return False

    async def get_ensemble_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        trade_value: float,
        market_category: str,
        strategy: Optional[str] = None,
        force_advanced: bool = False
    ) -> Optional[EnsembleResult]:
        """
        Get coordinated ensemble decision.

        Args:
            market_data: Market information
            portfolio_data: Portfolio state
            trade_value: Value of potential trade
            market_category: Category of the market
            strategy: Preferred ensemble strategy
            force_advanced: Force advanced ensemble even for low values

        Returns:
            EnsembleResult with detailed decision information
        """
        if not self._initialized:
            await self.initialize()

        start_time = datetime.now()

        try:
            # Determine strategy based on trade value and configuration
            if strategy is None:
                strategy = self._determine_optimal_strategy(trade_value, force_advanced)

            # Log decision request
            self.logger.info(
                "Requesting ensemble decision",
                trade_value=trade_value,
                market_category=market_category,
                strategy=strategy
            )

            # Get decision from ensemble engine
            result = await self._ensemble_engine.get_ensemble_decision(
                market_data, portfolio_data, trade_value, market_category, strategy
            )

            # Update metrics
            self._update_decision_metrics(result, start_time)

            # Log result
            if result.final_decision:
                self.logger.info(
                    f"Ensemble decision: {result.final_decision.action} {result.final_decision.side} "
                    f"(confidence: {result.final_decision.confidence:.1%}, uncertainty: {result.uncertainty_score:.2f})"
                )
            else:
                self.logger.info(
                    f"No ensemble decision - disagreement: {result.disagreement_detected}, "
                    f"uncertainty: {result.uncertainty_score:.2f}"
                )

            return result

        except Exception as e:
            self.logger.error(f"Error in ensemble decision: {e}")
            # Update error metrics
            self.metrics.total_decisions += 1
            await self._save_metrics()
            return None

    async def get_component_health(self) -> Dict[str, Any]:
        """
        Get health status of all ensemble components.

        Returns:
            Dictionary with component health information
        """
        health_status = {
            "coordinator": {
                "initialized": self._initialized,
                "last_health_check": self._last_health_check.isoformat(),
                "state_loaded": bool(self.state_file.exists())
            },
            "ensemble_engine": await self._get_component_health_info("ensemble_engine"),
            "performance_tracker": await self._get_component_health_info("performance_tracker"),
            "model_selector": await self._get_component_health_info("model_selector"),
            "cost_optimizer": await self._get_component_health_info("cost_optimizer"),
            "fallback_manager": await self._get_component_health_info("fallback_manager"),
            "provider_manager": await self._get_component_health_info("provider_manager")
        }

        return health_status

    async def get_ensemble_metrics(self) -> EnsembleMetrics:
        """
        Get current ensemble performance metrics.

        Returns:
            EnsembleMetrics object with current metrics
        """
        # Update timestamp
        self.metrics.last_updated = datetime.now()

        # Calculate success rate
        if self.metrics.total_decisions > 0:
            success_rate = self.metrics.successful_decisions / self.metrics.total_decisions
            self.metrics.statistics["success_rate"] = success_rate

        return self.metrics

    def _determine_optimal_strategy(self, trade_value: float, force_advanced: bool) -> str:
        """Determine optimal ensemble strategy based on trade value."""
        if force_advanced:
            return "cascading"

        # Use cascading strategy based on trade value thresholds
        if trade_value >= settings.trading.ensemble_cascading_high_value_threshold:
            return "cascading"
        elif trade_value >= settings.trading.ensemble_cascading_medium_value_threshold:
            return "weighted_voting" if settings.trading.ensemble_enable_weighted_voting else "consensus"
        else:
            return "confidence_based"

    async def _initialize_core_components(self) -> None:
        """Initialize core ensemble components."""
        self.logger.info("Initializing core components...")

        # Performance tracker
        self._performance_tracker = PerformanceTracker(self.db_manager)

        # Model selector
        self._model_selector = ModelSelector(self._performance_tracker)

        # Cost optimizer
        self._cost_optimizer = CostOptimizer(self.db_manager)

        self.logger.info("Core components initialized")

    async def _initialize_ensemble_engine(self) -> None:
        """Initialize ensemble engine."""
        self.logger.info("Initializing ensemble engine...")

        # Create ensemble configuration from settings
        config = EnsembleConfig(
            min_consensus_threshold=settings.trading.ensemble_consensus_threshold,
            disagreement_threshold=settings.trading.ensemble_disagreement_threshold,
            uncertainty_threshold=settings.trading.ensemble_uncertainty_threshold,
            enable_weighted_voting=settings.trading.ensemble_enable_weighted_voting,
            enable_confidence_calibration=settings.trading.ensemble_enable_confidence_calibration,
            performance_weight_factor=settings.trading.ensemble_performance_weight_factor,
            max_models_per_decision=settings.trading.ensemble_max_models_per_decision,
            timeout_seconds=settings.trading.ensemble_timeout_seconds
        )

        self._ensemble_engine = EnsembleEngine(
            self.db_manager,
            self._performance_tracker,
            self._model_selector,
            config
        )

        self.logger.info("Ensemble engine initialized")

    async def _initialize_management_components(self) -> None:
        """Initialize management components."""
        self.logger.info("Initializing management components...")

        # Fallback manager
        self._fallback_manager = FallbackManager()

        # Provider manager
        self._provider_manager = ProviderManager()

        self.logger.info("Management components initialized")

    async def _perform_health_check(self) -> None:
        """Perform health check on all components."""
        self.logger.info("Performing ensemble health check...")

        # Check each component
        components_to_check = [
            "ensemble_engine", "performance_tracker", "model_selector",
            "cost_optimizer", "fallback_manager", "provider_manager"
        ]

        for component_name in components_to_check:
            try:
                component = getattr(self, f"_{component_name}")
                if component:
                    # Perform basic health check
                    healthy = await self._check_component_health(component)
                    self.state.model_health_status[component_name] = healthy
                    self.logger.debug(f"Component {component_name} healthy: {healthy}")
            except Exception as e:
                self.logger.warning(f"Health check failed for {component_name}: {e}")
                self.state.model_health_status[component_name] = False

        self._last_health_check = datetime.now()

    async def _check_component_health(self, component: Any) -> bool:
        """Check health of individual component."""
        try:
            # Basic health check - component should respond to basic methods
            if hasattr(component, 'get_ensemble_decision'):
                # Ensemble engine
                return True
            elif hasattr(component, 'get_model_ranking'):
                # Model selector or performance tracker
                return True
            elif hasattr(component, 'calculate_cost_efficiency'):
                # Cost optimizer
                return True
            elif hasattr(component, 'check_provider_health'):
                # Fallback or provider manager
                return True
            else:
                return False
        except Exception:
            return False

    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        self.logger.info("Starting background tasks...")

        # Start health monitoring
        asyncio.create_task(self._health_monitoring_loop())

        # Start metrics collection
        asyncio.create_task(self._metrics_collection_loop())

        # Start state cleanup
        asyncio.create_task(self._state_cleanup_loop())

        self.logger.info("Background tasks started")

    async def _health_monitoring_loop(self) -> None:
        """Background health monitoring loop."""
        while True:
            try:
                await asyncio.sleep(settings.trading.ensemble_health_check_interval_seconds)
                await self._perform_health_check()
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _metrics_collection_loop(self) -> None:
        """Background metrics collection loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Collect every 5 minutes
                await self._collect_metrics()
                await self._save_metrics()
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _state_cleanup_loop(self) -> None:
        """Background state cleanup loop."""
        while True:
            try:
                await asyncio.sleep(3600)  # Cleanup every hour
                await self._cleanup_state()
                await self._save_state()
            except Exception as e:
                self.logger.error(f"Error in state cleanup loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying

    async def _collect_metrics(self) -> None:
        """Collect performance metrics from components."""
        try:
            # Collect from performance tracker
            if self._performance_tracker:
                model_rankings = self._model_selector.get_model_ranking() if self._model_selector else {}
                for model, ranking in model_rankings.items():
                    self.state.performance_cache[model] = ranking

            # Update statistics
            self.state.statistics["models_consulted"] = len(self.state.model_health_status)
            self.state.statistics["healthy_models"] = sum(1 for healthy in self.state.model_health_status.values() if healthy)

        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")

    async def _cleanup_state(self) -> None:
        """Clean up old state data."""
        try:
            # Clean up recent decisions (keep last 100)
            if len(self.state.recent_decisions) > 100:
                self.state.recent_decisions = self.state.recent_decisions[-100:]

            # Clean up error counts (reset if older than 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            for model in list(self.state.error_counts.keys()):
                # Check if model is healthy now
                if (self.state.model_health_status.get(model, False) and
                    self.state.error_counts.get(model, 0) > 0):
                    # Reset error counts for healthy models
                    self.state.error_counts[model] = 0

            self.state.last_cleanup_time = datetime.now()

        except Exception as e:
            self.logger.error(f"Error cleaning up state: {e}")

    def _update_decision_metrics(self, result: EnsembleResult, start_time: datetime) -> None:
        """Update metrics based on decision result."""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        self.metrics.total_decisions += 1
        self.metrics.response_time_ms = processing_time

        if result.final_decision:
            self.metrics.successful_decisions += 1
            self.metrics.average_confidence = (
                (self.metrics.average_confidence * (self.metrics.total_decisions - 1) +
                 result.final_decision.confidence) / self.metrics.total_decisions
            )
            self.metrics.average_uncertainty = (
                (self.metrics.average_uncertainty * (self.metrics.total_decisions - 1) +
                 result.uncertainty_score) / self.metrics.total_decisions
            )

        if result.disagreement_detected:
            self.metrics.ensemble_disagreements += 1
        else:
            self.metrics.ensemble_agreements += 1

    async def _get_component_health_info(self, component_name: str) -> Dict[str, Any]:
        """Get health information for a specific component."""
        component = getattr(self, f"_{component_name}", None)
        if component is None:
            return {"status": "not_initialized"}

        try:
            return {
                "status": "healthy",
                "initialized": True,
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }

    async def _load_state(self) -> None:
        """Load persistent state from disk."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'rb') as f:
                    self.state = pickle.load(f)
                self.logger.info("Ensemble state loaded from disk")
            else:
                self.logger.info("No existing state file, using defaults")
        except Exception as e:
            self.logger.warning(f"Failed to load ensemble state: {e}, using defaults")

    async def _save_state(self) -> None:
        """Save persistent state to disk."""
        try:
            with open(self.state_file, 'wb') as f:
                pickle.dump(self.state, f)
        except Exception as e:
            self.logger.error(f"Failed to save ensemble state: {e}")

    async def _save_metrics(self) -> None:
        """Save metrics to disk."""
        try:
            metrics_data = {
                "total_decisions": self.metrics.total_decisions,
                "successful_decisions": self.metrics.successful_decisions,
                "ensemble_agreements": self.metrics.ensemble_agreements,
                "ensemble_disagreements": self.metrics.ensemble_disagreements,
                "average_confidence": self.metrics.average_confidence,
                "average_uncertainty": self.metrics.average_uncertainty,
                "cost_savings": self.metrics.cost_savings,
                "response_time_ms": self.metrics.response_time_ms,
                "last_updated": self.metrics.last_updated.isoformat()
            }

            with open(self.metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    async def shutdown(self) -> None:
        """Shutdown ensemble coordinator and save state."""
        self.logger.info("Shutting down ensemble coordinator...")

        try:
            # Save current state
            await self._save_state()
            await self._save_metrics()

            self._initialized = False
            self.logger.info("Ensemble coordinator shutdown complete")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")