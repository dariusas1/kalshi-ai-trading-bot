"""
Cost Optimization Framework for Enhanced AI Model Integration.

Implements dynamic cost-per-performance modeling, budget-aware selection,
intelligent caching, real-time cost monitoring, and automated spending controls.
"""

import asyncio
import hashlib
import json
import statistics
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
import logging

from src.utils.performance_tracker import (
    PerformanceTracker,
    ModelPerformanceMetrics,
    CostPerformanceMetrics
)
from src.utils.database import DatabaseManager
from src.utils.logging_setup import TradingLoggerMixin


@dataclass
class CostEfficiencyMetrics:
    """Metrics for cost efficiency analysis."""
    model_name: str
    cost_efficiency_ratio: float  # accuracy per dollar
    cost_per_correct_prediction: float
    roi_score: float
    budget_utilization: float
    cost_trend: str  # "improving", "stable", "declining"


@dataclass
class BudgetStatus:
    """Current budget status information."""
    daily_limit: float
    spent: float
    remaining: float
    percentage_used: float
    status: str  # "healthy", "warning", "critical", "exhausted"
    projected_daily_spend: float
    recommended_actions: List[str]


@dataclass
class CacheEntry:
    """Cached model result with metadata."""
    cache_key: str
    model_name: str
    market_data: Dict[str, Any]
    result: Dict[str, Any]
    timestamp: datetime
    cost: float
    accuracy: float
    ttl_minutes: int
    hit_count: int = 0


@dataclass
class SpendingControl:
    """Automated spending control configuration."""
    enabled: bool
    max_hourly_spend: float
    alert_threshold: float  # percentage of daily budget
    auto_model_switching: bool
    cost_reduction_target: float  # percentage


@dataclass
class DynamicCostModel:
    """Dynamic cost model for a model in specific conditions."""
    model_name: str
    market_category: str
    base_cost_per_request: float
    cost_efficiency_factor: float
    accuracy_cost_tradeoff: float
    budget_sensitivity: float
    performance_window_hours: int


@dataclass
class CostOptimizationConfig:
    """Configuration for cost optimization framework."""
    # Dynamic modeling
    enable_dynamic_modeling: bool = True
    cost_performance_window_hours: int = 24
    min_predictions_for_modeling: int = 5

    # Budget controls
    daily_budget_limit: float = 50.0
    enable_budget_controls: bool = True
    budget_alert_threshold: float = 0.8
    auto_spending_controls: bool = True

    # Caching
    enable_intelligent_caching: bool = True
    cache_ttl_minutes: int = 30
    max_cache_size: int = 1000
    cache_similarity_threshold: float = 0.9

    # Real-time monitoring
    enable_real_time_monitoring: bool = True
    monitoring_interval_seconds: int = 60
    cost_tracking_retention_days: int = 30


class CostOptimizer(TradingLoggerMixin):
    """
    Advanced cost optimization framework for AI model selection and usage.

    Provides dynamic cost-per-performance modeling, budget-aware selection,
    intelligent caching, real-time monitoring, and automated spending controls.
    """

    def __init__(
        self,
        db_manager: DatabaseManager,
        performance_tracker: PerformanceTracker,
        config: Optional[CostOptimizationConfig] = None
    ):
        """
        Initialize cost optimizer.

        Args:
            db_manager: Database manager for persistence
            performance_tracker: Performance tracking system
            config: Cost optimization configuration
        """
        self.db_manager = db_manager
        self.performance_tracker = performance_tracker
        self.config = config or CostOptimizationConfig()

        # Intelligent caching system
        self.result_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

        # Dynamic cost models
        self.cost_models: Dict[str, DynamicCostModel] = {}

        # Real-time spending tracking
        self.current_spend = defaultdict(float)
        self.spend_history: List[Dict[str, Any]] = []
        self.last_cost_update = datetime.now()

        # Budget status
        self.daily_spend = 0.0
        self.budget_status: Optional[BudgetStatus] = None

        self.logger.info("Cost optimizer initialized", config=self.config.__dict__)

    async def calculate_cost_efficiency(
        self,
        model_name: str,
        market_category: Optional[str] = None,
        time_window_hours: int = 24
    ) -> float:
        """
        Calculate cost efficiency ratio for a model.

        Args:
            model_name: Name of the AI model
            market_category: Optional market category filter
            time_window_hours: Time window for analysis

        Returns:
            Cost efficiency ratio (0-1, higher is better)
        """
        try:
            # Get cost performance metrics
            cost_metrics = await self.performance_tracker.get_cost_performance_metrics(
                model_name, time_window_hours
            )

            # Get model ranking for context
            rankings = await self.performance_tracker.get_model_ranking(
                model_name=model_name,
                start_time=datetime.now() - timedelta(hours=time_window_hours)
            )

            if not rankings:
                return 0.0

            model_metrics = rankings[0]

            # Calculate comprehensive efficiency score
            efficiency_score = self._calculate_efficiency_score(cost_metrics, model_metrics)

            self.logger.debug(
                f"Calculated cost efficiency for {model_name}",
                efficiency_score=efficiency_score,
                cost_performance_ratio=cost_metrics.cost_performance_ratio,
                accuracy=model_metrics.accuracy
            )

            return efficiency_score

        except Exception as e:
            self.logger.error(f"Error calculating cost efficiency for {model_name}: {e}")
            return 0.0

    def _calculate_efficiency_score(
        self,
        cost_metrics: CostPerformanceMetrics,
        model_metrics: ModelPerformanceMetrics
    ) -> float:
        """Calculate comprehensive efficiency score from metrics."""
        # Normalize cost performance ratio (accuracy per dollar)
        normalized_cost_ratio = min(1.0, cost_metrics.cost_performance_ratio * 10)

        # Normalize ROI score
        normalized_roi = min(1.0, cost_metrics.roi_score / 100)

        # Budget efficiency (higher is better)
        normalized_budget_eff = min(1.0, cost_metrics.budget_efficiency / 100)

        # Decision quality factor
        quality_factor = model_metrics.avg_decision_quality

        # Response time factor (faster is better, normalized to 0-1)
        time_factor = max(0.0, 1.0 - (model_metrics.avg_response_time_ms / 5000))

        # Weighted combination
        efficiency_score = (
            normalized_cost_ratio * 0.3 +
            normalized_roi * 0.2 +
            normalized_budget_eff * 0.2 +
            quality_factor * 0.2 +
            time_factor * 0.1
        )

        return efficiency_score

    async def monitor_spend(
        self,
        model_costs: List[Dict[str, Any]],
        update_budget_status: bool = True
    ) -> Dict[str, Any]:
        """
        Monitor real-time spending and update budget status.

        Args:
            model_costs: List of model cost records
            update_budget_status: Whether to update budget status

        Returns:
            Current spending status
        """
        try:
            current_time = datetime.now()
            total_spend = 0.0

            # Process new cost records
            for cost_record in model_costs:
                model_name = cost_record["model"]
                cost = cost_record["cost"]

                # Update model-specific spend
                self.current_spend[model_name] += cost
                total_spend += cost

                # Track detailed cost history
                self.spend_history.append({
                    "model": model_name,
                    "cost": cost,
                    "timestamp": current_time,
                    "hour": current_time.hour
                })

            # Update daily spend
            self.daily_spend += total_spend

            # Update budget status if requested
            if update_budget_status:
                self.budget_status = await self._calculate_budget_status()

            # Clean old cost history (retention policy)
            cutoff_date = current_time - timedelta(days=self.config.cost_tracking_retention_days)
            self.spend_history = [
                record for record in self.spend_history
                if record["timestamp"] > cutoff_date
            ]

            spending_status = {
                "current_spend": self.daily_spend,
                "model_breakdown": dict(self.current_spend),
                "budget_status": self.budget_status.__dict__ if self.budget_status else None,
                "last_updated": current_time,
                "transaction_count": len(model_costs)
            }

            self.logger.debug(
                "Updated spend monitoring",
                total_spend=total_spend,
                daily_spend=self.daily_spend,
                transaction_count=len(model_costs)
            )

            return spending_status

        except Exception as e:
            self.logger.error(f"Error monitoring spend: {e}")
            return {"error": str(e)}

    async def enforce_budget_limits(self) -> BudgetStatus:
        """
        Enforce budget limits and generate spending controls.

        Returns:
            Current budget status
        """
        try:
            budget_status = await self._calculate_budget_status()

            # Generate spending controls if enabled
            if self.config.auto_spending_controls and self.config.enable_budget_controls:
                spending_controls = await self._generate_spending_controls(budget_status)
                budget_status.recommended_actions.extend(spending_controls)

            # Save budget status to database
            await self._save_budget_status(budget_status)

            self.logger.info(
                "Budget limit enforcement check",
                status=budget_status.status,
                percentage_used=budget_status.percentage_used,
                remaining_budget=budget_status.remaining
            )

            return budget_status

        except Exception as e:
            self.logger.error(f"Error enforcing budget limits: {e}")
            return BudgetStatus(
                daily_limit=self.config.daily_budget_limit,
                spent=0.0,
                remaining=self.config.daily_budget_limit,
                percentage_used=0.0,
                status="error",
                projected_daily_spend=0.0,
                recommended_actions=["Error calculating budget status"]
            )

    async def _calculate_budget_status(self) -> BudgetStatus:
        """Calculate current budget status."""
        # Get current daily spend
        today = datetime.now().date()
        daily_spend = await self._get_daily_spend(today)

        # Calculate remaining budget
        remaining = max(0.0, self.config.daily_budget_limit - daily_spend)
        percentage_used = (daily_spend / self.config.daily_budget_limit) * 100

        # Determine status
        if percentage_used >= 100:
            status = "exhausted"
        elif percentage_used >= 95:
            status = "critical"
        elif percentage_used >= self.config.budget_alert_threshold * 100:
            status = "warning"
        else:
            status = "healthy"

        # Project daily spend based on current rate
        current_hour = datetime.now().hour
        if current_hour > 0:
            hourly_rate = daily_spend / current_hour
            projected_daily_spend = hourly_rate * 24
        else:
            projected_daily_spend = daily_spend

        # Generate recommended actions
        recommended_actions = await self._generate_budget_recommendations(
            status, percentage_used, remaining
        )

        return BudgetStatus(
            daily_limit=self.config.daily_budget_limit,
            spent=daily_spend,
            remaining=remaining,
            percentage_used=percentage_used,
            status=status,
            projected_daily_spend=projected_daily_spend,
            recommended_actions=recommended_actions
        )

    async def _generate_spending_controls(self, budget_status: BudgetStatus) -> List[str]:
        """Generate automated spending control recommendations."""
        controls = []

        if budget_status.status in ["warning", "critical"]:
            controls.append("reduce_model_usage")
            controls.append("switch_to_cheaper_models")

        if budget_status.status == "critical":
            controls.append("enable_strict_caching")
            controls.append("reduce_ensemble_size")

        if budget_status.projected_daily_spend > budget_status.daily_limit:
            controls.append("implement_cost_throttling")

        return controls

    async def get_cached_result(
        self,
        model_name: str,
        market_data: Dict[str, Any],
        similarity_threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached result for similar market conditions.

        Args:
            model_name: Name of the model
            market_data: Current market data
            similarity_threshold: Minimum similarity threshold

        Returns:
            Cached result if available and similar
        """
        try:
            if not self.config.enable_intelligent_caching:
                return None

            threshold = similarity_threshold or self.config.cache_similarity_threshold

            # Generate cache key for exact match
            cache_key = self._generate_cache_key(model_name, market_data)

            # Check for exact match first
            if cache_key in self.result_cache:
                cache_entry = self.result_cache[cache_key]

                # Check TTL
                if self._is_cache_entry_valid(cache_entry):
                    cache_entry.hit_count += 1
                    self.cache_hits += 1

                    # Move to end (LRU)
                    self.result_cache.move_to_end(cache_key)

                    self.logger.debug(f"Cache hit for {model_name}", cache_key=cache_key)
                    return cache_entry.result
                else:
                    # Remove expired entry
                    del self.result_cache[cache_key]

            # Look for similar entries
            similar_entry = await self._find_similar_cache_entry(
                model_name, market_data, threshold
            )

            if similar_entry:
                similar_entry.hit_count += 1
                self.cache_hits += 1
                self.result_cache.move_to_end(similar_entry.cache_key)

                self.logger.debug(
                    f"Similar cache hit for {model_name}",
                    similarity_score=await self._calculate_similarity(
                        similar_entry.market_data, market_data
                    )
                )
                return similar_entry.result

            self.cache_misses += 1
            return None

        except Exception as e:
            self.logger.error(f"Error getting cached result: {e}")
            return None

    async def cache_model_result(
        self,
        model_name: str,
        market_data: Dict[str, Any],
        result: Dict[str, Any],
        cost: float = 0.0,
        accuracy: float = 0.0
    ) -> str:
        """
        Cache model result for future reuse.

        Args:
            model_name: Name of the model
            market_data: Market data context
            result: Model result to cache
            cost: Cost of the prediction
            accuracy: Accuracy of the prediction

        Returns:
            Cache key for the stored result
        """
        try:
            if not self.config.enable_intelligent_caching:
                return ""

            # Check if result should be cached
            if not await self._should_cache_result(model_name, market_data, result):
                return ""

            cache_key = self._generate_cache_key(model_name, market_data)

            # Create cache entry
            cache_entry = CacheEntry(
                cache_key=cache_key,
                model_name=model_name,
                market_data=market_data.copy(),
                result=result.copy(),
                timestamp=datetime.now(),
                cost=cost,
                accuracy=accuracy,
                ttl_minutes=self.config.cache_ttl_minutes
            )

            # Add to cache (LRU)
            self.result_cache[cache_key] = cache_entry

            # Enforce cache size limit
            if len(self.result_cache) > self.config.max_cache_size:
                # Remove oldest entries
                while len(self.result_cache) > self.config.max_cache_size:
                    oldest_key = next(iter(self.result_cache))
                    del self.result_cache[oldest_key]

            self.logger.debug(
                f"Cached result for {model_name}",
                cache_key=cache_key,
                cache_size=len(self.result_cache)
            )

            return cache_key

        except Exception as e:
            self.logger.error(f"Error caching model result: {e}")
            return ""

    def _generate_cache_key(self, model_name: str, market_data: Dict[str, Any]) -> str:
        """Generate cache key for model result."""
        # Normalize market data for consistent key generation
        key_data = {
            "model": model_name,
            "category": market_data.get("category", "unknown"),
            "volume_range": self._get_volume_range(market_data.get("volume", 0)),
            "price_range": self._get_price_range(market_data.get("price", 0)),
            "volatility": market_data.get("volatility", "unknown")
        }

        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_json.encode()).hexdigest()

    def _get_volume_range(self, volume: float) -> str:
        """Get volume range classification."""
        if volume < 1000:
            return "low"
        elif volume < 5000:
            return "medium"
        else:
            return "high"

    def _get_price_range(self, price: float) -> str:
        """Get price range classification."""
        if price < 0.3:
            return "low"
        elif price < 0.7:
            return "medium"
        else:
            return "high"

    async def _find_similar_cache_entry(
        self,
        model_name: str,
        market_data: Dict[str, Any],
        similarity_threshold: float
    ) -> Optional[CacheEntry]:
        """Find cache entry with similar market conditions."""
        try:
            best_entry = None
            best_similarity = 0.0

            for cache_key, cache_entry in self.result_cache.items():
                if cache_entry.model_name != model_name:
                    continue

                if not self._is_cache_entry_valid(cache_entry):
                    continue

                similarity = await self._calculate_similarity(
                    cache_entry.market_data, market_data
                )

                if similarity >= similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = cache_entry

            return best_entry

        except Exception as e:
            self.logger.error(f"Error finding similar cache entry: {e}")
            return None

    async def _calculate_similarity(
        self,
        cached_data: Dict[str, Any],
        current_data: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two market data sets."""
        try:
            similarity_factors = []

            # Category similarity (exact match)
            if cached_data.get("category") == current_data.get("category"):
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.0)

            # Volume similarity
            cached_volume = cached_data.get("volume", 0)
            current_volume = current_data.get("volume", 0)
            if cached_volume > 0 and current_volume > 0:
                volume_similarity = 1.0 - abs(cached_volume - current_volume) / max(cached_volume, current_volume)
                similarity_factors.append(volume_similarity)

            # Price similarity
            cached_price = cached_data.get("price", 0)
            current_price = current_data.get("price", 0)
            if cached_price > 0 and current_price > 0:
                price_similarity = 1.0 - abs(cached_price - current_price)
                similarity_factors.append(price_similarity)

            # Volatility similarity
            if cached_data.get("volatility") == current_data.get("volatility"):
                similarity_factors.append(1.0)
            else:
                similarity_factors.append(0.5)  # Partial match for different volatility

            return statistics.mean(similarity_factors) if similarity_factors else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def _is_cache_entry_valid(self, cache_entry: CacheEntry) -> bool:
        """Check if cache entry is still valid."""
        expiry_time = cache_entry.timestamp + timedelta(minutes=cache_entry.ttl_minutes)
        return datetime.now() < expiry_time

    async def _should_cache_result(
        self,
        model_name: str,
        market_data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> bool:
        """Determine if result should be cached."""
        try:
            # Don't cache skip decisions
            if result.get("action") == "SKIP":
                return False

            # Don't cache very low confidence results
            if result.get("confidence", 0) < 0.5:
                return False

            # Don't cache results for highly volatile markets
            if market_data.get("volatility") == "high":
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking if result should be cached: {e}")
            return False

    async def get_cache_efficiency_metrics(self) -> Dict[str, Any]:
        """Get cache efficiency and performance metrics."""
        try:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

            # Calculate memory usage
            cache_memory = len(self.result_cache) * 1024  # Rough estimate in bytes
            cache_memory_mb = cache_memory / (1024 * 1024)

            # Analyze cache entry statistics
            if self.result_cache:
                hit_counts = [entry.hit_count for entry in self.result_cache.values()]
                avg_hit_count = statistics.mean(hit_counts)
                max_hit_count = max(hit_counts)
            else:
                avg_hit_count = 0
                max_hit_count = 0

            metrics = {
                "hit_rate": hit_rate,
                "total_requests": total_requests,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_size": len(self.result_cache),
                "max_cache_size": self.config.max_cache_size,
                "memory_usage_mb": cache_memory_mb,
                "avg_hit_count": avg_hit_count,
                "max_hit_count": max_hit_count,
                "efficiency_score": hit_rate * min(1.0, len(self.result_cache) / self.config.max_cache_size)
            }

            self.logger.debug(
                "Cache efficiency metrics",
                hit_rate=hit_rate,
                cache_size=len(self.result_cache)
            )

            return metrics

        except Exception as e:
            self.logger.error(f"Error getting cache efficiency metrics: {e}")
            return {"error": str(e)}

    async def update_cost_performance_model(
        self,
        model_name: str,
        market_category: str,
        accuracy_score: float,
        cost_usd: float,
        response_time_ms: float
    ) -> None:
        """
        Update cost performance model with new data.

        Args:
            model_name: Name of the AI model
            market_category: Market category context
            accuracy_score: Accuracy of the prediction
            cost_usd: Cost in USD
            response_time_ms: Response time in milliseconds
        """
        try:
            if not self.config.enable_dynamic_modeling:
                return

            model_key = f"{model_name}_{market_category}"

            # Create or update dynamic cost model
            if model_key not in self.cost_models:
                self.cost_models[model_key] = DynamicCostModel(
                    model_name=model_name,
                    market_category=market_category,
                    base_cost_per_request=cost_usd,
                    cost_efficiency_factor=accuracy_score / (cost_usd + 0.001),
                    accuracy_cost_tradeoff=0.5,
                    budget_sensitivity=0.3,
                    performance_window_hours=self.config.cost_performance_window_hours
                )
            else:
                # Update existing model with new data
                model = self.cost_models[model_key]

                # Update cost efficiency factor
                new_efficiency = accuracy_score / (cost_usd + 0.001)
                model.cost_efficiency_factor = (
                    model.cost_efficiency_factor * 0.7 + new_efficiency * 0.3
                )

                # Update base cost with exponential smoothing
                model.base_cost_per_request = (
                    model.base_cost_per_request * 0.8 + cost_usd * 0.2
                )

                # Adjust accuracy-cost tradeoff based on recent performance
                if accuracy_score > 0.8:
                    model.accuracy_cost_tradeoff = min(1.0, model.accuracy_cost_tradeoff + 0.05)
                elif accuracy_score < 0.6:
                    model.accuracy_cost_tradeoff = max(0.0, model.accuracy_cost_tradeoff - 0.05)

            # Save to database
            await self._save_cost_model_data(model_key, self.cost_models[model_key])

            self.logger.debug(
                f"Updated cost performance model for {model_name}",
                market_category=market_category,
                efficiency_factor=self.cost_models[model_key].cost_efficiency_factor
            )

        except Exception as e:
            self.logger.error(f"Error updating cost performance model: {e}")

    async def select_models_with_budget(
        self,
        available_models: List[str],
        remaining_budget: float,
        trade_value: float,
        market_category: Optional[str] = None
    ) -> List[str]:
        """
        Select models considering budget constraints and cost efficiency.

        Args:
            available_models: List of available models
            remaining_budget: Remaining budget for the period
            trade_value: Value of the potential trade
            market_category: Market category for context

        Returns:
            Selected models ordered by preference
        """
        try:
            if not available_models:
                return []

            # Calculate model scores considering budget
            model_scores = []

            for model_name in available_models:
                score = await self._calculate_budget_aware_score(
                    model_name, remaining_budget, trade_value, market_category
                )
                model_scores.append((model_name, score))

            # Sort by score (descending)
            model_scores.sort(key=lambda x: x[1], reverse=True)

            # Select models within budget
            selected_models = []
            estimated_cost = 0.0

            for model_name, score in model_scores:
                model_cost = await self._estimate_model_cost(model_name, trade_value)

                if estimated_cost + model_cost <= remaining_budget or score > 0.7:
                    selected_models.append(model_name)
                    estimated_cost += model_cost

                    # Limit ensemble size based on trade value
                    max_models = 3 if trade_value > 50 else 2 if trade_value > 10 else 1
                    if len(selected_models) >= max_models:
                        break

            self.logger.info(
                "Budget-aware model selection",
                selected_models=selected_models,
                estimated_cost=estimated_cost,
                remaining_budget=remaining_budget
            )

            return selected_models

        except Exception as e:
            self.logger.error(f"Error selecting models with budget: {e}")
            return available_models[:1]  # Return first model as fallback

    async def _calculate_budget_aware_score(
        self,
        model_name: str,
        remaining_budget: float,
        trade_value: float,
        market_category: Optional[str] = None
    ) -> float:
        """Calculate model score considering budget constraints."""
        try:
            # Get cost efficiency
            cost_efficiency = await self.calculate_cost_efficiency(model_name, market_category)

            # Get estimated cost
            estimated_cost = await self._estimate_model_cost(model_name, trade_value)

            # Budget awareness factor
            budget_factor = 1.0
            if remaining_budget > 0:
                budget_affordability = 1.0 - (estimated_cost / remaining_budget)
                budget_factor = max(0.0, budget_affordability)
            else:
                budget_factor = 0.0

            # Trade value consideration
            value_factor = min(1.0, trade_value / 100.0)  # Normalize trade value

            # Combined score
            score = (
                cost_efficiency * 0.5 +
                budget_factor * 0.3 +
                value_factor * 0.2
            )

            return score

        except Exception as e:
            self.logger.error(f"Error calculating budget-aware score for {model_name}: {e}")
            return 0.5  # Default score

    async def _estimate_model_cost(self, model_name: str, trade_value: float) -> float:
        """Estimate cost for model prediction based on trade value."""
        # Base costs per model (approximate)
        base_costs = {
            "grok-4": 0.05,
            "grok-3": 0.03,
            "gpt-4": 0.04,
            "claude-3": 0.03
        }

        base_cost = base_costs.get(model_name, 0.04)

        # Adjust based on trade value (higher value trades might need more computation)
        if trade_value > 100:
            cost_multiplier = 1.2
        elif trade_value > 50:
            cost_multiplier = 1.1
        elif trade_value < 10:
            cost_multiplier = 0.8
        else:
            cost_multiplier = 1.0

        return base_cost * cost_multiplier

    async def _get_daily_spend(self, date: datetime.date) -> float:
        """Get total spending for a specific date."""
        try:
            # Calculate from spend history (includes all historical spend)
            daily_spend = 0.0
            for record in self.spend_history:
                if record["timestamp"].date() == date:
                    daily_spend += record["cost"]

            return daily_spend

        except Exception as e:
            self.logger.error(f"Error getting daily spend for {date}: {e}")
            return 0.0

    async def _generate_budget_recommendations(
        self,
        status: str,
        percentage_used: float,
        remaining_budget: float
    ) -> List[str]:
        """Generate budget recommendations based on current status."""
        recommendations = []

        if status == "healthy":
            recommendations.append("Normal spending - continue monitoring")

        elif status == "warning":
            recommendations.append("Consider reducing model usage frequency")
            if remaining_budget > 5.0:
                recommendations.append("Prefer cost-effective models for remaining trades")

        elif status == "critical":
            recommendations.append("Implement strict cost controls")
            recommendations.append("Use cheapest available models only")
            recommendations.append("Consider caching for all decisions")

        elif status == "exhausted":
            recommendations.append("Switch to emergency mode with cached decisions")
            recommendations.append("Skip non-essential trades until budget reset")

        return recommendations

    async def _save_cost_model_data(self, model_key: str, model: DynamicCostModel) -> None:
        """Save cost model data to database."""
        try:
            # This would save to database - for now, just log
            self.logger.debug(f"Saving cost model data for {model_key}")

        except Exception as e:
            self.logger.error(f"Error saving cost model data: {e}")

    async def _save_budget_status(self, budget_status: BudgetStatus) -> None:
        """Save budget status to database."""
        try:
            # This would save to database - for now, just log
            self.logger.debug(
                f"Saving budget status: {budget_status.status}",
                percentage_used=budget_status.percentage_used
            )

        except Exception as e:
            self.logger.error(f"Error saving budget status: {e}")

    async def get_optimization_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cost optimization performance metrics."""
        try:
            # Get cache metrics
            cache_metrics = await self.get_cache_efficiency_metrics()

            # Get cost savings from models
            model_savings = await self._calculate_model_savings()

            # Get budget utilization
            budget_utilization = (self.daily_spend / self.config.daily_budget_limit) if self.config.daily_budget_limit > 0 else 0

            metrics = {
                "cache_efficiency": cache_metrics,
                "model_savings": model_savings,
                "budget_utilization": budget_utilization,
                "daily_spend": self.daily_spend,
                "models_tracked": len(self.cost_models),
                "cache_size": len(self.result_cache),
                "optimization_enabled": {
                    "dynamic_modeling": self.config.enable_dynamic_modeling,
                    "budget_controls": self.config.enable_budget_controls,
                    "intelligent_caching": self.config.enable_intelligent_caching,
                    "real_time_monitoring": self.config.enable_real_time_monitoring
                }
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error getting optimization performance metrics: {e}")
            return {"error": str(e)}

    async def _calculate_model_savings(self) -> Dict[str, Any]:
        """Calculate cost savings from optimization strategies."""
        try:
            # This would calculate actual savings from various optimization strategies
            # For now, return estimated savings

            estimated_savings = {
                "cache_reuse_savings": self.cache_hits * 0.02,  # $0.02 per cache hit
                "budget_control_savings": max(0, self.daily_spend - self.config.daily_budget_limit) if self.daily_spend > self.config.daily_budget_limit else 0,
                "model_selection_savings": len(self.cost_models) * 0.01  # $0.01 per optimized model
            }

            total_savings = sum(estimated_savings.values())

            return {
                "breakdown": estimated_savings,
                "total_savings": total_savings,
                "savings_percentage": (total_savings / (self.daily_spend + total_savings)) * 100 if (self.daily_spend + total_savings) > 0 else 0
            }

        except Exception as e:
            self.logger.error(f"Error calculating model savings: {e}")
            return {"error": str(e)}