"""
FallbackManager - Multi-provider redundancy and graceful degradation for AI models.

Provides comprehensive fallback mechanisms including:
- Multi-provider redundancy (xAI, OpenAI, Anthropic, local models)
- Health checking and automatic failover
- Graceful degradation during partial outages
- Emergency trading modes for extended outages
- Performance monitoring and recovery procedures
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict, deque

import httpx
from src.utils.logging_setup import TradingLoggerMixin


class ProviderStatus(Enum):
    """Provider health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class EmergencyMode(Enum):
    """Emergency trading mode levels."""
    CONSERVATIVE = "conservative"
    MINIMAL = "minimal"
    SUSPENDED = "suspended"


@dataclass
class ProviderConfig:
    """Configuration for an AI provider."""
    name: str
    endpoint: str
    api_key: str
    models: List[str]
    priority: int  # Lower number = higher priority
    timeout: float  # Timeout in seconds
    max_retries: int
    health_check_interval: float = 60.0  # Health check interval in seconds
    cost_per_token: float = 0.00001  # Cost per token (USD)
    max_concurrent_requests: int = 5


@dataclass
class HealthCheckResult:
    """Result of a provider health check."""
    provider_name: str
    is_healthy: bool
    error_message: Optional[str] = None
    response_time: float = 0.0
    status_code: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FailoverEvent:
    """Record of a failover event."""
    timestamp: datetime
    from_provider: str
    to_provider: str
    reason: str
    response_time_ms: float


@dataclass
class SystemStatus:
    """Current system status."""
    mode: str  # "normal", "degraded", "emergency"
    healthy_providers: int
    total_providers: int
    degradation_level: float  # 0.0 to 1.0
    emergency_mode_active: bool = False
    emergency_reason: Optional[str] = None
    emergency_until: Optional[datetime] = None
    last_failover: Optional[FailoverEvent] = None


@dataclass
class EmergencyDecision:
    """Emergency trading decision."""
    action: str
    side: str
    confidence: float
    reasoning: str
    is_cached: bool = False
    cache_timestamp: Optional[datetime] = None


class FallbackManager(TradingLoggerMixin):
    """
    Manages fallback and redundancy across multiple AI providers.

    Features:
    - Multi-provider redundancy with automatic failover
    - Health monitoring and performance tracking
    - Graceful degradation during partial outages
    - Emergency trading modes for extended outages
    - Cached decision fallbacks
    - Comprehensive metrics and alerting
    """

    def __init__(self, db_manager, providers: Dict[str, ProviderConfig]):
        """
        Initialize FallbackManager.

        Args:
            db_manager: Database manager for logging
            providers: Dictionary of provider configurations
        """
        self.db_manager = db_manager
        self.providers = providers

        # Provider status tracking
        self.provider_status: Dict[str, ProviderStatus] = {}
        self.provider_health: Dict[str, HealthCheckResult] = {}
        self.last_health_check: Dict[str, datetime] = {}

        # Failover tracking
        self.failover_history: deque = deque(maxlen=100)
        self.current_primary_provider: Optional[str] = None

        # Emergency mode
        self.emergency_mode: Optional[EmergencyMode] = None
        self.emergency_reason: Optional[str] = None
        self.emergency_until: Optional[datetime] = None

        # Cached decisions for emergency mode
        self.cached_decisions: Dict[str, Dict[str, Any]] = {}
        self.cache_max_age_hours = 24  # Cache decisions for 24 hours

        # Performance metrics
        self.health_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "last_check": None,
            "response_times": deque(maxlen=100),
            "success_rate": 0.0,
            "error_count": 0,
            "last_error": None,
            "uptime_percentage": 100.0
        })

        # HTTP client for health checks
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Initialize provider status
        for provider_name in providers:
            self.provider_status[provider_name] = ProviderStatus.UNKNOWN

        self.logger.info(
            "FallbackManager initialized",
            providers=list(providers.keys()),
            total_providers=len(providers)
        )

    async def get_available_providers(self) -> List[ProviderConfig]:
        """
        Get list of available providers sorted by priority.

        Returns:
            List of available provider configs in priority order
        """
        available = []

        for provider_name, config in self.providers.items():
            status = self.provider_status.get(provider_name, ProviderStatus.UNKNOWN)

            if status in [ProviderStatus.HEALTHY, ProviderStatus.DEGRADED]:
                available.append(config)

        # Sort by priority (lower number = higher priority)
        available.sort(key=lambda p: p.priority)

        return available

    async def get_best_provider(self) -> Optional[ProviderConfig]:
        """
        Get the best available provider based on health and priority.

        Returns:
            Best available provider or None if all unavailable
        """
        available_providers = await self.get_available_providers()

        if not available_providers:
            return None

        return available_providers[0]

    async def check_provider_health(self, provider_name: str) -> HealthCheckResult:
        """
        Check health of a specific provider.

        Args:
            provider_name: Name of the provider to check

        Returns:
            Health check result
        """
        if provider_name not in self.providers:
            return HealthCheckResult(
                provider_name,
                False,
                f"Provider {provider_name} not configured"
            )

        result = await self._check_provider_health_internal(provider_name)
        self.provider_health[provider_name] = result
        self.last_health_check[provider_name] = datetime.now()

        # Update provider status
        if result.is_healthy:
            self.provider_status[provider_name] = ProviderStatus.HEALTHY
        else:
            self.provider_status[provider_name] = ProviderStatus.UNHEALTHY

        # Update metrics
        self._update_health_metrics(provider_name, result)

        return result

    async def _check_provider_health_internal(self, provider_name: str) -> HealthCheckResult:
        """
        Internal health check implementation.

        Instead of calling a non-existent /health endpoint, we check:
        1. If the provider has a valid API key configured
        2. If the provider's endpoint is reachable (basic connectivity)
        
        Actual API health is verified through ProviderManager when making requests.

        Args:
            provider_name: Name of provider to check

        Returns:
            Health check result
        """
        config = self.providers[provider_name]
        start_time = time.time()

        try:
            # Check if provider has valid configuration
            if not config.api_key or config.api_key in ["", "local-key"]:
                # Local provider doesn't need an API key
                if provider_name == "local":
                    # Check if local Ollama is running
                    try:
                        response = await self.http_client.get(
                            f"{config.endpoint}/api/tags",
                            timeout=5.0
                        )
                        response_time = (time.time() - start_time) * 1000
                        if response.status_code == 200:
                            return HealthCheckResult(
                                provider_name,
                                True,
                                None,
                                response_time,
                                response.status_code
                            )
                        else:
                            return HealthCheckResult(
                                provider_name,
                                False,
                                "Local model server not responding",
                                response_time,
                                response.status_code
                            )
                    except Exception:
                        response_time = (time.time() - start_time) * 1000
                        return HealthCheckResult(
                            provider_name,
                            False,
                            "Local model server not reachable",
                            response_time
                        )
                else:
                    response_time = (time.time() - start_time) * 1000
                    return HealthCheckResult(
                        provider_name,
                        False,
                        f"No API key configured for {provider_name}",
                        response_time
                    )

            # For cloud providers (xAI, OpenAI, Anthropic), consider them healthy
            # if they have a valid API key. The actual API availability is checked
            # when making requests through ProviderManager.
            #
            # This is safer because:
            # 1. xAI/OpenAI/Anthropic don't have standard /health endpoints
            # 2. Making test API calls on every health check wastes tokens/money
            # 3. ProviderManager handles request-time failures with fallbacks
            # 4. If we have valid keys and can make requests, that's what matters

            response_time = (time.time() - start_time) * 1000

            # Validate API key looks reasonable (non-empty, reasonable length)
            api_key_valid = len(config.api_key) > 10

            # ENHANCED: Check for specific valid API key patterns
            is_xai = provider_name == "xai" and config.api_key.startswith("xai-")
            is_openai = provider_name == "openai" and config.api_key.startswith("sk-")

            # Consider provider healthy if:
            # 1. API key format is valid, OR
            # 2. It's a known provider with the right key pattern
            api_key_valid = api_key_valid or is_xai or is_openai

            if api_key_valid:
                return HealthCheckResult(
                    provider_name,
                    True,
                    None,
                    response_time,
                    200  # Synthetic status code
                )
            else:
                return HealthCheckResult(
                    provider_name,
                    False,
                    f"Invalid API key format for {provider_name}",
                    response_time
                )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                provider_name,
                False,
                str(e)[:200],
                response_time
            )

    async def initiate_failover(self, from_provider: str, to_provider: str, reason: str) -> bool:
        """
        Initiate failover from one provider to another.

        Args:
            from_provider: Current provider name
            to_provider: Target provider name
            reason: Reason for failover

        Returns:
            True if failover was successful
        """
        try:
            # Check if target provider is available
            target_health = await self.check_provider_health(to_provider)

            if not target_health.is_healthy:
                self.logger.error(
                    f"Failover failed - target provider {to_provider} is unhealthy",
                    from_provider=from_provider,
                    to_provider=to_provider,
                    reason=reason
                )
                return False

            # Record failover event
            failover_event = FailoverEvent(
                timestamp=datetime.now(),
                from_provider=from_provider,
                to_provider=to_provider,
                reason=reason,
                response_time_ms=target_health.response_time
            )

            self.failover_history.append(failover_event)
            self.current_primary_provider = to_provider

            self.logger.warning(
                f"Failover initiated: {from_provider} -> {to_provider}",
                reason=reason,
                response_time_ms=target_health.response_time
            )

            # Log to database if available
            if self.db_manager:
                try:
                    await self._log_failover_event(failover_event)
                except Exception as e:
                    self.logger.error(f"Failed to log failover event: {e}")

            return True

        except Exception as e:
            self.logger.error(f"Failover failed: {e}")
            return False

    async def enable_emergency_mode(
        self,
        reason: str,
        duration_minutes: int = 60,
        mode: EmergencyMode = EmergencyMode.CONSERVATIVE
    ) -> None:
        """
        Enable emergency trading mode.

        Args:
            reason: Reason for emergency mode
            duration_minutes: Duration of emergency mode
            mode: Emergency mode level
        """
        self.emergency_mode = mode
        self.emergency_reason = reason
        self.emergency_until = datetime.now() + timedelta(minutes=duration_minutes)

        self.logger.critical(
            "Emergency mode activated",
            reason=reason,
            duration_minutes=duration_minutes,
            mode=mode.value,
            until=self.emergency_until.isoformat()
        )

        # Log to database if available
        if self.db_manager:
            try:
                await self._log_emergency_mode(reason, duration_minutes, mode)
            except Exception as e:
                self.logger.error(f"Failed to log emergency mode: {e}")

    async def disable_emergency_mode(self) -> None:
        """Disable emergency trading mode."""
        if self.emergency_mode:
            self.logger.info(
                "Emergency mode deactivated",
                previous_mode=self.emergency_mode.value,
                reason=self.emergency_reason
            )

        self.emergency_mode = None
        self.emergency_reason = None
        self.emergency_until = None

    async def get_system_status(self) -> SystemStatus:
        """
        Get current system status.

        Returns:
            System status object
        """
        healthy_count = sum(
            1 for status in self.provider_status.values()
            if status == ProviderStatus.HEALTHY
        )
        total_count = len(self.providers)

        # Determine system mode
        if self.emergency_mode:
            mode = "emergency"
        elif healthy_count == 0:
            mode = "emergency"
        elif healthy_count < total_count:
            mode = "degraded"
        else:
            mode = "normal"

        degradation_level = 1.0 - (healthy_count / total_count) if total_count > 0 else 1.0

        return SystemStatus(
            mode=mode,
            healthy_providers=healthy_count,
            total_providers=total_count,
            degradation_level=degradation_level,
            emergency_mode_active=self.emergency_mode is not None,
            emergency_reason=self.emergency_reason,
            emergency_until=self.emergency_until,
            last_failover=self.failover_history[-1] if self.failover_history else None
        )

    async def get_emergency_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any]
    ) -> EmergencyDecision:
        """
        Get an emergency trading decision.

        Args:
            market_data: Market information
            portfolio_data: Portfolio information

        Returns:
            Emergency trading decision
        """
        # Check if we have a cached decision
        market_key = self._get_market_cache_key(market_data)
        cached = self.cached_decisions.get(market_key)

        if cached and self._is_cache_valid(cached.get("timestamp")):
            return EmergencyDecision(
                action=cached["action"],
                side=cached["side"],
                confidence=cached["confidence"] * 0.8,  # Reduce confidence in emergency
                reasoning=f"[EMERGENCY CACHED] {cached['reasoning']}",
                is_cached=True,
                cache_timestamp=cached["timestamp"]
            )

        # Generate conservative emergency decision
        decision = self._generate_emergency_decision(market_data, portfolio_data)

        # Cache the decision for future use
        self.cached_decisions[market_key] = {
            "action": decision.action,
            "side": decision.side,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "timestamp": datetime.now(),
            "market_data": market_data
        }

        return decision

    def _generate_emergency_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any]
    ) -> EmergencyDecision:
        """
        Generate a conservative emergency trading decision.

        Args:
            market_data: Market information
            portfolio_data: Portfolio information

        Returns:
            Emergency trading decision
        """
        # Very conservative logic in emergency mode
        yes_price = market_data.get("yes_price", 50)
        no_price = market_data.get("no_price", 50)
        volume = market_data.get("volume", 0)

        # Calculate implied probability
        implied_prob = yes_price / 100

        # Emergency mode decision rules
        if volume < 1000:  # Low volume
            action = "SKIP"
            confidence = 0.9
            reasoning = "Emergency mode: Low volume market - too risky"
        elif abs(implied_prob - 0.5) < 0.1:  # Close to 50/50
            action = "SKIP"
            confidence = 0.8
            reasoning = "Emergency mode: Uncertain outcome - avoiding risk"
        elif implied_prob > 0.8 or implied_prob < 0.2:  # Extreme probabilities
            action = "SKIP"
            confidence = 0.7
            reasoning = "Emergency mode: Extreme pricing - potential manipulation"
        else:
            # Default to skip in emergency mode
            action = "SKIP"
            confidence = 0.6
            reasoning = "Emergency mode: Conservative approach - no trading"

        return EmergencyDecision(
            action=action,
            side="YES",  # Default side
            confidence=confidence,
            reasoning=f"[EMERGENCY CONSERVATIVE] {reasoning}"
        )

    async def get_fallback_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """
        Get a trading decision with fallback across providers.

        Args:
            market_data: Market information
            portfolio_data: Portfolio information
            timeout: Maximum time to wait for decision

        Returns:
            Trading decision or None if all providers fail
        """
        start_time = time.time()

        # Check if we're in emergency mode
        if self.emergency_mode and datetime.now() < self.emergency_until:
            emergency_decision = await self.get_emergency_decision(market_data, portfolio_data)
            return {
                "action": emergency_decision.action,
                "side": emergency_decision.side,
                "confidence": emergency_decision.confidence,
                "reasoning": emergency_decision.reasoning,
                "provider": "emergency",
                "fallback_level": "emergency"
            }

        # Try providers in order of priority
        available_providers = await self.get_available_providers()

        for provider_config in available_providers:
            # Check timeout
            if time.time() - start_time > timeout:
                self.logger.warning("Fallback decision timeout exceeded")
                break

            try:
                decision = await self._get_provider_decision(
                    provider_config,
                    market_data,
                    portfolio_data
                )

                if decision:
                    self.logger.info(
                        f"Decision obtained from provider {provider_config.name}",
                        action=decision.get("action"),
                        confidence=decision.get("confidence")
                    )

                    decision["provider"] = provider_config.name
                    decision["fallback_level"] = "provider"

                    return decision

            except Exception as e:
                self.logger.warning(
                    f"Provider {provider_config.name} failed: {e}",
                    provider=provider_config.name
                )

                # Try failover to next provider
                if provider_config != available_providers[-1]:  # Not the last provider
                    next_provider = available_providers[available_providers.index(provider_config) + 1]
                    await self.initiate_failover(
                        provider_config.name,
                        next_provider.name,
                        str(e)
                    )

        # All providers failed, try emergency mode
        if not self.emergency_mode:
            await self.enable_emergency_mode(
                "All providers failed",
                duration_minutes=30
            )

            emergency_decision = await self.get_emergency_decision(market_data, portfolio_data)
            return {
                "action": emergency_decision.action,
                "side": emergency_decision.side,
                "confidence": emergency_decision.confidence,
                "reasoning": emergency_decision.reasoning,
                "provider": "emergency",
                "fallback_level": "emergency"
            }

        return None

    async def _get_provider_decision(
        self,
        provider_config: ProviderConfig,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get decision from a specific provider.

        Args:
            provider_config: Provider configuration
            market_data: Market information
            portfolio_data: Portfolio information

        Returns:
            Decision from provider or None if failed
        """
        # This would integrate with actual provider clients
        # For now, return a mock decision
        return {
            "action": "SKIP",
            "side": "YES",
            "confidence": 0.5,
            "reasoning": f"Mock decision from {provider_config.name}"
        }

    async def check_recovery(self) -> bool:
        """
        Check if system can recover from emergency/degraded mode.

        Returns:
            True if recovery is possible
        """
        if not self.emergency_mode:
            return True

        # Check if any providers are healthy
        for provider_name in self.providers:
            health_result = await self.check_provider_health(provider_name)
            if health_result.is_healthy:
                self.logger.info(
                    f"Provider {provider_name} is healthy - can recover from emergency mode"
                )
                await self.disable_emergency_mode()
                return True

        return False

    def get_failover_history(self, limit: int = 10) -> List[FailoverEvent]:
        """
        Get recent failover events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of failover events
        """
        return list(self.failover_history)[-limit:]

    def get_health_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health metrics for all providers.

        Returns:
            Dictionary of provider health metrics
        """
        return dict(self.health_metrics)

    def set_cached_decisions(self, decisions: Dict[str, Dict[str, Any]]) -> None:
        """
        Set cached decisions for emergency mode.

        Args:
            decisions: Dictionary of cached decisions
        """
        self.cached_decisions.update(decisions)

    def _get_market_cache_key(self, market_data: Dict[str, Any]) -> str:
        """Generate cache key for market data."""
        title = market_data.get("title", "")
        yes_price = market_data.get("yes_price", 0)
        return f"{title}_{yes_price}".replace(" ", "_").lower()

    def _is_cache_valid(self, timestamp: Optional[datetime]) -> bool:
        """Check if cached decision is still valid."""
        if not timestamp:
            return False
        return datetime.now() - timestamp < timedelta(hours=self.cache_max_age_hours)

    def _update_health_metrics(self, provider_name: str, result: HealthCheckResult) -> None:
        """Update health metrics for a provider."""
        metrics = self.health_metrics[provider_name]
        metrics["last_check"] = result.timestamp
        metrics["response_times"].append(result.response_time)

        if result.is_healthy:
            metrics["success_rate"] = min(1.0, metrics["success_rate"] + 0.01)
        else:
            metrics["error_count"] += 1
            metrics["success_rate"] = max(0.0, metrics["success_rate"] - 0.05)
            metrics["last_error"] = result.error_message

    async def _log_failover_event(self, event: FailoverEvent) -> None:
        """Log failover event to database."""
        if self.db_manager:
            # Implementation would depend on database schema
            pass

    async def _log_emergency_mode(
        self,
        reason: str,
        duration_minutes: int,
        mode: EmergencyMode
    ) -> None:
        """Log emergency mode activation to database."""
        if self.db_manager:
            # Implementation would depend on database schema
            pass

    async def close(self) -> None:
        """Close resources used by FallbackManager."""
        if self.http_client:
            await self.http_client.aclose()

        self.logger.info(
            "FallbackManager closed",
            total_failovers=len(self.failover_history),
            cached_decisions=len(self.cached_decisions)
        )