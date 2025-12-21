"""
Enhanced AI client with integrated fallback and redundancy systems.

Integrates FallbackManager and ProviderManager with existing XAIClient and OpenAIClient
to provide seamless multi-provider redundancy, graceful degradation, and emergency modes.
"""

import asyncio
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from src.intelligence.fallback_manager import (
    FallbackManager,
    ProviderConfig,
    ProviderStatus,
    EmergencyDecision
)
from src.intelligence.provider_manager import (
    ProviderManager,
    AIRequest,
    AIResponse,
    ProviderType
)
from src.clients.xai_client import XAIClient, TradingDecision
from src.clients.openai_client import OpenAIClient
from src.config.settings import settings
from src.utils.logging_setup import TradingLoggerMixin


@dataclass
class EnhancedConfig:
    """Configuration for enhanced AI client."""
    enable_multi_provider: bool = True
    enable_fallback: bool = True
    enable_emergency_modes: bool = True
    health_check_interval: float = 60.0
    max_failover_attempts: int = 3
    emergency_mode_duration: int = 60  # minutes
    cache_decisions: bool = True
    cost_optimization: bool = True


class EnhancedAIClient(TradingLoggerMixin):
    """
    Enhanced AI client with comprehensive fallback and redundancy.

    Features:
    - Multi-provider redundancy with automatic failover
    - Graceful degradation during partial outages
    - Emergency trading modes for extended outages
    - Intelligent provider selection based on performance
    - Cost optimization and budget management
    - Comprehensive health monitoring
    """

    @property
    def ml_predictor(self):
        """Delegate to xAI client's ML predictor."""
        return self.xai_client.ml_predictor

    async def get_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        strategy: Optional[str] = None,
        query_type: Optional[str] = None,
        market_id: Optional[str] = None,
        log_query: bool = True
    ) -> Optional[str]:
        """
        Get completion with fallback support.
        """
        # 1. Try xAI (Primary)
        if not self.xai_client.is_api_exhausted:
            try:
                if log_query:
                    self.logger.info("Attempting completion with xAI")
                return await self.xai_client.get_completion(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    strategy=strategy or "unknown",
                    query_type=query_type or "completion",
                    market_id=market_id
                )
            except Exception as e:
                self.logger.warning(f"xAI completion failed: {e}, falling back to OpenAI")
        
        # 2. Fallback to OpenAI
        try:
            self.logger.info("Attempting completion with OpenAI (Fallback)")
            # OpenAI client expects messages, not raw prompt for chat models
            messages = [{"role": "user", "content": prompt}]
            response, _ = await self.openai_client._make_completion_request(
                messages, 
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        except Exception as e:
            self.logger.error(f"All AI providers failed for completion: {e}")
            return None

    def __init__(self, db_manager=None, kalshi_client=None, config: Optional[EnhancedConfig] = None):
        """
        Initialize EnhancedAIClient.

        Args:
            db_manager: Database manager for logging
            kalshi_client: Kalshi client for market data
            config: Enhanced client configuration
        """
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.config = config or EnhancedConfig()

        # Initialize existing clients for compatibility
        self.xai_client = XAIClient(db_manager, kalshi_client)
        self.openai_client = OpenAIClient()

        # Initialize provider configurations
        self.providers = self._create_provider_configs()

        # Initialize fallback and provider managers
        if self.config.enable_multi_provider:
            self.provider_manager = ProviderManager(self.providers)
            self.fallback_manager = FallbackManager(db_manager, self.providers)
        else:
            self.provider_manager = None
            self.fallback_manager = None

        # System state
        self.last_health_check = None
        self.emergency_decisions_cache = {}
        self.performance_metrics = {}

        self.logger.info(
            "EnhancedAIClient initialized",
            multi_provider=self.config.enable_multi_provider,
            fallback_enabled=self.config.enable_fallback,
            emergency_modes=self.config.enable_emergency_modes,
            total_providers=len(self.providers)
        )

        # Initialize providers if enabled
        if self.provider_manager:
            asyncio.create_task(self._initialize_providers())

    def _create_provider_configs(self) -> Dict[str, ProviderConfig]:
        """Create provider configurations from settings."""
        providers = {}

        # xAI provider
        if settings.api.xai_api_key:
            providers["xai"] = ProviderConfig(
                name="xai",
                endpoint="https://api.x.ai/v1",
                api_key=settings.api.xai_api_key,
                models=settings.trading.xai_models if hasattr(settings.trading, 'xai_models') else ["grok-4", "grok-3"],
                priority=1,
                timeout=settings.trading.ai_timeout,
                max_retries=settings.trading.ai_max_retries,
                cost_per_token=0.000015
            )

        # OpenAI provider
        if settings.api.openai_api_key:
            providers["openai"] = ProviderConfig(
                name="openai",
                endpoint=settings.api.openai_base_url or "https://api.openai.com/v1",
                api_key=settings.api.openai_api_key,
                models=settings.trading.openai_models if hasattr(settings.trading, 'openai_models') else ["gpt-4", "gpt-3.5-turbo"],
                priority=2,
                timeout=25.0,
                max_retries=3,
                cost_per_token=0.00001
            )

        # Anthropic provider
        if hasattr(settings.api, 'anthropic_api_key') and settings.api.anthropic_api_key:
            providers["anthropic"] = ProviderConfig(
                name="anthropic",
                endpoint="https://api.anthropic.com/v1",
                api_key=settings.api.anthropic_api_key,
                models=["claude-3-opus", "claude-3-sonnet"],
                priority=3,
                timeout=35.0,
                max_retries=2,
                cost_per_token=0.000025
            )

        # Local provider
        providers["local"] = ProviderConfig(
            name="local",
            endpoint="http://localhost:11434/v1",  # Ollama default
            api_key=os.getenv("OLLAMA_API_KEY", "local-key"),  # Local provider doesn't strictly need a key
            models=["llama-2", "mistral", "codellama"],
            priority=4,
            timeout=60.0,
            max_retries=1,
            cost_per_token=0.000001
        )

        return providers

    async def _initialize_providers(self) -> None:
        """Initialize all providers asynchronously."""
        if self.provider_manager:
            try:
                results = await self.provider_manager.initialize_all_providers()
                self.logger.info("Provider initialization results", results=results)

                # Start health monitoring
                if self.config.enable_fallback:
                    asyncio.create_task(self._start_health_monitoring())

            except Exception as e:
                self.logger.error(f"Failed to initialize providers: {e}")

    async def _start_health_monitoring(self) -> None:
        """Start continuous health monitoring."""
        while True:
            try:
                await self._check_system_health()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    async def _check_system_health(self) -> None:
        """Check health of all providers and update system state."""
        if not self.fallback_manager:
            return

        try:
            health_status = await self.fallback_manager.check_provider_health("xai") if "xai" in self.providers else None
            system_status = await self.fallback_manager.get_system_status()

            self.last_health_check = datetime.now()

            # Log system status changes
            if system_status.mode != "normal":
                self.logger.warning(
                    "System in degraded/emergency mode",
                    mode=system_status.mode,
                    healthy_providers=system_status.healthy_providers,
                    degradation_level=system_status.degradation_level
                )

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

    async def get_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str = "",
        ml_prediction=None
    ) -> Optional[TradingDecision]:
        """
        Get trading decision with full fallback and redundancy support.

        Args:
            market_data: Market information
            portfolio_data: Portfolio information
            news_summary: News summary
            ml_prediction: ML prediction (optional)

        Returns:
            Trading decision or None if all providers fail
        """
        try:
            # Check if multi-provider system is available and healthy
            if self._should_use_enhanced_system():
                return await self._get_enhanced_trading_decision(
                    market_data, portfolio_data, news_summary, ml_prediction
                )
            else:
                # Fall back to existing client logic
                return await self._get_legacy_trading_decision(
                    market_data, portfolio_data, news_summary, ml_prediction
                )

        except Exception as e:
            self.logger.error(f"Error getting trading decision: {e}")
            return None

    def _should_use_enhanced_system(self) -> bool:
        """Determine if enhanced multi-provider system should be used."""
        if not self.config.enable_multi_provider or not self.fallback_manager:
            return False

        # Check if we have healthy providers
        try:
            # Simple check - in production this would be async
            return len(self.providers) > 1
        except Exception:
            return False

    async def _get_enhanced_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str,
        ml_prediction
    ) -> Optional[TradingDecision]:
        """Get trading decision using enhanced multi-provider system."""
        try:
            # Prepare standardized request
            prompt = self._prepare_trading_prompt(market_data, portfolio_data, news_summary, ml_prediction)
            request = AIRequest(
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.trading.ai_temperature,
                max_tokens=settings.trading.ai_max_tokens
            )

            # Get decision from enhanced system
            decision_data = await self.fallback_manager.get_fallback_decision(
                market_data=market_data,
                portfolio_data=portfolio_data,
                timeout=30.0
            )

            if decision_data:
                return self._convert_to_trading_decision(decision_data, market_data)
            else:
                self.logger.warning("Enhanced system failed to provide decision")
                return None

        except Exception as e:
            self.logger.error(f"Enhanced trading decision failed: {e}")
            raise

    async def _get_legacy_trading_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str,
        ml_prediction
    ) -> Optional[TradingDecision]:
        """Get trading decision using legacy client logic."""
        try:
            # Try xAI client first
            if settings.api.xai_api_key:
                decision = await self.xai_client.get_trading_decision(
                    market_data, portfolio_data, news_summary, ml_prediction
                )
                if decision:
                    return decision

            # Fall back to OpenAI
            if settings.api.openai_api_key:
                openai_decision = await self.openai_client.get_trading_decision(
                    market_data, portfolio_data, news_summary
                )
                if openai_decision:
                    return TradingDecision(
                        action=openai_decision.action.upper(),
                        side=openai_decision.side.upper(),
                        confidence=openai_decision.confidence,
                        reasoning=f"[OPENAI FALLBACK] {openai_decision.reasoning}"
                    )

            return None

        except Exception as e:
            self.logger.error(f"Legacy trading decision failed: {e}")
            return None

    def _prepare_trading_prompt(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any],
        news_summary: str,
        ml_prediction
    ) -> str:
        """Prepare standardized trading prompt for all providers."""
        from src.utils.prompts import SIMPLIFIED_PROMPT_TPL

        # Extract key information
        title = market_data.get('title', 'Unknown Market')
        yes_price = market_data.get('yes_price', 50)
        no_price = market_data.get('no_price', 50)
        volume = market_data.get('volume', 0)

        # Calculate days to expiry
        days_to_expiry = market_data.get('days_to_expiry', 30)

        # Calculate max trade value
        max_trade_value = min(
            portfolio_data.get("balance", 0) * settings.trading.max_position_size_pct / 100,
            portfolio_data.get("balance", 0) * 0.05
        )

        prompt_params = {
            "title": title,
            "yes_price": yes_price,
            "no_price": no_price,
            "volume": volume,
            "days_to_expiry": days_to_expiry,
            "news_summary": news_summary[:1000],  # Limit news length
            "cash": portfolio_data.get("balance", 0),
            "max_trade_value": max_trade_value,
            "max_position_pct": settings.trading.max_position_size_pct,
            "ev_threshold": settings.trading.min_confidence_to_trade * 100
        }

        return SIMPLIFIED_PROMPT_TPL.format(**prompt_params)

    def _convert_to_trading_decision(
        self,
        decision_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Optional[TradingDecision]:
        """Convert standardized decision data to TradingDecision."""
        try:
            action = decision_data.get("action", "SKIP").upper()
            side = decision_data.get("side", "YES").upper()
            confidence = float(decision_data.get("confidence", 0.5))
            reasoning = decision_data.get("reasoning", "No reasoning provided")
            provider = decision_data.get("provider", "unknown")

            # Normalize action
            if action in ["BUY_YES", "BUY"]:
                action = "BUY"
            elif action in ["SKIP", "HOLD", "PASS"]:
                action = "SKIP"

            # Extract limit price if available
            limit_price = decision_data.get("limit_price")
            if limit_price:
                limit_price = int(limit_price)

            return TradingDecision(
                action=action,
                side=side,
                confidence=confidence,
                limit_price=limit_price,
                reasoning=f"[{provider.upper()}] {reasoning}"
            )

        except Exception as e:
            self.logger.error(f"Failed to convert decision data: {e}")
            return None

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self.fallback_manager:
            return {"mode": "legacy", "multi_provider": False}

        try:
            system_status = await self.fallback_manager.get_system_status()
            health_metrics = self.fallback_manager.get_health_metrics()

            status = {
                "mode": system_status.mode,
                "multi_provider": True,
                "healthy_providers": system_status.healthy_providers,
                "total_providers": system_status.total_providers,
                "degradation_level": system_status.degradation_level,
                "emergency_mode_active": system_status.emergency_mode_active,
                "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
                "health_metrics": health_metrics
            }

            if self.provider_manager:
                status["provider_stats"] = self.provider_manager.get_provider_stats()
                status["total_cost"] = self.provider_manager.get_total_cost()

            return status

        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {"mode": "error", "error": str(e)}

    async def enable_emergency_mode(
        self,
        reason: str,
        duration_minutes: int = 60
    ) -> None:
        """Enable emergency trading mode."""
        if not self.fallback_manager:
            self.logger.warning("Emergency mode not available - fallback manager disabled")
            return

        await self.fallback_manager.enable_emergency_mode(reason, duration_minutes)

    async def disable_emergency_mode(self) -> None:
        """Disable emergency trading mode."""
        if not self.fallback_manager:
            return

        await self.fallback_manager.disable_emergency_mode()

    async def get_emergency_decision(
        self,
        market_data: Dict[str, Any],
        portfolio_data: Dict[str, Any]
    ) -> Optional[TradingDecision]:
        """Get emergency trading decision."""
        if not self.fallback_manager:
            # Provide very conservative default decision
            return TradingDecision(
                action="SKIP",
                side="YES",
                confidence=0.3,
                reasoning="[EMERGENCY LEGACY] Conservative approach - no trading"
            )

        try:
            emergency_decision = await self.fallback_manager.get_emergency_decision(
                market_data, portfolio_data
            )

            if emergency_decision:
                return TradingDecision(
                    action=emergency_decision.action,
                    side=emergency_decision.side,
                    confidence=emergency_decision.confidence,
                    reasoning=emergency_decision.reasoning
                )

        except Exception as e:
            self.logger.error(f"Emergency decision failed: {e}")

        # Fallback conservative decision
        return TradingDecision(
            action="SKIP",
            side="YES",
            confidence=0.2,
            reasoning="[EMERGENCY FALLBACK] Ultra-conservative approach"
        )

    async def check_recovery(self) -> bool:
        """Check if system can recover from emergency/degraded mode."""
        if not self.fallback_manager:
            return True  # Legacy system always "recovers"

        return await self.fallback_manager.check_recovery()

    async def close(self) -> None:
        """Close all resources."""
        try:
            if self.provider_manager:
                await self.provider_manager.close()
            if self.fallback_manager:
                await self.fallback_manager.close()
            if self.xai_client:
                await self.xai_client.close()
            if self.openai_client:
                await self.openai_client.close()

            self.logger.info("EnhancedAIClient closed successfully")

        except Exception as e:
            self.logger.error(f"Error closing EnhancedAIClient: {e}")