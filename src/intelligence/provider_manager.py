"""
ProviderManager - Standardized interface for multiple AI providers.

Provides abstraction layer for:
- xAI (Grok models)
- OpenAI (GPT models)
- Anthropic (Claude models)
- Local models (Ollama/Llama/etc.)
- Unified API across all providers
- Request routing and load balancing
- Cost tracking and optimization
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import httpx
from openai import AsyncOpenAI

from src.intelligence.fallback_manager import ProviderConfig
from src.utils.logging_setup import TradingLoggerMixin


class ProviderType(Enum):
    """Supported provider types."""
    XAI = "xai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class AIRequest:
    """Standardized AI request across all providers."""
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = None


@dataclass
class AIResponse:
    """Standardized AI response from all providers."""
    content: str
    model_used: str
    provider: str
    tokens_used: int
    cost_usd: float
    response_time_ms: float
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = None


class BaseProvider(ABC):
    """Abstract base class for AI providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{config.name}")

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the provider."""
        pass

    @abstractmethod
    async def make_request(self, request: AIRequest) -> AIResponse:
        """Make a request to the provider."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models for this provider."""
        pass

    @abstractmethod
    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for given tokens and model."""
        pass


class XAIProvider(BaseProvider):
    """xAI provider implementation."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = None

    def _convert_message_to_sdk(self, message: Dict[str, str]) -> Any:
        """
        Convert a dictionary message to xAI SDK Message object.
        
        The xAI SDK's chat.append() method only accepts chat_pb2.Message objects
        (created via helper functions like user(), system(), assistant()) or
        Response objects from previous interactions.
        """
        from xai_sdk.chat import user as xai_user, system as xai_system, assistant as xai_assistant
        
        role = message.get("role", "user").lower()
        content = message.get("content", "")
        
        if role == "user":
            return xai_user(content)
        elif role == "system":
            return xai_system(content)
        elif role == "assistant":
            return xai_assistant(content)
        else:
            # Default to user message for unknown roles
            return xai_user(content)

    async def initialize(self) -> bool:
        """Initialize xAI client."""
        try:
            from xai_sdk import AsyncClient
            self.client = AsyncClient(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            self.logger.info("xAI client initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize xAI client: {e}")
            return False

    async def make_request(self, request: AIRequest) -> AIResponse:
        """Make request to xAI provider."""
        if not self.client:
            await self.initialize()

        start_time = time.time()
        model_to_use = request.model or self.config.models[0]

        try:
            # Create chat
            chat = self.client.chat.create(
                model=model_to_use,
                temperature=request.temperature,
                max_tokens=request.max_tokens or 4000
            )

            # Add messages - CONVERT dict messages to SDK format
            for message in request.messages:
                sdk_message = self._convert_message_to_sdk(message)
                chat.append(sdk_message)

            # Sample response
            response = await chat.sample()
            response_time_ms = (time.time() - start_time) * 1000

            # Calculate cost
            tokens_used = getattr(response.usage, 'total_tokens', len(response.content) // 4)
            cost = self.calculate_cost(tokens_used, model_to_use)

            return AIResponse(
                content=response.content,
                model_used=model_to_use,
                provider=ProviderType.XAI.value,
                tokens_used=tokens_used,
                cost_usd=cost,
                response_time_ms=response_time_ms,
                finish_reason=getattr(response, 'finish_reason', None),
                metadata={
                    "reasoning_tokens": getattr(response.usage, 'reasoning_tokens', 0) if hasattr(response, 'usage') else 0
                }
            )

        except Exception as e:
            self.logger.error(f"xAI request failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check xAI provider health."""
        try:
            if not self.client:
                await self.initialize()

            # Simple test request
            chat = self.client.chat.create(
                model=self.config.models[0],
                temperature=0.1,
                max_tokens=10
            )
            from xai_sdk.chat import user as xai_user
            chat.append(xai_user("test"))
            response = await chat.sample()

            return bool(response and response.content)
        except Exception as e:
            self.logger.error(f"xAI health check failed: {e}")
            return False

    async def list_models(self) -> List[str]:
        """List available xAI models."""
        return self.config.models.copy()

    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for xAI models."""
        # xAI pricing (approximate)
        if model == "grok-4":
            return tokens * 0.00002  # $0.02 per 1K tokens
        elif model == "grok-3":
            return tokens * 0.000015  # $0.015 per 1K tokens
        else:
            return tokens * 0.00001  # Default pricing


class OpenAIProvider(BaseProvider):
    """OpenAI provider implementation."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = None

    async def initialize(self) -> bool:
        """Initialize OpenAI client."""
        try:
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.endpoint,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries
            )
            self.logger.info("OpenAI client initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            return False

    async def make_request(self, request: AIRequest) -> AIResponse:
        """Make request to OpenAI provider."""
        if not self.client:
            await self.initialize()

        start_time = time.time()
        model_to_use = request.model or self.config.models[0]

        try:
            kwargs = {
                "model": model_to_use,
                "messages": request.messages,
                "temperature": request.temperature,
            }

            if request.max_tokens:
                kwargs["max_tokens"] = request.max_tokens

            if request.response_format:
                kwargs["response_format"] = request.response_format

            if request.stream:
                kwargs["stream"] = True

            response = await self.client.chat.completions.create(**kwargs)
            response_time_ms = (time.time() - start_time) * 1000

            content = ""
            if request.stream:
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        content += chunk.choices[0].delta.content
            else:
                content = response.choices[0].message.content

            # Calculate cost
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens
            cost = self.calculate_cost(total_tokens, model_to_use)

            return AIResponse(
                content=content,
                model_used=model_to_use,
                provider=ProviderType.OPENAI.value,
                tokens_used=total_tokens,
                cost_usd=cost,
                response_time_ms=response_time_ms,
                finish_reason=response.choices[0].finish_reason if response.choices else None,
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            )

        except Exception as e:
            self.logger.error(f"OpenAI request failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check OpenAI provider health."""
        try:
            if not self.client:
                await self.initialize()

            response = await self.client.chat.completions.create(
                model=self.config.models[0],
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                temperature=0
            )

            return bool(response and response.choices)
        except Exception as e:
            self.logger.error(f"OpenAI health check failed: {e}")
            return False

    async def list_models(self) -> List[str]:
        """List available OpenAI models."""
        return self.config.models.copy()

    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for OpenAI models."""
        # OpenAI pricing (approximate)
        if "gpt-4" in model:
            return tokens * 0.00003  # $0.03 per 1K tokens
        elif "gpt-3.5" in model:
            return tokens * 0.000002  # $0.002 per 1K tokens
        else:
            return tokens * 0.00001  # Default pricing


class AnthropicProvider(BaseProvider):
    """Anthropic provider implementation."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = None

    async def initialize(self) -> bool:
        """Initialize Anthropic client."""
        try:
            import anthropic
            self.client = anthropic.AsyncAnthropic(
                api_key=self.config.api_key,
                timeout=self.config.timeout
            )
            self.logger.info("Anthropic client initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {e}")
            return False

    async def make_request(self, request: AIRequest) -> AIResponse:
        """Make request to Anthropic provider."""
        if not self.client:
            await self.initialize()

        start_time = time.time()
        model_to_use = request.model or self.config.models[0]

        try:
            # Convert messages to Anthropic format
            messages = []
            system_message = None

            for msg in request.messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    messages.append(msg)

            kwargs = {
                "model": model_to_use,
                "messages": messages,
                "temperature": request.temperature,
            }

            if system_message:
                kwargs["system"] = system_message

            if request.max_tokens:
                kwargs["max_tokens"] = request.max_tokens
            else:
                kwargs["max_tokens"] = 4000

            response = await self.client.messages.create(**kwargs)
            response_time_ms = (time.time() - start_time) * 1000

            # Calculate cost
            input_tokens = response.usage.input_tokens if response.usage else 0
            output_tokens = response.usage.output_tokens if response.usage else 0
            total_tokens = input_tokens + output_tokens
            cost = self.calculate_cost(total_tokens, model_to_use)

            return AIResponse(
                content=response.content[0].text,
                model_used=model_to_use,
                provider=ProviderType.ANTHROPIC.value,
                tokens_used=total_tokens,
                cost_usd=cost,
                response_time_ms=response_time_ms,
                finish_reason=response.stop_reason,
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
            )

        except Exception as e:
            self.logger.error(f"Anthropic request failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check Anthropic provider health."""
        try:
            if not self.client:
                await self.initialize()

            response = await self.client.messages.create(
                model=self.config.models[0],
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                temperature=0
            )

            return bool(response and response.content)
        except Exception as e:
            self.logger.error(f"Anthropic health check failed: {e}")
            return False

    async def list_models(self) -> List[str]:
        """List available Anthropic models."""
        return self.config.models.copy()

    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for Anthropic models."""
        # Anthropic pricing (approximate)
        if "claude-3-opus" in model:
            return tokens * 0.000075  # $0.075 per 1K tokens
        elif "claude-3-sonnet" in model:
            return tokens * 0.000015  # $0.015 per 1K tokens
        elif "claude-3-haiku" in model:
            return tokens * 0.00000125  # $0.00125 per 1K tokens
        else:
            return tokens * 0.00001  # Default pricing


class LocalProvider(BaseProvider):
    """Local model provider implementation (Ollama, etc.)."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.client = None

    async def initialize(self) -> bool:
        """Initialize local provider client."""
        try:
            self.client = httpx.AsyncClient(
                base_url=self.config.endpoint,
                timeout=self.config.timeout
            )
            self.logger.info("Local provider client initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize local provider: {e}")
            return False

    async def make_request(self, request: AIRequest) -> AIResponse:
        """Make request to local provider."""
        if not self.client:
            await self.initialize()

        start_time = time.time()
        model_to_use = request.model or self.config.models[0]

        try:
            # Ollama API format
            payload = {
                "model": model_to_use,
                "messages": request.messages,
                "temperature": request.temperature,
            }

            if request.max_tokens:
                payload["num_predict"] = request.max_tokens
            else:
                payload["num_predict"] = 2048

            response = await self.client.post("/api/chat", json=payload)
            response.raise_for_status()

            response_data = response.json()
            response_time_ms = (time.time() - start_time) * 1000

            # Calculate cost (local models are essentially free, just energy cost)
            tokens_used = response_data.get("eval_count", 0) + response_data.get("prompt_eval_count", 0)
            cost = self.calculate_cost(tokens_used, model_to_use)

            return AIResponse(
                content=response_data["message"]["content"],
                model_used=model_to_use,
                provider=ProviderType.LOCAL.value,
                tokens_used=tokens_used,
                cost_usd=cost,
                response_time_ms=response_time_ms,
                finish_reason=response_data.get("done_reason"),
                metadata={
                    "eval_count": response_data.get("eval_count", 0),
                    "prompt_eval_count": response_data.get("prompt_eval_count", 0)
                }
            )

        except Exception as e:
            self.logger.error(f"Local provider request failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check local provider health."""
        try:
            if not self.client:
                await self.initialize()

            response = await self.client.get("/api/tags")
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Local provider health check failed: {e}")
            return False

    async def list_models(self) -> List[str]:
        """List available local models."""
        try:
            if not self.client:
                await self.initialize()

            response = await self.client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return models
        except Exception as e:
            self.logger.error(f"Failed to list local models: {e}")
            return self.config.models.copy()

    def calculate_cost(self, tokens: int, model: str) -> float:
        """Calculate cost for local models (essentially free)."""
        # Local models have minimal cost (electricity)
        return tokens * 0.000001  # $0.001 per 1M tokens (negligible)


class ProviderManager(TradingLoggerMixin):
    """
    Manages multiple AI providers with unified interface.

    Features:
    - Provider abstraction and standardization
    - Request routing and load balancing
    - Cost optimization and budget management
    - Performance monitoring and metrics
    - Automatic failover and recovery
    """

    def __init__(self, providers: Dict[str, ProviderConfig]):
        """
        Initialize ProviderManager.

        Args:
            providers: Dictionary of provider configurations
        """
        self.providers = {}
        self.provider_clients = {}
        self.request_stats = {}
        self.cost_tracker = {}

        # Initialize provider clients
        for name, config in providers.items():
            self.providers[name] = config
            self.provider_clients[name] = self._create_provider(config)
            self.request_stats[name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_response_time": 0.0
            }

        self.logger.info(
            "ProviderManager initialized",
            providers=list(providers.keys()),
            total_providers=len(providers)
        )

    def _create_provider(self, config: ProviderConfig) -> BaseProvider:
        """Create provider instance based on configuration."""
        if config.name.lower() == "xai":
            return XAIProvider(config)
        elif config.name.lower() == "openai":
            return OpenAIProvider(config)
        elif config.name.lower() == "anthropic":
            return AnthropicProvider(config)
        elif config.name.lower() == "local":
            return LocalProvider(config)
        else:
            raise ValueError(f"Unknown provider type: {config.name}")

    async def initialize_all_providers(self) -> Dict[str, bool]:
        """
        Initialize all providers.

        Returns:
            Dictionary of provider initialization results
        """
        results = {}

        for name, client in self.provider_clients.items():
            try:
                success = await client.initialize()
                results[name] = success
                self.logger.info(f"Provider {name} initialized: {success}")
            except Exception as e:
                results[name] = False
                self.logger.error(f"Failed to initialize provider {name}: {e}")

        return results

    async def make_request(
        self,
        request: AIRequest,
        preferred_provider: Optional[str] = None,
        fallback_providers: Optional[List[str]] = None
    ) -> AIResponse:
        """
        Make a request to the best available provider.

        Args:
            request: AI request to make
            preferred_provider: Preferred provider to use first
            fallback_providers: List of providers to try as fallbacks

        Returns:
            AI response from successful provider
        """
        start_time = time.time()
        provider_order = []

        # Determine provider order
        if preferred_provider and preferred_provider in self.provider_clients:
            provider_order.append(preferred_provider)

        if fallback_providers:
            provider_order.extend([p for p in fallback_providers if p in self.provider_clients])
        else:
            # Add all other providers by priority
            other_providers = sorted(
                [name for name in self.provider_clients.keys() if name not in provider_order],
                key=lambda p: self.providers[p].priority
            )
            provider_order.extend(other_providers)

        last_error = None

        for provider_name in provider_order:
            try:
                provider_client = self.provider_clients[provider_name]
                self.logger.debug(f"Trying provider: {provider_name}")

                # Make request
                response = await provider_client.make_request(request)

                # Update stats
                self._update_request_stats(provider_name, response, True, time.time() - start_time)

                self.logger.info(
                    f"Request successful with provider {provider_name}",
                    tokens_used=response.tokens_used,
                    cost_usd=response.cost_usd,
                    response_time_ms=response.response_time_ms
                )

                return response

            except Exception as e:
                last_error = e
                self._update_request_stats(provider_name, None, False, time.time() - start_time)
                self.logger.warning(
                    f"Provider {provider_name} failed: {e}",
                    provider=provider_name
                )
                continue

        # All providers failed
        self.logger.error(
            f"All providers failed for request",
            providers_tried=provider_order,
            last_error=str(last_error) if last_error else None
        )
        raise Exception(f"All providers failed. Last error: {last_error}")

    async def health_check_all(self) -> Dict[str, bool]:
        """
        Check health of all providers.

        Returns:
            Dictionary of provider health status
        """
        health_status = {}

        for name, client in self.provider_clients.items():
            try:
                health = await client.health_check()
                health_status[name] = health
                self.logger.debug(f"Provider {name} health: {health}")
            except Exception as e:
                health_status[name] = False
                self.logger.error(f"Health check failed for provider {name}: {e}")

        return health_status

    async def list_all_models(self) -> Dict[str, List[str]]:
        """
        List all available models from all providers.

        Returns:
            Dictionary of provider to list of models
        """
        all_models = {}

        for name, client in self.provider_clients.items():
            try:
                models = await client.list_models()
                all_models[name] = models
                self.logger.debug(f"Provider {name} models: {len(models)}")
            except Exception as e:
                self.logger.error(f"Failed to list models for provider {name}: {e}")
                all_models[name] = self.providers[name].models.copy()

        return all_models

    def get_provider_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all providers."""
        return dict(self.request_stats)

    def get_total_cost(self) -> float:
        """Get total cost across all providers."""
        return sum(stats["total_cost"] for stats in self.request_stats.values())

    def _update_request_stats(
        self,
        provider_name: str,
        response: Optional[AIResponse],
        success: bool,
        response_time: float
    ) -> None:
        """Update request statistics for a provider."""
        stats = self.request_stats[provider_name]
        stats["total_requests"] += 1

        if success and response:
            stats["successful_requests"] += 1
            stats["total_tokens"] += response.tokens_used
            stats["total_cost"] += response.cost_usd

            # Update average response time
            total_time = stats["avg_response_time"] * (stats["successful_requests"] - 1)
            stats["avg_response_time"] = (total_time + response_time) / stats["successful_requests"]
        else:
            stats["failed_requests"] += 1

    async def close(self) -> None:
        """Close all provider clients."""
        for name, client in self.provider_clients.items():
            try:
                if hasattr(client, 'aclose'):
                    await client.aclose()
                elif hasattr(client, 'close'):
                    await client.close()
                elif hasattr(client, 'client') and hasattr(client.client, 'aclose'):
                    await client.client.aclose()
                elif hasattr(client, 'client') and hasattr(client.client, 'close'):
                    await client.client.close()
            except Exception as e:
                self.logger.error(f"Error closing provider {name}: {e}")

        self.logger.info(
            "ProviderManager closed",
            total_cost=self.get_total_cost(),
            total_requests=sum(stats["total_requests"] for stats in self.request_stats.values())
        )