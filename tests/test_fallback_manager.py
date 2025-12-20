"""
Tests for FallbackManager class - multi-provider redundancy and graceful degradation.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json

from src.intelligence.fallback_manager import (
    FallbackManager,
    ProviderStatus,
    ProviderConfig,
    EmergencyMode,
    HealthCheckResult
)


class TestFallbackManager:
    """Test FallbackManager multi-provider redundancy functionality."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        return Mock()

    @pytest.fixture
    def fallback_manager(self, mock_db_manager):
        """Create a FallbackManager instance with test configurations."""
        providers = {
            "xai": ProviderConfig(
                name="xai",
                endpoint="https://api.x.ai/v1",
                api_key="test-xai-key",
                models=["grok-4", "grok-3"],
                priority=1,
                timeout=30.0,
                max_retries=3
            ),
            "openai": ProviderConfig(
                name="openai",
                endpoint="https://api.openai.com/v1",
                api_key="test-openai-key",
                models=["gpt-4", "gpt-3.5-turbo"],
                priority=2,
                timeout=25.0,
                max_retries=3
            ),
            "anthropic": ProviderConfig(
                name="anthropic",
                endpoint="https://api.anthropic.com/v1",
                api_key="test-anthropic-key",
                models=["claude-3-opus", "claude-3-sonnet"],
                priority=3,
                timeout=35.0,
                max_retries=2
            ),
            "local": ProviderConfig(
                name="local",
                endpoint="http://localhost:8080/v1",
                api_key="local-key",
                models=["llama-2", "mistral"],
                priority=4,
                timeout=45.0,
                max_retries=1
            )
        }

        return FallbackManager(mock_db_manager, providers)

    @pytest.mark.asyncio
    async def test_multi_provider_redundancy(self, fallback_manager):
        """Test automatic failover between multiple providers."""
        # Mock provider health checks with different status
        with patch.object(fallback_manager, '_check_provider_health_internal') as mock_health:
            # Configure mock responses
            mock_health.side_effect = [
                HealthCheckResult("xai", False, "API error", 0.0),
                HealthCheckResult("openai", True, "Healthy", 150.0),
                HealthCheckResult("anthropic", True, "Healthy", 200.0),
                HealthCheckResult("local", False, "Connection refused", 0.0)
            ]

            # Get available providers
            available_providers = await fallback_manager.get_available_providers()

            # Should return healthy providers in priority order
            assert len(available_providers) == 2
            assert available_providers[0].name == "openai"  # Priority 2, healthy
            assert available_providers[1].name == "anthropic"  # Priority 3, healthy

            # Verify health check was called for all providers
            assert mock_health.call_count == 4

    @pytest.mark.asyncio
    async def test_graceful_degradation_during_outages(self, fallback_manager):
        """Test system maintains reduced functionality during partial outages."""
        # Simulate partial provider failure
        with patch.object(fallback_manager, '_check_provider_health_internal') as mock_health:
            # Only xAI provider is healthy
            mock_health.side_effect = [
                HealthCheckResult("xai", True, "Healthy", 100.0),
                HealthCheckResult("openai", False, "Rate limit exceeded", 0.0),
                HealthCheckResult("anthropic", False, "Service unavailable", 0.0),
                HealthCheckResult("local", False, "Model loading", 0.0)
            ]

            # Check system status
            status = await fallback_manager.get_system_status()

            # Should indicate degraded mode
            assert status.mode == "degraded"
            assert status.healthy_providers == 1
            assert status.total_providers == 4
            assert status.degradation_level == 0.75  # 75% of providers unavailable

            # Should still be able to get a provider (graceful degradation)
            provider = await fallback_manager.get_best_provider()
            assert provider is not None
            assert provider.name == "xai"

    @pytest.mark.asyncio
    async def test_emergency_trading_modes_extended_outages(self, fallback_manager):
        """Test emergency trading mode activation during extended outages."""
        # Simulate complete provider failure
        with patch.object(fallback_manager, '_check_provider_health_internal') as mock_health:
            # All providers fail
            mock_health.side_effect = [
                HealthCheckResult("xai", False, "API down", 0.0),
                HealthCheckResult("openai", False, "Service outage", 0.0),
                HealthCheckResult("anthropic", False, "Maintenance", 0.0),
                HealthCheckResult("local", False, "Process crashed", 0.0)
            ]

            # Enable emergency mode
            await fallback_manager.enable_emergency_mode(
                reason="All providers unavailable",
                duration_minutes=60
            )

            # Check emergency mode status
            status = await fallback_manager.get_system_status()
            assert status.mode == "emergency"
            assert status.emergency_mode_active is True
            assert status.emergency_reason == "All providers unavailable"

            # Should provide emergency trading decisions
            decision = await fallback_manager.get_emergency_decision(
                market_data={"title": "Test Market", "yes_price": 60},
                portfolio_data={"balance": 1000}
            )

            # Emergency decisions should be conservative
            assert decision is not None
            assert decision.action == "SKIP"  # Conservative default
            assert "EMERGENCY MODE" in decision.reasoning
            assert decision.confidence <= 0.5  # Low confidence in emergency

    @pytest.mark.asyncio
    async def test_comprehensive_health_checking_automatic_failover(self, fallback_manager):
        """Test comprehensive health monitoring with automatic failover."""
        # Mock health check results over time
        health_results = [
            # Initial state: xAI healthy, others degraded
            [HealthCheckResult("xai", True, "Healthy", 120.0),
             HealthCheckResult("openai", False, "Slow response", 5000.0),
             HealthCheckResult("anthropic", False, "High error rate", 0.0),
             HealthCheckResult("local", False, "Connection timeout", 0.0)],

            # xAI fails, openai recovers
            [HealthCheckResult("xai", False, "API error", 0.0),
             HealthCheckResult("openai", True, "Recovered", 180.0),
             HealthCheckResult("anthropic", False, "Still degraded", 0.0),
             HealthCheckResult("local", False, "Still down", 0.0)],

            # openai degrades, anthropic recovers
            [HealthCheckResult("xai", False, "Still down", 0.0),
             HealthCheckResult("openai", False, "Rate limited", 0.0),
             HealthCheckResult("anthropic", True, "Recovered", 220.0),
             HealthCheckResult("local", False, "Still down", 0.0)]
        ]

        with patch.object(fallback_manager, '_check_provider_health_internal') as mock_health:
            mock_health.side_effect = health_results

            # Test initial state - xAI should be primary
            provider1 = await fallback_manager.get_best_provider()
            assert provider1.name == "xai"

            # Simulate time passing and xAI failure
            await asyncio.sleep(0.1)
            provider2 = await fallback_manager.get_best_provider()
            assert provider2.name == "openai"

            # Simulate further degradation and recovery
            await asyncio.sleep(0.1)
            provider3 = await fallback_manager.get_best_provider()
            assert provider3.name == "anthropic"

            # Verify failover history is tracked
            failover_history = fallback_manager.get_failover_history()
            assert len(failover_history) == 2  # xAI -> openai -> anthropic

            # Verify health metrics are tracked
            health_metrics = fallback_manager.get_health_metrics()
            assert "xai" in health_metrics
            assert "openai" in health_metrics
            assert "anthropic" in health_metrics

    @pytest.mark.asyncio
    async def test_fallback_performance_recovery(self, fallback_manager):
        """Test fallback system performance and recovery procedures."""
        # Test provider response time tracking
        with patch.object(fallback_manager, '_make_provider_request') as mock_request:
            # Configure mock to simulate different response times
            response_times = [100, 200, 3000, 150]  # milliseconds

            def mock_response(*args, **kwargs):
                provider_name = args[0] if args else "xai"
                response_time = response_times.pop(0) if response_times else 150

                # Simulate request
                if response_time > 2000:  # Simulate timeout
                    raise TimeoutError(f"Provider {provider_name} timeout")

                return {"decision": "BUY", "confidence": 0.7}

            mock_request.side_effect = mock_response

            # Test multiple requests with failover
            decisions = []
            providers_used = []

            for i in range(4):
                try:
                    decision = await fallback_manager.get_fallback_decision(
                        market_data={"title": f"Test Market {i}"},
                        portfolio_data={"balance": 1000}
                    )
                    decisions.append(decision)

                    # Track which provider was used
                    current_provider = await fallback_manager.get_best_provider()
                    providers_used.append(current_provider.name)

                except Exception as e:
                    decisions.append(None)

            # Should have successful decisions with failover
            successful_decisions = [d for d in decisions if d is not None]
            assert len(successful_decisions) >= 2  # At least 2 successful decisions

            # Test recovery procedure
            with patch.object(fallback_manager, '_check_provider_health_internal') as mock_health:
                # Initially all providers down
                mock_health.return_value = [
                    HealthCheckResult("xai", False, "Down", 0.0),
                    HealthCheckResult("openai", False, "Down", 0.0),
                    HealthCheckResult("anthropic", False, "Down", 0.0),
                    HealthCheckResult("local", False, "Down", 0.0)
                ]

                # System should be in emergency mode
                status = await fallback_manager.get_system_status()
                assert status.mode == "emergency"

                # Simulate recovery
                mock_health.return_value = [
                    HealthCheckResult("xai", True, "Recovered", 150.0),
                    HealthCheckResult("openai", True, "Recovered", 180.0),
                    HealthCheckResult("anthropic", True, "Recovered", 200.0),
                    HealthCheckResult("local", True, "Recovered", 300.0)
                ]

                # Force recovery check
                await fallback_manager.check_recovery()

                # Should exit emergency mode
                status = await fallback_manager.get_system_status()
                assert status.mode == "normal"
                assert status.emergency_mode_active is False


class TestEmergencyMode:
    """Test emergency trading mode functionality."""

    @pytest.fixture
    def emergency_fallback_manager(self, mock_db_manager):
        """Create a FallbackManager in emergency mode."""
        providers = {
            "xai": ProviderConfig("xai", "https://api.x.ai/v1", "key", ["grok-4"], 1, 30, 3)
        }
        manager = FallbackManager(mock_db_manager, providers)
        return manager

    @pytest.mark.asyncio
    async def test_emergency_mode_conservative_decisions(self, emergency_fallback_manager):
        """Test emergency mode makes conservative trading decisions."""
        await emergency_fallback_manager.enable_emergency_mode(
            reason="Test emergency",
            duration_minutes=30
        )

        # Test with various market scenarios
        test_cases = [
            {
                "market": {"title": "High Volatility Market", "yes_price": 80, "volume": 10000},
                "portfolio": {"balance": 1000},
                "expected_action": "SKIP"
            },
            {
                "market": {"title": "Low Risk Market", "yes_price": 55, "volume": 1000},
                "portfolio": {"balance": 500},
                "expected_action": "SKIP"  # Still conservative in emergency
            },
            {
                "market": {"title": "Medium Risk Market", "yes_price": 65, "volume": 5000},
                "portfolio": {"balance": 2000},
                "expected_action": "SKIP"  # Very conservative in emergency
            }
        ]

        for case in test_cases:
            decision = await emergency_fallback_manager.get_emergency_decision(
                case["market"],
                case["portfolio"]
            )

            # All emergency decisions should be SKIP (very conservative)
            assert decision.action == case["expected_action"]
            assert decision.confidence <= 0.5
            assert "EMERGENCY" in decision.reasoning
            assert "CONSERVATIVE" in decision.reasoning

    @pytest.mark.asyncio
    async def test_emergency_mode_cached_decisions(self, emergency_fallback_manager):
        """Test emergency mode uses cached decisions when available."""
        # Add some cached decisions
        cached_decisions = {
            "market_1": {
                "action": "BUY",
                "side": "YES",
                "confidence": 0.6,
                "reasoning": "Cached analysis",
                "timestamp": datetime.now() - timedelta(minutes=30)
            }
        }

        emergency_fallback_manager.set_cached_decisions(cached_decisions)

        await emergency_fallback_manager.enable_emergency_mode(
            reason="Test emergency with cache",
            duration_minutes=60
        )

        # Test with cached market
        decision = await emergency_fallback_manager.get_emergency_decision(
            {"title": "market_1", "yes_price": 60},
            {"balance": 1000}
        )

        # Should use cached decision if recent enough
        assert decision is not None
        # Emergency mode might still override cache for safety

        # Test with non-cached market
        decision2 = await emergency_fallback_manager.get_emergency_decision(
            {"title": "market_2", "yes_price": 60},
            {"balance": 1000}
        )

        # Should provide default conservative decision
        assert decision2.action == "SKIP"
        assert "NO CACHED" in decision2.reasoning or "DEFAULT" in decision2.reasoning


class TestProviderHealth:
    """Test provider health checking functionality."""

    @pytest.fixture
    def health_fallback_manager(self, mock_db_manager):
        """Create a FallbackManager for health testing."""
        providers = {
            "test_provider": ProviderConfig(
                "test_provider",
                "https://test.api.com/v1",
                "test-key",
                ["test-model"],
                1,
                10.0,
                2
            )
        }
        return FallbackManager(mock_db_manager, providers)

    @pytest.mark.asyncio
    async def test_health_check_timeout_handling(self, health_fallback_manager):
        """Test health check handles timeouts properly."""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Simulate timeout
            mock_get.side_effect = asyncio.TimeoutError("Request timeout")

            result = await health_fallback_manager._check_provider_health_internal(
                "test_provider"
            )

            assert result.is_healthy is False
            assert "timeout" in result.error_message.lower()
            assert result.response_time == 0.0

    @pytest.mark.asyncio
    async def test_health_check_rate_limit_handling(self, health_fallback_manager):
        """Test health check handles rate limiting properly."""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Simulate rate limit response
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {"retry-after": "60"}
            mock_get.return_value = mock_response

            result = await health_fallback_manager._check_provider_health_internal(
                "test_provider"
            )

            assert result.is_healthy is False
            assert "rate limit" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_health_check_success_metrics(self, health_fallback_manager):
        """Test health check records success metrics properly."""
        with patch('httpx.AsyncClient.get') as mock_get:
            # Simulate successful health check
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy", "models": ["test-model"]}
            mock_response.headers = {"x-response-time": "150"}
            mock_get.return_value = mock_response

            result = await health_fallback_manager._check_provider_health_internal(
                "test_provider"
            )

            assert result.is_healthy is True
            assert result.response_time > 0
            assert result.error_message is None

            # Verify metrics are recorded
            metrics = health_fallback_manager.get_health_metrics()
            assert "test_provider" in metrics
            assert metrics["test_provider"]["last_check"] > datetime.now() - timedelta(minutes=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])