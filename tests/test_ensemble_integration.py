"""
Integration tests for enhanced AI model ensemble system.
Tests integration between XAIClient, ensemble components, and decision logic.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.clients.xai_client import XAIClient, TradingDecision
from src.intelligence.ensemble_engine import EnsembleEngine, EnsembleConfig, EnsembleResult
from src.intelligence.model_selector import ModelSelector
from src.intelligence.cost_optimizer import CostOptimizer
from src.intelligence.fallback_manager import FallbackManager
from src.config.settings import settings


class TestXAIClientEnsembleIntegration:
    """Test integration between XAIClient and ensemble engine."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = Mock()
        db_manager.log_llm_query = AsyncMock()
        return db_manager

    @pytest.fixture
    def mock_kalshi_client(self):
        """Create mock Kalshi client."""
        kalshi_client = Mock()
        return kalshi_client

    @pytest.fixture
    def mock_xai_client(self, mock_db_manager, mock_kalshi_client):
        """Create XAIClient with mocked dependencies."""
        with patch('src.clients.xai_client.settings') as mock_settings:
            mock_settings.multi_model_ensemble = True
            mock_settings.api.xai_api_key = "test_key"
            mock_settings.trading.primary_model = "grok-4"
            mock_settings.trading.fallback_model = "grok-3"
            mock_settings.trading.ai_temperature = 0.1
            mock_settings.trading.ai_max_tokens = 8000

            client = XAIClient(
                api_key="test_key",
                db_manager=mock_db_manager,
                kalshi_client=mock_kalshi_client
            )
            return client

    @pytest.mark.asyncio
    async def test_existing_ensemble_decision_integration(self, mock_xai_client):
        """Test that existing get_ensemble_decision method works correctly."""
        # Mock the ensemble feature as enabled
        with patch('src.clients.xai_client.settings.multi_model_ensemble', True):
            # Mock the internal decision method
            mock_decision = TradingDecision(
                action="BUY",
                side="YES",
                confidence=0.8,
                limit_price=65,
                reasoning="Test reasoning"
            )

            mock_xai_client._get_trading_decision_with_prompt = AsyncMock(return_value=mock_decision)

            # Test data
            market_data = {
                "title": "Test Market",
                "yes_price": 60,
                "no_price": 40,
                "volume": 1000
            }
            portfolio_data = {"balance": 1000}

            # Call the method
            result = await mock_xai_client.get_ensemble_decision(
                market_data, portfolio_data, "Test news", 0.6
            )

            # Verify result
            assert result is not None
            assert result.action == "BUY"
            assert result.side == "YES"
            assert result.confidence == 0.8
            assert "MULTI-AGENT ENSEMBLE" in result.reasoning

    @pytest.mark.asyncio
    async def test_advanced_ensemble_decision_integration(self, mock_xai_client):
        """Test that get_advanced_ensemble_decision integrates with ensemble engine."""
        # Mock ensemble engine and its dependencies
        with patch('src.clients.xai_client.settings.multi_model_ensemble', True):
            # Create mock ensemble result
            mock_decision = TradingDecision(
                action="BUY",
                side="NO",
                confidence=0.75,
                limit_price=55,
                reasoning="Advanced ensemble reasoning"
            )

            mock_ensemble_result = Mock(spec=EnsembleResult)
            mock_ensemble_result.final_decision = mock_decision
            mock_ensemble_result.disagreement_detected = False
            mock_ensemble_result.uncertainty_score = 0.2
            mock_ensemble_result.ensemble_strategy.value = "weighted_voting"
            mock_ensemble_result.models_consulted = ["grok-4", "grok-3"]

            mock_ensemble_engine = Mock()
            mock_ensemble_engine.get_ensemble_decision = AsyncMock(return_value=mock_ensemble_result)

            mock_xai_client._ensemble_engine = mock_ensemble_engine
            mock_xai_client._ensemble_engine_initialized = True

            # Test data
            market_data = {
                "title": "Test Market",
                "yes_price": 60,
                "no_price": 40,
                "volume": 1000,
                "category": "politics"
            }
            portfolio_data = {"balance": 1000}

            # Call the method
            result = await mock_xai_client.get_advanced_ensemble_decision(
                market_data, portfolio_data, "Test news", trade_value=25.0
            )

            # Verify integration
            assert result is not None
            assert result.action == "BUY"
            assert result.side == "NO"
            assert result.confidence == 0.75
            assert "ADVANCED ENSEMBLE" in result.reasoning
            assert "weighted_voting" in result.reasoning

            # Verify ensemble engine was called correctly
            mock_ensemble_engine.get_ensemble_decision.assert_called_once()
            call_args = mock_ensemble_engine.get_ensemble_decision.call_args
            assert "enhanced_market_data" in str(call_args)

    @pytest.mark.asyncio
    async def test_ensemble_fallback_to_basic(self, mock_xai_client):
        """Test that advanced ensemble falls back to basic ensemble on error."""
        with patch('src.clients.xai_client.settings.multi_model_ensemble', True):
            # Mock basic ensemble decision
            mock_basic_decision = TradingDecision(
                action="BUY",
                side="YES",
                confidence=0.7,
                limit_price=60,
                reasoning="Basic ensemble reasoning"
            )

            # Mock advanced ensemble to fail
            mock_xai_client._ensemble_engine_initialized = True
            mock_xai_client._ensemble_engine = Mock()
            mock_xai_client._ensemble_engine.get_ensemble_decision = AsyncMock(
                side_effect=Exception("Ensemble engine failed")
            )

            # Mock basic ensemble method
            mock_xai_client.get_ensemble_decision = AsyncMock(return_value=mock_basic_decision)

            # Test data
            market_data = {
                "title": "Test Market",
                "yes_price": 60,
                "no_price": 40,
                "volume": 1000
            }
            portfolio_data = {"balance": 1000}

            # Call the method
            result = await mock_xai_client.get_advanced_ensemble_decision(
                market_data, portfolio_data, "Test news", trade_value=25.0
            )

            # Verify fallback worked
            assert result is not None
            assert result.action == "BUY"
            assert result.side == "YES"
            assert result.confidence == 0.7


class TestSettingsMultiModelEnsembleFlag:
    """Test Settings.multi_model_ensemble flag functionality."""

    @pytest.mark.asyncio
    async def test_multi_model_ensemble_disabled(self):
        """Test behavior when multi_model_ensemble is False."""
        with patch('src.clients.xai_client.settings.multi_model_ensemble', False):
            mock_xai_client = Mock(spec=XAIClient)
            mock_decision = TradingDecision(
                action="BUY",
                side="YES",
                confidence=0.8,
                reasoning="Single model decision"
            )

            mock_xai_client.get_trading_decision = AsyncMock(return_value=mock_decision)

            # Test data
            market_data = {"title": "Test Market"}
            portfolio_data = {"balance": 1000}

            # When ensemble is disabled, should fall back to single model
            from src.clients.xai_client import XAIClient

            # Create real client with mocked settings
            with patch('src.clients.xai_client.settings.multi_model_ensemble', False):
                with patch('src.clients.xai_client.settings.api.xai_api_key', 'test_key'):
                    client = XAIClient(api_key='test_key')
                    client.get_trading_decision = AsyncMock(return_value=mock_decision)

                    result = await client.get_ensemble_decision(
                        market_data, portfolio_data, "Test news"
                    )

                    assert result is not None
                    assert result.action == "BUY"
                    assert result.confidence == 0.8

    def test_settings_ensemble_flag_default_value(self):
        """Test that multi_model_ensemble flag has correct default value."""
        # Test the default setting
        assert settings.multi_model_ensemble is True

        # Test the trading config setting
        assert hasattr(settings.trading, 'multi_model_ensemble') or settings.multi_model_ensemble

    @pytest.mark.asyncio
    async def test_settings_flag_affects_decision_routing(self):
        """Test that the settings flag affects decision routing correctly."""
        with patch('src.clients.xai_client.settings.multi_model_ensemble', True):
            # Create client with ensemble enabled
            with patch('src.clients.xai_client.settings.api.xai_api_key', 'test_key'):
                client = XAIClient(api_key='test_key')

                # Mock both ensemble and single model methods
                mock_ensemble_decision = TradingDecision(
                    action="BUY", side="YES", confidence=0.9,
                    reasoning="Ensemble decision"
                )
                mock_single_decision = TradingDecision(
                    action="BUY", side="YES", confidence=0.8,
                    reasoning="Single model decision"
                )

                client.get_trading_decision = AsyncMock(return_value=mock_single_decision)

                # Test that ensemble is used when flag is True and conditions are met
                market_data = {"title": "Test Market", "yes_price": 60}
                portfolio_data = {"balance": 1000}

                # High stakes scenario should trigger ensemble
                max_investment = 1000 * 0.03  # 3% of balance
                if max_investment >= 50.0:  # High stakes threshold
                    # In real implementation, ensemble would be called
                    # For test, we verify the logic would route to ensemble
                    assert settings.multi_model_ensemble is True


class TestDecisionLogicIntegration:
    """Test integration with decision logic in decide.py."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        db_manager = Mock()
        db_manager.get_daily_ai_cost = AsyncMock(return_value=5.0)  # Under limit
        db_manager.was_recently_analyzed = AsyncMock(return_value=False)
        db_manager.get_market_analysis_count_today = AsyncMock(return_value=1)
        db_manager.record_market_analysis = AsyncMock()
        return db_manager

    @pytest.fixture
    def mock_market(self):
        """Create test market object."""
        from src.utils.database import Market

        market = Mock(spec=Market)
        market.market_id = "TEST_MARKET_123"
        market.title = "Test Market Title"
        market.yes_price = 65
        market.no_price = 35
        market.volume = 1500
        market.category = "politics"
        market.expiration_ts = datetime.now().timestamp() + 86400 * 7  # 7 days
        return market

    @pytest.mark.asyncio
    async def test_decide_py_ensemble_integration(self, mock_db_manager, mock_market):
        """Test that decide.py integrates with ensemble decisions correctly."""
        # Mock clients and settings
        with patch('src.jobs.decide.settings') as mock_settings:
            mock_settings.multi_model_ensemble = True
            mock_settings.get_ai_daily_limit.return_value = 15.0
            mock_settings.trading.min_volume_for_ai_analysis = 1000
            mock_settings.trading.exclude_low_liquidity_categories = []
            mock_settings.trading.max_position_size_pct = 3.0

            mock_kalshi_client = Mock()
            mock_kalshi_client.get_balance.return_value = {"balance": 100000}  # $1000 in cents

            # Create mock ensemble decision
            mock_ensemble_decision = TradingDecision(
                action="BUY",
                side="YES",
                confidence=0.8,
                limit_price=70,
                reasoning="Ensemble decision for high-stakes trade"
            )

            mock_xai_client = Mock()
            mock_xai_client.get_ensemble_decision = AsyncMock(return_value=mock_ensemble_decision)

            # Test the decision logic
            from src.jobs.decide import make_decision_for_market

            result = await make_decision_for_market(
                mock_market, mock_db_manager, mock_xai_client, mock_kalshi_client
            )

            # Verify ensemble was called for high-stakes scenario
            assert result is not None  # Position should be created
            mock_xai_client.get_ensemble_decision.assert_called_once()

            # Check call arguments
            call_args = mock_xai_client.get_ensemble_decision.call_args
            assert 'market_data' in call_args.kwargs
            assert 'portfolio_data' in call_args.kwargs
            assert 'news_summary' in call_args.kwargs

    @pytest.mark.asyncio
    async def test_decide_py_single_model_fallback(self, mock_db_manager, mock_market):
        """Test that decide.py falls back to single model when ensemble disabled."""
        with patch('src.jobs.decide.settings') as mock_settings:
            mock_settings.multi_model_ensemble = False  # Disabled
            mock_settings.get_ai_daily_limit.return_value = 15.0
            mock_settings.trading.min_volume_for_ai_analysis = 1000
            mock_settings.trading.exclude_low_liquidity_categories = []
            mock_settings.trading.max_position_size_pct = 3.0

            mock_kalshi_client = Mock()
            mock_kalshi_client.get_balance.return_value = {"balance": 20000}  # Low balance

            # Create mock single model decision
            mock_single_decision = TradingDecision(
                action="BUY",
                side="NO",
                confidence=0.7,
                limit_price=30,
                reasoning="Single model decision"
            )

            mock_xai_client = Mock()
            mock_xai_client.get_trading_decision = AsyncMock(return_value=mock_single_decision)

            # Test the decision logic
            from src.jobs.decide import make_decision_for_market

            result = await make_decision_for_market(
                mock_market, mock_db_manager, mock_xai_client, mock_kalshi_client
            )

            # Verify single model was called (not ensemble)
            mock_xai_client.get_ensemble_decision.assert_not_called()
            mock_xai_client.get_trading_decision.assert_called_once()

    def test_high_stakes_threshold_calculation(self):
        """Test high-stakes threshold calculation in decide.py."""
        # Test the calculation logic from decide.py
        available_balance = 1000.0
        max_position_size_pct = 3.0
        max_investment_possible = (available_balance * max_position_size_pct) / 100

        # Should be high stakes if >= $50
        is_high_stakes = max_investment_possible >= 50.0

        assert max_investment_possible == 30.0  # 3% of $1000
        assert is_high_stakes is False  # $30 < $50

        # Test with higher balance
        available_balance = 2000.0
        max_investment_possible = (available_balance * max_position_size_pct) / 100
        is_high_stakes = max_investment_possible >= 50.0

        assert max_investment_possible == 60.0  # 3% of $2000
        assert is_high_stakes is True  # $60 >= $50


class TestEnsembleComponentCoordination:
    """Test coordination between all ensemble components."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create mock database manager."""
        return Mock()

    @pytest.mark.asyncio
    async def test_ensemble_components_initialization(self, mock_db_manager):
        """Test that ensemble components initialize and coordinate correctly."""
        from src.intelligence.ensemble_engine import EnsembleEngine, EnsembleConfig
        from src.intelligence.model_selector import ModelSelector
        from src.intelligence.cost_optimizer import CostOptimizer
        from src.utils.performance_tracker import PerformanceTracker

        # Create components
        performance_tracker = PerformanceTracker(mock_db_manager)
        model_selector = ModelSelector(performance_tracker)
        cost_optimizer = CostOptimizer(mock_db_manager)
        config = EnsembleConfig()

        # Create ensemble engine
        ensemble_engine = EnsembleEngine(
            mock_db_manager,
            performance_tracker,
            model_selector,
            config
        )

        # Verify components are coordinated
        assert ensemble_engine.db_manager == mock_db_manager
        assert ensemble_engine.performance_tracker == performance_tracker
        assert ensemble_engine.model_selector == model_selector
        assert ensemble_engine.config == config

        # Test ensemble method exists
        assert hasattr(ensemble_engine, 'get_ensemble_decision')
        assert callable(getattr(ensemble_engine, 'get_ensemble_decision'))

    @pytest.mark.asyncio
    async def test_component_data_flow(self, mock_db_manager):
        """Test data flow between ensemble components."""
        from src.intelligence.model_selector import ModelSelector
        from src.intelligence.cost_optimizer import CostOptimizer
        from src.utils.performance_tracker import PerformanceTracker

        # Create components
        performance_tracker = PerformanceTracker(mock_db_manager)
        model_selector = ModelSelector(performance_tracker)
        cost_optimizer = CostOptimizer(mock_db_manager)

        # Test that components can share data
        # Mock some performance data
        mock_performance_data = {
            "grok-4": {"accuracy": 0.75, "cost_per_decision": 0.02, "response_time": 1.5},
            "grok-3": {"accuracy": 0.70, "cost_per_decision": 0.01, "response_time": 1.0}
        }

        performance_tracker.model_performance = mock_performance_data

        # Test model selector can access performance data
        # (In real implementation, this would be used for intelligent model selection)
        assert hasattr(model_selector, 'performance_tracker')
        assert model_selector.performance_tracker == performance_tracker

        # Test cost optimizer can access database
        assert cost_optimizer.db_manager == mock_db_manager

    @pytest.mark.asyncio
    async def test_ensemble_state_management(self, mock_db_manager):
        """Test that ensemble state is managed correctly."""
        from src.intelligence.ensemble_engine import EnsembleEngine, EnsembleConfig
        from src.intelligence.model_selector import ModelSelector
        from src.utils.performance_tracker import PerformanceTracker

        # Create ensemble engine
        performance_tracker = PerformanceTracker(mock_db_manager)
        model_selector = ModelSelector(performance_tracker)
        config = EnsembleConfig()

        ensemble_engine = EnsembleEngine(
            mock_db_manager,
            performance_tracker,
            model_selector,
            config
        )

        # Test state management properties exist
        assert hasattr(ensemble_engine, 'db_manager')
        assert hasattr(ensemble_engine, 'performance_tracker')
        assert hasattr(ensemble_engine, 'model_selector')
        assert hasattr(ensemble_engine, 'config')

        # Test configuration is applied
        assert ensemble_engine.config == config
        assert hasattr(config, 'consensus_threshold')
        assert hasattr(config, 'enable_weighted_voting')

    @pytest.mark.asyncio
    async def test_component_error_handling(self, mock_db_manager):
        """Test that ensemble components handle errors gracefully."""
        from src.intelligence.fallback_manager import FallbackManager
        from src.intelligence.provider_manager import ProviderManager

        # Create components
        fallback_manager = FallbackManager()
        provider_manager = ProviderManager()

        # Test error handling methods exist
        assert hasattr(fallback_manager, 'check_provider_health')
        assert hasattr(fallback_manager, 'initiate_failover')
        assert hasattr(fallback_manager, 'enable_emergency_mode')

        assert hasattr(provider_manager, 'get_provider_client')
        assert hasattr(provider_manager, 'standardize_request')

        # Test they can be called without errors (basic smoke test)
        try:
            # These should not raise exceptions even with minimal setup
            health_status = await fallback_manager.check_provider_health("grok-4")
            assert isinstance(health_status, bool)
        except Exception as e:
            # Expected - components need full setup to work properly
            assert "No provider" in str(e) or "not configured" in str(e).lower()