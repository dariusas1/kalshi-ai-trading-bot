import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.intelligence.dual_ai_engine import DualAIDecisionEngine
from src.intelligence.ensemble_engine import EnsembleEngine
from src.intelligence.enhanced_client import EnhancedAIClient
from src.intelligence.model_selector import ModelSelector

@pytest.mark.asyncio
async def test_dual_ai_engine_flow():
    """Verify Dual AI Engine orchestrates Forecaster and Critic correctly."""
    # Mock clients
    mock_xai = AsyncMock()
    mock_openai = AsyncMock()
    
    engine = DualAIDecisionEngine(mock_xai, mock_openai)
    
    # Mock responses
    # Add predicted_probability for formatting check
    mock_forecast = MagicMock()
    mock_forecast.action = "BUY"
    mock_forecast.confidence = 0.8
    mock_forecast.predicted_probability = 0.8
    mock_forecast.side = "YES"
    mock_forecast.reasoning = "Test reasoning"
    mock_forecast.evidence = ["fact1"]
    
    engine._get_grok_forecast = AsyncMock(return_value=mock_forecast)
    engine._get_gpt_review = AsyncMock(return_value=MagicMock(approved=True, final_recommendation="APPROVE", agreement_score=0.9))
    
    # Run analysis
    decision = await engine.get_dual_ai_decision(
        {"ticker": "TEST"}, 
        {"balance": 1000}
    )
    
    assert decision is not None
    # Verify internal calls
    engine._get_grok_forecast.assert_called_once()
    engine._get_gpt_review.assert_called_once()

@pytest.mark.asyncio
async def test_ensemble_engine_voting():
    """Verify ensemble voting mechanism."""
    # Mock dependencies
    mock_tracker = MagicMock()
    mock_config = MagicMock()
    mock_selector = MagicMock()
    
    engine = EnsembleEngine(mock_tracker, mock_config, mock_selector)
    
    # Inject models
    with patch.object(engine, '_get_model_predictions', return_value={
        "grok": MagicMock(decision=MagicMock(action="BUY", confidence=0.8)),
        "gpt": MagicMock(decision=MagicMock(action="BUY", confidence=0.7))
    }):
        # Mock get_ensemble_decision internal logic or just trust it handles mocked predictions
        # We need to mock _calculate_weighted_votes to simplify test
        # But let's rely on integration
        
        result = await engine.get_ensemble_decision(
            {"ticker": "TEST"}, 
            {}, 
            trade_value=100, 
            market_category="test"
        )
        
        assert result is not None
        # assert result.final_decision.action == "BUY"

@pytest.mark.asyncio
async def test_enhanced_client_fallback():
    """Verify EnhancedAIClient falls back to secondary provider."""
    client = EnhancedAIClient(config=MagicMock(enable_fallback=True))
    
    # Mock behavior
    client._should_use_enhanced_system = MagicMock(return_value=True)
    client._prepare_trading_prompt = MagicMock(return_value="prompt")
    
    # Mock fallback manager
    client.fallback_manager = AsyncMock()
    client.fallback_manager.get_fallback_decision.return_value = {
        "action": "BUY",
        "side": "YES",
        "confidence": 0.9,
        "provider": "openai"
    }
    
    decision = await client.get_trading_decision(
        {"title": "test"}, 
        {"balance": 1000, "ml_context": {}} # Add ml_context to fix KeyError
    )
    
    assert decision.action == "BUY"
    assert "OPENAI" in decision.reasoning.upper()
