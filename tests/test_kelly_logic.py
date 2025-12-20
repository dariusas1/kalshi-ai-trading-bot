
import pytest
from src.strategies.portfolio_optimization import _calculate_simple_kelly, AdvancedPortfolioOptimizer
from src.strategies.portfolio_optimization import MarketOpportunity
from src.utils.database import DatabaseManager
from src.clients.kalshi_client import KalshiClient
from src.clients.xai_client import XAIClient
from src.config.settings import settings

def create_test_opportunity(
    predicted_probability, 
    market_probability, 
    edge, 
    time_to_expiry=30.0,
    confidence=1.0
):
    return MarketOpportunity(
        market_id="test",
        market_title="test",
        predicted_probability=predicted_probability,
        market_probability=market_probability,
        confidence=confidence,
        edge=edge,
        volatility=0.5,
        expected_return=0.1,
        max_loss=0.6,
        time_to_expiry=time_to_expiry,
        correlation_score=0.0,
        kelly_fraction=0.0,
        fractional_kelly=0.0,
        risk_adjusted_fraction=0.0,
        sharpe_ratio=0.0,
        sortino_ratio=0.0,
        max_drawdown_contribution=0.0
    )

def test_calculate_simple_kelly_yes():
    # Case: YES side, AI prob 0.7, Market prob 0.6
    # p = 0.7, q = 0.3, b = (1-0.6)/0.6 = 0.4/0.6 = 2/3
    # Kelly = (2/3 * 0.7 - 0.3) / (2/3) = (1.4/3 - 0.9/3) / (2/3) = (0.5/3) / (2/3) = 0.25
    # Capped at 0.2 in _calculate_simple_kelly (hardcoded cap)
    opportunity = create_test_opportunity(0.7, 0.6, 0.1)
    kelly = _calculate_simple_kelly(opportunity)
    assert kelly == pytest.approx(0.2)

def test_calculate_simple_kelly_no():
    # Case: NO side, AI prob 0.3, Market prob 0.4
    # edge = 0.3 - 0.4 = -0.1 (Betting NO)
    # p_no = 1 - 0.3 = 0.7, q_no = 0.3
    # b_no = 0.4 / (1 - 0.4) = 0.4 / 0.6 = 2/3
    # Kelly = (2/3 * 0.7 - 0.3) / (2/3) = 0.25
    # Capped at 0.2
    opportunity = create_test_opportunity(0.3, 0.4, -0.1)
    kelly = _calculate_simple_kelly(opportunity)
    assert kelly == pytest.approx(0.2)

@pytest.mark.asyncio
async def test_advanced_kelly_logic_yes():
    # Mock optimizer
    optimizer = AdvancedPortfolioOptimizer(None, None, None)
    
    # Case: YES side, positive edge
    opportunity = create_test_opportunity(0.7, 0.6, 0.1, time_to_expiry=30.0)
    
    fractions = await optimizer._calculate_kelly_fractions([opportunity])
    
    # odds = 0.666, p = 0.7, kelly_standard = 0.25
    # kelly_fraction_multiplier = 0.55 (from settings)
    # fractional_kelly = 0.25 * 0.55 = 0.1375
    # max_single_position = 0.04 (from settings)
    # Expected approx 0.04
    
    assert opportunity.market_id in fractions
    assert fractions[opportunity.market_id] == pytest.approx(settings.trading.max_single_position)

@pytest.mark.asyncio
async def test_advanced_kelly_logic_no():
    # Mock optimizer
    optimizer = AdvancedPortfolioOptimizer(None, None, None)
    
    # Case: NO side, positive edge for NO (negative edge variable)
    opportunity = create_test_opportunity(0.3, 0.4, -0.1, time_to_expiry=30.0)
    
    fractions = await optimizer._calculate_kelly_fractions([opportunity])
    
    # Expected approx 0.04 (same logic as YES)
    
    assert opportunity.market_id in fractions
    assert fractions[opportunity.market_id] == pytest.approx(settings.trading.max_single_position)

@pytest.mark.asyncio
async def test_advanced_kelly_logic_small_edge_no_cap():
    # Test a case where it's NOT capped by max_single_position
    optimizer = AdvancedPortfolioOptimizer(None, None, None)
    
    # Case: YES side, very small edge
    # p = 0.51, market_prob = 0.50
    # b = 0.5/0.5 = 1.0
    # kelly_standard = (1.0 * 0.51 - 0.49) / 1.0 = 0.02
    # fractional_kelly = 0.02 * 0.55 = 0.011
    # max_single_position = 0.04
    # Expected approx 0.011
    
    opportunity = create_test_opportunity(0.51, 0.50, 0.01, time_to_expiry=30.0)
    fractions = await optimizer._calculate_kelly_fractions([opportunity])
    
    assert opportunity.market_id in fractions
    assert fractions[opportunity.market_id] == pytest.approx(0.011, rel=0.1)

@pytest.mark.asyncio
async def test_advanced_kelly_logic_no_edge():
    optimizer = AdvancedPortfolioOptimizer(None, None, None)
    opportunity = create_test_opportunity(0.5, 0.5, 0.0, time_to_expiry=30.0)
    fractions = await optimizer._calculate_kelly_fractions([opportunity])
    assert opportunity.market_id in fractions
    assert fractions[opportunity.market_id] == 0.0
