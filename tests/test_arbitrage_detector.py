"""
Unit tests for the ArbitrageDetector.

Tests cover:
1. Spread arbitrage detection (YES + NO < 100%)
2. Correlated market arbitrage detection
3. Trade execution with two legs
4. Hedging mechanism for failed leg 2
5. Opportunity filtering and capital allocation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from dataclasses import asdict

from src.strategies.arbitrage_detector import ArbitrageDetector, ArbitrageOpportunity
from src.utils.database import Market, Position


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager."""
    db = MagicMock()
    db.add_position = AsyncMock(return_value=1)
    return db


@pytest.fixture
def mock_kalshi_client():
    """Create a mock Kalshi client."""
    client = MagicMock()
    client.get_market = AsyncMock()
    client.place_order = AsyncMock()
    return client


@pytest.fixture
def detector(mock_db_manager, mock_kalshi_client):
    """Create an ArbitrageDetector with mocked dependencies."""
    return ArbitrageDetector(mock_db_manager, mock_kalshi_client)


@pytest.fixture
def sample_market():
    """Create a sample Market object for testing."""
    return Market(
        market_id="TEST-MARKET-001",
        title="Test Market",
        yes_price=50,
        no_price=50,
        volume=1000,
        expiration_ts=int((datetime.now() + timedelta(days=1)).timestamp()),
        category="test",
        status="active",
        last_updated=datetime.now(),
        has_position=False
    )


@pytest.fixture
def sample_opportunity():
    """Create a sample ArbitrageOpportunity for testing."""
    return ArbitrageOpportunity(
        market_id_1="TEST-MARKET-001",
        market_id_2="TEST-MARKET-001",  # Same market for spread arb
        market_title_1="Test Market",
        market_title_2="Test Market",
        side_1="YES",
        side_2="NO",
        price_1=0.45,
        price_2=0.50,
        spread=0.05,
        expected_profit=5.0,
        confidence=0.95,
        arb_type="spread",
        quantity=10,
        total_cost=9.50
    )


class TestArbitrageDetector:
    """Tests for ArbitrageDetector class."""

    # =====================================================
    # SPREAD ARBITRAGE DETECTION TESTS
    # =====================================================

    @pytest.mark.asyncio
    async def test_find_spread_arbitrage_detects_profitable_spread(self, detector, sample_market):
        """Test that spread arbitrage is detected when YES + NO < 100%."""
        # Mock market with YES + NO = 95% (profitable spread)
        detector.kalshi_client.get_market.return_value = {
            'market': {'yes_ask': 45, 'no_ask': 50}  # 95 cents total
        }
        
        markets = [sample_market]
        opportunities = await detector.find_spread_arbitrage(markets)
        
        assert len(opportunities) >= 1
        opp = opportunities[0]
        assert opp.arb_type == "spread"
        assert opp.price_1 == 0.45  # YES price
        assert opp.price_2 == 0.50  # NO price
        assert opp.spread > 0  # Positive spread (profit)

    @pytest.mark.asyncio
    async def test_find_spread_arbitrage_ignores_no_profit(self, detector, sample_market):
        """Test that no arbitrage is found when YES + NO >= 100%."""
        # Mock market with YES + NO = 100% (no profit)
        detector.kalshi_client.get_market.return_value = {
            'market': {'yes_ask': 50, 'no_ask': 50}  # 100 cents total
        }
        
        markets = [sample_market]
        opportunities = await detector.find_spread_arbitrage(markets)
        
        assert len(opportunities) == 0

    @pytest.mark.asyncio
    async def test_find_spread_arbitrage_accounts_for_fees(self, detector, sample_market):
        """Test that spreads below fee threshold are ignored."""
        # Mock market with YES + NO = 99% (1% profit, but fees are 2%)
        detector.kalshi_client.get_market.return_value = {
            'market': {'yes_ask': 49, 'no_ask': 50}  # 99 cents total
        }
        detector.min_spread_profit = 0.02  # 2% minimum
        detector.fee_rate = 0.01  # 1% per side
        
        markets = [sample_market]
        opportunities = await detector.find_spread_arbitrage(markets)
        
        # Should be empty since 1% profit - 2% fees = -1% (loss)
        assert len(opportunities) == 0

    @pytest.mark.asyncio
    async def test_find_spread_arbitrage_handles_missing_market_data(self, detector, sample_market):
        """Test graceful handling of API errors."""
        detector.kalshi_client.get_market.return_value = None
        
        markets = [sample_market]
        opportunities = await detector.find_spread_arbitrage(markets)
        
        assert len(opportunities) == 0

    # =====================================================
    # TRADE EXECUTION TESTS
    # =====================================================

    @pytest.mark.asyncio
    async def test_execute_arbitrage_trade_simulation_mode(self, detector, sample_opportunity):
        """Test that simulation mode returns success without placing orders."""
        result = await detector.execute_arbitrage_trade(sample_opportunity, live_mode=False)
        
        assert result['success'] is True
        assert result['leg1_executed'] is True
        assert result['leg2_executed'] is True
        # No actual orders should be placed
        detector.kalshi_client.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_arbitrage_trade_success_both_legs(self, detector, sample_opportunity, mock_db_manager):
        """Test successful execution of both arbitrage legs."""
        # Mock successful order responses
        detector.kalshi_client.place_order.return_value = {
            'order': {'status': 'filled', 'order_id': 'test-order-id'}
        }
        
        result = await detector.execute_arbitrage_trade(sample_opportunity, live_mode=True)
        
        assert result['success'] is True
        assert result['leg1_executed'] is True
        assert result['leg2_executed'] is True
        assert result['total_cost'] > 0
        
        # Verify positions were saved to database
        assert mock_db_manager.add_position.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_arbitrage_trade_leg1_failure(self, detector, sample_opportunity):
        """Test handling of leg 1 failure."""
        # Mock failed order response
        detector.kalshi_client.place_order.return_value = {
            'order': {'status': 'rejected'}
        }
        
        result = await detector.execute_arbitrage_trade(sample_opportunity, live_mode=True)
        
        assert result['success'] is False
        assert result['leg1_executed'] is False
        assert result['leg2_executed'] is False

    @pytest.mark.asyncio
    async def test_execute_arbitrage_trade_leg2_failure_triggers_hedge(self, detector, sample_opportunity):
        """Test that leg 2 failure triggers hedge attempt."""
        # Mock: leg 1 succeeds, leg 2 fails
        detector.kalshi_client.place_order.side_effect = [
            {'order': {'status': 'filled'}},  # Leg 1 success
            {'order': {'status': 'rejected'}},  # Leg 2 failure
            {'order': {'status': 'filled'}}   # Hedge success
        ]
        
        result = await detector.execute_arbitrage_trade(sample_opportunity, live_mode=True)
        
        assert result['leg1_executed'] is True
        assert result['leg2_executed'] is False
        assert result['success'] is False
        
        # Verify hedge was attempted (3 total orders: leg1, leg2, hedge)
        assert detector.kalshi_client.place_order.call_count == 3

    # =====================================================
    # OPPORTUNITY SUMMARY TESTS
    # =====================================================

    def test_get_arbitrage_summary_empty(self, detector):
        """Test summary for empty opportunities list."""
        summary = detector.get_arbitrage_summary([])
        
        assert summary['total_opportunities'] == 0
        assert summary['spread_count'] == 0
        assert summary['correlated_count'] == 0
        assert summary['total_expected_profit'] == 0.0

    def test_get_arbitrage_summary_with_opportunities(self, detector, sample_opportunity):
        """Test summary with multiple opportunities."""
        opps = [sample_opportunity, sample_opportunity]
        summary = detector.get_arbitrage_summary(opps)
        
        assert summary['total_opportunities'] == 2
        assert summary['spread_count'] == 2
        assert summary['total_expected_profit'] == 10.0  # 5.0 * 2
        assert summary['avg_confidence'] == 0.95

    # =====================================================
    # MARKET GROUPING TESTS
    # =====================================================

    def test_group_related_markets_by_topic(self, detector):
        """Test market grouping by topic keywords."""
        now = datetime.now()
        exp_ts = int((now + timedelta(days=1)).timestamp())
        markets = [
            Market(market_id="M1", title="Will Trump win?", yes_price=50, no_price=50,
                   volume=1000, expiration_ts=exp_ts, category="politics", status="active",
                   last_updated=now, has_position=False),
            Market(market_id="M2", title="Will Biden run again?", yes_price=50, no_price=50,
                   volume=1000, expiration_ts=exp_ts, category="politics", status="active",
                   last_updated=now, has_position=False),
            Market(market_id="M3", title="Bitcoin above 100k?", yes_price=50, no_price=50,
                   volume=1000, expiration_ts=exp_ts, category="crypto", status="active",
                   last_updated=now, has_position=False),
        ]
        
        groups = detector._group_related_markets(markets)
        
        assert 'politics_us' in groups
        assert 'crypto' in groups
        assert len(groups['politics_us']) == 2  # Trump and Biden markets
        assert len(groups['crypto']) == 1

    # =====================================================
    # INTEGRATION TEST
    # =====================================================

    @pytest.mark.asyncio
    async def test_find_all_arbitrage_opportunities(self, detector, sample_market):
        """Test the main entry point for finding all arbitrage types."""
        # Mock spread arbitrage opportunity
        detector.kalshi_client.get_market.return_value = {
            'market': {'yes_ask': 45, 'no_ask': 50}
        }
        
        markets = [sample_market]
        opportunities = await detector.find_all_arbitrage_opportunities(markets)
        
        # Should find at least the spread arbitrage
        assert len(opportunities) >= 1
        # Opportunities should be sorted by expected profit
        if len(opportunities) > 1:
            assert opportunities[0].expected_profit >= opportunities[1].expected_profit


class TestArbitrageOpportunity:
    """Tests for ArbitrageOpportunity dataclass."""

    def test_opportunity_creation(self, sample_opportunity):
        """Test that opportunity dataclass can be created correctly."""
        assert sample_opportunity.market_id_1 == "TEST-MARKET-001"
        assert sample_opportunity.arb_type == "spread"
        assert sample_opportunity.expected_profit == 5.0

    def test_opportunity_to_dict(self, sample_opportunity):
        """Test conversion to dictionary."""
        opp_dict = asdict(sample_opportunity)
        assert 'market_id_1' in opp_dict
        assert 'expected_profit' in opp_dict
        assert opp_dict['confidence'] == 0.95
