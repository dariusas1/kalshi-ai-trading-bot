
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
from src.strategies.market_making import AdvancedMarketMaker, MarketMakingOpportunity, LimitOrder
from src.utils.database import Market

class TestMarketMaker:
    """Verify market making strategy logic"""
    
    @pytest_asyncio.fixture
    async def market_maker(self, db_manager, mock_kalshi):
        mock_xai = AsyncMock()
        mock_xai.get_completion.return_value = '{"probability": 0.6, "confidence": 0.8, "volatility_factors": "Low", "stability": 0.9}'
        return AdvancedMarketMaker(db_manager, mock_kalshi, mock_xai)

    @pytest.mark.asyncio
    async def test_analyze_opportunities_filters_bad_markets(self, market_maker):
        """Should filter markets with extreme prices or no data"""
        # Mock markets
        good_market = Market(
            market_id="GOOD", title="Good Market", 
            subtitle="", ticker="GOOD", volume=1000, 
            yes_bid=40, yes_ask=45, expiration_ts=datetime.now().timestamp() + 86400,
            status="active"
        )
        bad_market = Market(
            market_id="BAD", title="Bad Market", 
            subtitle="", ticker="BAD", volume=1000, 
            yes_bid=99, yes_ask=100, expiration_ts=datetime.now().timestamp() + 86400,
            status="active"
        )
        
        # Mock get_market responses
        market_maker.kalshi_client.get_market = AsyncMock(side_effect=lambda mid: {
            'market': {'yes_price': 50, 'no_price': 50} if mid == "GOOD" else {'yes_price': 99, 'no_price': 1}
        })
        
        # Mock edge filter to pass GOOD market
        with patch('src.utils.edge_filter.EdgeFilter.calculate_edge') as mock_edge:
            mock_edge.return_value = MagicMock(passes_filter=True, edge_percentage=0.1)
            
            opportunities = await market_maker.analyze_market_making_opportunities([good_market, bad_market])
            
            assert len(opportunities) == 1
            assert opportunities[0].market_id == "GOOD"

    @pytest.mark.asyncio
    async def test_spread_calculation_logic(self, market_maker):
        """Should calculate logical spreads based on edge"""
        # Setup
        market = Market(
            market_id="TEST", title="Test", ticker="TEST", status="active",
            yes_bid=50, yes_ask=52, volume=1000, expiration_ts=datetime.now().timestamp()
        )
        yes_price = 0.50
        no_price = 0.50
        ai_prob = 0.60 # Edge on YES
        ai_conf = 0.8
        capital = 1000
        
        opp = await market_maker._calculate_market_making_opportunity(
            market, yes_price, no_price, ai_prob, ai_conf, capital
        )
        
        assert opp is not None
        # AI thinks YES is 0.60, market is 0.50.
        # So we should be willing to buy YES higher than 0.50? No, we are market making.
        # Market Making means we provide liquidity.
        # If we think fair value is 0.60, we might skew our quotes.
        
        # Check that spread is within bounds
        spread = opp.optimal_yes_ask - opp.optimal_yes_bid
        assert spread >= market_maker.min_spread
        assert spread <= market_maker.max_spread * 1.5 # Allow for adjustments
        
        # Check skews: Since AI says 0.60 (YES favored), we might bid higher on YES?
        # Logic in code: if yes_edge > 0 (0.6 - 0.5 = 0.1), 
        # optimal_yes_bid = yes_price + (spread / 2)
        # optimal_yes_ask = yes_price + spread
        # Basically moving the bracket UP to capture the undervalued asset?
        assert opp.optimal_yes_bid > yes_price 

    @pytest.mark.asyncio
    async def test_kelly_size_calculation(self, market_maker):
        """Should calculate sizes using Kelly Criterion"""
        # Test exact inputs
        yes_edge = 0.1
        no_edge = -0.1
        volatility = 0.05
        confidence = 0.8
        capital = 10000
        
        y_size, n_size = await market_maker._calculate_optimal_sizes(
            yes_edge, no_edge, volatility, confidence, capital, 0.5, 0.5
        )
        
        assert y_size > 0
        assert n_size > 0
        # Since YES edge is positive, YES size should be larger (or at least significant)
        # NO edge is negative, so NO size should be small (min size usually)
        assert y_size > n_size

    @pytest.mark.asyncio
    async def test_inventory_skew_adjustment(self, market_maker):
        """Should adjust sizes to balance inventory"""
        # Mock inventory state: Heavy on YES
        market_maker.net_yes_inventory = 1000
        market_maker.net_no_inventory = 0
        market_maker.db_manager.get_active_positions = AsyncMock(return_value=[
            MagicMock(side='YES', quantity=1000, strategy='market_making')
        ])
        
        base_yes = 100
        base_no = 100
        
        adj_yes, adj_no = await market_maker._calculate_skew_adjusted_sizes(base_yes, base_no)
        
        # Should reduce YES buying and increase NO buying
        assert adj_yes < base_yes
        assert adj_no > base_no
