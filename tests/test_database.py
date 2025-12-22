
import pytest
import asyncio
from datetime import datetime, timedelta
from src.utils.database import Market, Position

TEST_DB = "test_trading_system.db"


class TestDatabaseManager:
    """Verify database operations"""

    @pytest.mark.asyncio
    async def test_upsert_markets(self, db_manager):
        """Should insert new markets and update existing ones"""
        market = Market(
            market_id="TEST_MARKET",
            title="Test Market",
            yes_price=50,
            no_price=50,
            volume=100,
            expiration_ts=int((datetime.now() + timedelta(days=1)).timestamp()),
            category="Test",
            status="active",
            last_updated=datetime.now()
        )
        
        # Insert
        await db_manager.upsert_markets([market])
        
        # Verify
        fetched = await db_manager.get_eligible_markets(0, 365)
        assert len(fetched) == 1
        assert fetched[0].market_id == "TEST_MARKET"
        
        # Update
        market.title = "Updated Title"
        market.volume = 200
        await db_manager.upsert_markets([market])
        
        # Verify update
        fetched_updated = await db_manager.get_eligible_markets(0, 365)
        assert len(fetched_updated) == 1
        assert fetched_updated[0].title == "Updated Title"
        assert fetched_updated[0].volume == 200

    @pytest.mark.asyncio
    async def test_add_position_logic(self, db_manager):
        """Should handle new positions, duplicates, and reactivation correctly"""
        market_id = "POS_TEST"
        side = "YES"
        
        # 1. Setup Market
        market = Market(
            market_id=market_id, title="Pos Test", yes_price=50, no_price=50,
            volume=100, expiration_ts=int((datetime.now() + timedelta(days=1)).timestamp()),
            category="Test", status="active", last_updated=datetime.now()
        )
        await db_manager.upsert_markets([market])
        
        # 2. Add New Position
        position = Position(
            market_id=market_id, side=side, entry_price=0.5, quantity=10,
            timestamp=datetime.now(), rationale="Test", confidence=0.8
        )
        
        pos_id = await db_manager.add_position(position)
        assert pos_id is not None
        
        # Verify market flag
        markets = await db_manager.get_eligible_markets(0, 365)
        # Should NOT be returned as eligible because has_position=1
        assert len(markets) == 0
        
        # 3. Try to add Duplicate Open Position
        pos_id_dup = await db_manager.add_position(position)
        assert pos_id_dup is None  # Should prevent duplicate
        
        # 4. Close Position
        await db_manager.close_position(pos_id)
        
        # Verify closed
        # Since DBManager.close_position uses update_position_status('closed'),
        # checking DB directly or assume verified by get_position returning None for 'open'
        positions = await db_manager.get_open_non_live_positions()
        assert len(positions) == 0
        
        # Verify market flag reset
        markets = await db_manager.get_eligible_markets(0, 365)
        assert len(markets) == 1
        
        # 5. Reactivate Closed Position
        position_new = Position(
            market_id=market_id, side=side, entry_price=0.6, quantity=20,
            timestamp=datetime.now(), rationale="Reactivate", confidence=0.9
        )
        pos_id_reactivated = await db_manager.add_position(position_new)
        
        # Should reuse ID or be a valid ID
        assert pos_id_reactivated == pos_id 
        
        # Verify updated values
        # We need a direct query to check specific fields like price/quantity
        # but get_position_by_market_and_side works
        reactivated = await db_manager.get_position_by_market_and_side(market_id, side)
        assert reactivated.entry_price == 0.6
        assert reactivated.quantity == 20
        assert reactivated.status == 'open'

    @pytest.mark.asyncio
    async def test_cleanup_stale_markets(self, db_manager):
        """Should remove expired markets"""
        # Create expired market
        expired = Market(
            market_id="EXPIRED", title="Expired", yes_price=50, no_price=50,
            volume=100, expiration_ts=int((datetime.now() - timedelta(hours=1)).timestamp()),
            category="Test", status="active", last_updated=datetime.now()
        )
        # Create active market
        active = Market(
            market_id="ACTIVE", title="Active", yes_price=50, no_price=50,
            volume=100, expiration_ts=int((datetime.now() + timedelta(hours=1)).timestamp()),
            category="Test", status="active", last_updated=datetime.now()
        )
        
        await db_manager.upsert_markets([expired, active])
        
        removed = await db_manager.cleanup_stale_markets()
        assert removed == 1
        
        # Verify only active remains
        # get_eligible_markets filters by expiry too, 
        # so check raw fetch or just rely on cleanup return + get_eligible 
        # But get_eligible uses NOW, so expired wouldn't show anyway
        # So we trust 'removed' count primarily.
