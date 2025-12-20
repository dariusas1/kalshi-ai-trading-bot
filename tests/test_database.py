import asyncio
import json
import os
import pytest
import aiosqlite
from datetime import datetime, timedelta
from typing import List

from src.utils.database import DatabaseManager, Market

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

TEST_DB = "test_trading_system.db"
FIXTURE_PATH = "tests/fixtures/markets.json"


def load_and_prepare_markets(fixture_path: str) -> List[Market]:
    """Loads markets from a fixture and processes dynamic timestamps."""
    with open(fixture_path, 'r') as f:
        raw_markets = json.load(f)

    processed_markets = []
    now = datetime.now()
    for m in raw_markets:
        # Handle dynamic expiration timestamps like "NOW+5D"
        if isinstance(m["expiration_ts"], str) and "NOW+" in m["expiration_ts"]:
            days_to_add = int(m["expiration_ts"].split('+')[1].replace('D', ''))
            m["expiration_ts"] = int((now + timedelta(days=days_to_add)).timestamp())
        
        m["last_updated"] = datetime.now()
        processed_markets.append(Market(**m))
    return processed_markets


async def test_get_eligible_markets():
    """
    Test that get_eligible_markets correctly filters markets based on criteria.
    """
    db_path = TEST_DB
    if os.path.exists(db_path):
        os.remove(db_path)
    
    manager = DatabaseManager(db_path=db_path)
    await manager.initialize()
    
    markets = load_and_prepare_markets(FIXTURE_PATH)
    await manager.upsert_markets(markets)

    try:
        # Define filter criteria that match the "ELIGIBLE" markets in our fixture
        volume_min = 5000
        max_days_to_expiry = 7

        # Fetch eligible markets
        eligible_markets = await manager.get_eligible_markets(
            volume_min=volume_min,
            max_days_to_expiry=max_days_to_expiry
        )

        # Assertions
        assert len(eligible_markets) == 2, "Should find exactly two eligible markets"
        
        eligible_ids = {market.market_id for market in eligible_markets}
        assert "ELIGIBLE-1" in eligible_ids
        assert "ELIGIBLE-2-EDGE-CASE" in eligible_ids
        
        # Check that ineligible markets are not present
        assert "INELIGIBLE-LOW-VOLUME" not in eligible_ids
        assert "INELIGIBLE-LONG-EXPIRY" not in eligible_ids
        assert "INELIGIBLE-HAS-POSITION" not in eligible_ids
        assert "INELIGIBLE-CLOSED" not in eligible_ids
    finally:
        # Manual teardown
        if os.path.exists(db_path):
            os.remove(db_path)


async def test_update_position_status_resets_has_position():
    db_path = TEST_DB
    if os.path.exists(db_path):
        os.remove(db_path)

    manager = DatabaseManager(db_path=db_path)
    await manager.initialize()

    now = datetime.now()
    market = Market(
        market_id="TEST-MARKET-1",
        title="Test Market",
        yes_price=0.5,
        no_price=0.5,
        volume=1000,
        expiration_ts=int((now + timedelta(days=5)).timestamp()),
        category="Test",
        status="active",
        last_updated=now,
        has_position=False
    )
    await manager.upsert_markets([market])

    from src.utils.database import Position
    position = Position(
        market_id="TEST-MARKET-1",
        side="YES",
        entry_price=0.5,
        quantity=10,
        timestamp=now,
        strategy="directional_trading"
    )

    position_id = await manager.add_position(position)
    assert position_id is not None

    await manager.update_position_status(position_id, "closed")

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "SELECT has_position FROM markets WHERE market_id = ?",
            ("TEST-MARKET-1",)
        )
        row = await cursor.fetchone()
        assert row[0] == 0

    if os.path.exists(db_path):
        os.remove(db_path)


async def test_get_open_positions_includes_strategy():
    db_path = TEST_DB
    if os.path.exists(db_path):
        os.remove(db_path)

    manager = DatabaseManager(db_path=db_path)
    await manager.initialize()

    now = datetime.now()
    market = Market(
        market_id="TEST-MARKET-2",
        title="Test Market 2",
        yes_price=0.5,
        no_price=0.5,
        volume=1000,
        expiration_ts=int((now + timedelta(days=5)).timestamp()),
        category="Test",
        status="active",
        last_updated=now,
        has_position=False
    )
    await manager.upsert_markets([market])

    from src.utils.database import Position
    position = Position(
        market_id="TEST-MARKET-2",
        side="NO",
        entry_price=0.4,
        quantity=5,
        timestamp=now,
        strategy="quick_flip_scalping"
    )
    position_id = await manager.add_position(position)
    assert position_id is not None

    open_positions = await manager.get_open_positions()
    assert open_positions
    assert open_positions[0].strategy == "quick_flip_scalping"

    if os.path.exists(db_path):
        os.remove(db_path)
