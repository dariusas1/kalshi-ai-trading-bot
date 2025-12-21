
import pytest
import pytest_asyncio
import asyncio
import os
from unittest.mock import MagicMock
from tests.mocks.mock_kalshi import MockKalshiClient
from src.utils.database import DatabaseManager

@pytest.fixture
def mock_kalshi():
    return MockKalshiClient()

@pytest_asyncio.fixture
async def db_manager(tmp_path):
    # Use temp file DB for tests to persist across connections
    db_file = tmp_path / "test_trading.db"
    manager = DatabaseManager(db_path=str(db_file))
    await manager.initialize()
    return manager

@pytest.fixture
def mock_db_manager():
    manager = MagicMock(spec=DatabaseManager)
    manager.get_open_positions.return_value = []
    return manager

@pytest.fixture
async def initialized_db(db_manager):
    # Initialize implementation if needed
    # await db_manager.initialize() 
    return db_manager
