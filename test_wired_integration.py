import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock
import os

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.strategies.unified_trading_system import UnifiedAdvancedTradingSystem, TradingSystemConfig
from src.intelligence.enhanced_client import EnhancedAIClient
from src.intelligence.ensemble_coordinator import EnsembleCoordinator
from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager
from src.clients.xai_client import XAIClient

class TestWiredIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_unified_system_initialization(self):
        print("\nTesting UnifiedAdvancedTradingSystem initialization with EnhancedAIClient...")
        
        # Mocks
        mock_db = MagicMock(spec=DatabaseManager)
        mock_db.get_setting = AsyncMock(return_value="off")
        mock_kalshi = MagicMock(spec=KalshiClient)
        mock_kalshi.get_balance = AsyncMock(return_value={"balance": 100000})
        mock_kalshi.get_positions = AsyncMock(return_value={"positions": []})
        
        # Real/Mocked enhanced client
        enhanced_client = EnhancedAIClient(db_manager=mock_db, kalshi_client=mock_kalshi)
        # Mock the internal clients to prevent network calls
        enhanced_client.xai_client = MagicMock(spec=XAIClient)
        enhanced_client.openai_client = MagicMock()
        
        # Ensemble Coordinator
        ensemble_coordinator = EnsembleCoordinator(db_manager=mock_db)
        
        # Initialize Unified System
        system = UnifiedAdvancedTradingSystem(
            db_manager=mock_db,
            kalshi_client=mock_kalshi,
            xai_client=enhanced_client,
            ensemble_coordinator=ensemble_coordinator
        )
        
        # Verify wiring
        self.assertIsInstance(system.xai_client, EnhancedAIClient)
        self.assertIsInstance(system.ensemble_coordinator, EnsembleCoordinator)
        print("✅ UnifiedAdvancedTradingSystem accepted EnhancedAIClient and EnsembleCoordinator")
        
        # Run async_initialize
        await system.async_initialize()
        print("✅ async_initialize completed")

if __name__ == "__main__":
    unittest.main()
