"""
Market Ingestion Job

This job fetches active markets from the Kalshi API, transforms them into a structured format,
and upserts them into the database.
"""
import asyncio
import time
from datetime import datetime
from typing import Optional, List

from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager, Market
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


async def process_and_queue_markets(
    markets_data: List[dict],
    db_manager: DatabaseManager,
    queue: asyncio.Queue,
    existing_position_market_ids: set,
    logger,
):
    """
    Transforms market data, upserts to DB, and puts eligible markets on the queue.
    
    NOTE: KXMV prefix is now standard for all Kalshi sports markets (NFL, NBA, etc.)
    We no longer filter by ticker prefix. Instead, we rely on:
    - status='active' filter from the API
    - Category exclusions in settings
    """
    markets_to_upsert = []
    
    for market_data in markets_data:
        ticker = market_data.get("ticker", "")

        # A simple approach is to take the average of bid and ask.
        yes_price = (market_data.get("yes_bid", 0) + market_data.get("yes_ask", 0)) / 2
        no_price = (market_data.get("no_bid", 0) + market_data.get("no_ask", 0)) / 2

        volume = int(market_data.get("volume", 0))

        has_position = market_data["ticker"] in existing_position_market_ids

        market = Market(
            market_id=market_data["ticker"],
            title=market_data["title"],
            yes_price=yes_price / 100,
            no_price=no_price / 100,
            volume=volume,
            expiration_ts=int(
                datetime.fromisoformat(
                    market_data["expiration_time"].replace("Z", "+00:00")
                ).timestamp()
            ),
            category=market_data["category"],
            status=market_data["status"],
            last_updated=datetime.now(),
            has_position=has_position,
        )
        markets_to_upsert.append(market)

    if markets_to_upsert:
        await db_manager.upsert_markets(markets_to_upsert)
        logger.info(f"Successfully upserted {len(markets_to_upsert)} markets.")

        # 🎯 CATEGORY-SPECIFIC PARAMETERS
        from src.strategies.category_handlers import get_category_handler
        category_handler = get_category_handler()

        # Primary filtering criteria
        min_volume: float = settings.trading.min_volume_for_analysis  # Baseline
        
        eligible_markets = []
        for m in markets_to_upsert:
            # Get category-specific configuration
            config = category_handler.get_category_config(m.category)
            
            # Adjust volume threshold based on category position size multiplier
            # Smaller positions (riskier categories) need higher volume for safety
            adjusted_min_volume = min_volume / config.position_size_multiplier if config.position_size_multiplier > 0 else min_volume
            
            if m.volume >= adjusted_min_volume:
                # Check category filters from settings
                if (
                    (not settings.trading.preferred_categories or m.category in settings.trading.preferred_categories)
                    and (m.category not in settings.trading.excluded_categories)
                    and (m.category not in settings.trading.category_blacklist)
                ):
                    eligible_markets.append(m)

        logger.info(
            f"Found {len(eligible_markets)} eligible markets to process in this batch (using category-specific thresholds)."
        )
        for market in eligible_markets:
            await queue.put(market)

    else:
        logger.debug("No markets to upsert in this batch (empty after processing)")


async def run_ingestion(
    db_manager: DatabaseManager,
    queue: asyncio.Queue,
    market_ticker: Optional[str] = None,
):
    """
    Main function for the market ingestion job.

    Args:
        db_manager: DatabaseManager instance.
        queue: asyncio.Queue to put ingested markets into.
        market_ticker: Optional specific market ticker to ingest.
    """
    logger = get_trading_logger("market_ingestion")
    logger.info("Starting market ingestion job.", market_ticker=market_ticker)

    kalshi_client = KalshiClient()

    try:
        # Get all market IDs with existing positions
        existing_position_market_ids = await db_manager.get_markets_with_positions()

        if market_ticker:
            logger.info(f"Fetching single market: {market_ticker}")
            market_response = await kalshi_client.get_market(ticker=market_ticker)
            if market_response and "market" in market_response:
                await process_and_queue_markets(
                    [market_response["market"]],
                    db_manager,
                    queue,
                    existing_position_market_ids,
                    logger,
                )
            else:
                logger.warning(f"Could not find market with ticker: {market_ticker}")
        else:
            logger.info("Fetching all active markets from Kalshi API with pagination.")
            
            # 🧹 CRITICAL: Clean up stale/finalized markets BEFORE ingestion
            # This ensures we don't analyze expired boxing markets from yesterday
            stale_count = await db_manager.cleanup_stale_markets(max_age_hours=6)
            if stale_count > 0:
                logger.info(f"🧹 Cleaned up {stale_count} stale markets before ingestion")
            
            cursor = None
            while True:
                response = await kalshi_client.get_markets(limit=100, cursor=cursor, status="open")
                markets_page = response.get("markets", [])

                active_markets = markets_page # API filters for open markets
                
                if active_markets:
                    logger.info(
                        f"Fetched {len(markets_page)} markets, {len(active_markets)} are open/active."
                    )
                    await process_and_queue_markets(
                        active_markets,
                        db_manager,
                        queue,
                        existing_position_market_ids,
                        logger,
                    )

                cursor = response.get("cursor")
                if not cursor:
                    break

    except Exception as e:
        logger.error(
            "An error occurred during market ingestion.", error=str(e), exc_info=True
        )
    finally:
        await kalshi_client.close()
        logger.info("Market ingestion job finished.")
