"""
Cross-Market Arbitrage Detector

Implements arbitrage detection for correlated Kalshi markets:
1. Multi-outcome markets: Same event where YES/NO don't sum to 100%
2. Correlated markets: Related events with price discrepancies
3. Temporal arbitrage: Same event, different expiration dates

Key strategies:
- If YES + NO < 100¢, buy both sides for guaranteed profit
- If related markets show significant price divergence, trade the spread
- Identify mispriced outcomes based on event correlation
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re

from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager, Market, Position
from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger


@dataclass
class ArbitrageOpportunity:
    """Represents an arbitrage opportunity between markets."""
    market_id_1: str
    market_id_2: str
    market_title_1: str
    market_title_2: str
    
    # Trade details
    side_1: str  # "YES" or "NO"
    side_2: str  # "YES" or "NO"
    price_1: float  # Entry price for first leg
    price_2: float  # Entry price for second leg
    
    # Profit calculation
    spread: float  # Price difference (arbitrage profit %)
    expected_profit: float  # Expected profit in dollars per contract
    confidence: float  # Confidence in arbitrage opportunity (0-1)
    
    # Type of arbitrage
    arb_type: str  # "spread", "mispricing", or "temporal"
    
    # Position sizing
    quantity: int  # Contracts to trade
    total_cost: float  # Total capital required


class ArbitrageDetector:
    """
    Detects and executes arbitrage opportunities across Kalshi markets.
    
    Arbitrage Types:
    1. Spread Arbitrage: YES + NO < 100% (guaranteed profit)
    2. Correlated Market Arbitrage: Related events with price divergence
    3. Temporal Arbitrage: Same event at different timeframes
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        kalshi_client: KalshiClient
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.logger = get_trading_logger("arbitrage_detector")
        
        # Configuration
        self.min_spread_profit = 0.02  # Minimum 2% spread to trigger
        self.min_correlation_divergence = 0.10  # 10% price divergence for correlated
        self.max_capital_per_arb = 100.0  # Max capital per arbitrage trade
        self.fee_rate = 0.01  # Kalshi fee rate (estimated)
        
    async def find_all_arbitrage_opportunities(
        self, 
        markets: List[Market]
    ) -> List[ArbitrageOpportunity]:
        """
        Find all types of arbitrage opportunities across markets.
        """
        opportunities = []
        
        try:
            # 1. Find spread arbitrage (YES + NO < 100%)
            spread_opps = await self.find_spread_arbitrage(markets)
            opportunities.extend(spread_opps)
            
            # 2. Find correlated market arbitrage
            correlated_opps = []
            if settings.trading.cross_market_arbitrage:
                correlated_opps = await self.find_correlated_arbitrage(markets)
                opportunities.extend(correlated_opps)
            else:
                self.logger.info("Cross-market arbitrage disabled; skipping correlated arbitrage scan")
            
            # Sort by expected profit
            opportunities.sort(key=lambda x: x.expected_profit, reverse=True)
            
            self.logger.info(
                f"🎯 Found {len(opportunities)} arbitrage opportunities: "
                f"{len(spread_opps)} spread, {len(correlated_opps)} correlated"
            )
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding arbitrage opportunities: {e}")
            return []
    
    async def find_spread_arbitrage(
        self, 
        markets: List[Market]
    ) -> List[ArbitrageOpportunity]:
        """
        Find markets where YES + NO prices < 100 cents.
        This is a guaranteed profit opportunity.
        """
        opportunities = []
        
        for market in markets:
            try:
                # Get current orderbook
                market_data = await self.kalshi_client.get_market(market.market_id)
                if not market_data:
                    continue
                
                market_info = market_data.get('market', {})
                
                # Get best ask prices (price to BUY)
                yes_ask = market_info.get('yes_ask', 0) / 100  # Convert to dollars
                no_ask = market_info.get('no_ask', 0) / 100
                
                if yes_ask <= 0 or no_ask <= 0:
                    continue
                
                # Calculate total cost to buy both sides
                total_cost = yes_ask + no_ask
                
                # Check for spread arbitrage (cost < $1 = guaranteed profit)
                if total_cost < (1.0 - self.min_spread_profit):
                    profit_per_contract = 1.0 - total_cost
                    spread = profit_per_contract  # Guaranteed profit
                    
                    # Account for fees
                    net_profit = profit_per_contract - (2 * self.fee_rate)
                    
                    if net_profit > 0:
                        # Calculate position size
                        quantity = min(
                            100,  # Max contracts
                            int(self.max_capital_per_arb / total_cost)
                        )
                        
                        opportunity = ArbitrageOpportunity(
                            market_id_1=market.market_id,
                            market_id_2=market.market_id,  # Same market, both sides
                            market_title_1=market.title,
                            market_title_2=market.title,
                            side_1="YES",
                            side_2="NO",
                            price_1=yes_ask,
                            price_2=no_ask,
                            spread=spread,
                            expected_profit=net_profit * quantity,
                            confidence=0.95,  # High confidence for pure arbitrage
                            arb_type="spread",
                            quantity=quantity,
                            total_cost=total_cost * quantity
                        )
                        
                        opportunities.append(opportunity)
                        
                        self.logger.info(
                            f"💰 SPREAD ARBITRAGE: {market.market_id} - "
                            f"YES@{yes_ask:.2f} + NO@{no_ask:.2f} = {total_cost:.2f} "
                            f"(profit: ${net_profit * quantity:.2f})"
                        )
                        
            except Exception as e:
                self.logger.debug(f"Error checking spread for {market.market_id}: {e}")
                continue
        
        return opportunities
    
    async def find_correlated_arbitrage(
        self, 
        markets: List[Market]
    ) -> List[ArbitrageOpportunity]:
        """
        Find correlated markets with price divergence.
        
        Examples:
        - "Will it snow in NYC in December?" vs "Will NYC have a white Christmas?"
        - Same event with different strike prices
        - Related political/economic events
        """
        opportunities = []
        
        # Group markets by event series (similar topics)
        market_groups = self._group_related_markets(markets)
        
        for group_name, group_markets in market_groups.items():
            if len(group_markets) < 2:
                continue
                
            try:
                # Compare prices within group
                for i, market1 in enumerate(group_markets):
                    for market2 in group_markets[i+1:]:
                        opp = await self._compare_correlated_markets(market1, market2)
                        if opp:
                            opportunities.append(opp)
                            
            except Exception as e:
                self.logger.debug(f"Error in correlated arbitrage for {group_name}: {e}")
                continue
        
        return opportunities
    
    def _group_related_markets(
        self, 
        markets: List[Market]
    ) -> Dict[str, List[Market]]:
        """
        Group markets by similar topics/events for correlation analysis.
        """
        groups = {}
        
        for market in markets:
            # Extract key terms from market title
            title_lower = market.title.lower()
            
            # Group by common patterns
            group_key = None
            
            # Political events
            if 'trump' in title_lower or 'biden' in title_lower or 'election' in title_lower:
                group_key = 'politics_us'
            elif 'fed' in title_lower or 'interest rate' in title_lower:
                group_key = 'fed_rates'
            elif 'bitcoin' in title_lower or 'btc' in title_lower or 'crypto' in title_lower:
                group_key = 'crypto'
            elif 'sp500' in title_lower or 's&p' in title_lower or 'stock' in title_lower:
                group_key = 'stocks'
            elif 'weather' in title_lower or 'temperature' in title_lower:
                group_key = 'weather'
            elif 'nfl' in title_lower or 'nba' in title_lower or 'sports' in title_lower:
                group_key = 'sports'
            
            # Also group by series_ticker if available
            if hasattr(market, 'series_ticker') and market.series_ticker:
                group_key = f"series_{market.series_ticker}"
            
            if group_key:
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(market)
        
        return groups
    
    async def _compare_correlated_markets(
        self,
        market1: Market,
        market2: Market
    ) -> Optional[ArbitrageOpportunity]:
        """
        Compare two correlated markets for arbitrage opportunity.
        """
        try:
            # Get current prices
            data1 = await self.kalshi_client.get_market(market1.market_id)
            data2 = await self.kalshi_client.get_market(market2.market_id)
            
            if not data1 or not data2:
                return None
            
            info1 = data1.get('market', {})
            info2 = data2.get('market', {})
            
            yes_price_1 = info1.get('yes_ask', 50) / 100
            yes_price_2 = info2.get('yes_ask', 50) / 100
            
            # Calculate price divergence
            price_diff = abs(yes_price_1 - yes_price_2)
            
            # Check if divergence is significant
            if price_diff >= self.min_correlation_divergence:
                # Determine trade direction (buy cheaper, potentially sell more expensive)
                if yes_price_1 < yes_price_2:
                    buy_market = market1
                    buy_price = yes_price_1
                    sell_market = market2
                    sell_price = yes_price_2
                else:
                    buy_market = market2
                    buy_price = yes_price_2
                    sell_market = market1
                    sell_price = yes_price_1
                
                # Calculate expected profit (accounting for correlation risk)
                expected_edge = price_diff * 0.5  # Conservative estimate
                quantity = min(50, int(self.max_capital_per_arb / buy_price))
                expected_profit = expected_edge * quantity
                
                # Only proceed if profit exceeds fees
                if expected_profit > (2 * self.fee_rate * quantity):
                    return ArbitrageOpportunity(
                        market_id_1=buy_market.market_id,
                        market_id_2=sell_market.market_id,
                        market_title_1=buy_market.title,
                        market_title_2=sell_market.title,
                        side_1="YES",
                        side_2="NO",  # Hedge with NO on expensive market
                        price_1=buy_price,
                        price_2=1.0 - sell_price,  # NO price
                        spread=price_diff,
                        expected_profit=expected_profit,
                        confidence=0.6,  # Lower confidence for correlation arb
                        arb_type="correlated",
                        quantity=quantity,
                        total_cost=buy_price * quantity + (1.0 - sell_price) * quantity
                    )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error comparing markets: {e}")
            return None
    
    async def execute_arbitrage_trade(
        self,
        opportunity: ArbitrageOpportunity,
        live_mode: bool = False
    ) -> Dict:
        """
        Execute an arbitrage trade (both legs).
        """
        results = {
            'success': False,
            'leg1_executed': False,
            'leg2_executed': False,
            'total_cost': 0.0,
            'expected_profit': opportunity.expected_profit
        }
        
        try:
            import uuid
            
            if not live_mode:
                self.logger.info(
                    f"📝 SIMULATED ARB: {opportunity.arb_type} - "
                    f"Buy {opportunity.side_1} {opportunity.market_id_1} @ {opportunity.price_1:.2f}, "
                    f"Buy {opportunity.side_2} {opportunity.market_id_2} @ {opportunity.price_2:.2f}"
                )
                results['success'] = True
                results['leg1_executed'] = True
                results['leg2_executed'] = True
                return results
            
            buffer_cents = getattr(settings.trading, "market_order_price_buffer_cents", 2)

            # Execute leg 1 (IOC to avoid partial exposure)
            order1_params = {
                "ticker": opportunity.market_id_1,
                "client_order_id": str(uuid.uuid4()),
                "side": opportunity.side_1.lower(),
                "action": "buy",
                "count": opportunity.quantity,
                "type_": "market",
                "time_in_force": "immediate_or_cancel"
            }
            if opportunity.side_1.lower() == "yes":
                order1_params["yes_price"] = min(99, int(opportunity.price_1 * 100 + buffer_cents))
            else:
                order1_params["no_price"] = min(99, int(opportunity.price_1 * 100 + buffer_cents))
            
            response1 = await self.kalshi_client.place_order(**order1_params)
            
            order1 = response1.get('order', {}) if response1 else {}
            order1_status = order1.get('status')
            if order1_status in ['filled', 'executed']:
                results['leg1_executed'] = True
                results['total_cost'] += opportunity.price_1 * opportunity.quantity
                
                self.logger.info(
                    f"✅ ARB LEG 1 EXECUTED: {opportunity.side_1} {opportunity.quantity} "
                    f"{opportunity.market_id_1} @ {opportunity.price_1:.2f}"
                )
            else:
                self.logger.error(f"❌ ARB LEG 1 FAILED: {response1}")
                return results
            
            # Execute leg 2 only if leg 1 succeeded (for spread arb)
            order2_params = {
                "ticker": opportunity.market_id_2,
                "client_order_id": str(uuid.uuid4()),
                "side": opportunity.side_2.lower(),
                "action": "buy",
                "count": opportunity.quantity,
                "type_": "market",
                "time_in_force": "immediate_or_cancel"
            }
            if opportunity.side_2.lower() == "yes":
                order2_params["yes_price"] = min(99, int(opportunity.price_2 * 100 + buffer_cents))
            else:
                order2_params["no_price"] = min(99, int(opportunity.price_2 * 100 + buffer_cents))
            
            response2 = await self.kalshi_client.place_order(**order2_params)
            
            order2 = response2.get('order', {}) if response2 else {}
            order2_status = order2.get('status')
            if order2_status in ['filled', 'executed']:
                results['leg2_executed'] = True
                results['total_cost'] += opportunity.price_2 * opportunity.quantity
                results['success'] = True
                
                self.logger.info(
                    f"✅ ARB LEG 2 EXECUTED: {opportunity.side_2} {opportunity.quantity} "
                    f"{opportunity.market_id_2} @ {opportunity.price_2:.2f}"
                )
                self.logger.info(
                    f"💰 ARBITRAGE COMPLETE: Expected profit ${opportunity.expected_profit:.2f}"
                )
            else:
                self.logger.error(f"❌ ARB LEG 2 FAILED: {response2}")
                await self._attempt_arb_hedge(opportunity, buffer_cents)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error executing arbitrage trade: {e}")
            return results

    async def _attempt_arb_hedge(self, opportunity: ArbitrageOpportunity, buffer_cents: int) -> None:
        """Attempt to neutralize exposure if the second leg fails."""
        try:
            import uuid

            hedge_action = {
                "ticker": opportunity.market_id_1,
                "client_order_id": str(uuid.uuid4()),
                "side": opportunity.side_1.lower(),
                "action": "sell",
                "count": opportunity.quantity,
                "type_": "market",
                "time_in_force": "immediate_or_cancel"
            }
            if opportunity.side_1.lower() == "yes":
                hedge_action["yes_price"] = max(1, int(opportunity.price_1 * 100 - buffer_cents))
            else:
                hedge_action["no_price"] = max(1, int(opportunity.price_1 * 100 - buffer_cents))

            hedge_response = await self.kalshi_client.place_order(**hedge_action)
            hedge_order = hedge_response.get('order', {}) if hedge_response else {}
            hedge_status = hedge_order.get('status')
            if hedge_status in ['filled', 'executed']:
                self.logger.warning(
                    "⚠️ Hedge executed to neutralize failed arbitrage leg",
                    market_id=opportunity.market_id_1,
                    side=opportunity.side_1,
                    quantity=opportunity.quantity
                )
            else:
                self.logger.error(
                    "❌ Hedge failed; manual review required",
                    market_id=opportunity.market_id_1,
                    side=opportunity.side_1,
                    quantity=opportunity.quantity,
                    response=hedge_response
                )
        except Exception as e:
            self.logger.error(f"Error attempting arbitrage hedge: {e}")
    
    def get_arbitrage_summary(
        self, 
        opportunities: List[ArbitrageOpportunity]
    ) -> Dict:
        """
        Get summary statistics for arbitrage opportunities.
        """
        if not opportunities:
            return {
                'total_opportunities': 0,
                'spread_count': 0,
                'correlated_count': 0,
                'total_expected_profit': 0.0,
                'total_capital_required': 0.0
            }
        
        spread_opps = [o for o in opportunities if o.arb_type == 'spread']
        correlated_opps = [o for o in opportunities if o.arb_type == 'correlated']
        
        return {
            'total_opportunities': len(opportunities),
            'spread_count': len(spread_opps),
            'correlated_count': len(correlated_opps),
            'total_expected_profit': sum(o.expected_profit for o in opportunities),
            'total_capital_required': sum(o.total_cost for o in opportunities),
            'avg_confidence': sum(o.confidence for o in opportunities) / len(opportunities),
            'best_opportunity': opportunities[0] if opportunities else None
        }
