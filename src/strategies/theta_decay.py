"""
Time-Decay (Theta) Exploitation Strategy

This module exploits theta decay on prediction markets near expiry:
- Targets YES-heavy markets (high YES price) close to expiration
- Shorts the NO side to capture time decay on overpriced positions
- Uses aggressive position sizing for near-expiry opportunities

Key insight: Markets with extreme YES prices near expiry often revert
as uncertainty resolves. Betting against the "obvious" outcome captures
the theta decay premium.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import asyncio

from src.clients.kalshi_client import KalshiClient
from src.utils.database import DatabaseManager, Market, Position
from src.utils.logging_setup import get_trading_logger


@dataclass
class ThetaOpportunity:
    """Represents a theta decay trading opportunity."""
    market_id: str
    market_title: str
    yes_price: float
    no_price: float
    hours_to_expiry: float
    
    # Theta analysis
    theta_edge: float  # Expected profit from time decay
    overpriced_side: str  # Which side is overpriced ("YES" or "NO")
    recommended_action: str  # "BUY_NO" (short YES) or "BUY_YES" (short NO)
    
    # Position sizing
    suggested_position_pct: float  # Suggested position as % of capital
    max_loss: float  # Maximum potential loss
    expected_profit: float  # Expected profit if decay occurs
    
    # Risk assessment
    risk_score: float  # 0-1, higher = more risky
    confidence: float  # 0-1, confidence in the opportunity
    
    def __str__(self) -> str:
        return (
            f"ThetaOpportunity({self.market_id}): "
            f"{self.recommended_action} @ {self.hours_to_expiry:.1f}h to expiry, "
            f"theta_edge={self.theta_edge:.1%}, conf={self.confidence:.1%}"
        )


class ThetaDecayStrategy:
    """
    Exploits time decay on prediction markets near expiry.
    
    Strategy logic:
    1. Find markets with extreme YES prices (>75¢) near expiry (<24h)
    2. These markets have high "theta" - time premium baked in
    3. As expiry approaches, prices tend to revert toward 50/50 or true probability
    4. Buy the underpriced side (NO when YES is overpriced) to capture decay
    
    Risk management:
    - Only trade markets with sufficient liquidity
    - Use aggressive but bounded position sizing
    - Stop-loss if price moves against us significantly
    """
    
    def __init__(
        self,
        kalshi_client: KalshiClient,
        db_manager: DatabaseManager,
        # Configuration
        min_yes_price: float = 0.75,  # Minimum YES price to consider
        max_yes_price: float = 0.95,  # Maximum YES price (too extreme)
        max_hours_to_expiry: int = 24,  # Maximum hours before expiry
        min_hours_to_expiry: float = 1.0,  # Minimum hours (avoid settlement issues)
        theta_allocation: float = 0.10,  # 10% of capital for theta trades
        max_position_pct: float = 0.05,  # Max 5% per theta trade
        min_volume: int = 500  # Minimum trading volume
    ):
        self.kalshi_client = kalshi_client
        self.db_manager = db_manager
        self.logger = get_trading_logger("theta_decay")
        
        # Configuration
        self.min_yes_price = min_yes_price
        self.max_yes_price = max_yes_price
        self.max_hours_to_expiry = max_hours_to_expiry
        self.min_hours_to_expiry = min_hours_to_expiry
        self.theta_allocation = theta_allocation
        self.max_position_pct = max_position_pct
        self.min_volume = min_volume
    
    async def find_theta_opportunities(
        self, 
        markets: Optional[List[Market]] = None
    ) -> List[ThetaOpportunity]:
        """
        Find markets with theta decay opportunities.
        
        Args:
            markets: Optional list of markets to analyze. If None, fetches eligible markets.
            
        Returns:
            List of ThetaOpportunity objects sorted by expected profit
        """
        opportunities = []
        
        try:
            # Get markets if not provided
            if markets is None:
                markets = await self.db_manager.get_eligible_markets(
                    volume_min=self.min_volume,
                    max_days_to_expiry=1  # Only markets expiring within 1 day
                )
            
            self.logger.info(f"Scanning {len(markets)} markets for theta opportunities")
            
            for market in markets:
                try:
                    opportunity = await self._analyze_market_for_theta(market)
                    if opportunity:
                        opportunities.append(opportunity)
                        self.logger.info(f"Found theta opportunity: {opportunity}")
                except Exception as e:
                    self.logger.debug(f"Error analyzing {market.market_id}: {e}")
                    continue
            
            # Sort by expected profit (descending)
            opportunities.sort(key=lambda x: x.expected_profit, reverse=True)
            
            self.logger.info(f"Found {len(opportunities)} theta opportunities")
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding theta opportunities: {e}")
            return []
    
    async def _analyze_market_for_theta(self, market: Market) -> Optional[ThetaOpportunity]:
        """
        Analyze a single market for theta decay opportunity.
        """
        try:
            # Get current market data
            market_data = await self.kalshi_client.get_market(market.market_id)
            if not market_data:
                return None
            
            # Extract market info
            market_info = market_data.get('market', market_data)
            yes_price = market_info.get('yes_price', market_info.get('yes_bid', 0))
            no_price = market_info.get('no_price', market_info.get('no_bid', 0))
            
            # Normalize prices (convert from cents if needed)
            if yes_price > 1:
                yes_price = yes_price / 100
            if no_price > 1:
                no_price = no_price / 100
            
            # Calculate hours to expiry
            expiry_ts = market_info.get('expiration_time') or market_info.get('close_time')
            if expiry_ts:
                if isinstance(expiry_ts, str):
                    try:
                        expiry_dt = datetime.fromisoformat(expiry_ts.replace('Z', '+00:00'))
                        hours_to_expiry = (expiry_dt.timestamp() - time.time()) / 3600
                    except:
                        hours_to_expiry = market.expiration_ts - time.time() if hasattr(market, 'expiration_ts') else 48
                        hours_to_expiry = hours_to_expiry / 3600
                else:
                    hours_to_expiry = (expiry_ts - time.time()) / 3600
            elif hasattr(market, 'expiration_ts') and market.expiration_ts:
                hours_to_expiry = (market.expiration_ts - time.time()) / 3600
            else:
                return None  # Can't determine expiry
            
            # Filter by time to expiry
            if hours_to_expiry > self.max_hours_to_expiry or hours_to_expiry < self.min_hours_to_expiry:
                return None
            
            # Check for theta opportunity
            # Case 1: YES is overpriced (high YES price)
            if yes_price >= self.min_yes_price and yes_price <= self.max_yes_price:
                return self._create_theta_opportunity(
                    market=market,
                    market_info=market_info,
                    yes_price=yes_price,
                    no_price=no_price,
                    hours_to_expiry=hours_to_expiry,
                    overpriced_side="YES"
                )
            
            # Case 2: NO is overpriced (low YES price = high NO price)
            if (1 - yes_price) >= self.min_yes_price and (1 - yes_price) <= self.max_yes_price:
                return self._create_theta_opportunity(
                    market=market,
                    market_info=market_info,
                    yes_price=yes_price,
                    no_price=no_price,
                    hours_to_expiry=hours_to_expiry,
                    overpriced_side="NO"
                )
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Error analyzing market {market.market_id}: {e}")
            return None
    
    def _create_theta_opportunity(
        self,
        market: Market,
        market_info: Dict,
        yes_price: float,
        no_price: float,
        hours_to_expiry: float,
        overpriced_side: str
    ) -> ThetaOpportunity:
        """
        Create a ThetaOpportunity from market analysis.
        """
        # Calculate theta metrics
        if overpriced_side == "YES":
            # YES is overpriced, buy NO to capture decay
            entry_price = no_price if no_price > 0 else (1 - yes_price)
            theta_edge = self._calculate_theta_edge(yes_price, hours_to_expiry)
            recommended_action = "BUY_NO"
            max_loss = entry_price  # Lose entry if NO goes to 0
            expected_profit = entry_price * theta_edge  # Profit if decay occurs
        else:
            # NO is overpriced, buy YES to capture decay
            entry_price = yes_price
            theta_edge = self._calculate_theta_edge(1 - yes_price, hours_to_expiry)
            recommended_action = "BUY_YES"
            max_loss = entry_price
            expected_profit = entry_price * theta_edge
        
        # Calculate position sizing
        # More aggressive for higher edge, less time to expiry
        time_factor = 1 - (hours_to_expiry / self.max_hours_to_expiry)  # More aggressive closer to expiry
        edge_factor = min(1.0, theta_edge / 0.10)  # Scale by edge (10% = full size)
        
        suggested_position_pct = self.max_position_pct * time_factor * edge_factor
        suggested_position_pct = max(0.01, min(self.max_position_pct, suggested_position_pct))
        
        # Risk assessment
        # Higher risk if: very close to expiry, extreme prices, low volume
        expiry_risk = max(0, 1 - hours_to_expiry / 6)  # Risk increases in last 6 hours
        price_risk = abs(yes_price - 0.5) / 0.5  # Risk of extreme prices
        volume = market.volume if hasattr(market, 'volume') else 1000
        volume_risk = 1 - min(1.0, volume / 5000)  # Lower volume = higher risk
        
        risk_score = (expiry_risk * 0.4 + price_risk * 0.3 + volume_risk * 0.3)
        
        # Confidence calculation
        confidence = (1 - risk_score) * min(1.0, theta_edge / 0.05)  # Base confidence on edge and risk
        
        return ThetaOpportunity(
            market_id=market.market_id,
            market_title=market.title if hasattr(market, 'title') else market.market_id,
            yes_price=yes_price,
            no_price=no_price,
            hours_to_expiry=hours_to_expiry,
            theta_edge=theta_edge,
            overpriced_side=overpriced_side,
            recommended_action=recommended_action,
            suggested_position_pct=suggested_position_pct,
            max_loss=max_loss,
            expected_profit=expected_profit,
            risk_score=risk_score,
            confidence=confidence
        )
    
    def _calculate_theta_edge(self, high_price: float, hours_to_expiry: float) -> float:
        """
        Calculate the theta decay edge for a market.
        
        The edge is based on:
        1. How overpriced the dominant side is (distance from fair value)
        2. How little time remains (forcing resolution)
        
        Theta decay accelerates as expiry approaches (like options).
        """
        # Estimate "fair" probability as slightly less extreme than current price
        # Markets at 80% YES typically resolve to ~70% or less in reality
        price_premium = high_price - 0.65  # Assume 65% is "fair" for extreme prices
        price_premium = max(0, price_premium)
        
        # Time decay factor - accelerates near expiry
        # Based on square root of time (like options theta)
        if hours_to_expiry > 0:
            time_decay = 1 - np.sqrt(hours_to_expiry / self.max_hours_to_expiry)
        else:
            time_decay = 1.0
        
        # Combined theta edge
        theta_edge = price_premium * time_decay * 0.5  # 50% of premium captured
        
        return min(0.25, theta_edge)  # Cap at 25% edge
    
    async def execute_theta_trades(
        self,
        opportunities: List[ThetaOpportunity],
        available_capital: float
    ) -> Dict:
        """
        Execute theta decay trades for the given opportunities.
        
        Args:
            opportunities: List of theta opportunities to trade
            available_capital: Total capital available for theta trades
            
        Returns:
            Dictionary with execution results
        """
        results = {
            'trades_attempted': 0,
            'trades_executed': 0,
            'total_exposure': 0.0,
            'positions_created': []
        }
        
        theta_budget = available_capital * self.theta_allocation
        remaining_budget = theta_budget
        
        for opp in opportunities:
            if remaining_budget <= 0:
                break
            
            try:
                # Calculate position size
                position_value = min(
                    available_capital * opp.suggested_position_pct,
                    remaining_budget
                )
                
                if position_value < 5:  # Minimum $5 position
                    continue
                
                results['trades_attempted'] += 1
                
                # Determine side and price
                if opp.recommended_action == "BUY_NO":
                    side = "NO"
                    entry_price = opp.no_price
                else:
                    side = "YES"
                    entry_price = opp.yes_price
                
                # Calculate quantity
                quantity = max(1, int(position_value / entry_price))
                
                # Check for existing position
                existing = await self.db_manager.get_position_by_market_and_side(
                    opp.market_id, side
                )
                if existing:
                    self.logger.info(f"Already have position in {opp.market_id} {side}, skipping")
                    continue
                
                # Create position
                position = Position(
                    market_id=opp.market_id,
                    side=side,
                    entry_price=entry_price,
                    quantity=quantity,
                    timestamp=datetime.now(),
                    rationale=f"THETA DECAY: {opp.overpriced_side} overpriced @ {opp.hours_to_expiry:.1f}h to expiry, edge={opp.theta_edge:.1%}",
                    confidence=opp.confidence,
                    live=False,
                    strategy="theta_decay",
                    # Set aggressive exit levels for near-expiry trades
                    stop_loss_price=entry_price * 0.7,  # 30% stop loss
                    take_profit_price=entry_price * 1.3,  # 30% take profit
                    max_hold_hours=int(opp.hours_to_expiry)  # Hold until expiry
                )
                
                position_id = await self.db_manager.add_position(position)
                if position_id:
                    position.id = position_id
                    results['trades_executed'] += 1
                    results['total_exposure'] += quantity * entry_price
                    results['positions_created'].append(position)
                    remaining_budget -= quantity * entry_price
                    
                    self.logger.info(
                        f"✅ Created theta position: {opp.market_id} {side} x{quantity} "
                        f"@ {entry_price:.2f}, edge={opp.theta_edge:.1%}"
                    )
                
            except Exception as e:
                self.logger.error(f"Error executing theta trade for {opp.market_id}: {e}")
                continue
        
        self.logger.info(
            f"Theta execution complete: {results['trades_executed']}/{results['trades_attempted']} trades, "
            f"${results['total_exposure']:.2f} exposure"
        )
        
        return results


async def run_theta_decay_strategy(
    db_manager: DatabaseManager,
    kalshi_client: KalshiClient,
    available_capital: float
) -> Dict:
    """
    Main entry point for running the theta decay strategy.
    
    Args:
        db_manager: Database manager
        kalshi_client: Kalshi API client
        available_capital: Total available capital
        
    Returns:
        Execution results dictionary
    """
    logger = get_trading_logger("theta_decay_main")
    
    try:
        strategy = ThetaDecayStrategy(kalshi_client, db_manager)
        
        # Find opportunities
        opportunities = await strategy.find_theta_opportunities()
        
        if not opportunities:
            logger.info("No theta decay opportunities found")
            return {'trades_executed': 0, 'opportunities_found': 0}
        
        logger.info(f"Found {len(opportunities)} theta opportunities, executing trades...")
        
        # Execute trades
        results = await strategy.execute_theta_trades(opportunities, available_capital)
        results['opportunities_found'] = len(opportunities)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in theta decay strategy: {e}")
        return {'error': str(e)}
