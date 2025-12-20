"""
Risk Manager Module

Implements automatic position sizing reduction and portfolio rebalancing
to manage risk when violations are detected.

Features:
- Position sizing reduction when volatility/drawdown/correlation exceed limits
- Portfolio rebalancing to align actual vs target strategy allocations
- Partial position selling to reduce exposure without fully closing
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import uuid

from src.utils.database import DatabaseManager, Position
from src.clients.kalshi_client import KalshiClient
from src.utils.logging_setup import get_trading_logger


@dataclass
class ReductionResult:
    """Result of position sizing reduction."""
    positions_reduced: int
    capital_freed: float
    positions_fully_closed: int
    positions_partially_reduced: int
    details: List[str]


@dataclass
class RebalanceResult:
    """Result of portfolio rebalancing."""
    rebalanced: bool
    summary: str
    strategy_adjustments: Dict[str, float]
    positions_sold: int
    capital_reallocated: float


class RiskManager:
    """
    Manages portfolio risk through position sizing reduction and rebalancing.
    
    Implements:
    - Automatic position sizing reduction when risk limits are exceeded
    - Portfolio rebalancing to maintain target strategy allocations
    - Partial position selling to reduce exposure proportionally
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager, 
        kalshi_client: KalshiClient,
        config: Any = None
    ):
        self.db_manager = db_manager
        self.kalshi_client = kalshi_client
        self.config = config
        self.logger = get_trading_logger("risk_manager")
        
        # Default configuration if not provided
        self.max_reduction_per_cycle = 0.30  # Max 30% reduction per position per cycle
        self.min_position_value = 5.0  # Don't reduce below $5
        self.rebalance_threshold = 0.10  # 10% drift triggers rebalancing
        
    async def reduce_position_sizing(
        self, 
        results: Any,
        violations: List[str]
    ) -> Dict[str, Any]:
        """
        Reduce position sizes when risk violations are detected.
        
        Risk Priority:
        1. Volatility violation → reduce largest/most volatile positions
        2. Drawdown violation → reduce positions with highest loss potential
        3. Correlation violation → reduce most correlated positions
        
        Args:
            results: TradingSystemResults with current portfolio metrics
            violations: List of violation description strings
            
        Returns:
            Dictionary with reduction results
        """
        result = {
            'positions_reduced': 0,
            'positions_fully_closed': 0,
            'capital_freed': 0.0,
            'details': []
        }
        
        try:
            # Parse violation types
            has_volatility_violation = any('vol' in v.lower() for v in violations)
            has_drawdown_violation = any('drawdown' in v.lower() for v in violations)
            has_correlation_violation = any('correlation' in v.lower() for v in violations)
            
            # Get current open positions
            positions = await self.db_manager.get_open_positions()
            
            if not positions:
                self.logger.info("No positions to reduce")
                return result
            
            # Calculate reduction severity based on violation count and severity
            reduction_factor = self._calculate_reduction_factor(
                results, violations,
                has_volatility_violation, has_drawdown_violation, has_correlation_violation
            )
            
            self.logger.info(f"📉 Risk reduction triggered: factor={reduction_factor:.1%}, violations={len(violations)}")
            
            # Get positions sorted by priority for reduction
            positions_to_reduce = await self._prioritize_positions_for_reduction(
                positions,
                has_volatility_violation,
                has_drawdown_violation,
                has_correlation_violation
            )
            
            # Calculate total reduction needed
            total_position_value = sum(p.entry_price * p.quantity for p in positions)
            target_reduction = total_position_value * reduction_factor
            
            # Reduce positions until we hit the target
            capital_freed = 0.0
            for position in positions_to_reduce:
                if capital_freed >= target_reduction:
                    break
                    
                # Calculate reduction for this position
                position_value = position.entry_price * position.quantity
                reduction_amount = min(
                    position_value * self.max_reduction_per_cycle,
                    target_reduction - capital_freed
                )
                
                # Determine if we should partially reduce or fully close
                remaining_value = position_value - reduction_amount
                
                if remaining_value < self.min_position_value:
                    # Full close
                    success = await self._close_position_for_risk(position)
                    if success:
                        result['positions_fully_closed'] += 1
                        result['positions_reduced'] += 1
                        capital_freed += position_value
                        result['details'].append(
                            f"CLOSED {position.market_id} ({position.side}): ${position_value:.2f}"
                        )
                else:
                    # Partial reduction
                    reduction_pct = reduction_amount / position_value
                    sold_value = await self._partial_sell_position(position, reduction_pct)
                    if sold_value > 0:
                        result['positions_reduced'] += 1
                        capital_freed += sold_value
                        result['details'].append(
                            f"REDUCED {position.market_id} ({position.side}): "
                            f"-${sold_value:.2f} ({reduction_pct:.0%})"
                        )
            
            result['capital_freed'] = capital_freed
            
            self.logger.info(
                f"📉 Risk reduction complete: {result['positions_reduced']} positions, "
                f"${result['capital_freed']:.2f} freed"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in position sizing reduction: {e}")
            result['details'].append(f"Error: {e}")
            return result
    
    def _calculate_reduction_factor(
        self,
        results: Any,
        violations: List[str],
        has_vol: bool,
        has_dd: bool,
        has_corr: bool
    ) -> float:
        """
        Calculate how much to reduce based on violation severity.
        
        Returns a factor between 0.05 (5%) and 0.25 (25%).
        """
        base_factor = 0.05  # Minimum 5% reduction
        
        # Add based on violation count
        base_factor += len(violations) * 0.03
        
        # Add based on severity
        if has_vol and self.config:
            vol_excess = results.portfolio_volatility - self.config.max_portfolio_volatility
            if vol_excess > 0:
                # 5% extra reduction for each 10% over limit
                base_factor += (vol_excess / 0.10) * 0.05
        
        if has_dd and self.config:
            dd_excess = results.max_portfolio_drawdown - self.config.max_drawdown_limit
            if dd_excess > 0:
                # 5% extra reduction for each 5% over limit
                base_factor += (dd_excess / 0.05) * 0.05
        
        if has_corr and self.config:
            corr_excess = results.correlation_score - self.config.max_correlation_exposure
            if corr_excess > 0:
                # 3% extra reduction for each 10% over limit
                base_factor += (corr_excess / 0.10) * 0.03
        
        # Cap at 25% maximum reduction per cycle
        return min(0.25, base_factor)
    
    async def _prioritize_positions_for_reduction(
        self,
        positions: List[Position],
        has_volatility_violation: bool,
        has_drawdown_violation: bool,
        has_correlation_violation: bool
    ) -> List[Position]:
        """
        Sort positions by priority for reduction.
        
        Priority factors:
        - Larger positions (more capital at risk)
        - Lower confidence (less conviction)
        - Older positions (may be stale)
        - No stop-loss (higher risk)
        """
        scored_positions = []
        
        for position in positions:
            score = 0.0
            
            # Factor 1: Position size (larger = higher priority to reduce)
            position_value = position.entry_price * position.quantity
            score += min(5.0, position_value / 20)  # Max 5 points for $100+ positions
            
            # Factor 2: Low confidence (lower = higher priority)
            if position.confidence:
                if position.confidence < 0.6:
                    score += 3.0
                elif position.confidence < 0.7:
                    score += 1.5
            else:
                score += 2.0  # No confidence = treat as uncertain
            
            # Factor 3: Age (older = higher priority)
            age_hours = (datetime.now() - position.timestamp).total_seconds() / 3600
            if age_hours > 72:  # 3+ days
                score += 2.0
            elif age_hours > 24:  # 1+ day
                score += 1.0
            
            # Factor 4: No stop-loss (higher risk)
            if not position.stop_loss_price:
                score += 2.0
            
            # Factor 5: Strategy-specific prioritization
            if has_volatility_violation:
                # Prioritize reducing market making positions (typically higher exposure)
                if position.strategy == 'market_making':
                    score += 1.5
            
            if has_drawdown_violation:
                # Prioritize positions that might be underwater
                # (require real-time price check for accuracy)
                score += 0.5
            
            if has_correlation_violation:
                # Prioritize directional positions (can be highly correlated)
                if position.strategy in ['portfolio_optimization', 'directional_trading']:
                    score += 1.0
            
            scored_positions.append((position, score))
        
        # Sort by score descending (highest priority first)
        scored_positions.sort(key=lambda x: x[1], reverse=True)
        
        return [p[0] for p in scored_positions]
    
    async def _close_position_for_risk(self, position: Position) -> bool:
        """
        Close a position due to risk management.
        Places a market sell order to immediately exit.
        """
        try:
            from src.config.settings import settings
            
            live_mode = getattr(settings.trading, 'live_trading_enabled', False)
            
            if live_mode:
                # 🚨 Validate ticker before placing order
                if not position.market_id or len(position.market_id) < 5:
                    self.logger.error(f"❌ Invalid ticker: {position.market_id}")
                    return False
                
                # Place market sell order  
                order_id = str(uuid.uuid4())
                
                # Build order params with required price
                order_params = {
                    "ticker": position.market_id,
                    "client_order_id": order_id,
                    "side": position.side.lower(),
                    "action": "sell",
                    "count": position.quantity,
                    "type_": "market"
                }
                
                # Add required price for market order
                if position.side.lower() == "yes":
                    order_params["yes_price"] = 1  # Sell at any price (1 cent minimum)
                else:
                    order_params["no_price"] = 1
                
                response = await self.kalshi_client.place_order(**order_params)
                
                if response and 'order' in response:
                    self.logger.info(f"✅ Risk closure order placed: {position.market_id}")
                    # Update position status in database
                    await self.db_manager.update_position_status(position.id, "closed")
                    return True
                else:
                    self.logger.error(f"❌ Failed to close position for risk: {response}")
                    return False
            else:
                # Paper trading mode - just mark as closed
                await self.db_manager.update_position_status(position.id, "closed")
                self.logger.info(f"📝 [PAPER] Closed position for risk: {position.market_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error closing position {position.market_id}: {e}")
            return False
    
    async def _partial_sell_position(
        self, 
        position: Position, 
        reduction_pct: float
    ) -> float:
        """
        Sell a portion of a position to reduce exposure.
        
        Args:
            position: The position to partially reduce
            reduction_pct: Percentage to reduce (0.0 to 1.0)
            
        Returns:
            Dollar value sold (0 if failed)
        """
        try:
            from src.config.settings import settings
            
            contracts_to_sell = max(1, int(position.quantity * reduction_pct))
            
            # Don't sell more than we have
            contracts_to_sell = min(contracts_to_sell, position.quantity)
            
            live_mode = getattr(settings.trading, 'live_trading_enabled', False)
            
            if live_mode:
                # 🚨 Validate ticker before placing order
                if not position.market_id or len(position.market_id) < 5:
                    self.logger.error(f"❌ Invalid ticker for partial sell: {position.market_id}")
                    return 0.0
                
                # Place market sell order for partial quantity
                order_id = str(uuid.uuid4())
                
                # Build order params with required price
                order_params = {
                    "ticker": position.market_id,
                    "client_order_id": order_id,
                    "side": position.side.lower(),
                    "action": "sell",
                    "count": contracts_to_sell,
                    "type_": "market"
                }
                
                # Add required price for market order
                if position.side.lower() == "yes":
                    order_params["yes_price"] = 1  # Sell at any price
                else:
                    order_params["no_price"] = 1
                
                response = await self.kalshi_client.place_order(**order_params)
                
                if response and 'order' in response:
                    sold_value = contracts_to_sell * position.entry_price
                    
                    # Update position quantity in database
                    new_quantity = position.quantity - contracts_to_sell
                    if new_quantity <= 0:
                        await self.db_manager.update_position_status(position.id, "closed")
                    else:
                        await self._update_position_quantity(position.id, new_quantity)
                    
                    self.logger.info(
                        f"✅ Partial sell: {position.market_id} "
                        f"-{contracts_to_sell} contracts (${sold_value:.2f})"
                    )
                    return sold_value
                else:
                    self.logger.error(f"❌ Failed to partially sell: {response}")
                    return 0.0
            else:
                # Paper trading mode
                sold_value = contracts_to_sell * position.entry_price
                new_quantity = position.quantity - contracts_to_sell
                
                if new_quantity <= 0:
                    await self.db_manager.update_position_status(position.id, "closed")
                else:
                    await self._update_position_quantity(position.id, new_quantity)
                
                self.logger.info(
                    f"📝 [PAPER] Partial sell: {position.market_id} "
                    f"-{contracts_to_sell} contracts (${sold_value:.2f})"
                )
                return sold_value
                
        except Exception as e:
            self.logger.error(f"Error in partial sell for {position.market_id}: {e}")
            return 0.0
    
    async def _update_position_quantity(self, position_id: int, new_quantity: int) -> None:
        """Update position quantity in database after partial sale."""
        try:
            import aiosqlite
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                await db.execute(
                    "UPDATE positions SET quantity = ? WHERE id = ?",
                    (new_quantity, position_id)
                )
                await db.commit()
        except Exception as e:
            self.logger.error(f"Error updating position quantity: {e}")
    
    async def rebalance_portfolio(self) -> Dict[str, Any]:
        """
        Rebalance portfolio to align with target strategy allocations.
        
        Process:
        1. Calculate current allocation per strategy
        2. Compare with target allocation from config
        3. For over-allocated strategies: sell to reduce
        4. Under-allocated strategies will naturally fill on next trading cycle
        
        Returns:
            Dictionary with rebalancing results
        """
        result = {
            'rebalanced': False,
            'summary': 'No rebalancing needed',
            'strategy_adjustments': {},
            'positions_sold': 0,
            'capital_reallocated': 0.0
        }
        
        try:
            # Get current positions grouped by strategy
            positions = await self.db_manager.get_open_positions()
            
            if not positions:
                result['summary'] = 'No positions to rebalance'
                return result
            
            # Calculate current allocation by strategy
            current_allocation = self._calculate_current_allocation(positions)
            
            # Get target allocation from config
            target_allocation = self._get_target_allocation()
            
            # Calculate drift from target
            drift = {}
            needs_rebalance = False
            
            for strategy, target in target_allocation.items():
                current = current_allocation.get(strategy, 0.0)
                diff = current - target
                drift[strategy] = diff
                
                if abs(diff) > self.rebalance_threshold:
                    needs_rebalance = True
            
            if not needs_rebalance:
                result['summary'] = f'Allocation drift within threshold ({self.rebalance_threshold:.0%})'
                return result
            
            self.logger.info(f"⚖️ Rebalancing triggered. Drift: {drift}")
            
            # Rebalance over-allocated strategies
            total_sold = 0.0
            positions_sold = 0
            
            for strategy, drift_pct in drift.items():
                if drift_pct > self.rebalance_threshold:
                    # Strategy is over-allocated - reduce positions
                    strategy_positions = [
                        p for p in positions 
                        if self._position_matches_strategy(p, strategy)
                    ]
                    
                    if not strategy_positions:
                        continue
                    
                    # Calculate how much to sell
                    total_strategy_value = sum(p.entry_price * p.quantity for p in strategy_positions)
                    target_reduction = total_strategy_value * drift_pct
                    
                    # Sort by priority (low confidence, old, large)
                    strategy_positions.sort(
                        key=lambda p: (
                            -(p.confidence or 0.5),  # Lower confidence first
                            (datetime.now() - p.timestamp).total_seconds(),  # Older first
                            -(p.entry_price * p.quantity)  # Larger first
                        )
                    )
                    
                    reduction_achieved = 0.0
                    for position in strategy_positions:
                        if reduction_achieved >= target_reduction:
                            break
                        
                        position_value = position.entry_price * position.quantity
                        reduction_needed = min(
                            target_reduction - reduction_achieved,
                            position_value * 0.5  # Max 50% reduction per position
                        )
                        
                        if reduction_needed / position_value > 0.1:  # Min 10% reduction
                            reduction_pct = reduction_needed / position_value
                            sold_value = await self._partial_sell_position(position, reduction_pct)
                            
                            if sold_value > 0:
                                reduction_achieved += sold_value
                                positions_sold += 1
                    
                    result['strategy_adjustments'][strategy] = -reduction_achieved
                    total_sold += reduction_achieved
            
            result['rebalanced'] = positions_sold > 0
            result['positions_sold'] = positions_sold
            result['capital_reallocated'] = total_sold
            result['summary'] = (
                f"Reduced {positions_sold} positions across over-allocated strategies, "
                f"freed ${total_sold:.2f}"
            )
            
            self.logger.info(f"⚖️ Rebalancing complete: {result['summary']}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in portfolio rebalancing: {e}")
            result['summary'] = f'Error: {e}'
            return result
    
    def _calculate_current_allocation(self, positions: List[Position]) -> Dict[str, float]:
        """Calculate current portfolio allocation by strategy."""
        if not positions:
            return {}
        
        # Group by strategy
        strategy_values = {}
        total_value = 0.0
        
        for position in positions:
            strategy = self._normalize_strategy_name(position.strategy or 'unknown')
            value = position.entry_price * position.quantity
            
            strategy_values[strategy] = strategy_values.get(strategy, 0.0) + value
            total_value += value
        
        if total_value == 0:
            return {}
        
        # Convert to percentages
        return {s: v / total_value for s, v in strategy_values.items()}
    
    def _get_target_allocation(self) -> Dict[str, float]:
        """Get target allocation from config."""
        if self.config:
            return {
                'market_making': getattr(self.config, 'market_making_allocation', 0.30),
                'directional_trading': getattr(self.config, 'directional_trading_allocation', 0.40),
                'quick_flip': getattr(self.config, 'quick_flip_allocation', 0.30),
                'arbitrage': getattr(self.config, 'arbitrage_allocation', 0.0)
            }
        
        # Default allocation
        return {
            'market_making': 0.30,
            'directional_trading': 0.40,
            'quick_flip': 0.30,
            'arbitrage': 0.0
        }
    
    def _normalize_strategy_name(self, strategy: str) -> str:
        """Normalize strategy names to match allocation keys."""
        mapping = {
            'market_making': 'market_making',
            'portfolio_optimization': 'directional_trading',
            'directional_trading': 'directional_trading',
            'quick_flip_scalping': 'quick_flip',
            'quick_flip': 'quick_flip',
            'arbitrage': 'arbitrage',
            'theta_decay': 'directional_trading'  # Group theta with directional
        }
        return mapping.get(strategy, 'directional_trading')
    
    def _position_matches_strategy(self, position: Position, strategy: str) -> bool:
        """Check if position belongs to a normalized strategy category."""
        position_strategy = self._normalize_strategy_name(position.strategy or 'unknown')
        return position_strategy == strategy
