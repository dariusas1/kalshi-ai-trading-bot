"""
Dynamic Allocation Manager

Performance-based capital allocation that adjusts strategy allocations
based on historical win rates and PnL performance.

Replaces static 30/40/30 split with dynamic rebalancing.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from src.utils.database import DatabaseManager
from src.utils.logging_setup import get_trading_logger
from src.config.settings import settings


@dataclass
class StrategyAllocation:
    """Allocation percentages for each strategy."""
    market_making: float = 0.30
    directional_trading: float = 0.40
    quick_flip: float = 0.30
    arbitrage: float = 0.00
    
    def validate(self) -> bool:
        """Ensure allocations sum to 1.0."""
        total = self.market_making + self.directional_trading + self.quick_flip + self.arbitrage
        return abs(total - 1.0) < 0.01


class DynamicAllocationManager:
    """
    Manages dynamic capital allocation based on strategy performance.
    
    Features:
    - Fetches strategy performance from database
    - Calculates win rate and PnL weighting
    - Returns adjusted allocation percentages
    - Uses Bayesian smoothing for strategies with few trades
    """
    
    # Minimum and maximum allocation per strategy
    MIN_ALLOCATION = 0.15  # 15% minimum
    MAX_ALLOCATION = 0.50  # 50% maximum
    MIN_TRADES_FOR_ADJUSTMENT = 10  # Need at least 10 trades to adjust from baseline
    
    # Baseline allocations (used when insufficient data)
    BASELINE = StrategyAllocation(
        market_making=0.30,
        directional_trading=0.40,
        quick_flip=0.30,
        arbitrage=0.00
    )
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.logger = get_trading_logger("dynamic_allocation")
    
    async def get_dynamic_allocation(self, baseline: Optional[StrategyAllocation] = None) -> StrategyAllocation:
        """
        Calculate dynamic allocation based on strategy performance.
        
        Args:
            baseline: Optional user-defined baseline to fall back to.
            
        Returns:
            StrategyAllocation with adjusted percentages based on win rates and PnL.
        """
        fallback = baseline or self.BASELINE
        try:
            # Get performance metrics by strategy
            performance = await self.db_manager.get_performance_by_strategy()
            
            if not performance:
                self.logger.info("📊 No strategy performance data - using baseline allocation")
                return fallback
            
            # Check if we have enough trades for any strategy
            total_trades = sum(p.get('completed_trades', 0) for p in performance.values())
            if total_trades < self.MIN_TRADES_FOR_ADJUSTMENT:
                self.logger.info(f"📊 Insufficient trade history ({total_trades} trades) - using baseline allocation")
                return fallback
            
            # Calculate performance scores for each strategy
            scores = self._calculate_performance_scores(performance)
            
            # Convert scores to allocations (passing fallback for bounds)
            allocation = self._scores_to_allocation(scores)
            
            self.logger.info(
                f"📊 Dynamic allocation calculated: "
                f"MM={allocation.market_making:.1%}, "
                f"Dir={allocation.directional_trading:.1%}, "
                f"QF={allocation.quick_flip:.1%}"
            )
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic allocation: {e}")
            return fallback
    
    def _calculate_performance_scores(self, performance: Dict) -> Dict[str, float]:
        """
        Calculate a composite performance score for each strategy.
        
        Score = (win_rate * 0.6) + (normalized_pnl * 0.4)
        Uses Bayesian smoothing for strategies with few trades.
        """
        scores = {}
        
        # Strategy name mapping (database names to allocation names)
        strategy_mapping = {
            'market_making': 'market_making',
            'portfolio_optimization': 'directional_trading',
            'directional_trading': 'directional_trading',
            'quick_flip_scalping': 'quick_flip',
            'quick_flip': 'quick_flip'
        }
        
        # Calculate raw scores with Bayesian smoothing
        for db_strategy, data in performance.items():
            allocation_strategy = strategy_mapping.get(db_strategy)
            if not allocation_strategy:
                continue
            
            trades = data.get('completed_trades', 0)
            win_rate = data.get('win_rate_pct', 50.0) / 100.0  # Convert to 0-1
            total_pnl = data.get('total_pnl', 0.0)
            
            # Bayesian smoothing: weight toward 50% win rate for few trades
            # Prior: 50% win rate with strength of 5 trades
            prior_weight = 5
            smoothed_win_rate = (win_rate * trades + 0.5 * prior_weight) / (trades + prior_weight)
            
            # Normalize PnL to 0-1 scale (assuming reasonable range of -$100 to +$100)
            normalized_pnl = max(0, min(1, (total_pnl + 100) / 200))
            
            # Composite score: 60% win rate, 40% PnL
            score = (smoothed_win_rate * 0.6) + (normalized_pnl * 0.4)

            # Performance gating for underperforming strategies
            if trades >= self.MIN_TRADES_FOR_ADJUSTMENT:
                min_win = getattr(settings.trading, "min_strategy_win_rate", 0.45)
                min_pnl = getattr(settings.trading, "min_strategy_pnl", 0.0)
                if win_rate < min_win or total_pnl < min_pnl:
                    score = min(score, 0.2)
            
            # Aggregate scores for strategies that map to same allocation
            if allocation_strategy in scores:
                # Average the scores if multiple database strategies map to same allocation
                scores[allocation_strategy] = (scores[allocation_strategy] + score) / 2
            else:
                scores[allocation_strategy] = score
            
            self.logger.debug(
                f"Strategy {db_strategy}: trades={trades}, win_rate={win_rate:.1%}, "
                f"smoothed={smoothed_win_rate:.1%}, pnl=${total_pnl:.2f}, score={score:.3f}"
            )
        
        # Ensure all strategies have a score (use neutral 0.5 for missing)
        for strategy in ['market_making', 'directional_trading', 'quick_flip']:
            if strategy not in scores:
                scores[strategy] = 0.5
        
        return scores
    
    def _scores_to_allocation(self, scores: Dict[str, float]) -> StrategyAllocation:
        """
        Convert performance scores to allocation percentages.
        
        Higher scores get higher allocations, with min/max constraints.
        """
        # Get scores for active strategies (exclude arbitrage for now)
        active_strategies = ['market_making', 'directional_trading', 'quick_flip']
        active_scores = {s: scores.get(s, 0.5) for s in active_strategies}
        
        # Calculate total score for normalization
        total_score = sum(active_scores.values())
        
        if total_score == 0:
            return self.BASELINE
        
        # Convert to raw allocations
        raw_allocations = {s: score / total_score for s, score in active_scores.items()}
        
        # Apply min/max constraints and renormalize
        constrained = {}
        for strategy, alloc in raw_allocations.items():
            constrained[strategy] = max(self.MIN_ALLOCATION, min(self.MAX_ALLOCATION, alloc))
        
        # Renormalize to sum to 1.0
        total = sum(constrained.values())
        normalized = {s: alloc / total for s, alloc in constrained.items()}
        
        return StrategyAllocation(
            market_making=normalized['market_making'],
            directional_trading=normalized['directional_trading'],
            quick_flip=normalized['quick_flip'],
            arbitrage=0.0
        )


async def get_dynamic_strategy_allocation(
    db_manager: DatabaseManager, 
    baseline: Optional[StrategyAllocation] = None
) -> StrategyAllocation:
    """
    Convenience function to get dynamic allocation.
    
    Args:
        db_manager: Database manager instance
        baseline: Optional baseline allocation to use if insufficient data
        
    Returns:
        StrategyAllocation with performance-based percentages
    """
    manager = DynamicAllocationManager(db_manager)
    return await manager.get_dynamic_allocation(baseline=baseline)
