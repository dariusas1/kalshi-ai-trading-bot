"""
Edge-Based Filtering Module

Implements the >10% edge filtering requirement recommended by Grok4 performance analysis.
Provides consistent edge filtering across all trading strategies.

Key Features:
- 10% minimum edge requirement for trading
- Adaptive edge thresholds based on confidence and risk
- Edge calculation utilities for different position types
- Filtering logic for market opportunities
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import math
import logging

# Module-level stats tracking for diagnostic purposes
_filter_stats = {
    'passed': 0,
    'rejected': 0,
    'last_summary_at': 0
}
_logger = logging.getLogger("edge_filter")


@dataclass
class EdgeFilterResult:
    """Result of edge filtering analysis."""
    passes_filter: bool
    edge_magnitude: float
    edge_percentage: float
    side: str  # "YES" or "NO"
    reason: str
    confidence_adjusted_edge: float


class EdgeFilter:
    """
    Centralized edge filtering following Grok4 recommendations.
    
    UPDATED: More aggressive thresholds to allow more trading opportunities.
    """
    
    # PERMISSIVE: Allow more trading opportunities with lower edge requirements
    # These thresholds determine minimum edge required for different confidence levels
    MIN_EDGE_REQUIREMENT = 0.05       # REDUCED: 5% minimum edge (was 8%)
    HIGH_CONFIDENCE_EDGE = 0.04       # REDUCED: 4% edge for high confidence >=80% (was 5%)
    MEDIUM_CONFIDENCE_EDGE = 0.06     # 6% edge for medium confidence 60-80% (quality over quantity)
    LOW_CONFIDENCE_EDGE = 0.10        # REDUCED: 10% edge for low confidence <60% (was 15%)

    # Confidence threshold for quality trades
    MIN_CONFIDENCE_FOR_TRADE = 0.50   # 50% minimum confidence (unchanged)
    MAX_ACCEPTABLE_RISK = 0.5         # 50% max position risk (unchanged)

    # Market quality standards
    MIN_VOLUME_FOR_HIGH_EDGE = 1000   # REDUCED: Lower volume requirement (was 2000)
    MIN_SPREAD_QUALITY = 0.02         # Tighter spread requirements (unchanged)

    
    @classmethod
    def calculate_edge(
        cls,
        ai_probability: float,
        market_probability: float,
        confidence: Optional[float] = None
    ) -> EdgeFilterResult:
        """
        Calculate edge and determine if it meets filtering criteria.
        
        Args:
            ai_probability: AI predicted probability (0.0 to 1.0)
            market_probability: Current market price/probability (0.0 to 1.0)
            confidence: AI confidence level (0.0 to 1.0)
            
        Returns:
            EdgeFilterResult with filtering decision and details
        """
        
        # Validate inputs
        ai_probability = max(0.01, min(0.99, ai_probability))
        market_probability = max(0.01, min(0.99, market_probability))
        confidence = confidence or 0.7
        
        # Calculate raw edge (probability difference)
        edge_magnitude = ai_probability - market_probability
        edge_percentage = abs(edge_magnitude)
        
        # ⚖️ BIAS CORRECTION: LLMs can be overly negative/conservative
        # Penalize NO edges to ensure we only take really strong NO positions
        if edge_magnitude < 0:
            edge_percentage *= 0.8  # 20% penalty for NO bets (requires stronger signal)
        
        # Determine position side based on edge direction
        if edge_magnitude > 0:
            side = "YES"  # AI thinks YES is underpriced
        else:
            side = "NO"   # AI thinks NO is underpriced (YES is overpriced)
        
        # Confidence-adjusted edge thresholds
        if confidence >= 0.8:
            required_edge = cls.HIGH_CONFIDENCE_EDGE     # 8% for high confidence
        elif confidence >= 0.6:
            required_edge = cls.MEDIUM_CONFIDENCE_EDGE   # 10% for medium confidence
        else:
            required_edge = cls.LOW_CONFIDENCE_EDGE      # 15% for low confidence

        # Settings-driven overrides
        try:
            from src.config.settings import settings
            min_confidence = getattr(settings.trading, "min_confidence_threshold", cls.MIN_CONFIDENCE_FOR_TRADE)
            min_edge = getattr(settings.trading, "min_trade_edge", cls.MIN_EDGE_REQUIREMENT)
            required_edge = max(required_edge, min_edge)
        except Exception:
            min_confidence = cls.MIN_CONFIDENCE_FOR_TRADE
        
        # Calculate confidence-adjusted edge
        confidence_adjusted_edge = edge_percentage * confidence
        
        # Check if edge meets requirements (use > instead of >= to avoid floating point precision issues)
        passes_basic_edge = edge_percentage > (required_edge - 0.001)  # Allow tiny tolerance for floating point
        passes_confidence = confidence >= min_confidence
        
        # Generate filtering decision and reason
        if not passes_confidence:
            passes_filter = False
            reason = f"Confidence {confidence:.1%} below minimum {cls.MIN_CONFIDENCE_FOR_TRADE:.1%}"
        elif not passes_basic_edge:
            passes_filter = False
            reason = f"Edge {edge_percentage:.1%} below required {required_edge:.1%} for confidence {confidence:.1%}"
        else:
            passes_filter = True
            reason = f"Meets requirements: {edge_percentage:.1%} edge, {confidence:.1%} confidence"
        
        # Update stats tracking
        if passes_filter:
            _filter_stats['passed'] += 1
        else:
            _filter_stats['rejected'] += 1
        
        # Log summary every 20 decisions
        total_decisions = _filter_stats['passed'] + _filter_stats['rejected']
        if total_decisions > 0 and total_decisions % 20 == 0 and total_decisions != _filter_stats['last_summary_at']:
            _filter_stats['last_summary_at'] = total_decisions
            pass_rate = _filter_stats['passed'] / total_decisions * 100
            _logger.info(
                f"📊 Edge filter stats: {_filter_stats['passed']} passed, "
                f"{_filter_stats['rejected']} rejected ({pass_rate:.1f}% pass rate)"
            )
        
        return EdgeFilterResult(
            passes_filter=passes_filter,
            edge_magnitude=edge_magnitude,
            edge_percentage=edge_percentage,
            side=side,
            reason=reason,
            confidence_adjusted_edge=confidence_adjusted_edge
        )
    
    @classmethod
    def filter_opportunities(
        cls,
        opportunities: List[Dict[str, Any]],
        require_edge_filter: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Filter a list of market opportunities based on edge requirements.
        
        Args:
            opportunities: List of opportunity dictionaries with edge, confidence, etc.
            require_edge_filter: Whether to apply edge filtering (default True)
            
        Returns:
            Filtered list of opportunities that meet edge requirements
        """
        if not require_edge_filter:
            return opportunities
        
        filtered_opportunities = []
        
        for opp in opportunities:
            # Extract required fields
            ai_prob = opp.get('predicted_probability', 0.5)
            market_prob = opp.get('market_probability', 0.5)
            confidence = opp.get('confidence', 0.7)
            
            # Calculate edge
            edge_result = cls.calculate_edge(ai_prob, market_prob, confidence)
            
            # Add to filtered list if it passes
            if edge_result.passes_filter:
                # Add edge analysis to opportunity
                opp['edge_filter_result'] = edge_result
                opp['filtered_edge'] = edge_result.edge_magnitude
                opp['edge_percentage'] = edge_result.edge_percentage
                opp['recommended_side'] = edge_result.side
                
                filtered_opportunities.append(opp)
        
        return filtered_opportunities
    
    @classmethod
    def should_trade_market(
        cls,
        ai_probability: float,
        market_probability: float,
        confidence: float,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, str, EdgeFilterResult]:
        """
        Comprehensive trading decision based on edge filtering.
        
        Args:
            ai_probability: AI predicted probability
            market_probability: Market price/probability  
            confidence: AI confidence level
            additional_filters: Optional additional filtering criteria
            
        Returns:
            (should_trade, reason, edge_result)
        """
        
        # Calculate edge
        edge_result = cls.calculate_edge(ai_probability, market_probability, confidence)
        
        # Basic edge filter
        if not edge_result.passes_filter:
            return False, edge_result.reason, edge_result
        
        # Additional filters if provided
        if additional_filters:
            volume = additional_filters.get('volume', 0)
            min_volume = additional_filters.get('min_volume', 1000)
            time_to_expiry = additional_filters.get('time_to_expiry_days', 0)
            
            # Get max time to expiry from settings (default 1 day = 24 hours)
            try:
                from src.config.settings import settings
                default_max_expiry = settings.trading.max_time_to_expiry_days
            except Exception:
                default_max_expiry = 1  # Default to 1 day if settings unavailable
            
            max_time_to_expiry = additional_filters.get('max_time_to_expiry', default_max_expiry)
            
            if volume < min_volume:
                return False, f"Volume {volume} below minimum {min_volume}", edge_result

            # STRICT: Reject markets expiring beyond max time regardless of confidence
            if time_to_expiry > max_time_to_expiry:
                return False, f"Market expires in {time_to_expiry:.1f} days, exceeds {max_time_to_expiry} day limit", edge_result
            
            # Additional check for lower confidence on longer-term markets
            try:
                from src.config.settings import settings
                long_term_min_conf = settings.trading.min_confidence_long_term
            except Exception:
                long_term_min_conf = cls.MIN_CONFIDENCE_FOR_TRADE

            # For markets closer to max expiry, require higher confidence
            if time_to_expiry > (max_time_to_expiry * 0.5) and confidence < long_term_min_conf:
                return False, f"Near-expiry market needs confidence {long_term_min_conf:.1%}, got {confidence:.1%}", edge_result
        
        return True, f"TRADE APPROVED: {edge_result.reason}", edge_result
    
    @classmethod
    def get_edge_summary(cls, edge_results: List[EdgeFilterResult]) -> Dict[str, Any]:
        """
        Generate summary statistics for a list of edge filtering results.
        
        Returns:
            Dictionary with summary statistics
        """
        if not edge_results:
            return {"total": 0, "passed": 0, "pass_rate": 0.0}
        
        total = len(edge_results)
        passed = sum(1 for r in edge_results if r.passes_filter)
        pass_rate = passed / total
        
        avg_edge = sum(r.edge_percentage for r in edge_results) / total
        avg_confidence_adj = sum(r.confidence_adjusted_edge for r in edge_results) / total
        
        yes_positions = sum(1 for r in edge_results if r.side == "YES")
        no_positions = sum(1 for r in edge_results if r.side == "NO")
        
        return {
            "total_opportunities": total,
            "passed_filter": passed,
            "filtered_out": total - passed,
            "pass_rate": pass_rate,
            "average_edge": avg_edge,
            "average_confidence_adjusted_edge": avg_confidence_adj,
            "yes_positions": yes_positions,
            "no_positions": no_positions,
            "side_balance": yes_positions / total if total > 0 else 0.0
        }


# Convenience functions for backward compatibility
def calculate_edge(ai_prob: float, market_prob: float, confidence: float = 0.7) -> EdgeFilterResult:
    """Convenience function for edge calculation."""
    return EdgeFilter.calculate_edge(ai_prob, market_prob, confidence)


def passes_edge_filter(ai_prob: float, market_prob: float, confidence: float = 0.7) -> bool:
    """Simple boolean check for edge filtering."""
    result = EdgeFilter.calculate_edge(ai_prob, market_prob, confidence)
    return result.passes_filter


def get_minimum_edge_for_confidence(confidence: float) -> float:
    """Get the minimum edge required for a given confidence level."""
    if confidence >= 0.8:
        return EdgeFilter.HIGH_CONFIDENCE_EDGE
    elif confidence >= 0.6:
        return EdgeFilter.MEDIUM_CONFIDENCE_EDGE
    else:
        return EdgeFilter.LOW_CONFIDENCE_EDGE 
