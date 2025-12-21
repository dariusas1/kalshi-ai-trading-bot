"""
Circuit Breaker Module

Implements a three-tier trading halt system for production safety:
1. Order-level: Per-order validation (existing in risk_manager)
2. Daily-level: Daily loss limits (existing in settings)  
3. Hourly-level: 5% hourly loss triggers GLOBAL PAUSE

When tripped:
- All trading immediately stops
- Manual intervention required to resume
- Logged for forensic review

Based on production best practices from the Kalshi trading infrastructure article.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple, Optional
import logging

from src.utils.database import DatabaseManager
from src.config.settings import settings


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    hourly_loss_threshold_pct: float = 0.05  # 5% of account triggers pause
    cooldown_minutes: int = 60  # How long the "hourly" window is
    require_manual_reset: bool = True  # Whether operator must manually reset
    enabled: bool = True  # Master switch for circuit breaker


class CircuitBreaker:
    """
    Circuit Breaker for Trading System.
    
    Monitors hourly P&L and halts all trading if losses exceed threshold.
    This is a critical safety mechanism to prevent cascading losses during
    system failures, bad market conditions, or model misbehavior.
    
    Usage:
        circuit_breaker = CircuitBreaker(db_manager)
        
        # Before each trading cycle
        can_trade, reason = await circuit_breaker.check_can_trade()
        if not can_trade:
            logger.critical(f"Trading halted: {reason}")
            return
        
        # After each trade
        await circuit_breaker.record_trade_pnl(trade_pnl, account_balance)
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager, 
        config: Optional[CircuitBreakerConfig] = None
    ):
        self.db = db_manager
        self.config = config or CircuitBreakerConfig(
            hourly_loss_threshold_pct=getattr(
                settings.trading, 'circuit_breaker_hourly_loss_pct', 0.05
            ),
            require_manual_reset=getattr(
                settings.trading, 'circuit_breaker_require_manual_reset', True
            ),
            enabled=getattr(
                settings.trading, 'circuit_breaker_enabled', True
            )
        )
        self.logger = logging.getLogger("circuit_breaker")
    
    async def check_can_trade(self) -> Tuple[bool, str]:
        """
        Check if trading is allowed.
        
        Returns:
            Tuple of (can_trade: bool, reason: str)
            - (True, "OK") if trading is allowed
            - (False, reason) if trading is blocked
        """
        if not self.config.enabled:
            return True, "Circuit breaker disabled"
        
        try:
            state = await self.db.get_circuit_breaker_state()
            
            # Check if manually paused
            if state.get("is_paused"):
                reason = state.get("pause_reason", "Unknown reason")
                paused_at = state.get("paused_at", "Unknown time")
                return False, f"Circuit breaker ACTIVE since {paused_at}: {reason}"
            
            # Check if in manual override mode (forced trading)
            if state.get("manual_override"):
                self.logger.warning("⚠️ Trading with manual override - circuit breaker bypassed")
                return True, "Manual override active"
            
            return True, "OK"
            
        except Exception as e:
            # On error, be conservative and block trading
            self.logger.error(f"Circuit breaker check error: {e}")
            return False, f"Circuit breaker error: {e}"
    
    async def record_trade_pnl(self, pnl: float, account_balance: float) -> bool:
        """
        Record a trade's P&L and check if circuit breaker should trip.
        
        Args:
            pnl: Profit/loss from the trade (negative for losses)
            account_balance: Current account balance for percentage calculation
        
        Returns:
            True if trading can continue, False if circuit breaker tripped
        """
        if not self.config.enabled:
            return True
        
        try:
            result = await self.db.update_circuit_breaker_hourly_loss(pnl, account_balance)
            
            hourly_loss = result.get("hourly_loss", 0)
            loss_pct = result.get("loss_pct", 0)
            should_trip = result.get("should_trip", False)
            
            if should_trip:
                reason = (
                    f"Hourly loss threshold exceeded: "
                    f"${abs(hourly_loss):.2f} ({loss_pct:.1%}) > {self.config.hourly_loss_threshold_pct:.0%}"
                )
                await self.force_pause(reason)
                return False
            
            # Log warning if approaching threshold
            if loss_pct >= self.config.hourly_loss_threshold_pct * 0.5:
                self.logger.warning(
                    f"⚠️ Approaching circuit breaker threshold: "
                    f"${abs(hourly_loss):.2f} ({loss_pct:.1%})"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error recording trade P&L: {e}")
            return True  # Don't block trading on logging errors
    
    async def force_pause(self, reason: str) -> bool:
        """
        Manually pause all trading.
        
        Args:
            reason: Description of why trading was paused
        
        Returns:
            True if successfully paused
        """
        self.logger.critical(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")
        self.logger.critical("⛔ ALL TRADING HALTED - Manual reset required")
        
        try:
            await self.db.set_circuit_breaker_paused(True, reason)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set circuit breaker: {e}")
            return False
    
    async def reset(self) -> bool:
        """
        Reset the circuit breaker (operator intervention).
        Only call this after investigating why the circuit breaker tripped.
        
        Returns:
            True if successfully reset
        """
        self.logger.info("✅ Circuit breaker reset by operator")
        
        try:
            await self.db.reset_circuit_breaker()
            return True
        except Exception as e:
            self.logger.error(f"Failed to reset circuit breaker: {e}")
            return False
    
    async def get_status(self) -> dict:
        """
        Get the current circuit breaker status for monitoring.
        
        Returns:
            Dict with status information
        """
        state = await self.db.get_circuit_breaker_state()
        
        return {
            "enabled": self.config.enabled,
            "is_paused": state.get("is_paused", False),
            "pause_reason": state.get("pause_reason"),
            "paused_at": state.get("paused_at"),
            "hourly_loss": state.get("hourly_loss", 0),
            "hourly_window_start": state.get("hourly_window_start"),
            "threshold_pct": self.config.hourly_loss_threshold_pct,
            "require_manual_reset": self.config.require_manual_reset
        }
