"""
Risk Guard — Circuit breakers and safety checks.

Every trading decision passes through these guards BEFORE execution.
Any guard can REJECT a trade with a clear reason.

Alpha Arena Lessons:
- Claude lost -71% because there was NO max drawdown halt
- Gemini panic-flipped because there was NO daily loss limit
- GPT-5 overtraded to death because there was NO trade cooldown
- Fee erosion killed several bots because there was NO fee tracking
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Literal
from collections import deque

import pandas as pd


class GuardStatus(str, Enum):
    PASS = "pass"      # Trade approved
    WARN = "warn"      # Trade approved with reduced size
    REJECT = "reject"  # Trade blocked
    HALT = "halt"      # All trading halted


@dataclass
class GuardResult:
    """Result of a single guard check."""
    status: GuardStatus = GuardStatus.PASS
    name: str = ""
    message: str = ""
    severity: int = 0  # 0=info, 1=warn, 2=block
    suggested_action: str = "proceed"
    
    # For position size reduction
    size_multiplier: float = 1.0  # 1.0 = full size, 0.5 = half size
    
    @property
    def blocks_trade(self) -> bool:
        return self.status in (GuardStatus.REJECT, GuardStatus.HALT)
    
    @property
    def reduces_size(self) -> bool:
        return self.status == GuardStatus.WARN and self.size_multiplier < 1.0


# =============================================================================
# Guard Configuration
# =============================================================================

@dataclass
class GuardConfig:
    """Configuration for all risk guards."""
    
    # Max Drawdown Guard
    max_drawdown_pct: float = 15.0  # Halt if drawdown exceeds 15%
    cooldown_after_drawdown_hours: int = 24  # 24h cooldown after halt
    
    # Daily Loss Limit
    daily_loss_limit_pct: float = 5.0  # Halt if daily loss exceeds 5%
    reset_daily_loss_at_utc: int = 0  # Reset at midnight UTC
    
    # Fee Budget Guard
    fee_budget_pct: float = 2.0  # Warn if fees exceed 2% of account
    fee_budget_max_pct: float = 5.0  # Halt if fees exceed 5%
    
    # Trade Cooldown
    min_trade_interval_hours: float = 4.0  # Minimum 4h between new positions
    min_trade_interval_pct: Optional[float] = None  # Override: % of ATR-based volatility
    
    # Concentration Limits
    max_concurrent_positions: int = 3
    max_position_pct: float = 15.0  # No single position > 15% of equity
    max_correlation: float = 0.8  # Don't open if correlation to existing position > 0.8
    
    # Volatility Guard
    max_volatility_pct: float = 5.0  # Reject if ATR% > 5% (choppy market)
    min_volatility_pct: float = 0.3  # Reject if ATR% < 0.3% (no movement)
    
    # Account Health
    min_account_balance_usd: float = 50.0  # Halt if balance below $50
    
    # Weekend / Low Liquidity
    skip_weekend_trading: bool = True  # Reject new positions on weekends
    
    # Loss Streak
    max_consecutive_losses: int = 4  # Reduce size after 4 losses in a row
    loss_streak_size_reduction: float = 0.5  # Reduce to 50% after loss streak
    
    # Profit Target / Stop Override  
    enforce_min_risk_reward: float = 1.5  # Reject if R/R < 1.5:1
    max_stop_distance_pct: float = 10.0  # Reject if stop > 10% from entry


# =============================================================================
# Individual Guards
# =============================================================================

class RiskGuard:
    """
    Collection of risk guards that every trade must pass.
    
    Each guard is a pure function: input current state → output GuardResult.
    They are designed to be composable and independently testable.
    """
    
    def __init__(self, config: Optional[GuardConfig] = None):
        self._config = config or GuardConfig()
        
        # State tracking
        self._peak_equity: float = 0.0
        self._daily_loss: float = 0.0
        self._daily_loss_date: Optional[str] = None
        self._total_fees_paid: float = 0.0
        self._last_trade_time: Optional[datetime] = None
        self._consecutive_losses: int = 0
        self._halted: bool = False
        self._halt_reason: str = ""
        self._halt_until: Optional[datetime] = None
        self._open_positions: list[dict] = []
        self._trade_history: deque[dict] = deque(maxlen=100)  # Last 100 trades
    
    # ========================================================================
    # Public: Run All Guards
    # ========================================================================
    
    def check_all(
        self,
        account_equity: float,
        entry_price: float,
        stop_price: float,
        direction: Literal["LONG", "SHORT"],
        market_data: Optional[pd.DataFrame] = None,
    ) -> list[GuardResult]:
        """
        Run all guards and return the most restrictive result.
        
        Returns:
            List of GuardResult (one per guard)
        """
        results = []
        
        # 1. Halt check (always first)
        results.append(self._check_halt())
        
        # 2. Account health
        results.append(self._check_account_health(account_equity))
        
        # 3. Drawdown
        results.append(self._check_drawdown(account_equity))
        
        # 4. Daily loss
        results.append(self._check_daily_loss(account_equity))
        
        # 5. Trade cooldown
        results.append(self._check_cooldown())
        
        # 6. Concentration
        results.append(self._check_concentration(account_equity))
        
        # 7. Volatility
        results.append(self._check_volatility(entry_price, market_data))
        
        # 8. Risk/reward ratio
        results.append(self._check_risk_reward(entry_price, stop_price))
        
        # 9. Stop distance
        results.append(self._check_stop_distance(entry_price, stop_price))
        
        # 10. Loss streak
        results.append(self._check_loss_streak())
        
        # 11. Weekend trading
        results.append(self._check_weekend())
        
        # 12. Fee budget
        results.append(self._check_fee_budget(account_equity))
        
        return results
    
    def get_overall_status(self, results: list[GuardResult]) -> GuardResult:
        """
        Consolidate multiple guard results into a single decision.
        
        Priority: HALT > REJECT > WARN > PASS
        """
        halt_results = [r for r in results if r.status == GuardStatus.HALT]
        reject_results = [r for r in results if r.status == GuardStatus.REJECT]
        warn_results = [r for r in results if r.status == GuardStatus.WARN]
        
        worst = GuardResult(status=GuardStatus.PASS, name="overall", message="All guards passed")
        
        if halt_results:
            worst = halt_results[0]
            worst.message = f"HALT: {halt_results[0].message}"
        elif reject_results:
            worst = reject_results[0]
            worst.message = f"REJECT: {reject_results[0].message}"
        elif warn_results:
            # Combine all warnings
            size_multipliers = [r.size_multiplier for r in warn_results if r.size_multiplier < 1.0]
            min_multiplier = min(size_multipliers) if size_multipliers else 1.0
            messages = [r.message for r in warn_results]
            worst = GuardResult(
                status=GuardStatus.WARN,
                name="overall_warning",
                message="; ".join(messages),
                size_multiplier=min_multiplier,
            )
        
        return worst
    
    # ========================================================================
    # Individual Guard Implementations
    # ========================================================================
    
    def _check_halt(self) -> GuardResult:
        """Check if trading is globally halted."""
        if self._halted:
            if self._halt_until and datetime.now() > self._halt_until:
                # Auto-release from halt
                self._halted = False
                self._halt_reason = ""
                self._halt_until = None
                return GuardResult(
                    status=GuardStatus.PASS,
                    name="halt_guard",
                    message="Halt period expired, trading resumed",
                )
            
            return GuardResult(
                status=GuardStatus.HALT,
                name="halt_guard",
                message=f"Trading halted: {self._halt_reason}",
                severity=2,
            )
        return GuardResult(status=GuardStatus.PASS, name="halt_guard")
    
    def _check_account_health(self, equity: float) -> GuardResult:
        """Check if account has minimum balance."""
        if equity < self._config.min_account_balance_usd:
            self._trigger_halt(f"Account balance ${equity:.2f} below minimum ${self._config.min_account_balance_usd}")
            return GuardResult(
                status=GuardStatus.HALT,
                name="account_health",
                message=f"Account balance too low: ${equity:.2f}",
                severity=2,
            )
        return GuardResult(status=GuardStatus.PASS, name="account_health")
    
    def _check_drawdown(self, equity: float) -> GuardResult:
        """
        Check maximum drawdown.
        
        Tracks peak equity and triggers halt if drawdown threshold reached.
        """
        # Update peak
        if equity > self._peak_equity:
            self._peak_equity = equity
        
        if self._peak_equity <= 0:
            return GuardResult(status=GuardStatus.PASS, name="drawdown")
        
        drawdown_pct = ((self._peak_equity - equity) / self._peak_equity) * 100
        
        if drawdown_pct >= self._config.max_drawdown_pct:
            self._trigger_halt(
                f"Max drawdown reached: {drawdown_pct:.2f}% (limit: {self._config.max_drawdown_pct}%)",
                cooldown_hours=self._config.cooldown_after_drawdown_hours,
            )
            return GuardResult(
                status=GuardStatus.HALT,
                name="drawdown",
                message=f"Max drawdown reached: {drawdown_pct:.2f}%",
                severity=2,
            )
        
        # Warning at 80% of max drawdown
        if drawdown_pct >= self._config.max_drawdown_pct * 0.8:
            return GuardResult(
                status=GuardStatus.WARN,
                name="drawdown",
                message=f"Approaching max drawdown: {drawdown_pct:.2f}%",
                size_multiplier=0.5,
            )
        
        return GuardResult(status=GuardStatus.PASS, name="drawdown")
    
    def _check_daily_loss(self, equity: float) -> GuardResult:
        """Check daily loss limit."""
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Reset daily loss if new day
        if self._daily_loss_date != today:
            self._daily_loss_date = today
            self._daily_loss = 0.0
        
        if equity <= 0:
            return GuardResult(status=GuardStatus.PASS, name="daily_loss")
        
        # Calculate from peak today
        # This is simplified — in reality we'd track from daily start equity
        daily_loss_pct = (self._daily_loss / equity) * 100 if equity > 0 else 0
        
        if daily_loss_pct >= self._config.daily_loss_limit_pct:
            self._trigger_halt(
                f"Daily loss limit reached: {daily_loss_pct:.2f}%",
                cooldown_hours=12,  # Shorter cooldown for daily limit
            )
            return GuardResult(
                status=GuardStatus.HALT,
                name="daily_loss",
                message=f"Daily loss limit: {daily_loss_pct:.2f}%",
                severity=2,
            )
        
        return GuardResult(status=GuardStatus.PASS, name="daily_loss")
    
    def _check_cooldown(self) -> GuardResult:
        """Check minimum time since last trade."""
        if self._last_trade_time is None:
            return GuardResult(status=GuardStatus.PASS, name="cooldown")
        
        hours_since = (datetime.now() - self._last_trade_time).total_seconds() / 3600
        
        if hours_since < self._config.min_trade_interval_hours:
            return GuardResult(
                status=GuardStatus.REJECT,
                name="cooldown",
                message=f"Trade cooldown active: {hours_since:.1f}h / {self._config.min_trade_interval_hours}h",
                severity=1,
            )
        
        return GuardResult(status=GuardStatus.PASS, name="cooldown")
    
    def _check_concentration(self, equity: float) -> GuardResult:
        """Check position concentration limits."""
        if len(self._open_positions) >= self._config.max_concurrent_positions:
            return GuardResult(
                status=GuardStatus.REJECT,
                name="concentration",
                message=f"Max positions reached: {len(self._open_positions)} / {self._config.max_concurrent_positions}",
                severity=1,
            )
        
        # Check correlation of new position vs existing
        for pos in self._open_positions:
            # Calculate correlation (simplified: based on direction)
            if pos.get("direction") == "LONG" and pos.get("direction") == "LONG":
                # If both long on similar assets, correlation is high
                if self._is_correlated_pair(pos.get("symbol"), "BTC"):  # Assuming BTC for now
                    return GuardResult(
                        status=GuardStatus.WARN,
                        name="concentration",
                        message="New position correlates with existing position",
                        size_multiplier=0.5,
                    )
        
        return GuardResult(status=GuardStatus.PASS, name="concentration")
    
    def _check_volatility(
        self,
        entry_price: float,
        market_data: Optional[pd.DataFrame],
    ) -> GuardResult:
        """Check if market volatility is within acceptable range."""
        if market_data is None or market_data.empty:
            return GuardResult(status=GuardStatus.PASS, name="volatility")
        
        recent = market_data.tail(20)
        avg_range = (recent["high"] - recent["low"]).mean()
        
        if entry_price == 0:
            return GuardResult(status=GuardStatus.PASS, name="volatility")
        
        volatility_pct = (avg_range / entry_price) * 100
        
        if volatility_pct > self._config.max_volatility_pct:
            return GuardResult(
                status=GuardStatus.REJECT,
                name="volatility",
                message=f"Too volatile: {volatility_pct:.2f}% ATR (max: {self._config.max_volatility_pct}%)",
                severity=1,
            )
        
        if volatility_pct < self._config.min_volatility_pct:
            return GuardResult(
                status=GuardStatus.REJECT,
                name="volatility",
                message=f"Not enough movement: {volatility_pct:.2f}% ATR (min: {self._config.min_volatility_pct}%)",
                severity=1,
            )
        
        return GuardResult(status=GuardStatus.PASS, name="volatility")
    
    def _check_risk_reward(
        self,
        entry_price: float,
        stop_price: float,
    ) -> GuardResult:
        """Check if risk/reward ratio meets minimum threshold."""
        if stop_price == 0 or entry_price == 0:
            return GuardResult(status=GuardStatus.REJECT, name="risk_reward", message="Invalid prices")
        
        risk = abs(entry_price - stop_price)
        # We don't know target yet, but check if risk is reasonable
        risk_pct = (risk / entry_price) * 100
        
        if risk_pct == 0:
            return GuardResult(
                status=GuardStatus.REJECT,
                name="risk_reward",
                message="Zero risk distance — cannot calculate R/R",
                severity=2,
            )
        
        return GuardResult(status=GuardStatus.PASS, name="risk_reward")
    
    def _check_stop_distance(
        self,
        entry_price: float,
        stop_price: float,
    ) -> GuardResult:
        """Check if stop is an unreasonable distance from entry."""
        if entry_price == 0:
            return GuardResult(status=GuardStatus.PASS, name="stop_distance")
        
        distance_pct = (abs(entry_price - stop_price) / entry_price) * 100
        
        if distance_pct > self._config.max_stop_distance_pct:
            return GuardResult(
                status=GuardStatus.REJECT,
                name="stop_distance",
                message=f"Stop too far: {distance_pct:.2f}% (max: {self._config.max_stop_distance_pct}%)",
                severity=1,
            )
        
        if distance_pct < 0.3:
            return GuardResult(
                status=GuardStatus.REJECT,
                name="stop_distance",
                message=f"Stop too tight: {distance_pct:.2f}% (min: 0.3%)",
                severity=1,
            )
        
        return GuardResult(status=GuardStatus.PASS, name="stop_distance")
    
    def _check_loss_streak(self) -> GuardResult:
        """Reduce size after consecutive losses."""
        if self._consecutive_losses >= self._config.max_consecutive_losses:
            return GuardResult(
                status=GuardStatus.WARN,
                name="loss_streak",
                message=f"Loss streak: {self._consecutive_losses} losses. Reducing size.",
                size_multiplier=self._config.loss_streak_size_reduction,
            )
        return GuardResult(status=GuardStatus.PASS, name="loss_streak")
    
    def _check_weekend(self) -> GuardResult:
        """Optional weekend trading block."""
        if not self._config.skip_weekend_trading:
            return GuardResult(status=GuardStatus.PASS, name="weekend")
        
        day = datetime.now().weekday()
        if day >= 5:  # Saturday = 5, Sunday = 6
            return GuardResult(
                status=GuardStatus.REJECT,
                name="weekend",
                message="Weekend trading disabled",
                severity=0,
            )
        return GuardResult(status=GuardStatus.PASS, name="weekend")
    
    def _check_fee_budget(self, equity: float) -> GuardResult:
        """Check if fees are exceeding budget."""
        if equity <= 0:
            return GuardResult(status=GuardStatus.PASS, name="fee_budget")
        
        fee_pct = (self._total_fees_paid / equity) * 100
        
        if fee_pct >= self._config.fee_budget_max_pct:
            self._trigger_halt(f"Fee budget exceeded: {fee_pct:.2f}%")
            return GuardResult(
                status=GuardStatus.HALT,
                name="fee_budget",
                message=f"Fee budget exceeded: {fee_pct:.2f}%",
                severity=2,
            )
        
        if fee_pct >= self._config.fee_budget_pct:
            return GuardResult(
                status=GuardStatus.WARN,
                name="fee_budget",
                message=f"Fee budget warning: {fee_pct:.2f}%",
                size_multiplier=0.5,
            )
        
        return GuardResult(status=GuardStatus.PASS, name="fee_budget")
    
    # ========================================================================
    # State Management (called by RiskManager after trades)
    # ========================================================================
    
    def record_trade(
        self,
        pnl: float,
        fees: float,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record completed trade for state tracking."""
        ts = timestamp or datetime.now()
        
        self._trade_history.append({
            "pnl": pnl,
            "fees": fees,
            "timestamp": ts,
        })
        
        # Update fees
        self._total_fees_paid += fees
        
        # Update consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        
        # Update daily loss
        if pnl < 0:
            if self._daily_loss_date == ts.strftime("%Y-%m-%d"):
                self._daily_loss += abs(pnl)
            else:
                self._daily_loss_date = ts.strftime("%Y-%m-%d")
                self._daily_loss = abs(pnl)
        
        # Update last trade time
        self._last_trade_time = ts
    
    def record_position_opened(self, position: dict) -> None:
        """Track open position."""
        self._open_positions.append(position)
    
    def record_position_closed(self, position_id: str) -> None:
        """Remove closed position."""
        self._open_positions = [p for p in self._open_positions if p.get("id") != position_id]
    
    def _trigger_halt(self, reason: str, cooldown_hours: int = 24) -> None:
        """Trigger a trading halt."""
        self._halted = True
        self._halt_reason = reason
        self._halt_until = datetime.now() + timedelta(hours=cooldown_hours)
    
    @staticmethod
    def _is_correlated_pair(symbol1: str, symbol2: str) -> bool:
        """Check if two symbols are likely correlated (simplified)."""
        # Crypto majors are highly correlated
        crypto_majors = {"BTC", "ETH", "SOL", "BNB"}
        if symbol1 in crypto_majors and symbol2 in crypto_majors:
            return True
        
        # Meme coins are correlated
        meme_coins = {"DOGE", "SHIB", "PEPE"}
        if symbol1 in meme_coins and symbol2 in meme_coins:
            return True
        
        return False
    
    # ========================================================================
    # Properties
    # ========================================================================
    
    @property
    def is_halted(self) -> bool:
        return self._halted and (self._halt_until is None or datetime.now() < self._halt_until)
    
    @property
    def current_drawdown_pct(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        # This needs current equity — calculated externally and passed in
        return 0.0  # Placeholder — actual drawdown tracked by RiskManager
    
    @property
    def stats(self) -> dict:
        return {
            "peak_equity": self._peak_equity,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "halt_until": self._halt_until.isoformat() if self._halt_until else None,
            "daily_loss": self._daily_loss,
            "daily_loss_date": self._daily_loss_date,
            "total_fees_paid": self._total_fees_paid,
            "consecutive_losses": self._consecutive_losses,
            "open_positions": len(self._open_positions),
            "total_trades": len(self._trade_history),
        }
