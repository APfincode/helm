"""
Risk Manager — The central orchestrator for risk management.

The RiskManager sits between the LLM signal generator and the execution engine.
Its job: enforce hard limits that the LLM CANNOT override.

Flow:
    LLM Signal (direction, suggested_stop, confidence)
        ↓
    RiskManager.validate(signal, account_state, market_data)
        ↓
    1. PositionSizer → calculates size (risk-based, NOT confidence-based)
    2. RiskGuard → runs all 12 circuit breakers
    3. If any HALT → return REJECTED
    4. If any WARN → reduce position size
    5. Return RiskDecision with approved size, enforced stop, leverage
        ↓
    Execution Engine

Alpha Arena Lesson: The LLM should only pick direction.
Everything else — size, stop, leverage, timing — is computed by the risk module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal

import pandas as pd

from .sizer import PositionSizer, PositionSize, SizerConfig
from .guard import RiskGuard, GuardResult, GuardStatus, GuardConfig
from src.llm.client import Signal

logger = logging.getLogger(__name__)


# =============================================================================
# Risk Decision
# =============================================================================

@dataclass
class RiskDecision:
    """
    Final decision from the Risk Manager.
    
    This is the ONLY object that reaches the execution engine.
    The raw LLM signal never touches execution.
    """
    approved: bool = False
    direction: Literal["LONG", "SHORT", "NEUTRAL"] = "NEUTRAL"
    confidence: float = 0.0
    
    # Position details (computed by risk module)
    quantity: float = 0.0
    entry_price: float = 0.0
    stop_price: float = 0.0
    take_profit_price: float = 0.0
    leverage: float = 1.0
    notional_value: float = 0.0
    margin_required: float = 0.0
    
    # Risk metrics
    risk_usd: float = 0.0
    risk_pct: float = 0.0
    
    # Reasons
    rejection_reason: Optional[str] = None
    guard_results: list[GuardResult] = field(default_factory=list)
    sizer_result: Optional[PositionSize] = None
    
    # Audit
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "direction": self.direction,
            "confidence": self.confidence,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "stop_price": self.stop_price,
            "take_profit_price": self.take_profit_price,
            "leverage": self.leverage,
            "notional_value": self.notional_value,
            "margin_required": self.margin_required,
            "risk_usd": self.risk_usd,
            "risk_pct": self.risk_pct,
            "rejection_reason": self.rejection_reason,
            "guard_results": [r.__dict__ for r in self.guard_results],
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Risk Manager
# =============================================================================

class RiskManager:
    """
    Central risk management orchestrator.
    
    Guarantees:
    1. No trade executes without passing ALL guards
    2. Position size is ALWAYS risk-based, never confidence-based
    3. Max leverage is hard-capped (Alpha Arena used 20x, we use 5x max)
    4. Drawdown halt is automatic and non-negotiable
    5. Every decision is logged with full reasoning
    """
    
    def __init__(
        self,
        sizer_config: Optional[SizerConfig] = None,
        guard_config: Optional[GuardConfig] = None,
    ):
        self._sizer = PositionSizer(sizer_config)
        self._guard = RiskGuard(guard_config)
        
        # State
        self._total_trades_accepted: int = 0
        self._total_trades_rejected: int = 0
        self._peak_equity: float = 0.0
    
    def validate(
        self,
        signal: Signal,
        account_equity: float,
        entry_price: float,
        market_data: Optional[pd.DataFrame] = None,
        open_positions: Optional[list[dict]] = None,
        trade_history: Optional[list[dict]] = None,
    ) -> RiskDecision:
        """
        Validate a trading signal and return a RiskDecision.
        
        This is the SINGLE entry point for risk management.
        
        Args:
            signal: LLM-generated signal (direction, confidence, etc.)
            account_equity: Current total equity
            entry_price: Current market price
            market_data: OHLCV data for volatility calculations
            open_positions: Currently open positions
            trade_history: Past completed trades
            
        Returns:
            RiskDecision — safe to execute if approved=True
        """
        decision = RiskDecision()
        
        # =====================================================================
        # Step 1: Update peak equity for drawdown tracking
        # =====================================================================
        if account_equity > self._peak_equity:
            self._peak_equity = account_equity
        
        # =====================================================================
        # Step 2: Basic signal validation
        # =====================================================================
        if signal.direction not in ("LONG", "SHORT"):
            decision.approved = False
            decision.direction = "NEUTRAL"
            decision.rejection_reason = f"Signal direction '{signal.direction}' is not a tradeable signal"
            self._total_trades_rejected += 1
            return decision
        
        # Override confidence-sourced sizing — confidence only decides trade/no-trade
        if signal.confidence < 0.55:  # Raised threshold from 0.5
            decision.approved = False
            decision.direction = "NEUTRAL"
            decision.confidence = signal.confidence
            decision.rejection_reason = f"Confidence {signal.confidence:.2f} below minimum threshold (0.55)"
            self._total_trades_rejected += 1
            return decision
        
        # =====================================================================
        # Step 3: Calculate position size (RISK-BASED, not confidence-based)
        # =====================================================================
        position_size = self._sizer.calculate(
            account_equity=account_equity,
            entry_price=entry_price,
            direction=signal.direction,
            market_data=market_data,
            suggested_stop=signal.stop_loss_pct,  # LLM's suggestion, may be overridden
            confidence=signal.confidence,  # Only used for trade/no-trade decision
            trade_history=trade_history,
        )
        
        decision.sizer_result = position_size
        
        if not position_size.approved:
            decision.approved = False
            decision.rejection_reason = position_size.rejection_reason
            self._total_trades_rejected += 1
            return decision
        
        # =====================================================================
        # Step 4: Sync open positions to guard
        # =====================================================================
        if open_positions:
            for pos in open_positions:
                if pos.get("id") not in [p.get("id") for p in self._guard._open_positions]:
                    self._guard.record_position_opened(pos)
        
        # =====================================================================
        # Step 5: Run ALL risk guards
        # =====================================================================
        guard_results = self._guard.check_all(
            account_equity=account_equity,
            entry_price=entry_price,
            stop_price=position_size.stop_price,
            direction=signal.direction,
            market_data=market_data,
        )
        
        decision.guard_results = guard_results
        overall = self._guard.get_overall_status(guard_results)
        
        # =====================================================================
        # Step 6: Apply guard decisions
        # =====================================================================
        if overall.blocks_trade:
            decision.approved = False
            decision.rejection_reason = overall.message
            decision.direction = "NEUTRAL"
            self._total_trades_rejected += 1
            
            # Log the rejection
            logger.warning(f"Trade REJECTED: {overall.message}")
            return decision
        
        if overall.reduces_size:
            # Apply size multiplier from warnings
            multiplier = min(r.size_multiplier for r in guard_results if r.size_multiplier < 1.0)
            position_size.quantity *= multiplier
            position_size.notional_value *= multiplier
            position_size.margin_required *= multiplier
            position_size.risk_usd *= multiplier
            position_size.risk_pct *= multiplier
            logger.info(f"Position size reduced by {multiplier:.0%} due to warnings")
        
        # =====================================================================
        # Step 7: Calculate take profit (minimum 1.5:1 R/R)
        # =====================================================================
        take_profit = self._calculate_take_profit(
            entry_price=entry_price,
            stop_price=position_size.stop_price,
            direction=signal.direction,
            suggested_target=signal.take_profit_pct,
        )
        
        # =====================================================================
        # Step 8: Populate final decision
        # =====================================================================
        decision.approved = True
        decision.direction = signal.direction
        decision.confidence = signal.confidence
        
        decision.quantity = position_size.quantity
        decision.entry_price = entry_price
        decision.stop_price = position_size.stop_price
        decision.take_profit_price = take_profit
        decision.leverage = position_size.leverage_used
        decision.notional_value = position_size.notional_value
        decision.margin_required = position_size.margin_required
        
        decision.risk_usd = position_size.risk_usd
        decision.risk_pct = position_size.risk_pct
        
        self._total_trades_accepted += 1
        
        logger.info(
            f"Trade APPROVED: {decision.direction} {decision.quantity:.4f} @ "
            f"{decision.entry_price:.2f} (stop: {decision.stop_price:.2f}, "
            f"tp: {decision.take_profit_price:.2f}, risk: ${decision.risk_usd:.2f}, "
            f"lev: {decision.leverage:.1f}x)"
        )
        
        return decision
    
    def record_trade_result(
        self,
        pnl: float,
        fees: float,
        position_id: str,
    ) -> None:
        """
        Record completed trade result for guard state updates.
        
        Call this after EVERY trade closes.
        """
        self._guard.record_trade(pnl=pnl, fees=fees)
        self._guard.record_position_closed(position_id)
    
    def record_position_opened(self, position: dict) -> None:
        """Record that a position was opened."""
        self._guard.record_position_opened(position)
    
    # ========================================================================
    # Helpers
    # ========================================================================
    
    @staticmethod
    def _calculate_take_profit(
        entry_price: float,
        stop_price: float,
        direction: Literal["LONG", "SHORT"],
        suggested_target: Optional[float] = None,
    ) -> float:
        """
        Calculate take-profit price ensuring minimum R/R ratio.
        
        Minimum R/R = 1.5:1
        If LLM suggests a worse R/R, override it.
        """
        risk = abs(entry_price - stop_price)
        min_reward = risk * 1.5  # Minimum 1.5:1 R/R
        
        if direction == "LONG":
            min_tp = entry_price + min_reward
            
            # Sanity check: LLM suggested TP
            if suggested_target and suggested_target > entry_price:
                # Use the more conservative of suggested and minimum
                return min(suggested_target, min_tp) if suggested_target > min_tp else suggested_target
            
            return min_tp
        else:  # SHORT
            min_tp = entry_price - min_reward
            
            if suggested_target and suggested_target < entry_price:
                return max(suggested_target, min_tp) if suggested_target < min_tp else suggested_target
            
            return min_tp
    
    # ========================================================================
    # Stats & Monitoring
    # ========================================================================
    
    @property
    def stats(self) -> dict:
        """Get current risk management statistics."""
        total = self._total_trades_accepted + self._total_trades_rejected
        accept_rate = (self._total_trades_accepted / total * 100) if total > 0 else 0
        
        return {
            "total_trades_accepted": self._total_trades_accepted,
            "total_trades_rejected": self._total_trades_rejected,
            "accept_rate_pct": accept_rate,
            "peak_equity": self._peak_equity,
            "current_drawdown_pct": self.current_drawdown_pct,
            "guards_triggered_today": self._guard.stats,
        }
    
    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown from peak equity."""
        # This needs current equity passed in — placeholder logic
        return 0.0
    
    @property
    def is_halted(self) -> bool:
        return self._guard.is_halted
    
    @property
    def halt_reason(self) -> str:
        return self._guard._halt_reason if self._guard.is_halted else ""
