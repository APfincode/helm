"""
Position Sizer — Calculates position size based on RISK, not confidence.

Alpha Arena Lesson: Qwen3 won by YOLOing 100% of capital on one trade.
If BTC had dumped 15%, it would have been liquidated.

Our approach: Position size = max_risk_usd / (entry_price - stop_price)
This means: you lose the SAME amount whether the trade is "high confidence" or "low confidence"
The ONLY thing that changes is how MUCH you stand to gain.

Methods:
- FIXED_RISK: Risk exactly N% of account per trade (default: 1%)
- VOLATILITY_ADJUSTED: Reduce size when ATR is high (choppy market = smaller)
- KELLY_FRACTIONAL: Kelly Criterion with half-Kelly fraction (premium only)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Literal

import pandas as pd


class SizingMethod(str, Enum):
    FIXED_RISK = "fixed_risk"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    KELLY_FRACTIONAL = "kelly_fractional"


@dataclass
class PositionSize:
    """
    Complete position sizing result.
    
    The LLM should NEVER see this directly — the risk manager uses
    position_size.quantity for execution while the LLM only knows
    it got a "LONG" or "SHORT" signal.
    """
    quantity: float = 0.0  # Base currency units (e.g., BTC)
    notional_value: float = 0.0  # USD value of position
    margin_required: float = 0.0  # USD margin at current leverage
    leverage_used: float = 1.0
    
    # Risk metrics
    risk_usd: float = 0.0  # Amount at risk if stop is hit
    risk_pct: float = 0.0  # Risk as % of account
    stop_price: float = 0.0
    entry_price: float = 0.0
    
    # Sizing details
    method: SizingMethod = SizingMethod.FIXED_RISK
    sizing_reason: str = ""
    
    # Validation
    approved: bool = True
    rejection_reason: Optional[str] = None
    
    @property
    def is_zero(self) -> bool:
        return self.quantity == 0.0 or not self.approved
    
    def to_dict(self) -> dict:
        return {
            "quantity": self.quantity,
            "notional_value": self.notional_value,
            "margin_required": self.margin_required,
            "leverage_used": self.leverage_used,
            "risk_usd": self.risk_usd,
            "risk_pct": self.risk_pct,
            "stop_price": self.stop_price,
            "entry_price": self.entry_price,
            "method": self.method.value,
            "sizing_reason": self.sizing_reason,
            "approved": self.approved,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class SizerConfig:
    """Configuration for position sizing."""
    # Fixed Risk parameters (default: 1% per trade)
    fixed_risk_pct: float = 1.0  # Risk 1% of account per trade
    
    # Volatility adjustment
    atr_lookback: int = 14
    atr_multiple: float = 2.0  # Stop distance = ATR * multiple
    max_volatility_pct: float = 5.0  # If ATR% > 5%, position size → 0 (chop)
    
    # Kelly Criterion
    kelly_fraction: float = 0.3  # Use 30% of full Kelly (half-Kelly = 0.5)
    kelly_min_trades: int = 30  # Need 30+ trades for Kelly estimate
    
    # Universal limits
    max_position_pct: float = 0.15  # No single position > 15% of capital
    max_leverage: float = 5.0  # NEVER exceed 5x (Alpha Arena was 20x)
    min_trade_size_usd: float = 10.0  # Minimum $10 trade
    
    @property
    def max_risk_per_trade(self) -> float:
        """Return max risk per trade as percentage."""
        return self.fixed_risk_pct


class PositionSizer:
    """
    Calculates position sizes that make it IMPOSSIBLE to blow up.
    
    The formula is simple:
        position_size = account_risk_usd / (entry_price - stop_price)
    
    This means:
    - If stop is close (tight setup) → smaller position (same risk)
    - If stop is far (volatile setup) → smaller position (same risk)
    - If account is small → smaller position
    - Risk is ALWAYS the same % of account, regardless of "confidence"
    """
    
    def __init__(self, config: Optional[SizerConfig] = None):
        self._config = config or SizerConfig()
    
    def calculate(
        self,
        account_equity: float,
        entry_price: float,
        direction: Literal["LONG", "SHORT"],
        market_data: Optional[pd.DataFrame] = None,
        suggested_stop: Optional[float] = None,
        suggested_take_profit: Optional[float] = None,
        confidence: float = 0.5,
        method: SizingMethod = SizingMethod.FIXED_RISK,
        trade_history: Optional[list[dict]] = None,
    ) -> PositionSize:
        """
        Calculate position size.
        
        This is the ONLY method for determining position size.
        The LLM's "confidence" is IGNORED for sizing — it only affects
        whether we take the trade at all (confidence < 0.6 → NEUTRAL).
        
        Args:
            account_equity: Current total equity (cash + unrealized PnL)
            entry_price: Price at entry
            direction: LONG or SHORT
            market_data: OHLCV data for volatility calculation
            suggested_stop: LLM's suggested stop (used as reference, may be overridden)
            suggested_take_profit: LLM's suggested target
            confidence: LLM confidence (ONLY used for trade/no-trade, not sizing)
            method: SizingMethod to use
            trade_history: Past trades for Kelly Criterion
            
        Returns:
            PositionSize with complete sizing details
        """
        result = PositionSize(
            entry_price=entry_price,
            method=method,
        )
        
        # =====================================================================
        # Step 1: Validate inputs
        # =====================================================================
        if account_equity <= 0:
            result.approved = False
            result.rejection_reason = "Account equity is zero or negative"
            return result
        
        if entry_price <= 0:
            result.approved = False
            result.rejection_reason = "Invalid entry price"
            return result
        
        # =====================================================================
        # Step 2: Calculate stop loss distance
        # =====================================================================
        stop_price = self._calculate_stop_price(
            entry_price=entry_price,
            direction=direction,
            market_data=market_data,
            suggested_stop=suggested_stop,
        )
        
        result.stop_price = stop_price
        
        # Validate stop distance
        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0:
            result.approved = False
            result.rejection_reason = "Stop loss distance is zero — cannot calculate risk"
            return result
        
        # =====================================================================
        # Step 3: Calculate risk amount (in USD)
        # =====================================================================
        risk_usd = account_equity * (self._config.fixed_risk_pct / 100)
        result.risk_usd = risk_usd
        result.risk_pct = self._config.fixed_risk_pct
        
        # =====================================================================
        # Step 4: Calculate raw position size
        # =====================================================================
        # position_size = risk_usd / stop_distance
        # Example: $100k account, 1% risk = $1,000
        # BTC entry $50,000, stop $49,000 → stop_distance = $1,000
        # quantity = $1,000 / $1,000 = 1.0 BTC
        raw_quantity = risk_usd / stop_distance
        
        # =====================================================================
        # Step 5: Apply volatility adjustment (if method = VOLATILITY_ADJUSTED)
        # =====================================================================
        if method == SizingMethod.VOLATILITY_ADJUSTED and market_data is not None:
            vol_factor = self._volatility_factor(market_data)
            raw_quantity *= vol_factor
            result.sizing_reason = f"Volatility-adjusted: factor={vol_factor:.2f}"
        
        # =====================================================================
        # Step 6: Apply Kelly Criterion (if method = KELLY_FRACTIONAL)
        # =====================================================================
        if method == SizingMethod.KELLY_FRACTIONAL and trade_history:
            kelly_size = self._kelly_size(
                entry_price=entry_price,
                stop_price=stop_price,
                trade_history=trade_history,
                account_equity=account_equity,
            )
            raw_quantity = min(raw_quantity, kelly_size)
            result.sizing_reason = f"Kelly-fractional capped at {kelly_size:.4f}"
        
        # =====================================================================
        # Step 7: Apply hard limits
        # =====================================================================
        notional = raw_quantity * entry_price
        margin = notional / self._config.max_leverage
        
        # Cap position to max % of equity
        max_notional = account_equity * self._config.max_position_pct
        if notional > max_notional:
            raw_quantity = max_notional / entry_price
            notional = max_notional
            margin = notional / self._config.max_leverage
            result.sizing_reason += "; capped by max_position_pct"
        
        # Cap leverage
        implied_leverage = notional / margin if margin > 0 else 0
        if implied_leverage > self._config.max_leverage:
            margin = notional / self._config.max_leverage
            result.sizing_reason += "; leverage capped"
        
        # Minimum trade size
        if notional < self._config.min_trade_size_usd:
            result.approved = False
            result.rejection_reason = f"Position too small: ${notional:.2f} < min ${self._config.min_trade_size_usd}"
            return result
        
        # =====================================================================
        # Step 8: Populate result
        # =====================================================================
        result.quantity = raw_quantity
        result.notional_value = notional
        result.margin_required = margin
        result.leverage_used = self._config.max_leverage if notional > account_equity else notional / account_equity
        
        # Recalculate actual risk
        actual_risk = raw_quantity * stop_distance
        result.risk_usd = actual_risk
        result.risk_pct = (actual_risk / account_equity) * 100
        
        result.approved = True
        if not result.sizing_reason:
            result.sizing_reason = f"Fixed risk: ${actual_risk:.2f} ({result.risk_pct:.2f}%) at {entry_price:.2f}"
        
        return result
    
    # ========================================================================
    # Stop Loss Calculation
    # ========================================================================
    
    def _calculate_stop_price(
        self,
        entry_price: float,
        direction: Literal["LONG", "SHORT"],
        market_data: Optional[pd.DataFrame],
        suggested_stop: Optional[float],
    ) -> float:
        """
        Calculate stop loss price.
        
        Priority:
        1. Use suggested stop IF it provides reasonable risk
        2. Fallback to volatility-based stop (ATR * N)
        3. Fallback to percentage-based stop (default 2%)
        """
        # Validate suggested stop
        if suggested_stop is not None and suggested_stop > 0:
            distance = abs(entry_price - suggested_stop)
            distance_pct = (distance / entry_price) * 100
            
            # Only accept stops between 0.5% and 10%
            if 0.5 <= distance_pct <= 10.0:
                return suggested_stop
        
        # Volatility-based stop
        if market_data is not None and not market_data.empty:
            atr = self._calculate_atr(market_data, self._config.atr_lookback)
            if atr > 0:
                atr_pct = (atr / entry_price) * 100
                # If ATR% is > 5%, market is too volatile — return conservative stop
                if atr_pct > self._config.max_volatility_pct:
                    return self._percentage_stop(entry_price, direction, pct=3.0)
                
                # Normal volatility: ATR * 2
                if direction == "LONG":
                    return entry_price - (atr * self._config.atr_multiple)
                else:
                    return entry_price + (atr * self._config.atr_multiple)
        
        # Default percentage stop
        return self._percentage_stop(entry_price, direction)
    
    @staticmethod
    def _percentage_stop(
        entry_price: float,
        direction: Literal["LONG", "SHORT"],
        pct: float = 2.0,
    ) -> float:
        """
        Calculate percentage-based stop loss.
        Default: 2% away from entry.
        """
        if direction == "LONG":
            return entry_price * (1 - pct / 100)
        else:
            return entry_price * (1 + pct / 100)
    
    # ========================================================================
    # Volatility Adjustment
    # ========================================================================
    
    def _volatility_factor(self, market_data: pd.DataFrame) -> float:
        """
        Calculate position size reduction factor based on volatility.
        
        High volatility → smaller positions.
        Very high volatility → 0 (don't trade chop).
        """
        if len(market_data) < self._config.atr_lookback + 1:
            return 1.0
        
        recent = market_data.tail(20)
        avg_range = (recent["high"] - recent["low"]).mean()
        current_price = recent["close"].iloc[-1]
        
        if current_price == 0:
            return 1.0
        
        avg_range_pct = (avg_range / current_price) * 100
        
        # Volatility factor
        if avg_range_pct > self._config.max_volatility_pct:
            # Too volatile — reduce to 50%
            return 0.5
        elif avg_range_pct > 3.0:
            # Elevated volatility — reduce to 75%
            return 0.75
        else:
            # Normal volatility
            return 1.0
    
    @staticmethod
    def _calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(data) < period + 1:
            return 0.0
        
        high = data["high"]
        low = data["low"]
        close = data["close"]
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return float(atr) if pd.notna(atr) else 0.0
    
    # ========================================================================
    # Kelly Criterion
    # ========================================================================
    
    def _kelly_size(
        self,
        entry_price: float,
        stop_price: float,
        trade_history: list[dict],
        account_equity: float,
    ) -> float:
        """
        Calculate position size using fractional Kelly Criterion.
        
        Kelly % = W/R - (1-W)
        Where:
        - W = win rate (probability of winning)
        - R = average win / average loss (reward-to-risk ratio)
        
        Fractional Kelly = full_kelly * kelly_fraction (default: 30%)
        """
        if len(trade_history) < self._config.kelly_min_trades:
            return float("inf")  # No cap if insufficient data
        
        wins = [t for t in trade_history if t.get("pnl", 0) > 0]
        losses = [t for t in trade_history if t.get("pnl", 0) < 0]
        
        if not wins or not losses:
            return float("inf")
        
        win_rate = len(wins) / len(trade_history)
        avg_win = sum(t["pnl"] for t in wins) / len(wins)
        avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses))
        
        if avg_loss == 0:
            return float("inf")
        
        r_ratio = avg_win / avg_loss
        
        # Kelly fraction
        kelly = win_rate - ((1 - win_rate) / r_ratio)
        kelly = max(0, min(kelly, 0.5))  # Cap at 50% Kelly for safety
        
        fractional_kelly = kelly * self._config.kelly_fraction
        
        # Convert Kelly to position size
        stop_distance = abs(entry_price - stop_price)
        risk_amount = account_equity * fractional_kelly
        
        if stop_distance == 0:
            return float("inf")
        
        return risk_amount / stop_distance
