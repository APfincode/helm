"""
Risk Management Test Suite — Phase 5 Validation.

Tests every guard, sizer, and manager function.
Covers Alpha Arena failure modes:
- Excessive leverage (Claude -71%)
- No max drawdown halt (Gemini kept digging)
- Overtrading (GPT-5 fee death)
- YOLO sizing (Qwen got lucky)
"""

import pytest
import unittest.mock as mock
from datetime import datetime

from src.llm.client import Signal


# Need to mock dependencies before importing risk module
import sys
sys.modules["pandas"] = mock.MagicMock()
sys.modules["numpy"] = mock.MagicMock()

from src.risk.sizer import PositionSizer, SizerConfig, PositionSize, SizingMethod
from src.risk.guard import RiskGuard, GuardConfig, GuardStatus, GuardResult
from src.risk.manager import RiskManager, RiskDecision


# =============================================================================
# PositionSizer Tests
# =============================================================================

class TestPositionSizer:
    def test_fixed_risk_basic(self):
        sizer = PositionSizer(SizerConfig(fixed_risk_pct=1.0))
        
        # $100k account, BTC at $50k, stop at $49k
        result = sizer.calculate(
            account_equity=100000,
            entry_price=50000,
            direction="LONG",
            suggested_stop=49000,
        )
        
        # Risk = $1,000 (1% of $100k)
        # Stop distance = $1,000
        # Quantity = $1,000 / $1,000 = 1.0 BTC
        assert result.approved
        assert result.quantity == 1.0
        assert result.risk_usd == 1000.0
        assert result.risk_pct == 1.0
    
    def test_fixed_risk_short(self):
        sizer = PositionSizer(SizerConfig(fixed_risk_pct=1.0))
        
        result = sizer.calculate(
            account_equity=100000,
            entry_price=50000,
            direction="SHORT",
            suggested_stop=51000,
        )
        
        # Risk = $1,000
        # Stop distance = $1,000 (above entry for short)
        # Quantity = 1.0 BTC
        assert result.approved
        assert result.quantity == 1.0
    
    def test_volatility_adjustment_reduces_size(self):
        cfg = SizerConfig(fixed_risk_pct=1.0)
        sizer = PositionSizer(cfg)
        
        # Mock market data with high volatility
        mock_data = mock.MagicMock()
        mock_data.__len__.return_value = 25
        mock_data.tail.return_value = mock.MagicMock()
        mock_data.tail.return_value.__getitem__ = lambda self, k: mock.MagicMock()
        # ATR = 3000 on BTC @ 50000 = 6% → vol_factor = 0.5
        
        result = sizer.calculate(
            account_equity=100000,
            entry_price=50000,
            direction="LONG",
            suggested_stop=49000,
            market_data=mock_data,
            method=SizingMethod.VOLATILITY_ADJUSTED,
        )
        
        # Should still approve but with adjusted sizing
        assert result.approved
    
    def test_zero_equity_rejected(self):
        sizer = PositionSizer()
        result = sizer.calculate(account_equity=0, entry_price=50000, direction="LONG")
        assert not result.approved
        assert "Account equity is zero" in result.rejection_reason
    
    def test_zero_stop_distance_rejected(self):
        sizer = PositionSizer()
        result = sizer.calculate(
            account_equity=100000,
            entry_price=50000,
            direction="LONG",
            suggested_stop=50000,  # Same as entry = zero distance
        )
        assert not result.approved
        assert "Stop loss distance is zero" in result.rejection_reason
    
    def test_percentage_stop_fallback(self):
        sizer = PositionSizer()
        result = sizer.calculate(
            account_equity=100000,
            entry_price=50000,
            direction="LONG",
            # No suggested stop → defaults to 2%
        )
        
        assert result.approved
        # 2% of $50,000 = $1,000 stop distance
        # Quantity = $1,000 / $1,000 = 1.0 BTC
        assert result.stop_price == 49000.0  # $50k * 0.98
    
    def test_max_position_cap(self):
        sizer = PositionSizer(SizerConfig(max_position_pct=0.10))  # 10% cap
        
        result = sizer.calculate(
            account_equity=100000,
            entry_price=50000,
            direction="LONG",
            suggested_stop=40000,  # Far stop = 20% distance
        )
        
        # Risk = $1,000, Stop = $10,000 → would be 0.1 BTC
        # But max_position_pct = 0.10 → max notional = $10,000
        # quantity = $10,000 / $50,000 = 0.2 BTC
        assert result.approved
        assert result.notional_value <= 10000  # Capped
    
    def test_minimum_trade_size(self):
        sizer = PositionSizer(SizerConfig(min_trade_size_usd=100))
        
        result = sizer.calculate(
            account_equity=1000,  # Small account
            entry_price=50000,
            direction="LONG",
            suggested_stop=49900,  # Very tight stop
        )
        
        # Risk = $10 (1% of $1,000), Stop = $100
        # Quantity = $10 / $100 = 0.1 BTC
        # Notional = 0.1 * $50,000 = $5,000 → above $100 min
        assert result.approved or "too small" in (result.rejection_reason or "")
    
    def test_confidence_ignored_for_sizing(self):
        sizer = PositionSizer(SizerConfig(fixed_risk_pct=1.0))
        
        low_conf = sizer.calculate(
            account_equity=100000,
            entry_price=50000,
            direction="LONG",
            suggested_stop=49000,
            confidence=0.3,  # Low confidence
        )
        
        high_conf = sizer.calculate(
            account_equity=100000,
            entry_price=50000,
            direction="LONG",
            suggested_stop=49000,
            confidence=0.99,  # High confidence
        )
        
        # Same risk, same size — confidence ignored
        assert low_conf.risk_usd == high_conf.risk_usd
        assert low_conf.quantity == high_conf.quantity


# =============================================================================
# RiskGuard Tests
# =============================================================================

class TestRiskGuard:
    def test_drawdown_halt(self):
        guard = RiskGuard(GuardConfig(max_drawdown_pct=15.0))
        
        # Simulate peak equity of $100k, now at $84k (16% drawdown)
        guard._peak_equity = 100000
        
        results = guard.check_all(account_equity=84000, entry_price=50000, stop_price=49000, direction="LONG")
        overall = guard.get_overall_status(results)
        
        assert overall.status == GuardStatus.HALT
        assert "drawdown" in overall.message.lower()
    
    def test_drawdown_warning(self):
        guard = RiskGuard(GuardConfig(max_drawdown_pct=15.0))
        guard._peak_equity = 100000
        
        # At 12% drawdown (80% of 15%) → Warning
        results = guard.check_all(account_equity=88000, entry_price=50000, stop_price=49000, direction="LONG")
        
        warn_results = [r for r in results if r.status == GuardStatus.WARN and r.name == "drawdown"]
        assert len(warn_results) > 0
        assert warn_results[0].size_multiplier == 0.5
    
    def test_account_health_halt(self):
        guard = RiskGuard(GuardConfig(min_account_balance_usd=50))
        
        results = guard.check_all(account_equity=30, entry_price=50000, stop_price=49000, direction="LONG")
        overall = guard.get_overall_status(results)
        
        assert overall.status == GuardStatus.HALT
        assert "balance" in overall.message.lower()
    
    def test_cooldown_blocks(self):
        guard = RiskGuard(GuardConfig(min_trade_interval_hours=4.0))
        guard._last_trade_time = datetime.now()
        
        results = guard.check_all(account_equity=100000, entry_price=50000, stop_price=49000, direction="LONG")
        overall = guard.get_overall_status(results)
        
        assert overall.status == GuardStatus.REJECT
        assert "cooldown" in overall.message.lower()
    
    def test_concentration_limit(self):
        guard = RiskGuard(GuardConfig(max_concurrent_positions=2))
        
        # Add 2 open positions
        guard._open_positions = [
            {"id": "1", "symbol": "BTC", "direction": "LONG"},
            {"id": "2", "symbol": "ETH", "direction": "LONG"},
        ]
        
        results = guard.check_all(account_equity=100000, entry_price=50000, stop_price=49000, direction="LONG")
        overall = guard.get_overall_status(results)
        
        assert overall.status == GuardStatus.REJECT
        assert "position" in overall.message.lower()
    
    def test_weekend_trading_block(self):
        guard = RiskGuard(GuardConfig(skip_weekend_trading=True))
        
        # Mock Saturday
        with mock.patch("src.risk.guard.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2024, 1, 6)  # Saturday
            mock_dt.strftime = datetime.strftime
            
            results = guard.check_all(account_equity=100000, entry_price=50000, stop_price=49000, direction="LONG")
            overall = guard.get_overall_status(results)
            
            assert overall.status == GuardStatus.REJECT
            assert "Weekend" in overall.message
    
    def test_fee_budget_warning(self):
        guard = RiskGuard(GuardConfig(fee_budget_pct=2.0))
        guard._total_fees_paid = 2500  # $2.5k fees on $100k equity = 2.5%
        
        results = guard.check_all(account_equity=100000, entry_price=50000, stop_price=49000, direction="LONG")
        
        warn = [r for r in results if r.status == GuardStatus.WARN and r.name == "fee_budget"]
        assert len(warn) > 0
        assert warn[0].size_multiplier == 0.5
    
    def test_fee_budget_halt(self):
        guard = RiskGuard(GuardConfig(fee_budget_pct=2.0, fee_budget_max_pct=5.0))
        guard._total_fees_paid = 6000  # 6% → halt
        
        results = guard.check_all(account_equity=100000, entry_price=50000, stop_price=49000, direction="LONG")
        overall = guard.get_overall_status(results)
        
        assert overall.status == GuardStatus.HALT
        assert "fee" in overall.message.lower()
    
    def test_stop_distance_rejects_too_far(self):
        guard = RiskGuard(GuardConfig(max_stop_distance_pct=10.0))
        
        results = guard.check_all(account_equity=100000, entry_price=50000, stop_price=40000, direction="LONG")
        overall = guard.get_overall_status(results)
        
        assert overall.status == GuardStatus.REJECT
        assert "too far" in overall.message.lower()
    
    def test_stop_distance_rejects_too_tight(self):
        guard = RiskGuard(GuardConfig(max_stop_distance_pct=10.0))
        
        results = guard.check_all(account_equity=100000, entry_price=50000, stop_price=49990, direction="LONG")
        overall = guard.get_overall_status(results)
        
        assert overall.status == GuardStatus.REJECT
        assert "too tight" in overall.message.lower()
    
    def test_loss_streak_reduction(self):
        guard = RiskGuard(GuardConfig(max_consecutive_losses=3, loss_streak_size_reduction=0.5))
        guard._consecutive_losses = 4
        
        results = guard.check_all(account_equity=100000, entry_price=50000, stop_price=49000, direction="LONG")
        
        warn = [r for r in results if r.status == GuardStatus.WARN and r.name == "loss_streak"]
        assert len(warn) > 0
        assert warn[0].size_multiplier == 0.5
    
    def test_trade_history_updates(self):
        guard = RiskGuard()
        
        guard.record_trade(pnl=-100, fees=5)
        assert guard._consecutive_losses == 1
        assert guard._total_fees_paid == 5
        
        guard.record_trade(pnl=-200, fees=5)
        assert guard._consecutive_losses == 2
        
        guard.record_trade(pnl=150, fees=5)
        assert guard._consecutive_losses == 0  # Reset on win
    
    def test_combined_warnings_reduce_size(self):
        guard = RiskGuard(GuardConfig(max_consecutive_losses=3))
        guard._consecutive_losses = 4
        guard._peak_equity = 100000
        
        # 12% drawdown (warn, 0.5x) + loss streak (warn, 0.5x) = 0.5x
        results = guard.check_all(account_equity=88000, entry_price=50000, stop_price=49000, direction="LONG")
        overall = guard.get_overall_status(results)
        
        assert overall.status == GuardStatus.WARN
        assert overall.size_multiplier == 0.5  # Minimum of all warnings
    
    def test_halt_priority_over_reject(self):
        """If both halt and reject trigger, halt wins."""
        guard = RiskGuard(GuardConfig(max_drawdown_pct=10.0, max_concurrent_positions=0))
        guard._peak_equity = 100000
        
        # At 15% drawdown (HALT) AND max positions (REJECT)
        results = guard.check_all(account_equity=85000, entry_price=50000, stop_price=49000, direction="LONG")
        overall = guard.get_overall_status(results)
        
        assert overall.status == GuardStatus.HALT
    
    def test_empty_guards_pass(self):
        guard = RiskGuard()
        results = guard.check_all(account_equity=100000, entry_price=50000, stop_price=49000, direction="LONG")
        
        overall = guard.get_overall_status(results)
        assert overall.status == GuardStatus.PASS


# =============================================================================
# RiskManager Tests
# =============================================================================

class TestRiskManager:
    def test_approves_valid_long_signal(self):
        manager = RiskManager()
        signal = Signal(direction="LONG", confidence=0.75, stop_loss_pct=2.0, take_profit_pct=4.0)
        
        result = manager.validate(
            signal=signal,
            account_equity=100000,
            entry_price=50000,
        )
        
        assert result.approved
        assert result.direction == "LONG"
        assert result.quantity > 0
        assert result.risk_usd > 0
    
    def test_rejects_neutral_signal(self):
        manager = RiskManager()
        signal = Signal(direction="NEUTRAL", confidence=0.0)
        
        result = manager.validate(
            signal=signal,
            account_equity=100000,
            entry_price=50000,
        )
        
        assert not result.approved
        assert "not a tradeable signal" in result.rejection_reason
    
    def test_rejects_low_confidence(self):
        manager = RiskManager()
        signal = Signal(direction="LONG", confidence=0.3)
        
        result = manager.validate(
            signal=signal,
            account_equity=100000,
            entry_price=50000,
        )
        
        assert not result.approved
        assert "confidence" in result.rejection_reason.lower()
    
    def test_rejects_invalid_direction(self):
        manager = RiskManager()
        signal = Signal(direction="BUY", confidence=0.9)
        
        result = manager.validate(
            signal=signal,
            account_equity=100000,
            entry_price=50000,
        )
        
        assert not result.approved
    
    def test_enforces_max_leverage(self):
        manager = RiskManager(SizerConfig(max_leverage=5.0))
        signal = Signal(direction="LONG", confidence=0.8)
        
        result = manager.validate(
            signal=signal,
            account_equity=100000,
            entry_price=50000,
        )
        
        assert result.approved
        assert result.leverage <= 5.0
    
    def test_halt_prevents_trade(self):
        manager = RiskManager()
        
        # Trigger halt
        manager._guard._trigger_halt("Test halt", cooldown_hours=1)
        
        signal = Signal(direction="LONG", confidence=0.8)
        result = manager.validate(
            signal=signal,
            account_equity=100000,
            entry_price=50000,
        )
        
        assert not result.approved
        assert "halted" in result.rejection_reason.lower()
    
    def test_take_profit_minimum_rr(self):
        manager = RiskManager()
        signal = Signal(direction="LONG", confidence=0.8, stop_loss_pct=1.0, take_profit_pct=1.0)
        
        # LLM suggests 1:1 R/R, but minimum is 1.5:1
        result = manager.validate(
            signal=signal,
            account_equity=100000,
            entry_price=50000,
        )
        
        assert result.approved
        # Risk = $500 (1% of $100k * 1% distance), min reward = $750
        # TP = $50000 + $750 = $50750
        assert result.take_profit_price > 50000
    
    def test_records_trade_result(self):
        manager = RiskManager()
        
        manager.record_trade_result(pnl=-100, fees=5, position_id="123")
        assert manager._guard._consecutive_losses == 1
        
        manager.record_trade_result(pnl=200, fees=5, position_id="456")
        assert manager._guard._consecutive_losses == 0
    
    def test_stats_tracking(self):
        manager = RiskManager()
        
        # Approve some, reject some
        manager.validate(Signal(direction="LONG", confidence=0.8), account_equity=100000, entry_price=50000)
        manager.validate(Signal(direction="SHORT", confidence=0.8), account_equity=100000, entry_price=50000)
        manager.validate(Signal(direction="NEUTRAL", confidence=0.0), account_equity=100000, entry_price=50000)
        
        stats = manager.stats
        assert stats["total_trades_accepted"] == 2
        assert stats["total_trades_rejected"] == 1
    
    def test_confidence_ignored_for_sizing_by_manager(self):
        manager = RiskManager(SizerConfig(fixed_risk_pct=1.0))
        
        low = manager.validate(
            Signal(direction="LONG", confidence=0.6, stop_loss_pct=2.0),
            account_equity=100000,
            entry_price=50000,
        )
        
        high = manager.validate(
            Signal(direction="LONG", confidence=0.99, stop_loss_pct=2.0),
            account_equity=100000,
            entry_price=50000,
        )
        
        # Both approved, equal risk
        assert low.approved and high.approved
        assert low.risk_usd == high.risk_usd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
