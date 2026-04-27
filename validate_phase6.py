#!/usr/bin/env python3
"""
Phase 6 Validation — Execution Engine.

Tests without external dependencies:
- Position tracking
- Paper trading simulation
- Execution flow
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch

# Mock all external dependencies
sys.modules["pandas"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["httpx"] = MagicMock()
sys.modules["aiosqlite"] = MagicMock()
sys.modules["cryptography"] = MagicMock()
sys.modules["cryptography.hazmat"] = MagicMock()
sys.modules["pydantic"] = MagicMock()
sys.modules["pydantic"].BaseModel = type("BaseModel", (), {"model_dump": lambda self: {}, "__init__": lambda self, **kwargs: None})
sys.modules["pydantic"].Field = lambda *a, **k: None
sys.modules["pydantic"].field_validator = lambda *a, **k: lambda f: f
sys.modules["pydantic_core"] = MagicMock()
sys.modules["tenacity"] = MagicMock()
sys.modules["tenacity"].retry = MagicMock(return_value=lambda f: f)
sys.modules["tenacity"].stop_after_attempt = MagicMock()
sys.modules["tenacity"].wait_exponential = MagicMock()

# Auto-detect repo root
_repo_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(_repo_root))

from src.llm.client import Signal
from src.execution.position_tracker import PositionTracker, PositionState, PositionStatus
from src.execution.paper_trading import PaperTradingExecutor, PaperAccount
from src.execution.execution_engine import ExecutionEngine, ExecutionMode, ExecutionConfig

print("=" * 60)
print("PHASE 6: EXECUTION ENGINE — VALIDATION")
print("=" * 60)

errors = []
passed = 0

def test(name, condition, detail=""):
    global passed, errors
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        errors.append(f"FAIL: {name} — {detail}")
        print(f"  FAIL: {name} — {detail}")

# =====================================================================
# 1. Position Tracker
# =====================================================================
print("\n[1] Position Tracker")

tracker = PositionTracker()

# Add position
pos = PositionState(
    id="test_1",
    coin="BTC",
    direction="LONG",
    entry_price=50000,
    entry_time=datetime.now(),
    quantity=1.0,
    leverage=5.0,
    stop_loss_price=49000,
    take_profit_price=52000,
    risk_usd=1000,
    margin_used=10000,
)
tracker.add_position(pos)
test("Position added", tracker.open_position_count == 1)
test("Position retrievable", tracker.get_position("test_1") is not None)

# Update prices
prices = {"BTC": 51000}
triggers = tracker.update_prices(prices)
test("PnL updated", abs(pos.unrealized_pnl - 5000) < 10)  # (51000-50000)*1*5

# No trigger yet
prices = {"BTC": 50500}
triggers = tracker.update_prices(prices)
test("No trigger at 50500", len(triggers) == 0)

# Stop trigger
prices = {"BTC": 48900}
triggers = tracker.update_prices(prices)
test("Stop triggered at 48900", len(triggers) == 1)
test("Trigger is stop_loss", triggers[0]["trigger"] == "stop_loss")

# TP trigger  
# Reset position
pos2 = PositionState(
    id="test_2",
    coin="BTC",
    direction="LONG",
    entry_price=50000,
    entry_time=datetime.now(),
    quantity=1.0,
    leverage=5.0,
    stop_loss_price=49000,
    take_profit_price=52000,
    risk_usd=1000,
    margin_used=10000,
)
tracker2 = PositionTracker()
tracker2.add_position(pos2)
prices = {"BTC": 52100}
triggers2 = tracker2.update_prices(prices)
test("TP triggered at 52100", len(triggers2) == 1)
test("Trigger is take_profit", triggers2[0]["trigger"] == "take_profit")

# Close position
closed = tracker2.close_position("test_2", exit_price=52100, reason="take_profit", fees=10)
test("Position closed", closed is not None)
test("Realized PnL positive", closed.realized_pnl > 0)
test("Position removed from open", tracker2.open_position_count == 0)
test("Position in history", len(tracker2._history) == 1)

# Stats
test("Stats tracked", tracker2.stats["total_opened"] == 1)
test("Stats total_tp_hit", tracker2.stats["total_tp_hit"] == 1)

# Short position
pos3 = PositionState(
    id="test_3",
    coin="BTC",
    direction="SHORT",
    entry_price=50000,
    entry_time=datetime.now(),
    quantity=1.0,
    leverage=5.0,
    stop_loss_price=51000,
    take_profit_price=48000,
    risk_usd=1000,
    margin_used=10000,
)
tracker3 = PositionTracker()
tracker3.add_position(pos3)
prices = {"BTC": 51100}
triggers3 = tracker3.update_prices(prices)
test("Short stop triggered", len(triggers3) == 1)

# =====================================================================
# 2. Paper Trading
# =====================================================================
print("\n[2] Paper Trading")

async def test_paper():
    paper = PaperTradingExecutor(initial_equity=10000)
    
    # Open position
    result = await paper.place_market_order(
        coin="BTC", side="LONG", quantity=0.1, current_price=50000
    )
    test("Paper order filled", result.success and result.status.value == "filled")
    test("Fee tracked", result.fee_paid > 0)
    
    # Check account
    summary = await paper.get_account_summary()
    test("Account summary exists", "equity" in summary)
    test("Cash reduced", summary["cash"] < 10000)
    
    # Position exists
    positions = await paper.get_positions()
    test("Position created", len(positions) == 1)
    
    # Update price
    await paper.update_prices({"BTC": 51000})
    summary2 = await paper.get_account_summary()
    test("Price update changes equity", summary2["equity"] != summary["equity"])
    
    # Close position
    close_result = await paper.place_market_order(
        coin="BTC", side="SHORT", quantity=0.1, current_price=51000, reduce_only=True
    )
    test("Close order filled", close_result.success)
    
    # Final stats
    stats = paper.stats
    test("Stats include total_trades", stats["total_trades"] == 2)
    
    return True

# Run async test
asyncio.run(test_paper())

# =====================================================================
# 3. Execution Engine (Smoke Test)
# =====================================================================
print("\n[3] Execution Engine")

# Mock signal generator
class MockSignalGen:
    def __init__(self): self._init = False
    async def init(self): self._init = True
    async def generate(self, data): return Signal(direction="NEUTRAL", confidence=0.0)

# Mock risk manager
class MockRiskMgr:
    def __init__(self):
        self.stats = {"total_trades_accepted": 0, "total_trades_rejected": 0}
    def validate(self, **kwargs):
        from src.risk.manager import RiskDecision
        d = RiskDecision()
        d.approved = False
        d.rejection_reason = "test"
        return d
    def record_position_opened(self, pos): pass
    def record_trade_result(self, **kwargs): pass

# Create engine
config = ExecutionConfig(mode=ExecutionMode.PAPER, symbols=["BTC"], signal_interval_minutes=1)
engine = ExecutionEngine(config, MockSignalGen(), MockRiskMgr())

test("Engine created", engine is not None)
test("Default mode is PAPER", config.mode == ExecutionMode.PAPER)

# Can't switch to LIVE without confirmation
try:
    engine.enable_live_mode(confirmed=False)
    test("LIVE mode requires confirmation", False)
except ValueError:
    test("LIVE mode requires confirmation", True)

# Stats
stats = engine.stats
test("Stats tracked", "mode" in stats)

# =====================================================================
# Summary
# =====================================================================
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {len(errors)} failed")
print("=" * 60)

if errors:
    print("\nErrors:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\n ALL VALIDATIONS PASSED")
    print(" Phase 6 Execution Engine is structurally sound.")
    sys.exit(0)
