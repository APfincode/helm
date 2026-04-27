"""
Execution Module — Phase 6.

Live execution bridge connecting RiskManager decisions to Hyperliquid.

Flow:
    LLM Signal → RiskManager → ExecutionEngine → Hyperliquid API
                                   ↓
                            Paper Trading (optional)
                                   ↓
                            Position Tracker
                                   ↓
                            Audit Logger

Key Principles:
1. NEVER send raw LLM output to the exchange
2. All orders pre-validated by RiskManager
3. Paper trading mode for live validation without real money
4. Hyperliquid rate limits: 1200 calls / 10 min
5. Every order attempt is logged (success or failure)
6. Reduce-only position closure (never increase risk accidentally)
"""

from .hyperliquid_executor import HyperliquidExecutor, ExecutorConfig, OrderResult
from .paper_trading import PaperTradingExecutor
from .position_tracker import PositionTracker, PositionState
from .execution_engine import ExecutionEngine, ExecutionMode

__all__ = [
    "HyperliquidExecutor",
    "ExecutorConfig",
    "OrderResult",
    "PaperTradingExecutor",
    "PositionTracker",
    "PositionState",
    "ExecutionEngine",
    "ExecutionMode",
]
