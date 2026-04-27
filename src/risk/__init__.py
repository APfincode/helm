"""
Risk Management Module — Phase 5.

Hard overrides that make it IMPOSSIBLE for the LLM to blow up the account.
Architecture: LLM suggests direction → Risk Manager controls EVERYTHING else.

Key principle from Alpha Arena: LLMs CANNOT be trusted with risk management.
Claude lost -71% because it had discretion over leverage and position sizing.
Our harness removes ALL discretion from the LLM — it only picks direction.
"""

from .sizer import PositionSizer, SizingMethod
from .guard import RiskGuard, GuardResult
from .manager import RiskManager, RiskDecision

__all__ = [
    "PositionSizer",
    "SizingMethod",
    "RiskGuard",
    "GuardResult",
    "RiskManager",
    "RiskDecision",
]
