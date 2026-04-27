"""
LLM module — Multi-provider LLM client with failsafe signal generation.

Safety layers:
1. Multi-provider failover (if primary fails, try backup)
2. Strict response validation (Pydantic schemas)
3. Prompt injection detection (before sending)
4. Automatic NEUTRAL fallback on any error
5. Token/cost tracking with budget limits

Usage:
    from src.llm.client import LLMClient
    from src.llm.prompt_engine import PromptEngine
    from src.llm.signal_generator import SignalGenerator
    
    client = LLMClient()
    engine = PromptEngine("basic_signal")
    
    prompt = engine.render(market_data)
    signal = await client.generate_signal(prompt)
    
    # On ANY failure, signal is NEUTRAL with explanation
"""

from .client import LLMClient, LLMError, LLMResponseError, Signal
from .prompt_engine import PromptEngine
from .tracker import TokenTracker, CostTracker
from .signal_generator import SignalGenerator

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMResponseError",
    "Signal",
    "PromptEngine",
    "TokenTracker",
    "CostTracker",
    "SignalGenerator",
]
