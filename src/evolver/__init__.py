"""
Self-Evolving Prompts (Darwinian Evolver) — Phase 4.

Composable, contrastive prompt evolution system that:
1. Maintains a population of prompt variants
2. Mutates them using composable text operators
3. Evaluates fitness on REAL Hyperliquid + Binance historical data
4. Selects winners and breeds next generation
5. Promotes best variant to production SignalGenerator

Key concepts:
- Composability: mutations stack (focus_on_volume + focus_on_rsi = hybrid)
- Contrast: tracking which mutation caused which fitness change
- Safety: every mutated prompt passes injection detection before evaluation
- Budget: evolution cost tracking prevents runaway spend
"""

from .models import PromptVariant, Population, EvolutionConfig, FitnessScore
from .mutation import MutationEngine, ComposableMutation
from .fitness import CrossExchangeFitness, FitnessWeights
from .backtest_adapter import BacktestEvaluator
from .prompt_pool import PromptPool
from .runner import EvolutionRunner

__all__ = [
    "PromptVariant",
    "Population",
    "EvolutionConfig",
    "FitnessScore",
    "MutationEngine",
    "ComposableMutation",
    "CrossExchangeFitness",
    "FitnessWeights",
    "BacktestEvaluator",
    "PromptPool",
    "EvolutionRunner",
]
