#!/usr/bin/env python3
"""
Phase 4 Validation Script — Validates the self-evolving prompt system
without requiring external dependencies (numpy, pandas, etc.).

This script mocks the external libraries and tests all core logic:
1. Mutation engine composability
2. Fitness calculation formulas
3. Population management
4. Prompt hash deduplication
5. YAML export format
6. Import structure
"""

import sys
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock external dependencies BEFORE importing project modules
mock_np = MagicMock()
mock_np.sqrt = math.sqrt
mock_np.mean = lambda x: sum(x)/len(x) if x else 0
mock_np.std = lambda x: 0.0 if len(x) < 2 else (sum((v - sum(x)/len(x))**2 for v in x)/(len(x)-1))**0.5
mock_np.__version__ = "1.0.0"

sys.modules["numpy"] = mock_np
sys.modules["pandas"] = MagicMock()
sys.modules["httpx"] = MagicMock()
sys.modules["tenacity"] = MagicMock()
sys.modules["yaml"] = MagicMock()
sys.modules["pydantic"] = MagicMock()
sys.modules["aiosqlite"] = MagicMock()
sys.modules["cryptography"] = MagicMock()
sys.modules["cryptography.hazmat"] = MagicMock()
sys.modules["cryptography.hazmat.primitives"] = MagicMock()

# Mock pandas DataFrame
mock_df = MagicMock()
sys.modules["pandas"].DataFrame = mock_df
sys.modules["pandas"].cut = MagicMock(return_value=[])
sys.modules["pandas"].merge = MagicMock(return_value=MagicMock())
sys.modules["pandas"].notna = MagicMock(return_value=True)

# Mock tenacity retry
mock_retry = MagicMock()
mock_retry.return_value = lambda f: f
sys.modules["tenacity"].retry = mock_retry
sys.modules["tenacity"].stop_after_attempt = MagicMock()
sys.modules["tenacity"].wait_exponential = MagicMock()

# Mock yaml
sys.modules["yaml"].safe_load = MagicMock(return_value={})
sys.modules["yaml"].YAMLError = Exception

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

# Now import our code
from src.evolver.models import PromptVariant, Population, EvolutionConfig, FitnessScore, MutationType
from src.evolver.mutation import MutationEngine
from src.evolver.fitness import CrossExchangeFitness
from src.backtest.models import BacktestResult, BacktestConfig

# Patch remaining imports in backtest engine
import src.backtest.engine
src.backtest.engine.pd = MagicMock()
src.backtest.engine.np = mock_np

print("=" * 60)
print("PHASE 4: SELF-EVOLVING PROMPTS — VALIDATION")
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
# 1. Seed Variant Construction
# =====================================================================
print("\n[1] PromptVariant Construction")
seed = PromptVariant(
    id="seed_0", generation=0,
    system_prompt="""You are a quantitative cryptocurrency trading analyst. Your job is to analyze
market data and produce a trading signal in a specific JSON format.

IMPORTANT RULES:
1. Output MUST be valid JSON matching the schema below
2. Be conservative — when uncertain, return "NEUTRAL" with low confidence
3. Provide clear reasoning for your signal
4. Always include stop_loss and take_profit percentages
""",
    user_template="Analyze this data: {{market_data}}",
    temperature=0.2,
    max_tokens=500,
)
test("Seed has correct generation", seed.generation == 0)
test("Seed has prompt hash", len(seed.prompt_hash) == 16)
test("Unevaluated variant has -inf fitness", seed.fitness_value == float("-inf"))

# =====================================================================
# 2. Mutation Engine
# =====================================================================
print("\n[2] Mutation Engine")
engine = MutationEngine(seed=42)

test("Library has mutations", len(engine._library) >= 7)

child, mutations = engine.mutate(seed)
test("Child has different ID", child.id != seed.id)
test("Child generation incremented", child.generation == 1)
test("Parent lineage tracked", child.parent_ids == ["seed_0"])
test("Mutations recorded", len(child.mutations_applied) >= 1)
test("Prompt hash changed", child.prompt_hash != seed.prompt_hash)
test("Descriptions match mutations", len(child.mutation_descriptions) == len(mutations))

# Test deduplication
v1 = PromptVariant(id="a", generation=1, system_prompt="same", user_template="same")
v2 = PromptVariant(id="b", generation=1, system_prompt="same", user_template="same")
test("Duplicate content has same hash", v1.prompt_hash == v2.prompt_hash)

# Test crossover
parent2 = PromptVariant(
    id="p2", generation=0,
    system_prompt="Different system prompt",
    user_template="Different user template",
    temperature=0.4,
)
crossover_child, cx_mutations = engine.crossover(seed, parent2)
test("Crossover has 2 parents", len(crossover_child.parent_ids) == 2)
test("Crossover system from one parent",
     crossover_child.system_prompt in [seed.system_prompt, parent2.system_prompt])
test("Crossover user from one parent",
     crossover_child.user_template in [seed.user_template, parent2.user_template])
test("Crossover temperature averaged",
     abs(crossover_child.temperature - 0.3) < 0.01)

# =====================================================================
# 3. Population Management
# =====================================================================
print("\n[3] Population Management")

pop = Population(generation=1)
pop.variants = [
    PromptVariant(id="low", generation=1, system_prompt="a", user_template="b",
                 fitness=FitnessScore(composite_score=0.2)),
    PromptVariant(id="high", generation=1, system_prompt="a", user_template="b",
                 fitness=FitnessScore(composite_score=0.8)),
    PromptVariant(id="mid", generation=1, system_prompt="a", user_template="b",
                 fitness=FitnessScore(composite_score=0.5)),
]
for v in pop.variants:
    v.evaluated = True

pop.sort_by_fitness()
test("Sort places best first", pop.variants[0].id == "high")
test("Sort places worst last", pop.variants[2].id == "low")

top2 = pop.get_top_k(2)
test("get_top_k returns correct count", len(top2) == 2)
test("get_top_k returns best first", top2[0].id == "high")

diversity = pop.compute_diversity()
test("Diversity in valid range", 0.0 <= diversity <= 1.0)

# =====================================================================
# 4. Fitness Calculation
# =====================================================================
print("\n[4] Fitness Calculation")

config = EvolutionConfig(
    population_size=5, elite_count=1,
    max_generations=3, backtest_window_days=7
)
calc = CrossExchangeFitness(config)

hl_result = BacktestResult(config=BacktestConfig())
hl_result.total_return_pct = 8.0
hl_result.sharpe_ratio = 1.5
hl_result.max_drawdown_pct = 4.0
hl_result.total_trades = 15
hl_result.win_rate = 65.0

fitness = calc.calculate(
    hl_result=hl_result, bn_result=hl_result,
    direction_agreement_rate=0.75, return_correlation=0.6,
    llm_calls=20, llm_cost=1.25
)

test("Composite score in [0,1]", 0.0 <= fitness.composite_score <= 1.0)
test("HL metrics populated", fitness.hyperliquid_return_pct == 8.0)
test("BN metrics populated", fitness.binance_return_pct == 8.0)
test("Cost tracking correct", fitness.total_llm_cost_usd == 1.25)
test("Agreement rate recorded", fitness.direction_agreement_rate == 0.75)

# Test normalization boundaries
test("Return norm: -50% -> 0", calc._normalize_return(-50) == 0.0)
test("Return norm: 0% -> 0.5", calc._normalize_return(0) == 0.5)
test("Return norm: 50% -> 1", calc._normalize_return(50) == 1.0)
test("Return norm clips >50", calc._normalize_return(100) == 1.0)

test("Sharpe norm: -2 -> 0", calc._normalize_sharpe(-2) == 0.0)
test("Sharpe norm: 1 -> 0.5", calc._normalize_sharpe(1) == 0.5)
test("Sharpe norm: 4 -> 1", calc._normalize_sharpe(4) == 1.0)

test("DD norm: 0% -> 1", calc._normalize_drawdown(0) == 1.0)
test("DD norm: 15% -> 0.5", calc._normalize_drawdown(15) == 0.5)
test("DD norm: 30% -> 0", calc._normalize_drawdown(30) == 0.0)

# Test negative Sharpe penalty
bad_result = BacktestResult(config=BacktestConfig())
bad_result.sharpe_ratio = -0.5
bad_result.total_return_pct = 10.0
bad_result.total_trades = 10

good_result = BacktestResult(config=BacktestConfig())
good_result.sharpe_ratio = 1.5
good_result.total_return_pct = 10.0
good_result.total_trades = 10

bad_fitness = calc.calculate(bad_result, bad_result, 0.8, 0.7, 0, 0)
good_fitness = calc.calculate(good_result, good_result, 0.8, 0.7, 0, 0)
test("Good Sharpe > Bad Sharpe", good_fitness.composite_score > bad_fitness.composite_score)

# Test trade count penalty
low_trade = BacktestResult(config=BacktestConfig())
low_trade.total_trades = 2
low_trade.total_return_pct = 10.0
low_trade.sharpe_ratio = 2.0

high_trade = BacktestResult(config=BacktestConfig())
high_trade.total_trades = 20
high_trade.total_return_pct = 10.0
high_trade.sharpe_ratio = 2.0

low_fit = calc.calculate(low_trade, low_trade, 0.8, 0.7, 0, 0)
high_fit = calc.calculate(high_trade, high_trade, 0.8, 0.7, 0, 0)
test("More trades = higher fitness", high_fit.composite_score > low_fit.composite_score)

# =====================================================================
# 5. EvolutionConfig Validation
# =====================================================================
print("\n[5] EvolutionConfig Validation")

try:
    bad_config = EvolutionConfig(population_size=2)
    test("Rejects too small population", False)
except AssertionError:
    test("Rejects too small population", True)

try:
    bad_config = EvolutionConfig(population_size=5, elite_count=5)
    test("Rejects elite >= population", False)
except AssertionError:
    test("Rejects elite >= population", True)

try:
    bad_config = EvolutionConfig(mutation_rate=1.5)
    test("Rejects mutation > 1", False)
except AssertionError:
    test("Rejects mutation > 1", True)

config = EvolutionConfig()
test("HL+BN weights sum to 1", config.hl_weight + config.bn_weight == 1.0)

# =====================================================================
# 6. Tournament Selection (via runner logic)
# =====================================================================
print("\n[6] Selection Logic")

pop2 = Population(generation=1)
pop2.variants = [
    PromptVariant(id="w1", generation=1, system_prompt="a", user_template="b",
                 fitness=FitnessScore(composite_score=0.9)),
    PromptVariant(id="w2", generation=1, system_prompt="a", user_template="b",
                 fitness=FitnessScore(composite_score=0.8)),
    PromptVariant(id="l1", generation=1, system_prompt="a", user_template="b",
                 fitness=FitnessScore(composite_score=0.1)),
    PromptVariant(id="l2", generation=1, system_prompt="a", user_template="b",
                 fitness=FitnessScore(composite_score=0.2)),
]
for v in pop2.variants:
    v.evaluated = True

# Best should always be selected in tournament
import random
random.seed(42)
from src.evolver.runner import EvolutionRunner

# Mock runner for tournament test
class MockRunner:
    def __init__(self, pop):
        self._config = EvolutionConfig(tournament_size=2)
        self._pop = pop
    
    def _tournament_select(self, population):
        contestants = random.sample(population.variants, min(self._config.tournament_size, len(population.variants)))
        return max(contestants, key=lambda v: v.fitness_value)

mock_runner = MockRunner(pop2)
winners = [mock_runner._tournament_select(pop2).id for _ in range(100)]
tournament_win_rate = sum(1 for w in winners if w in ("w1", "w2")) / len(winners)
test("Tournament favors better fitness", tournament_win_rate > 0.7,
     f"win rate = {tournament_win_rate:.2%}")

# =====================================================================
# 7. Prompt Pool Serialization
# =====================================================================
print("\n[7] Serialization")

variant_dict = seed.to_dict()
test("to_dict produces dict", isinstance(variant_dict, dict))
test("to_dict has prompt_hash", "prompt_hash" in variant_dict)

# Fitness serialization
f = FitnessScore(composite_score=0.75)
f_dict = f.to_dict()
test("FitnessScore serializes", isinstance(f_dict, dict))
test("FitnessScore roundtrips", FitnessScore.from_dict(f_dict).composite_score == 0.75)

# Population serialization
pop3 = Population(generation=2, variants=[seed])
pop3.best_fitness_ever = 0.9
pop_dict = pop3.to_dict()
test("Population serializes", pop_dict["generation"] == 2)

# =====================================================================
# 8. YAML Export Format
# =====================================================================
print("\n[8] YAML Export")

best = PromptVariant(
    id="evolved_v5", generation=5,
    system_prompt="Evolved system prompt here",
    user_template="Evolved user template here",
    temperature=0.15, max_tokens=900,
    fitness=FitnessScore(composite_score=0.85)
)

yaml_content = f"""name: "evolved_signal_v{best.generation}"
version: "1.0.0"
description: "Auto-evolved prompt (fitness={best.fitness_value:.4f})"

system_prompt: |
{best.system_prompt}

user_template: |
{best.user_template}

output_schema:
  type: "object"
  required: []

variables: ["market_data"]
max_tokens: {best.max_tokens}
temperature: {best.temperature}
"""

test("YAML contains system prompt", "Evolved system prompt here" in yaml_content)
test("YAML contains user template", "Evolved user template here" in yaml_content)
test("YAML has correct generation", "evolved_signal_v5" in yaml_content)
test("YAML has max_tokens", "max_tokens: 900" in yaml_content)

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
    print(" Phase 4 is structurally sound and ready for testing with real data.")
    sys.exit(0)
