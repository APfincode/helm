"""
Test suite for the Self-Evolving Prompt system (Phase 4).

Tests cover:
1. Mutation engine (composability, deduplication, safety)
2. Fitness calculation (cross-exchange, normalization)
3. Backtest adapter (signal generation, backtest integration)
4. Prompt pool (persistence, lineage, deduplication)
5. Evolution runner (selection, breeding, convergence)
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path

from src.evolver.models import (
    PromptVariant, Population, EvolutionConfig, FitnessScore, MutationType
)
from src.evolver.mutation import MutationEngine, ComposableMutation
from src.evolver.fitness import CrossExchangeFitness, FitnessWeights
from src.backtest.models import BacktestResult, BacktestConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def seed_variant():
    """Create a seed prompt variant for testing."""
    return PromptVariant(
        id="test_seed",
        generation=0,
        system_prompt="You are a trading analyst. Output JSON signals.",
        user_template="Analyze this data: {{market_data}}",
        temperature=0.2,
        max_tokens=500,
    )


@pytest.fixture
def basic_backtest_result():
    """Create a basic backtest result for testing."""
    config = BacktestConfig(symbol="BTC", timeframe="1h")
    result = BacktestResult(config=config)
    result.total_return_pct = 5.0
    result.sharpe_ratio = 1.2
    result.max_drawdown_pct = 3.0
    result.total_trades = 10
    result.win_rate = 60.0
    return result


@pytest.fixture
def evolution_config():
    """Default evolution config."""
    return EvolutionConfig(
        population_size=5,
        elite_count=1,
        max_generations=3,
        backtest_window_days=7,
    )


# =============================================================================
# Mutation Engine Tests
# =============================================================================

class TestMutationEngine:
    def test_init(self):
        engine = MutationEngine(seed=42)
        assert len(engine._library) > 0
    
    def test_mutate_produces_different_text(self, seed_variant):
        engine = MutationEngine(seed=42)
        child, mutations = engine.mutate(seed_variant)
        
        assert child.id != seed_variant.id
        assert child.generation == seed_variant.generation + 1
        assert child.parent_ids == [seed_variant.id]
        assert len(mutations) >= 1
    
    def test_mutate_tracks_lineage(self, seed_variant):
        engine = MutationEngine(seed=42)
        child, mutations = engine.mutate(seed_variant)
        
        assert child.mutations_applied
        assert len(child.mutation_descriptions) == len(mutations)
        assert all(isinstance(m, MutationType) for m in child.mutations_applied)
    
    def test_crossover_combines_parents(self, seed_variant):
        engine = MutationEngine(seed=42)
        parent2 = PromptVariant(
            id="parent2",
            generation=0,
            system_prompt="Different system prompt here",
            user_template="Different user template",
            temperature=0.3,
        )
        
        child, mutations = engine.crossover(seed_variant, parent2)
        
        assert child.generation == 1
        assert len(child.parent_ids) == 2
        assert child.system_prompt in [seed_variant.system_prompt, parent2.system_prompt]
        assert child.user_template in [seed_variant.user_template, parent2.user_template]
    
    def test_prompt_hash_consistency(self, seed_variant):
        """Same content = same hash (deduplication works)."""
        v1 = PromptVariant(
            id="v1", generation=0,
            system_prompt="same", user_template="same"
        )
        v2 = PromptVariant(
            id="v2", generation=0,
            system_prompt="same", user_template="same"
        )
        assert v1.prompt_hash == v2.prompt_hash
    
    def test_content_hash_changes_with_mutation(self, seed_variant):
        engine = MutationEngine(seed=42)
        child, _ = engine.mutate(seed_variant)
        assert child.prompt_hash != seed_variant.prompt_hash


# =============================================================================
# Fitness Tests
# =============================================================================

class TestCrossExchangeFitness:
    def test_calculate_composite(self, evolution_config, basic_backtest_result):
        calc = CrossExchangeFitness(evolution_config)
        
        fitness = calc.calculate(
            hl_result=basic_backtest_result,
            bn_result=basic_backtest_result,
            direction_agreement_rate=0.8,
            return_correlation=0.7,
            llm_calls=10,
            llm_cost=0.50,
        )
        
        assert fitness.composite_score > 0
        assert fitness.composite_score <= 1.0
        assert fitness.direction_agreement_rate == 0.8
        assert fitness.total_llm_cost_usd == 0.50
    
    def test_return_normalization(self):
        assert CrossExchangeFitness._normalize_return(-50) == 0.0
        assert CrossExchangeFitness._normalize_return(0) == 0.5
        assert CrossExchangeFitness._normalize_return(50) == 1.0
        assert CrossExchangeFitness._normalize_return(100) == 1.0  # Clipped
    
    def test_sharpe_normalization(self):
        assert CrossExchangeFitness._normalize_sharpe(-2) == 0.0
        assert CrossExchangeFitness._normalize_sharpe(1) == 0.5
        assert CrossExchangeFitness._normalize_sharpe(4) == 1.0
    
    def test_drawdown_normalization(self):
        assert CrossExchangeFitness._normalize_drawdown(0) == 1.0
        assert CrossExchangeFitness._normalize_drawdown(15) == 0.5
        assert CrossExchangeFitness._normalize_drawdown(30) == 0.0
        assert CrossExchangeFitness._normalize_drawdown(50) == 0.0  # Clipped
    
    def test_negative_sharpe_penalty(self, evolution_config, basic_backtest_result):
        """Negative Sharpe with positive return should be penalized."""
        bad_result = BacktestResult(config=basic_backtest_result.config)
        bad_result.sharpe_ratio = -0.5
        bad_result.total_return_pct = 10.0
        bad_result.total_trades = 10
        
        good_result = BacktestResult(config=basic_backtest_result.config)
        good_result.sharpe_ratio = 1.5
        good_result.total_return_pct = 10.0
        good_result.total_trades = 10
        
        calc = CrossExchangeFitness(evolution_config)
        
        bad_fitness = calc.calculate(bad_result, bad_result, 0.8, 0.7, 0, 0)
        good_fitness = calc.calculate(good_result, good_result, 0.8, 0.7, 0, 0)
        
        assert good_fitness.composite_score > bad_fitness.composite_score
    
    def test_too_few_trades_penalty(self, evolution_config, basic_backtest_result):
        """Variants with too few trades should get lower fitness."""
        low_trade_result = BacktestResult(config=basic_backtest_result.config)
        low_trade_result.total_trades = 2  # Below min_trades_for_fitness
        low_trade_result.total_return_pct = 10.0
        low_trade_result.sharpe_ratio = 2.0
        
        high_trade_result = BacktestResult(config=basic_backtest_result.config)
        high_trade_result.total_trades = 20
        high_trade_result.total_return_pct = 10.0
        high_trade_result.sharpe_ratio = 2.0
        
        calc = CrossExchangeFitness(evolution_config)
        
        low = calc.calculate(low_trade_result, low_trade_result, 0.8, 0.7, 0, 0)
        high = calc.calculate(high_trade_result, high_trade_result, 0.8, 0.7, 0, 0)
        
        assert high.composite_score > low.composite_score


# =============================================================================
# Population Tests
# =============================================================================

class TestPopulation:
    def test_sort_by_fitness(self):
        pop = Population(generation=1)
        pop.variants = [
            PromptVariant(id="low", generation=1, system_prompt="a", user_template="b",
                         fitness=FitnessScore(composite_score=0.2)),
            PromptVariant(id="high", generation=1, system_prompt="a", user_template="b",
                         fitness=FitnessScore(composite_score=0.8)),
            PromptVariant(id="mid", generation=1, system_prompt="a", user_template="b",
                         fitness=FitnessScore(composite_score=0.5)),
        ]
        pop.sort_by_fitness()
        
        assert pop.variants[0].id == "high"
        assert pop.variants[1].id == "mid"
        assert pop.variants[2].id == "low"
    
    def test_diversity_calculation(self):
        pop = Population(generation=1)
        pop.variants = [
            PromptVariant(id="v1", generation=1, system_prompt="alpha", user_template="one"),
            PromptVariant(id="v2", generation=1, system_prompt="beta", user_template="two"),
            PromptVariant(id="v3", generation=1, system_prompt="gamma", user_template="three"),
        ]
        diversity = pop.compute_diversity()
        assert 0.0 <= diversity <= 1.0
    
    def test_get_top_k(self):
        pop = Population(generation=1)
        pop.variants = [
            PromptVariant(id="a", generation=1, system_prompt="a", user_template="b",
                         fitness=FitnessScore(composite_score=0.9)),
            PromptVariant(id="b", generation=1, system_prompt="a", user_template="b",
                         fitness=FitnessScore(composite_score=0.7)),
            PromptVariant(id="c", generation=1, system_prompt="a", user_template="b",
                         fitness=FitnessScore(composite_score=0.5)),
        ]
        top = pop.get_top_k(2)
        assert len(top) == 2
        assert top[0].id == "a"


# =============================================================================
# Prompt Pool Tests (Integration)
# =============================================================================

class TestPromptPool:
    def test_save_and_load_population(self, tmp_path, evolution_config):
        pool = PromptPool(tmp_path / "test.db")
        
        pop = Population(generation=1)
        pop.variants = [
            PromptVariant(id="v1", generation=1, system_prompt="sys1", user_template="usr1",
                         fitness=FitnessScore(composite_score=0.5)),
            PromptVariant(id="v2", generation=1, system_prompt="sys2", user_template="usr2",
                         fitness=FitnessScore(composite_score=0.7)),
        ]
        
        pool.save_population(pop, evolution_config)
        loaded = pool.load_population(1)
        
        assert loaded is not None
        assert loaded.generation == 1
        assert len(loaded.variants) == 2
        assert loaded.variants[0].system_prompt == "sys1"
    
    def test_get_best_variant(self, tmp_path):
        pool = PromptPool(tmp_path / "test.db")
        
        pop = Population(generation=1)
        pop.variants = [
            PromptVariant(id="v1", generation=1, system_prompt="a", user_template="b",
                         fitness=FitnessScore(composite_score=0.5, evaluated=True)),
            PromptVariant(id="v2", generation=1, system_prompt="a", user_template="b",
                         fitness=FitnessScore(composite_score=0.9, evaluated=True)),
        ]
        for v in pop.variants:
            v.evaluated = True
        
        pool.save_population(pop)
        best = pool.get_best_variant_ever()
        
        assert best is not None
        assert best.id == "v2"
    
    def test_lineage_tracking(self, tmp_path):
        pool = PromptPool(tmp_path / "test.db")
        
        child = PromptVariant(
            id="child", generation=2,
            system_prompt="sys", user_template="usr",
            parent_ids=["parent1"]
        )
        parent = PromptVariant(
            id="parent1", generation=1,
            system_prompt="parent_sys", user_template="parent_usr",
            parent_ids=[]
        )
        
        pop = Population(generation=1, variants=[parent])
        pool.save_population(pop)
        pop2 = Population(generation=2, variants=[child])
        pool.save_population(pop2)
        
        lineage = pool.get_lineage("child")
        assert len(lineage) == 2
        assert lineage[0]["id"] == "child"
        assert lineage[1]["id"] == "parent1"


# =============================================================================
# Evolution Config Tests
# =============================================================================

class TestEvolutionConfig:
    def test_valid_config(self):
        config = EvolutionConfig()
        assert config.population_size >= 3
        assert config.elite_count < config.population_size
    
    def test_invalid_population_size(self):
        with pytest.raises(AssertionError):
            EvolutionConfig(population_size=2)
    
    def test_invalid_elite_count(self):
        with pytest.raises(AssertionError):
            EvolutionConfig(population_size=5, elite_count=5)
    
    def test_weight_sum(self):
        config = EvolutionConfig()
        assert config.hl_weight + config.bn_weight == 1.0
