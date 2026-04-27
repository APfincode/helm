"""
Evolution Runner — Main orchestrator for the Darwinian prompt evolution loop.

The runner manages the full lifecycle:
1. Initialize population (from seed or previous generation)
2. Evaluate fitness on Hyperliquid + Binance (via BacktestEvaluator)
3. Select elites and tournament winners
4. Breed children (mutation + crossover)
5. Replace population with new generation
6. Check convergence and budget limits
7. Promote best variant to production

Safety gates:
- Budget cap: stops evolution if cost exceeds max_evolution_cost_usd
- Convergence: stops if best fitness doesn't improve for N generations
- Injection check: all new prompts scanned before evaluation
- Lineage deduplication: prevents duplicate variants by content hash
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import Optional

from .models import PromptVariant, Population, EvolutionConfig, FitnessScore
from .mutation import MutationEngine
from .fitness import CrossExchangeFitness, FitnessWeights
from .backtest_adapter import BacktestEvaluator
from .prompt_pool import PromptPool
from src.llm.client import LLMClient
from src.backtest.models import BacktestConfig
from src.security.input_validator import PromptInjectionDetector


logger = logging.getLogger(__name__)


class EvolutionRunner:
    """
    Main entry point for prompt evolution.
    
    Usage:
        runner = EvolutionRunner(config, llm_client, pool)
        await runner.run_evolution(seed_variant)
    """
    
    def __init__(
        self,
        config: EvolutionConfig,
        llm_client: LLMClient,
        pool: PromptPool,
        fitness_weights: Optional[FitnessWeights] = None,
        seed: Optional[int] = None,
    ):
        self._config = config
        self._llm_client = llm_client
        self._pool = pool
        self._fitness_weights = fitness_weights
        
        self._mutation_engine = MutationEngine(seed=seed)
        self._fitness_calculator = CrossExchangeFitness(config, fitness_weights)
        self._injection_detector = PromptInjectionDetector()
        
        # State
        self._current_generation = 0
        self._best_fitness_history: list[float] = []
        self._total_cost = 0.0
        self._halted = False
        self._halt_reason = ""
    
    # ========================================================================
    # Main Evolution Loop
    # ========================================================================
    
    async def run_evolution(
        self,
        seed_variant: Optional[PromptVariant] = None,
        resume_from_generation: Optional[int] = None,
    ) -> PromptVariant:
        """
        Run the full evolutionary process.
        
        Returns:
            The best PromptVariant found across all generations
        """
        # Initialize or resume
        if resume_from_generation is not None:
            population = self._pool.load_population(resume_from_generation)
            if population is None:
                logger.warning(f"Generation {resume_from_generation} not found, starting fresh")
                population = await self._init_population(seed_variant)
        else:
            population = await self._init_population(seed_variant)
        
        self._current_generation = population.generation
        
        logger.info(
            f"Starting evolution: pop_size={self._config.population_size}, "
            f"max_gens={self._config.max_generations}, "
            f"budget=${self._config.max_evolution_cost_usd}"
        )
        
        # Evolution loop
        for gen in range(self._current_generation, self._config.max_generations):
            if self._halted:
                break
            
            logger.info(f"=== Generation {gen} ===")
            
            # Evaluate fitness for all unevaluated variants
            population = await self._evaluate_population(population)
            
            # Log results
            self._log_generation(population)
            
            # Save to pool
            self._pool.save_population(population, self._config)
            
            # Check termination criteria
            if self._should_terminate(population):
                break
            
            # Check budget
            if self._total_cost >= self._config.max_evolution_cost_usd:
                logger.warning(f"Budget exceeded: ${self._total_cost:.2f} / ${self._config.max_evolution_cost_usd}")
                self._halted = True
                self._halt_reason = "budget_exceeded"
                break
            
            # Create next generation
            population = self._breed_next_generation(population)
            self._current_generation = population.generation
        
        # Final evaluation (if not already done)
        if not self._halted:
            population = await self._evaluate_population(population)
            self._pool.save_population(population, self._config)
        
        # Return best variant ever
        best = self._pool.get_best_variant_ever()
        if best and best.fitness:
            self._pool.save_best_variant(best, reason=f"End of evolution at gen {self._current_generation}")
            logger.info(
                f"Evolution complete. Best variant: {best.id} "
                f"(gen={best.generation}, fitness={best.fitness.composite_score:.4f})"
            )
        
        return best
    
    # ========================================================================
    # Initialization
    # ========================================================================
    
    async def _init_population(self, seed: Optional[PromptVariant] = None) -> Population:
        """Create generation 0 from a seed variant."""
        if seed is None:
            # Create default seed from basic template
            from src.llm.prompt_engine import PromptEngine
            engine = PromptEngine("basic_signal")
            seed = PromptVariant(
                id="seed_0",
                generation=0,
                system_prompt=engine._template.system_prompt,
                user_template=engine._template.user_template,
                temperature=engine._template.temperature,
                max_tokens=engine._template.max_tokens,
            )
        
        # Create population with seed + mutated variants
        variants = [seed]
        seen_hashes = {seed.prompt_hash}
        
        attempts = 0
        max_attempts = self._config.population_size * 3
        
        while len(variants) < self._config.population_size and attempts < max_attempts:
            attempts += 1
            
            # Create a mutation of the seed
            parent = random.choice(variants)
            child, _ = self._mutation_engine.mutate(parent)
            
            # Skip if duplicate
            if child.prompt_hash in seen_hashes:
                continue
            
            # Check injection
            if self._config.require_injection_check:
                if self._injection_detector.is_suspicious(child.full_prompt_text):
                    logger.warning(f"Injection detected in variant {child.id}, skipping")
                    continue
            
            # Check length
            if len(child.full_prompt_text) > self._config.max_prompt_length:
                logger.warning(f"Variant {child.id} too long, skipping")
                continue
            
            seen_hashes.add(child.prompt_hash)
            variants.append(child)
        
        population = Population(
            generation=0,
            variants=variants,
        )
        
        logger.info(f"Initialized population with {len(variants)} unique variants")
        return population
    
    # ========================================================================
    # Evaluation
    # ========================================================================
    
    async def _evaluate_population(self, population: Population) -> Population:
        """Evaluate fitness for all unevaluated variants in the population."""
        # Backtest config for each exchange
        hl_config = BacktestConfig(
            symbol="BTC",
            timeframe="1h",
            taker_fee_pct=0.035,  # Hyperliquid taker
            maker_fee_pct=0.01,
            initial_capital=10000,
        )
        bn_config = BacktestConfig(
            symbol="BTC",
            timeframe="1h",
            taker_fee_pct=0.05,  # Binance Futures taker
            maker_fee_pct=0.02,
            initial_capital=10000,
        )
        
        # Calculate backtest window
        end = datetime.now()
        start = end - timedelta(days=self._config.backtest_window_days)
        
        # Create evaluator
        evaluator = BacktestEvaluator(
            llm_client=self._llm_client,
            hl_config=hl_config,
            bn_config=bn_config,
            signal_interval=self._config.signal_interval_hours,
        )
        
        # Evaluate variants (with concurrency limit to avoid rate limits)
        unevaluated = [v for v in population.variants if not v.evaluated]
        logger.info(f"Evaluating {len(unevaluated)} unevaluated variants")
        
        # Evaluate one at a time to manage LLM costs
        for variant in unevaluated:
            if self._total_cost >= self._config.max_evolution_cost_usd:
                logger.warning("Budget limit reached during evaluation")
                break
            
            try:
                fitness = await evaluator.evaluate(
                    variant=variant,
                    start_date=start,
                    end_date=end,
                    symbol="BTC",
                    timeframe="1h",
                )
                
                variant.fitness = fitness
                variant.evaluated = True
                self._total_cost += fitness.total_llm_cost_usd
                
                logger.info(
                    f"Evaluated {variant.id}: composite={fitness.composite_score:.4f}, "
                    f"cost=${fitness.total_llm_cost_usd:.4f}"
                )
                
            except Exception as e:
                logger.error(f"Evaluation failed for {variant.id}: {e}")
                variant.evaluated = True
                variant.evaluation_error = str(e)
                variant.fitness = FitnessScore()
                variant.fitness.composite_score = -999.0
        
        # Update population metadata
        evaluated = [v for v in population.variants if v.evaluated and v.fitness]
        if evaluated:
            population.avg_fitness = sum(v.fitness_value for v in evaluated) / len(evaluated)
            best = population.get_best()
            if best and best.fitness:
                if best.fitness.composite_score > population.best_fitness_ever:
                    population.best_fitness_ever = best.fitness.composite_score
                    population.best_variant_id = best.id
        
        population.evaluated_at = datetime.now()
        
        return population
    
    # ========================================================================
    # Breeding
    # ========================================================================
    
    def _breed_next_generation(self, population: Population) -> Population:
        """Create the next generation through selection and reproduction."""
        population.sort_by_fitness()
        
        # Elites automatically survive
        elites = population.get_top_k(self._config.elite_count)
        
        # Fill remaining slots through tournament selection
        children = []
        seen_hashes = {e.prompt_hash for e in elites}
        
        attempts = 0
        max_attempts = self._config.population_size * 5
        
        while len(children) < (self._config.population_size - len(elites)) and attempts < max_attempts:
            attempts += 1
            
            # Tournament selection
            if random.random() < self._config.crossover_rate:
                # Crossover: two parents
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                child, mutations = self._mutation_engine.crossover(parent1, parent2)
            else:
                # Asexual: mutation only
                parent = self._tournament_select(population)
                child, mutations = self._mutation_engine.mutate(parent)
            
            # Deduplication
            if child.prompt_hash in seen_hashes:
                continue
            
            # Injection check
            if self._config.require_injection_check:
                if self._injection_detector.is_suspicious(child.full_prompt_text):
                    continue
            
            # Length check
            if len(child.full_prompt_text) > self._config.max_prompt_length:
                continue
            
            seen_hashes.add(child.prompt_hash)
            children.append(child)
        
        # Build new population (elites + children)
        new_variants = elites + children
        
        new_population = Population(
            generation=population.generation + 1,
            variants=new_variants,
        )
        
        # Preserve best-ever reference
        new_population.best_fitness_ever = population.best_fitness_ever
        new_population.best_variant_id = population.best_variant_id
        
        logger.info(
            f"Generation {new_population.generation}: "
            f"{len(elites)} elites + {len(children)} children = {len(new_variants)} total"
        )
        
        return new_population
    
    def _tournament_select(self, population: Population) -> PromptVariant:
        """Tournament selection: pick best from random subset."""
        k = min(self._config.tournament_size, len(population.variants))
        contestants = random.sample(population.variants, k)
        return max(contestants, key=lambda v: v.fitness_value)
    
    # ========================================================================
    # Termination
    # ========================================================================
    
    def _should_terminate(self, population: Population) -> bool:
        """Check if evolution has converged."""
        best = population.get_best()
        if not best or not best.fitness:
            return False
        
        self._best_fitness_history.append(best.fitness.composite_score)
        
        # Keep only last N generations
        n = self._config.convergence_generations
        if len(self._best_fitness_history) > n:
            self._best_fitness_history = self._best_fitness_history[-n:]
        
        # Check if no improvement for N generations
        if len(self._best_fitness_history) >= n:
            max_fitness = max(self._best_fitness_history)
            min_fitness = min(self._best_fitness_history)
            improvement = max_fitness - min_fitness
            
            if improvement < self._config.convergence_threshold:
                logger.info(
                    f"Converged: no improvement ({improvement:.6f}) "
                    f"for {n} generations"
                )
                self._halted = True
                self._halt_reason = "converged"
                return True
        
        return False
    
    # ========================================================================
    # Logging
    # ========================================================================
    
    def _log_generation(self, population: Population) -> None:
        """Log generation statistics."""
        population.compute_diversity()
        evaluated = [v for v in population.variants if v.evaluated and v.fitness]
        
        if not evaluated:
            logger.warning(f"Gen {population.generation}: No evaluated variants")
            return
        
        fitnesses = [v.fitness.composite_score for v in evaluated]
        best = population.get_best()
        
        logger.info(
            f"Gen {population.generation} stats: "
            f"best={max(fitnesses):.4f}, worst={min(fitnesses):.4f}, "
            f"avg={sum(fitnesses)/len(fitnesses):.4f}, "
            f"diversity={population.diversity_score:.3f}, "
            f"evaluated={len(evaluated)}/{len(population.variants)}"
        )
        
        if best and best.fitness:
            logger.info(
                f"Best variant: {best.id} | "
                f"HL: return={best.fitness.hyperliquid_return_pct:.2f}% "
                f"sharpe={best.fitness.hyperliquid_sharpe:.2f} "
                f"dd={best.fitness.hyperliquid_max_dd_pct:.2f}% | "
                f"BN: return={best.fitness.binance_return_pct:.2f}% "
                f"sharpe={best.fitness.binance_sharpe:.2f} "
                f"dd={best.fitness.binance_max_dd_pct:.2f}% | "
                f"agreement={best.fitness.direction_agreement_rate:.2%}"
            )
    
    # ========================================================================
    # Exports
    # ========================================================================
    
    def export_best_prompt(self, output_path: Optional[str] = None) -> str:
        """Export the best prompt as a YAML template for production use."""
        best = self._pool.get_best_variant_ever()
        if not best:
            return ""
        
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
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(yaml_content)
            logger.info(f"Exported evolved prompt to {output_path}")
        
        return yaml_content
