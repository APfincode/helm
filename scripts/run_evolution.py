"""
Run Prompt Evolution — CLI entry point for Darwinian prompt evolution.

Usage:
    python scripts/run_evolution.py --generations 10 --population 8 --budget 30
    
    # Resume from generation 5:
    python scripts/run_evolution.py --resume 5
    
    # Export best prompt to production:
    python scripts/run_evolution.py --export config/prompts/evolved_best.yaml
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evolver.runner import EvolutionRunner
from src.evolver.models import EvolutionConfig, PromptVariant
from src.evolver.prompt_pool import PromptPool
from src.llm.client import LLMClient
from src.llm.prompt_engine import PromptEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Evolve trading prompt templates")
    parser.add_argument("--generations", type=int, default=10, help="Max generations")
    parser.add_argument("--population", type=int, default=8, help="Population size")
    parser.add_argument("--budget", type=float, default=30.0, help="Max evolution cost USD")
    parser.add_argument("--backtest-days", type=int, default=90, help="Backtest window in days")
    parser.add_argument("--seed-template", type=str, default="basic_signal", help="Template to evolve from")
    parser.add_argument("--resume", type=int, default=None, help="Resume from generation N")
    parser.add_argument("--export", type=str, default=None, help="Export best prompt to file")
    parser.add_argument("--db", type=str, default="data/prompt_evolution.db", help="Evolution database path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Config
    config = EvolutionConfig(
        population_size=args.population,
        max_generations=args.generations,
        max_evolution_cost_usd=args.budget,
        backtest_window_days=args.backtest_days,
    )
    
    logger.info(f"Evolution config: {config.to_dict()}")
    
    # Initialize components
    pool = PromptPool(Path(args.db))
    llm_client = LLMClient()
    
    # Seed variant
    seed = None
    if args.resume is None:
        # Load seed template
        try:
            engine = PromptEngine(args.seed_template)
            seed = PromptVariant(
                id="seed_0",
                generation=0,
                system_prompt=engine._template.system_prompt,
                user_template=engine._template.user_template,
                temperature=engine._template.temperature,
                max_tokens=engine._template.max_tokens,
            )
            logger.info(f"Loaded seed template: {args.seed_template}")
        except Exception as e:
            logger.error(f"Failed to load seed template: {e}")
            sys.exit(1)
    
    # Run evolution
    runner = EvolutionRunner(
        config=config,
        llm_client=llm_client,
        pool=pool,
        seed=args.seed,
    )
    
    best_variant = await runner.run_evolution(
        seed_variant=seed,
        resume_from_generation=args.resume,
    )
    
    if best_variant:
        print(f"\n{'='*60}")
        print(f"BEST VARIANT: {best_variant.id}")
        print(f"Generation: {best_variant.generation}")
        print(f"Fitness: {best_variant.fitness_value:.4f}")
        print(f"HL Return: {best_variant.fitness.hyperliquid_return_pct:.2f}%")
        print(f"BN Return: {best_variant.fitness.binance_return_pct:.2f}%")
        print(f"HL Sharpe: {best_variant.fitness.hyperliquid_sharpe:.2f}")
        print(f"BN Sharpe: {best_variant.fitness.binance_sharpe:.2f}")
        print(f"Max DD: {best_variant.fitness.avg_max_dd_pct:.2f}%")
        print(f"Agreement: {best_variant.fitness.direction_agreement_rate:.1%}")
        print(f"{'='*60}\n")
        
        if args.export:
            runner.export_best_prompt(args.export)
            print(f"Exported to: {args.export}")
    else:
        print("No best variant found (evolution may have failed)")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
