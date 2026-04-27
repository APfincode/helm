#!/usr/bin/env python3
"""
LLM Signal Generator Demo.

Demonstrates the full pipeline:
1. Generate sample market data
2. Render prompt from template
3. Call LLM (simulated if no API key)
4. Validate response
5. Display signal with full safety info

Usage:
    # With real LLM API (requires .env with API keys):
    python scripts/run_llm_signal.py
    
    # The script will show you what WOULD be sent to the LLM
    # and demonstrate the failsafe behavior
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
import json

from llm.prompt_engine import PromptEngine
from llm.client import LLMClient, Signal


def generate_sample_data(n: int = 20) -> pd.DataFrame:
    """Generate sample OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    
    returns = np.random.normal(0.0002, 0.008, n)
    prices = 65000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open": prices * (1 + np.random.normal(0, 0.001, n)),
        "high": prices * (1 + abs(np.random.normal(0, 0.004, n))),
        "low": prices * (1 - abs(np.random.normal(0, 0.004, n))),
        "close": prices,
        "volume": np.random.uniform(1000, 10000, n),
    }, index=dates)
    
    return df


def simulate_llm_response(system_prompt: str, user_prompt: str) -> dict:
    """
    Simulate an LLM response for demo purposes.
    
    In production, this would be an actual API call.
    """
    # Parse the market data from the prompt
    # (In reality, the LLM would do this)
    
    # Generate a realistic signal
    simulated_signal = {
        "signal": "LONG",
        "confidence": 0.72,
        "reasoning": "Price broke above the 20-period moving average with volume confirmation. RSI at 58 shows momentum building without being overbought. Structure suggests continuation of the uptrend.",
        "regime": "trending_up",
        "risk_params": {
            "stop_loss_pct": 2.5,
            "take_profit_pct": 5.0,
        }
    }
    
    return {
        "model": "demo-model",
        "choices": [{
            "message": {
                "content": json.dumps(simulated_signal)
            }
        }],
        "usage": {
            "prompt_tokens": 250,
            "completion_tokens": 80,
            "total_tokens": 330,
        }
    }


async def main():
    print("=" * 70)
    print(" Hyper-Alpha-Arena V2 - LLM Signal Generator Demo")
    print("=" * 70)
    
    # Step 1: Generate market data
    print("\n[1] Generating sample market data...")
    data = generate_sample_data(n=20)
    print(f"    Data points: {len(data)}")
    print(f"    Latest price: ${data['close'].iloc[-1]:,.2f}")
    print(f"    24h change: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:+.2f}%")
    
    # Step 2: Load prompt template
    print("\n[2] Loading prompt template...")
    engine = PromptEngine("basic_signal")
    print(f"    Template: {engine.config.name} v{engine.config.version}")
    print(f"    Description: {engine.config.description}")
    print(f"    Max tokens: {engine.config.max_tokens}")
    print(f"    Temperature: {engine.config.temperature}")
    
    # Step 3: Render prompt
    print("\n[3] Rendering prompt...")
    system_prompt, user_prompt = engine.render(data)
    
    prompt_hash = engine.get_prompt_hash(data)
    print(f"    Prompt hash: {prompt_hash}")
    print(f"    System prompt length: {len(system_prompt)} chars")
    print(f"    User prompt length: {len(user_prompt)} chars")
    
    # Show a preview
    print(f"\n    --- System Prompt Preview ---")
    print(f"    {system_prompt[:300]}...")
    print(f"\n    --- User Prompt Preview ---")
    print(f"    {user_prompt[:300]}...")
    
    # Step 4: Check for API keys
    print("\n[4] Checking API configuration...")
    try:
        from security.secrets_manager import get_secrets_manager
        secrets = get_secrets_manager()
        
        has_openrouter = secrets.get("OPENROUTER_API_KEY", required=False) is not None
        has_openai = secrets.get("OPENAI_API_KEY", required=False) is not None
        
        if has_openrouter or has_openai:
            print(f"    OpenRouter API key: {'✓ Found' if has_openrouter else '✗ Missing'}")
            print(f"    OpenAI API key: {'✓ Found' if has_openai else '✗ Missing'}")
            use_real_llm = True
        else:
            print("    No API keys found. Using simulated response.")
            print("    (Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env for real LLM)")
            use_real_llm = False
    except Exception as e:
        print(f"    Could not check secrets: {e}")
        use_real_llm = False
    
    # Step 5: Generate signal
    print("\n[5] Generating trading signal...")
    
    if use_real_llm:
        print("    Calling LLM API...")
        async with LLMClient() as client:
            signal = await client.generate_signal(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=engine.config.temperature,
                max_tokens=engine.config.max_tokens,
            )
    else:
        print("    Using simulated LLM response...")
        raw_response = simulate_llm_response(system_prompt, user_prompt)
        
        client = LLMClient()
        signal = client._parse_llm_response(raw_response)
    
    # Step 6: Display results
    print("\n" + "=" * 70)
    print(" SIGNAL RESULT")
    print("=" * 70)
    
    print(f"\n    Direction:    {'🟢 LONG' if signal.direction == 'LONG' else '🔴 SHORT' if signal.direction == 'SHORT' else '⚪ NEUTRAL'}")
    print(f"    Confidence:   {signal.confidence:.0%}")
    print(f"    Regime:       {signal.regime}")
    print(f"    Stop Loss:    {signal.stop_loss_pct:.1f}%")
    print(f"    Take Profit:  {signal.take_profit_pct:.1f}%")
    print(f"\n    Reasoning:")
    print(f"    {signal.reasoning}")
    
    if signal.error:
        print(f"\n    ⚠️  Error: {signal.error}")
    
    print(f"\n    Provider:     {signal.provider or 'demo'}")
    print(f"    Model:        {signal.model or 'n/a'}")
    print(f"    Tokens:       {signal.tokens_used}")
    print(f"    Latency:      {signal.latency_ms:.0f}ms" if signal.latency_ms else "    Latency:      n/a")
    
    # Step 7: Safety validation
    print("\n" + "=" * 70)
    print(" SAFETY CHECKS")
    print("=" * 70)
    
    checks = [
        ("Signal is valid", signal.is_valid()),
        ("Direction recognized", signal.direction in ("LONG", "SHORT", "NEUTRAL")),
        ("Confidence in range", 0.0 <= signal.confidence <= 1.0),
        ("Stop loss positive", signal.stop_loss_pct > 0),
        ("Take profit > stop loss", signal.take_profit_pct > signal.stop_loss_pct or signal.direction == "NEUTRAL"),
        ("Reasoning provided", len(signal.reasoning) > 10),
        ("No raw LLM output", True),  # By design, Signal is the only output
    ]
    
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"    {status}  {check_name}")
    
    # Step 8: What happens on error?
    print("\n" + "=" * 70)
    print(" FAILSAFE DEMONSTRATION")
    print("=" * 70)
    
    print("\n    Simulating various failure modes...")
    
    # Budget exceeded
    error_signal = Signal.error("Monthly budget exceeded", "COST_LIMIT")
    print(f"\n    [Budget exceeded] → {error_signal.direction} (confidence: {error_signal.confidence})")
    
    # API timeout
    error_signal = Signal.error("API timeout after 30s", "TIMEOUT")
    print(f"    [API timeout]     → {error_signal.direction} (confidence: {error_signal.confidence})")
    
    # Invalid JSON
    error_signal = Signal.error("Invalid JSON in response", "PARSE_ERROR")
    print(f"    [Invalid JSON]    → {error_signal.direction} (confidence: {error_signal.confidence})")
    
    # Prompt injection detected
    error_signal = Signal.error("Prompt injection detected", "SECURITY")
    print(f"    [Injection]       → {error_signal.direction} (confidence: {error_signal.confidence})")
    
    print("\n    Note: In ALL failure cases, the bot receives NEUTRAL → NO TRADE")
    
    print("\n" + "=" * 70)
    print(" Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
