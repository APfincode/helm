#!/usr/bin/env python3
"""
Phase 7 Validation — Perp Market Microstructure + Macro Context.

Tests without external dependencies (all HTTP mocked):
- PerpMicrostructure data model and computed properties
- MacroContext data model and formatting
- UnifiedMarketContext prompt rendering
- MarketContextBuilder assembly logic
- SignalGenerator integration with market context
- Prompt template with market_context_section variable
"""

import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
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

from src.market_context.perp_microstructure import PerpMicrostructure, PerpMicrostructureFetcher
from src.market_context.macro_context import FearGreedReading, NewsItem, MacroContext, MacroContextFetcher
from src.market_context.context_builder import UnifiedMarketContext, MarketContextBuilder

print("=" * 60)
print("PHASE 7: PERP MARKET MICROSTRUCTURE — VALIDATION")
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
# 1. PerpMicrostructure Model
# =====================================================================
print("\n[1] PerpMicrostructure Data Model")

micro = PerpMicrostructure(
    symbol="BTC",
    timestamp=datetime.utcnow(),
    funding_rate=0.000125,
    open_interest_usd=2_220_000_000,
    oi_change_24h=5.5,
    premium_pct=0.03,
    day_return=2.1,
    day_volume_usd=1_500_000_000,
    mark_price=95400,
    oracle_price=95380,
)

# Test predominant_bias: positive funding + positive premium + OI up + price up = long-heavy
test("Predominant bias: long-heavy", micro.predominant_bias == "long-heavy")

# Test extreme funding detection
micro_low = PerpMicrostructure(symbol="BTC", timestamp=datetime.utcnow(), funding_rate=0.001)
test("Extreme funding: high positive", micro_low.is_extreme_funding)

micro_neg = PerpMicrostructure(symbol="BTC", timestamp=datetime.utcnow(), funding_rate=-0.001)
test("Extreme funding: high negative", micro_neg.is_extreme_funding)

micro_normal = PerpMicrostructure(symbol="BTC", timestamp=datetime.utcnow(), funding_rate=0.0001)
test("Extreme funding: normal not extreme", not micro_normal.is_extreme_funding)

# Test prompt context formatting
ctx = micro.to_prompt_context()
test("Prompt context has funding_rate_8h", "funding_rate_8h" in ctx)
test("Prompt context has open_interest", "open_interest" in ctx)
test("Prompt context has predominant_bias", "predominant_bias" in ctx)
test("Prompt context extreme_funding is bool", isinstance(ctx.get("extreme_funding"), bool))

# Test short-heavy bias
micro_short = PerpMicrostructure(
    symbol="BTC", timestamp=datetime.utcnow(),
    funding_rate=-0.0002, premium_pct=-0.08, oi_change_24h=8, day_return=-3.0
)
test("Predominant bias: short-heavy", micro_short.predominant_bias == "short-heavy")

# Test neutral bias
micro_neutral = PerpMicrostructure(symbol="BTC", timestamp=datetime.utcnow(), funding_rate=0.0)
test("Predominant bias: neutral", micro_neutral.predominant_bias == "neutral")

# =====================================================================
# 2. MacroContext Model
# =====================================================================
print("\n[2] MacroContext Data Model")

fg = FearGreedReading(value=65, classification="Greed", timestamp=datetime.utcnow())
test("F&G extreme: not extreme", not fg.is_extreme)

test("F&G extreme: extreme fear", FearGreedReading(15, "Extreme Fear", datetime.utcnow()).is_extreme)
test("F&G extreme: extreme greed", FearGreedReading(85, "Extreme Greed", datetime.utcnow()).is_extreme)

news = NewsItem(
    title="SEC approves Bitcoin ETF",
    source="CoinDesk",
    tickers=["BTC"],
    published_at=datetime.utcnow(),
    sentiment_score=0.8,
)
test("NewsItem created", news.title == "SEC approves Bitcoin ETF")

macro = MacroContext(
    fear_greed=fg,
    fear_greed_history=[
        FearGreedReading(55, "Greed", datetime.utcnow() - timedelta(days=2)),
        FearGreedReading(60, "Greed", datetime.utcnow() - timedelta(days=1)),
        fg,
    ],
    breaking_news=[news],
)

mctx = macro.to_prompt_context()
test("Macro context has fear_greed", "fear_greed" in mctx)
test("Macro context has breaking_news", "breaking_news" in mctx)
test("Macro context has fear_greed_trend", "fear_greed_trend" in mctx)
test("F&G trend computed", mctx["fear_greed_trend"] != "")

# =====================================================================
# 3. UnifiedMarketContext
# =====================================================================
print("\n[3] UnifiedMarketContext Prompt Rendering")

unified = UnifiedMarketContext(
    symbol="BTC",
    timestamp=datetime.utcnow(),
    funding_rate_8h="0.0125%",
    funding_annualized="13.7%",
    funding_percentile="75%",
    open_interest="$2.22B",
    oi_change_24h="+5.5%",
    perp_premium="0.030%",
    day_return="+2.1%",
    volume_24h="$1.50B",
    predominant_bias="long-heavy (crowded longs — potential short setup)",
    extreme_funding=False,
    fear_greed="65 (Greed)",
    fear_greed_trend="increasing greed (+10 points)",
    breaking_news="- [BTC] SEC approves Bitcoin ETF [JUST IN]",
)

prompt_section = unified.to_prompt_section()
test("Prompt section contains Hyperliquid", "Hyperliquid Perp Market Context" in prompt_section)
test("Prompt section contains Funding", "Funding Rate" in prompt_section)
test("Prompt section contains Fear & Greed", "Fear & Greed" in prompt_section)
test("Prompt section contains Breaking News", "Breaking News" in prompt_section)
test("Prompt section contains bias", "Predominant Bias" in prompt_section)

# =====================================================================
# 4. Prompt Template Loading
# =====================================================================
print("\n[4] Prompt Template with Market Context")

import yaml

template_path = str(_repo_root / "config" / "prompts" / "basic_signal.yaml")
with open(template_path, "r") as f:
    tmpl = yaml.safe_load(f)

test("Template has market_context_section variable", "market_context_section" in tmpl.get("variables", []))
test("Template references {{market_context_section}}", "{{market_context_section}}" in tmpl.get("user_template", ""))
test("Template instructs LLM to consider funding", "funding" in tmpl.get("user_template", ""))
test("Template instructs LLM to consider F&G", "Fear & Greed" in tmpl.get("user_template", ""))

# =====================================================================
# 5. MarketContextBuilder Assembly (Mocked)
# =====================================================================
print("\n[5] MarketContextBuilder Assembly")

# Test _assemble with partial data
builder = MarketContextBuilder()
result = builder._assemble("BTC", None, None)
test("Assembly with no data still returns context", result.symbol == "BTC")

result2 = builder._assemble("ETH", micro, macro)
test("Assembly with full data has funding", result2.funding_rate_8h != "")
test("Assembly with full data has F&G", result2.fear_greed != "Unavailable")
test("Assembly with full data has news", result2.breaking_news != "")

# =====================================================================
# 6. SignalGenerator Integration (Mocked)
# =====================================================================
print("\n[6] SignalGenerator with Market Context Integration")

from src.llm.signal_generator import SignalGenerator

# Mock LLM client that captures prompts
class MockLLMClient:
    def __init__(self):
        self.last_user_prompt = ""
        self._initialized = True
    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass
    async def generate_signal(self, system_prompt, user_prompt, **kwargs):
        self.last_user_prompt = user_prompt
        # Return a neutral signal
        from src.llm.client import Signal
        return Signal(direction="NEUTRAL", confidence=0.5)
    def get_stats(self): return {}
    def is_valid(self): return True

# Create generator with mocked components
mock_client = MockLLMClient()
mock_engine = MagicMock()
mock_engine.render.return_value = ("system", "user with {{market_context_section}}")
mock_engine.render_with_history.return_value = ("system", "user with history")
mock_engine.config.max_tokens = 500
mock_engine.config.temperature = 0.2

gen = SignalGenerator(
    template_name="basic_signal",
    client=mock_client,
    prompt_engine=mock_engine,
    enable_market_context=True,
)

# Mock the market context builder
gen._shared_context_builder = MagicMock()
gen._shared_context_builder.build = AsyncMock(return_value=unified)
gen._initialized = True

import pandas as pd
df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]})

# Run generate with symbol
signal = asyncio.run(gen.generate(df, symbol="BTC"))
test("SignalGenerator accepts symbol param", signal is not None)
test("Mock engine render called with market_context_section", "market_context_section" in str(mock_engine.render.call_args))

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
    print(" Phase 7 Perp Microstructure is structurally sound.")
    sys.exit(0)
