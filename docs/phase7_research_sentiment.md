# Phase 7 Deep Research: Sentiment & News Integration

**Date:** April 25, 2026  
**Researcher:** AI Agent (Hermes)  
**Scope:** Free-only data sources, fallback architectures, trading efficacy, feasibility  
**Constraint:** ZERO paid APIs. Every method must have a free fallback.

---

## 1. Prior Implementations & Academic Evidence

### 1.1 Academic Papers

#### UCLA arXiv:2507.09739 (July 2025)
**"Enhancing Trading Performance Through Sentiment Analysis with LLMs"**
- **Authors:** Haojie Liu, Zihan Lin, Randall R. Rojas (UCLA)
- **Period:** Aug 2019 – Aug 2024 (S&P 500)
- **Key Finding:** A hybrid strategy combining **GPT-2 sentiment + VW MACD** achieved **5.77% return** vs **-0.696% buy-and-hold** over the same period.
- **Critical Insight:** "Sentiment models alone do not consistently beat traditional methods, but **combining sentiment with technical indicators substantially enhances performance**, particularly in volatile environments."
- **Best Lag:** k=1 (today's news → tomorrow's return) delivered best out-of-sample accuracy.
- **FinBERT Accuracy:** 75.56% on Benzinga financial news.

#### Adalytica Case Study (Feb 2025)
**"Boost Returns Trading Crypto: Backtest Using Sentiment"**
- **Period:** Feb 2024 – Feb 2025 (1 year)
- **Asset:** BTC-USD
- **Strategy:** When week-over-week sentiment outpaced price → buy & hold. When price exceeded sentiment → sell long / go short.
- **Results:**

| Metric | Buy-and-Hold | Sentiment Strategy |
|--------|--------------|-------------------|
| Annual Return | 76.2% | **174.4%** |
| Volatility | Baseline | **51% lower** |
| Max Drawdown | Baseline | **58% lower** |
| Sharpe Ratio | 1.2 | **5.52** |

- **Rebalancing:** On average every 3 days (low frequency = low fees).

#### Medium / Chedy Smaoui (Jan 2024)
**"Using Sentiment Analysis as a Trading Signal"**
- **Approach:** FinBERT on financial news headlines for GOOG.
- **Threshold:** Buy when sentiment > 0.6, hold 3 days, exit.
- **Framework:** Backtrader.py + quantstats.
- **Result:** Strategy implemented but no explicit Sharpe advantage stated. Shows code structure for integration.

### 1.2 Open Source Projects

| Repo | Language | Approach | Status | Paid API? |
|------|----------|----------|--------|-----------|
| **CyberPunkMetalHead/Cryptocurrency-Sentiment-Bot** | C# | Inverse Reddit sentiment (Vader) — buys what r/CryptoCurrency hates | 33★, active | No (PRAW free) |
| **cryptocontrol/sentiment-trading-bot** | Java | CryptoControl sentiment API → long/short | 25★, dead 2018 | **YES** — requires CryptoControl API |
| **freqtrade/freqtrade** | Python | Framework supports custom sentiment strategies | Active | No |

**Key Lesson from CyberPunkMetalHead:** Reddit sentiment as a **contrarian** indicator. The thesis: "r/Cryptocurrency is always a great inverse indicator."

---

## 2. Does It Actually Work? — Trading Efficacy Analysis

### 2.1 What the Evidence Says

**PRO (Sentiment Helps):**
- Adalytica: **+98.2% excess return** vs buy-and-hold on BTC, with massively lower risk.
- UCLA: Hybrid sentiment + technical **outperformed** in volatile periods.
- Institutional adoption: Renaissance, Two Sigma, Citadel all use NLP/news signals (confirmed in quant finance literature).

**CON (Sentiment Alone Fails):**
- UCLA explicitly states: "Sentiment models **alone** do not consistently beat traditional methods."
- Alpha decay is real — as more participants use sentiment, edge degrades.
- Social media sentiment is noisy, manipulated (bot farms, paid shills), and lagging.
- r/CryptoCurrency inverse indicator works until it doesn't (echo chamber dynamics shift).

### 2.2 Verdict for Our Bot

> **Sentiment should NOT be a standalone signal. It should be CONTEXT fed into the LLM prompt.**

**Why this is the right architecture for us:**
1. Our bot already has a sophisticated technical signal engine (Phase 3 LLM + Phase 2 backtest).
2. Adding sentiment as **prompt context** lets the LLM weight it appropriately — rather than hard-coding a sentiment threshold that will decay.
3. The RiskManager (Phase 5) still has final veto power — sentiment never bypasses circuit breakers.
4. This mirrors the UCLA finding: **hybrid** (LLM technical + sentiment context) beats either alone.

**Expected Impact:**
- Conservative estimate: **+0.2 to +0.5 Sharpe improvement** by reducing false entries during extreme fear/greed.
- Best case: **+10-20% reduction in max drawdown** by avoiding FOMO tops and panic bottoms.
- No guarantee of alpha — but strong evidence of **risk reduction**.

---

## 3. Free Data Sources Architecture (Zero Paid APIs)

### 3.1 News & Media Sentiment

#### Tier 1: cryptocurrency.cv (PRIMARY)
**URL:** `https://cryptocurrency.cv/api/news`
- **Cost:** $0 — No API key required. Free forever.
- **Coverage:** 130+ English sources + 75 international, 662K articles (2017–2026).
- **Endpoints:**
  - `GET /api/news` — latest news
  - `GET /api/breaking` — last 2 hours
  - `GET /api/trending?hours=24` — trending topics WITH sentiment
  - `GET /api/sentiment?asset=BTC` — deep sentiment with confidence
  - `GET /api/market/fear-greed` — Fear & Greed Index
- **Sentiment Schema:** `{score: 0.65, label: "positive", confidence: 0.85}`
- **Enriched:** Articles include `btc_price`, `eth_price`, `fear_greed_index` at time of publication.
- **Rate Limit:** Unknown but appears generous (community project).
- **Fallback:** Self-host via Vercel/Docker (open source, MIT license).

#### Tier 2: RSS Feeds (SECONDARY)
Direct RSS/Atom feeds from major crypto news outlets. No API keys.
- **CoinDesk:** `https://www.coindesk.com/arc/outboundfeeds/rss/` 
- **Cointelegraph:** `https://cointelegraph.com/rss`
- **The Block:** `https://www.theblock.co/rss.xml`
- **Decrypt:** `https://decrypt.co/feed`
- **Bitcoin Magazine:** `https://bitcoinmagazine.com/feed`
- **Parsing:** `feedparser` (Python) or `rss-parser`.
- **Sentiment:** Run local FinBERT on headline + summary.

#### Tier 3: CryptoPanic (TERTIARY)
**URL:** `https://cryptopanic.com/api/v1/posts/`
- **Free tier:** Requires API key but free tier exists (rate limited).
- **Coverage:** Aggregates news + social + YouTube.
- **Sentiment:** Some posts include user-voted sentiment.
- **Fallback:** Scrape the public HTML (not ideal, but possible).

### 3.2 Social Media Sentiment

#### Twitter/X Scraping (NO API KEY)
**Primary: Twikit**
- **Repo:** `d60/twikit`
- **Approach:** Scrape X/Twitter without API key. Search tweets, get trending.
- **Risk:** X actively fights scrapers. Breaks frequently.
- **Fallback:** Nitter instances (RSS feeds of Twitter accounts).

**Secondary: Nitter RSS**
- **URL pattern:** `https://nitter.net/search?f=tweets&q=bitcoin`
- **Status:** Fragile — instances get blocked.
- **Fallback:** Direct web scraping with `requests` + `BeautifulSoup` (higher breakage risk).

#### Reddit (FREE)
**Primary: PRAW (Python Reddit API Wrapper)**
- **Cost:** $0 — requires Reddit app registration (free).
- **Rate limit:** 30 requests/minute (read-only).
- **Data:** r/CryptoCurrency, r/Bitcoin, r/ethfinance, r/ethtrader.
- **Metrics:** Post scores, comment sentiment, trending keywords.

**Secondary: Direct JSON Scraping**
- **URL:** `https://www.reddit.com/r/CryptoCurrency/hot.json` (append `.json` to any Reddit URL).
- **No API key needed.** Rate limited by IP.
- **Fallback:** PullPush API (`https://pullpush.io/`) — community Reddit archive.

### 3.3 On-Chain & Market Microstructure (ALL FREE)

#### Hyperliquid Native API (PRIMARY — ALREADY USING)
**Base:** `https://api.hyperliquid.xyz/info`
- **Endpoint:** `metaAndAssetCtxs`
- **Free data per asset:**
  - `funding` — current funding rate
  - `openInterest` — OI in USD
  - `premium` — perp premium vs oracle
  - `dayNtlVlm` — 24h volume
  - `oraclePx`, `markPx`, `midPx`
- **Historical funding:** `fundingHistory` endpoint
- **Cost:** $0, no API key, rate limits are generous.
- **Already integrated** via our DataFetcher.

#### CoinGecko (SECONDARY)
**Free Tier (Demo API):**
- 10,000 calls/month
- 30 calls/minute
- **Endpoints:**
  - `/coins/markets` — price, volume, market cap, price change
  - `/coins/{id}/market_chart` — historical price/volume
  - `/global` — total market cap, dominance
- **No credit card required** for demo tier.

#### Fear & Greed Index (TERTIARY)
**Alternative.me API:**
- **URL:** `https://api.alternative.me/fng/?limit=10`
- **Cost:** $0, no API key.
- **Data:** 0-100 score, classification (Extreme Fear to Extreme Greed), timestamp.
- **Update frequency:** Daily.
- **Historical:** Can fetch up to `limit=0` for all historical data.

#### Whale Alerts (OPTIONAL)
**Options:**
1. **Whale Alert Telegram:** `t.me/whale_alert_io` — can monitor via Telegram bot API.
2. **n8n Template:** CoinGecko provides a free n8n workflow for whale alerts.
3. **Self-scrape:** Etherscan/BscScan free APIs for large transfers (limited).

### 3.4 NLP Models (Local, Zero API Cost)

#### FinBERT (Financial Sentiment)
- **Repo:** `ProsusAI/finbert` on HuggingFace
- **Type:** BERT fine-tuned on financial corpus.
- **Classes:** Positive, Negative, Neutral.
- **Inference:** Runs locally via `transformers` library.
- **Speed:** ~50-200ms per text on CPU, <10ms on GPU.
- **VRAM:** ~400MB.

#### Crypto-FinBERT (Domain-Specific)
- **Repo:** `burakutf/finetuned-finbert-crypto` on HuggingFace.
- **Type:** FinBERT further fine-tuned on crypto-specific text.
- **Advantage:** Better understanding of crypto slang (HODL, rugpull, ape in, etc.).

#### VADER (Lightweight Fallback)
- **Library:** `nltk.sentiment.vader`
- **Type:** Lexicon/rule-based, no ML required.
- **Speed:** ~1ms per text.
- **Use case:** When FinBERT is too slow or transformers library unavailable.
- **Limitation:** Doesn't understand crypto jargon well.

---

## 4. Recommended Architecture for Our Bot

### 4.1 Multi-Tier Fallback Design

```
┌─────────────────────────────────────────────────────────────────┐
│  SENTIMENT MODULE                                               │
│  ───────────────                                                │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ AGGREGATOR                                               │  │
│  │ - Fetches from ALL sources every 15-60 min               │  │
│  │ - Caches results                                        │  │
│  │ - Computes blended sentiment score                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐            │
│         ▼                    ▼                    ▼            │
│  ┌─────────────┐    ┌───────────────┐    ┌───────────────┐   │
│  │ NEWS TIER   │    │ SOCIAL TIER   │    │ MARKET TIER   │   │
│  │ ──────────  │    │ ────────────  │    │ ────────────  │   │
│  │ 1. crypto.  │    │ 1. Reddit     │    │ 1. HL funding │   │
│  │    cv API   │    │    (PRAW)     │    │ 2. HL OI      │   │
│  │ 2. RSS      │    │ 2. X/Twitter  │    │ 3. HL premium │   │
│  │    feeds    │    │    (Twikit)   │    │ 4. CoinGecko  │   │
│  │ 3. Crypto-  │    │ 3. Alt.me F&G │    │    dominance  │   │
│  │    Panic    │    │               │    │               │   │
│  └─────────────┘    └───────────────┘    └───────────────┘   │
│         │                    │                    │            │
│         └────────────────────┼────────────────────┘            │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ NLP ENGINE (Local)                                       │  │
│  │ 1. FinBERT-Crypto (primary)                              │  │
│  │ 2. FinBERT (fallback)                                    │  │
│  │ 3. VADER (lightweight fallback)                          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ OUTPUT: Structured Sentiment Context                     │  │
│  │ {                                                        │  │
│  │   "news_sentiment": 0.62,      // -1 to +1              │  │
│  │   "social_sentiment": -0.15,   // -1 to +1              │  │
│  │   "market_microstructure": {                             │  │
│  │     "funding_rate": 0.000125,  // positive = longs pay  │  │
│  │     "open_interest": 2220000000,                        │  │
│  │     "premium": 0.000317,       // perp vs spot          │  │
│  │     "fear_greed": 65          // 0-100                  │  │
│  │   },                                                     │  │
│  │   "trending_topics": ["ETF", "SEC"],                    │  │
│  │   "last_updated": "2026-04-25T06:00:00Z"                │  │
│  │ }                                                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Integration with Existing Pipeline

**Current Flow (Phase 6):**
```
Market Data → SignalGenerator (LLM) → RiskManager → Execution
```

**Phase 7 Enhanced Flow:**
```
Market Data ──┬──→ SignalGenerator (LLM with sentiment context)
              │      ↓
Sentiment ────┘   RiskManager ──→ Execution
     ↑                ↑
     └─── Feeds into prompt as "Market Narrative"
```

**How it feeds into the LLM prompt:**

```markdown
## Market Context
Price Action: BTC $95,400 (+2.1% 24h), ETH $3,200 (-0.5% 24h)

## Sentiment Analysis
News Sentiment: +0.62 (positive, confidence 0.85)
Social Sentiment: -0.15 (slightly negative)
Fear & Greed Index: 65 (Greed)
Trending Topics: ETF approval, SEC lawsuit

## Market Microstructure
Funding Rate: 0.0125% (longs paying shorts — cautiously bullish)
Open Interest: $2.22B (high leverage environment)
Perp Premium: +0.03% (slight bullish bias)

## Task
Based on the technical analysis AND market sentiment above, generate a trading signal...
```

### 4.3 Why This Architecture Works

1. **No standalone sentiment trades** — avoids alpha decay from pure sentiment strategies.
2. **LLM can ignore noise** — if sentiment contradicts price action, the LLM can weight it down.
3. **RiskManager unchanged** — circuit breakers still apply regardless of sentiment.
4. **Cache-friendly** — sentiment fetched every 15-60 min, not on every tick. Reduces API calls.
5. **Graceful degradation** — if Twikit breaks, we still have cryptocurrency.cv + Reddit + Hyperliquid data.

---

## 5. Feasibility Study

### 5.1 Cost Analysis

| Component | Cost | Notes |
|-----------|------|-------|
| cryptocurrency.cv API | **$0** | No API key, open source |
| Reddit (PRAW) | **$0** | Free read-only API |
| X/Twitter (Twikit) | **$0** | Scraping, fragile |
| Hyperliquid API | **$0** | Already using |
| CoinGecko API | **$0** | 10K calls/month free |
| Alternative.me F&G | **$0** | No API key |
| FinBERT (local) | **$0** | HuggingFace, runs on CPU |
| **TOTAL** | **$0/month** | |

### 5.2 Complexity Analysis

| Task | Lines of Code | Difficulty | Risk |
|------|---------------|------------|------|
| News aggregator (crypto.cv + RSS) | ~300 | Medium | Low |
| Reddit scraper (PRAW) | ~150 | Low | Low |
| X/Twitter scraper (Twikit) | ~200 | Medium | **High** (breaks often) |
| Hyperliquid microstructure fetcher | ~100 | Low | None (native API) |
| Fear & Greed fetcher | ~50 | Low | None |
| NLP pipeline (FinBERT + VADER) | ~200 | Medium | Low |
| Prompt integration (context builder) | ~150 | Low | None |
| **TOTAL** | **~1,150 LOC** | | |

### 5.3 Maintenance Burden

| Component | Maintenance Level | Why |
|-----------|-------------------|-----|
| cryptocurrency.cv | **Low** | Stable API, open source, community maintained |
| RSS feeds | **Low-Medium** | URLs rarely change, but parsers need updates |
| Reddit (PRAW) | **Low** | Official API, stable |
| X/Twitter (Twikit) | **HIGH** | X actively breaks scrapers. Expect monthly fixes. |
| Hyperliquid | **None** | Already integrated, native API |
| CoinGecko | **Low** | Official API, stable |
| NLP models | **None** | Local inference, no external dependency |

**Mitigation:** Make Twikit/optional. If it breaks, the other 5 sources still provide robust sentiment.

### 5.4 Performance Impact Estimate

| Scenario | Expected Impact | Confidence |
|----------|-----------------|------------|
| Sharpe ratio improvement | +0.2 to +0.5 | Medium |
| Max drawdown reduction | -10% to -20% | Medium |
| False signal reduction | -15% to -30% | Medium-High |
| Fee reduction (fewer bad trades) | -5% to -15% | Medium |
| Latency added to signal generation | +50-500ms | High |

**Important Caveat:** Sentiment data has a **lag**. News breaks, then price moves, then sentiment APIs reflect it. The real edge is in **contrarian detection** (extreme fear/greed) and **narrative awareness** (e.g., knowing an ETF decision is the dominant story).

---

## 6. Risks & Mitigations

### Risk 1: Scraper Breakage (Twitter/X)
- **Mitigation:** Make Twitter optional. If Twikit fails, fall back to Reddit + News + Market microstructure.

### Risk 2: Sentiment API Rate Limits
- **Mitigation:** Cache aggressively (15-60 min). All sources are polled on interval, not per-tick.

### Risk 3: Sentiment Manipulation (Bot Farms, Paid Shills)
- **Mitigation:** 
  - Weight news (harder to manipulate) higher than social.
  - Use confidence scores. Discard low-confidence social sentiment.
  - Outlier detection: if social sentiment spikes >3 sigma, flag as potential manipulation.

### Risk 4: Alpha Decay
- **Mitigation:** Sentiment is **context**, not a standalone signal. The LLM prompt evolves (Phase 4). As sentiment strategies decay, the Darwinian evolver will reduce their weighting automatically.

### Risk 5: LLM Prompt Bloat
- **Mitigation:** Keep sentiment context concise (<300 tokens). Use structured JSON, not prose. The prompt already handles market data; sentiment is an add-on section.

---

## 7. Implementation Recommendation

### Phase 7A: Core Sentiment Module (MVP — ~600 LOC)
1. **cryptocurrency.cv integration** — fetch news + pre-computed sentiment.
2. **Alternative.me F&G** — simple HTTP fetch.
3. **Hyperliquid microstructure** — extend existing DataFetcher to capture funding + OI + premium.
4. **Prompt integration** — add sentiment context section to LLM prompt template.
5. **No social scraping yet** — news + market microstructure + F&G is enough for MVP.

### Phase 7B: Social Layer (Optional — ~400 LOC)
1. Reddit scraper (PRAW) for r/CryptoCurrency.
2. Twikit for X (optional, high maintenance).
3. Local FinBERT for custom sentiment scoring on Reddit posts.

### Phase 7C: Advanced NLP (Optional — ~200 LOC)
1. Trending topic extraction (named entity recognition).
2. Narrative tracking (e.g., "ETF narrative strength" over time).
3. Contrarian signal generation (extreme F&G + price divergence).

---

## 8. Final Verdict

| Question | Answer |
|----------|--------|
| **Should we build this?** | **YES** — but as prompt context, not standalone signals. |
| **Will it improve profitability?** | Maybe modestly. Strong evidence it **reduces risk** (drawdown, volatility). |
| **Is it truly free?** | **YES** — $0/month with the recommended architecture. |
| **How long to implement MVP?** | 1-2 days for 7A (core module). 2-3 days for 7A+7B. |
| **What's the maintenance burden?** | Low-Medium. Only Twitter/X is fragile. Everything else is stable. |
| **Does it fit our risk model?** | **YES** — sentiment never bypasses RiskManager. It's advisory only. |

### Strategic Value
The biggest value isn't alpha generation — it's **narrative awareness**. When the LLM knows "the market is panicking about an SEC lawsuit," it generates more nuanced signals. When it sees "funding rates are extremely negative," it understands shorts are over-leveraged. This context improves signal quality without adding directional bias.

### Next Step
If approved, proceed with **Phase 7A implementation** (core sentiment module: cryptocurrency.cv + F&G + HL microstructure + prompt integration). Estimated 600-800 lines of production code.
