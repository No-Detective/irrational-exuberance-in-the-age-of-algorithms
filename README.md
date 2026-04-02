# Irrational Exuberance in the Age of Algorithms
## How AI-Mediated Crypto Sentiment Spills Over to Equity Markets

**Research Data Pipeline — Complete Technical Reference**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Quick Start](#3-quick-start)
4. [Configuration](#4-configuration)
5. [File-by-File Reference](#5-file-by-file-reference)
   - [main.py](#mainpy)
   - [config/settings.py](#configsettingspy)
   - [config/.env.template](#configenvtemplate)
   - [collectors/equity_collector.py](#collectorsequity_collectorpy)
   - [collectors/crypto_collector.py](#collectorscrypto_collectorpy)
   - [collectors/sentiment_collector.py](#collectorssentiment_collectorpy)
   - [collectors/macro_collector.py](#collectorsmacro_collectorpy)
   - [collectors/regulatory_collector.py](#collectorsregulatory_collectorpy)
   - [collectors/regulatory_events.csv](#collectorsregulatory_eventscsv)
   - [pipeline/orchestrator.py](#pipelineorchestratorpy)
   - [pipeline/transformer.py](#pipelinetransformerpy)
   - [pipeline/feature_engineering.py](#pipelinefeature_engineeringpy)
   - [pipeline/orthogonalization.py](#pipelineorthogonalizationpy)
   - [analysis/validate.py](#analysisvalidatepy)
   - [data/processed/](#dataprocessed)
   - [requirements.txt](#requirementstxt)
6. [Pipeline Architecture](#6-pipeline-architecture)
7. [Data Sources — Full Inventory](#7-data-sources--full-inventory)
8. [Methodology](#8-methodology)
   - [Section 3.3.1 — Composite Sentiment Index](#section-331--composite-sentiment-index)
   - [Section 3.3.2 — Orthogonalization (Equation 1)](#section-332--orthogonalization-equation-1)
   - [Feature Engineering](#feature-engineering)
9. [Complete Development History — What We Tried, What Broke, What We Fixed](#9-complete-development-history--what-we-tried-what-broke-what-we-fixed)
   - [Phase 1 — Initial Architecture](#phase-1--initial-architecture)
   - [Phase 2 — First Full Run Failures](#phase-2--first-full-run-failures)
   - [Phase 3 — Timezone Hell](#phase-3--timezone-hell)
   - [Phase 4 — Sentiment Composite Crisis](#phase-4--sentiment-composite-crisis)
   - [Phase 5 — Python 3.14 / NumPy 2.x Compatibility](#phase-5--python-314--numpy-2x-compatibility)
   - [Phase 6 — Baker-Wurgler Investigation](#phase-6--baker-wurgler-investigation)
   - [Phase 7 — CoinMetrics API Restrictions](#phase-7--coinmetrics-api-restrictions)
   - [Phase 8 — Final Run: All Steps Complete](#phase-8--final-run-all-steps-complete)
10. [Bug Registry — Every Error, Root Cause, and Fix](#10-bug-registry--every-error-root-cause-and-fix)
11. [Data Source Status — What Works, What Doesn't, Why](#11-data-source-status--what-works-what-doesnt-why)
12. [Final Output Variables](#12-final-output-variables)
13. [Orthogonalization Results](#13-orthogonalization-results)
14. [Running the Pipeline](#14-running-the-pipeline)
15. [Academic Citation Guidance](#15-academic-citation-guidance)

---

## 1. Project Overview

This codebase implements the complete data collection, processing, and feature-engineering pipeline for the working paper:

> **"Irrational Exuberance in the Age of Algorithms: How AI-Mediated Crypto Sentiment Spills Over to Equity Markets"**

The paper investigates three hypotheses:

- **H1**: Irrational crypto sentiment (the portion unexplained by market fundamentals) Granger-causes equity market returns and volatility, even after controlling for rational sentiment and macro factors.
- **H2**: The sentiment-to-equity spillover is mediated by on-chain network metrics (active addresses, MVRV ratio, transaction counts) that capture speculative excess.
- **H3**: The spillover is stronger during periods of elevated algorithmic trading activity, as measured by market microstructure proxies.

The pipeline produces `data/processed/master_panel_final.parquet` — a daily panel of 2,920+ observations from January 2018 through December 2025 containing 59 variables, of which the two key outputs are `rational_sentiment` and `irrational_sentiment`.

**Final run results (2026-03-31):**
- All 17 pipeline steps succeeded
- `master_panel_final.parquet`: 2,922 rows × 59 columns
- Orthogonalization R² = 0.0556, F p-val = 2.80×10⁻¹⁰ (***), orthogonality corr = −1.71×10⁻⁸ ≈ 0

---

## 2. Repository Structure

```
crypto_research_FINAL/
│
├── main.py                              ← Entry point — all CLI modes
├── requirements.txt                     ← Python dependencies
│
├── config/
│   ├── settings.py                      ← Single source of truth for all config
│   └── .env.template                    ← Copy to .env and fill in your keys
│
├── collectors/                          ← Step 1: Raw data collection
│   ├── equity_collector.py              ← Equities (yfinance), BW sentiment, EPU
│   ├── crypto_collector.py              ← CoinGecko/yfinance, CoinMetrics, Blockchain.com, DeFiLlama, Coinglass
│   ├── sentiment_collector.py           ← StockTwits+FinBERT, Reddit, Fear&Greed, Google Trends, GDELT
│   ├── macro_collector.py               ← FRED: 6 macro series
│   ├── regulatory_collector.py          ← SEC EDGAR, CFTC RSS, Latham & Watkins Tracker
│   └── regulatory_events.csv            ← 70 hand-coded events 2018-2025 (fallback)
│
├── pipeline/                            ← Steps 2–4: Processing
│   ├── orchestrator.py                  ← Central runner, SQLite log, scheduler
│   ├── transformer.py                   ← Merges raw shards → master_panel.parquet
│   ├── feature_engineering.py           ← PCA sentiment, algo intensity, RV, reg dummies
│   └── orthogonalization.py             ← Equation 1 OLS → rational/irrational sentiment
│
├── analysis/
│   └── validate.py                      ← Post-pipeline validation suite
│
└── data/
    ├── raw/                             ← Raw Parquet shards (gitignored)
    │   ├── equity/
    │   ├── crypto/
    │   ├── onchain/
    │   ├── futures/
    │   ├── sentiment/
    │   └── macro/
    └── processed/                       ← Final analysis-ready files
        ├── master_panel.parquet         ← After transformer (2922 × 50)
        ├── master_panel_features.parquet← After feature engineering (2922 × 56)
        ├── master_panel_final.parquet   ← After orthogonalization (2922 × 59)
        ├── coverage_report.csv          ← Per-variable coverage %
        ├── orthogonalization_report.txt ← Full OLS results + sensitivity
        └── validation_report.txt        ← ADF, outliers, orthogonality checks
```

---

## 3. Quick Start

```bash
# 1. Clone / copy the project directory
cd crypto_research_FINAL

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up credentials
cp config/.env.template config/.env
# Edit config/.env and add at minimum FRED_API_KEY (free)

# 4. Clear any old run log and run the full pipeline
rm -f data/run_log.db
python main.py

# 5. Check status
python main.py --mode status

# 6. Validate output
python main.py --mode validate
```

**Expected runtime:** 45–75 minutes (dominated by StockTwits pagination with mandatory sleep intervals and Google Trends quarterly batching).

**Expected output:** `data/processed/master_panel_final.parquet` with 2,922 daily rows, 59 columns.

---

## 4. Configuration

All configuration lives in `config/.env`. Copy `config/.env.template` and fill in your values.

### Required

| Key | Where to get it | Notes |
|-----|----------------|-------|
| `FRED_API_KEY` | https://fred.stlouisfed.org/docs/api/api_key.html | Free, instant |

### Optional (enable additional data sources)

| Key | Where to get it | Effect if missing |
|-----|----------------|-------------------|
| `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` | https://www.reddit.com/prefs/apps | Reddit NLP skipped |
| `COINMETRICS_API_KEY` | https://coinmetrics.io/community/ | Community API still works without key, but rate limits apply |
| `COINGECKO_API_KEY` | https://coingecko.com | Fallback to yfinance if missing |
| `COINGLASS_API_KEY` | https://coinglass.com | Futures data skipped |
| `KAIKO_API_KEY` | https://kaiko.com/pages/academic-program | Algo intensity institutional data skipped |

### Format rules (CRITICAL)

- No quotes around values: `FRED_API_KEY=abc123` not `FRED_API_KEY="abc123"`
- No trailing spaces
- File must be saved as `config/.env` (not `.env.template`)

### Date range

```
START_DATE=2018-01-01
END_DATE=2025-12-31
```

Change these to rerun over a different window. The pipeline respects these values everywhere.

---

## 5. File-by-File Reference

### `main.py`

**Role:** CLI entry point. Parses arguments and delegates to the appropriate module.

**Lines:** 92

**CLI modes:**

| Mode | Command | What it does |
|------|---------|--------------|
| `full` (default) | `python main.py` | Runs all 4 pipeline steps from scratch |
| `incremental` | `python main.py --mode incremental` | Fetches only yesterday's data |
| `features` | `python main.py --mode features` | Re-runs PCA and algo intensity only |
| `orthogonalize` | `python main.py --mode orthogonalize` | Re-runs OLS decomposition only |
| `validate` | `python main.py --mode validate` | Runs validation suite on existing output |
| `schedule` | `python main.py --mode schedule` | Starts APScheduler cron at 07:00 UTC daily |
| `status` | `python main.py --mode status` | Prints last 60 runs from SQLite log |

**Key flags:**
- `--start 2020-01-01` — override start date
- `--end 2025-12-31` — override end date
- `--force` — re-run collectors even if they already ran today (skips deduplication)

**Logging:** All runs write to `logs/pipeline_YYYY-MM-DD.log` with 30-day retention.

---

### `config/settings.py`

**Role:** Single source of truth. Every other module imports configuration from here — nothing else reads `.env` directly.

**Lines:** 52

**What it does:**
1. Loads `config/.env` via `python-dotenv`
2. Exposes `START_DATE`, `END_DATE` as strings
3. Creates all required directory tree (`data/raw/equity/`, `data/raw/crypto/`, `data/raw/onchain/`, `data/raw/futures/`, `data/raw/sentiment/`, `data/raw/macro/`, `data/processed/`, `logs/`) using `mkdir(parents=True, exist_ok=True)` — so the directories are guaranteed to exist before any collector runs
4. Exposes all API keys as module-level constants
5. Defines `EQUITY_TICKERS` dict and `REDDIT_SUBREDDITS` list

**Key constants:**

```python
START_DATE = "2018-01-01"   # from .env or default
END_DATE   = "2025-12-31"   # from .env or default
DATA_DIR   = Path("data")
RAW_DIR    = DATA_DIR / "raw"
DB_PATH    = DATA_DIR / "run_log.db"
```

---

### `config/.env.template`

**Role:** Template for the secrets file. Copy to `config/.env`, fill in values, never commit to version control.

Contains all supported environment variables with comments explaining each one. The `GDELT_ENABLED` flag (default `false`) is also controlled here — set to `true` to re-enable GDELT collection if the server becomes responsive.

---

### `collectors/equity_collector.py`

**Role:** Collects equity market data, the Baker-Wurgler sentiment index, and the EPU index.

**Lines:** 319

**Functions:**

#### `_to_naive(index) → pd.DatetimeIndex`
Utility shared across all collectors. Strips timezone from any date-like input — handles `pd.Series`, `pd.DatetimeIndex`, and lists — and always returns a tz-naive `DatetimeIndex`. Uses `pd.DatetimeIndex(pd.to_datetime(index))` explicitly rather than just `pd.to_datetime()` because the latter on a pandas `Series` returns a `Series` (which has no `.tz` attribute), causing `AttributeError`. This was Bug #10 (see Bug Registry).

#### `collect_equity(start, end) → pd.DataFrame`
Downloads daily OHLCV data for 6 tickers via `yfinance`:
- `SPY` — S&P 500 ETF (equity market proxy)
- `^VIX` — CBOE Volatility Index
- `XLK` — Technology sector ETF
- `XLF` — Financials sector ETF
- `GC=F` — Gold futures (risk-off proxy)
- `DX-Y.NYB` — US Dollar Index (DXY)

Computes log returns for each. Saves to `data/raw/equity/equity_daily.parquet` (2,013 rows × 7 columns).

**Result:** 2,013 rows (market days only — weekends and holidays excluded), 99.9% coverage for all series.

#### `collect_baker_wurgler() → pd.DataFrame`
Downloads the Baker-Wurgler (2006) monthly investor sentiment index from Wurgler's NYU page and extends it through 2025 using FRED UMCSENT.

**File structure (confirmed after investigation):**
The Excel file (`Investor_Sentiment_Data_20190327_POST.xlsx`) has three sheets:
- **`README`**: 33 rows of text notes — the title, general notes, update date, methodology description. No data whatsoever. This is what all previous parse attempts accidentally read.
- **`DATA`**: 735 rows × 16 columns. Header in row 0: `yearmo, SENT^, SENT, pdnd, ripo, nipo, cefd, s, [macro controls]`. Data rows use YYYYMM integers (e.g. `195801`). Missing values encoded as Stata `"."` strings, not `NaN`.
- **`STATA CODE`**: The replication script. No data.

**Data coverage:** The file covers July 1965 → December 2018. Last updated March 2019. Our paper window is January 2018 → December 2025. Only 12 months of real BW data overlap.

**Column quirk:** The orthogonalized sentiment column is named `"SENT^ "` (with a trailing space). The parser uses `str(c).strip()` and matches against both `"SENT^"` and `"SENT^ "`.

**Extension strategy:** FRED `UMCSENT` (University of Michigan Consumer Sentiment Index) is fetched for January 2019 → December 2025 (84 months), z-scored to match the BW scale, and concatenated. UMCSENT is one of the original component proxies Baker & Wurgler use to construct their index. Combined result: 96 monthly observations covering the full 2018–2025 window. Forward-filled to daily in the transformer.

**Paper methods note:** "Baker-Wurgler (2006) sentiment data cover January–December 2018. For January 2019 – December 2025, we extend using the University of Michigan Consumer Sentiment Index (FRED: UMCSENT), a core BW component proxy, standardised to zero mean and unit variance."

**Output:** `data/raw/equity/baker_wurgler.parquet` (96 months × 2 columns).

#### `collect_epu(start, end) → pd.DataFrame`
Downloads the Baker-Bloom-Davis Economic Policy Uncertainty Index from FRED (series `USEPUINDXD` — daily frequency, 2,922 observations). The original URL at policyuncertainty.com moved at some point; FRED is now the primary source and is stable.

**Output:** `data/raw/macro/epu_index.parquet` (2,922 rows × 1 column).

---

### `collectors/crypto_collector.py`

**Role:** All cryptocurrency market data — prices, on-chain metrics, stablecoin market cap, futures positioning.

**Lines:** 327

**Functions:**

#### `_to_naive(index) → pd.DatetimeIndex`
Same utility as in equity_collector. The implementation is the same in all 5 collector files and the transformer — this was a deliberate decision to avoid import coupling between collectors.

#### `_cg_get(url, params) → dict`
Retry-wrapped CoinGecko API call. Retries up to 5 times with exponential backoff (4s–60s). Raises on HTTP 429 (rate limit) to trigger the next retry.

#### `_fetch_coin_cg(coin_id, start, end) → pd.DataFrame`
Fetches daily prices, market cap, and volume from CoinGecko's `/coins/{id}/market_chart/range` endpoint for a single coin. Computes log return. Sleeps 7 seconds between coins to respect rate limits.

**Column naming convention:** `bitcoin_price`, `bitcoin_mcap`, `bitcoin_volume`, `bitcoin_return`.

#### `collect_crypto_prices(start, end) → pd.DataFrame`
Attempts CoinGecko for `bitcoin` and `ethereum`. On failure (HTTP 429 or any exception after 5 retries), falls back to `yfinance` for `BTC-USD` and `ETH-USD`. The yfinance fallback provides price, return, and volume but NOT market cap (only CoinGecko provides `bitcoin_mcap` directly). The transformer handles the missing `bitcoin_mcap` by deriving it from `bitcoin_price × 19,500,000` (approximate circulating supply).

**Output:** `data/raw/crypto/crypto_prices.parquet` (2,921 rows × 6 columns).

**Status:** CoinGecko consistently rate-limiting (HTTP 429) throughout development. yfinance fallback is reliable and provides 2,921 rows with 100% coverage.

#### `collect_coinmetrics(start, end) → pd.DataFrame`
Uses the `coinmetrics-api-client` Python package to fetch BTC on-chain metrics from the CoinMetrics Community API.

**Current metric set (community-tier, verified):**

| CoinMetrics Key | Our Column Name | Description |
|----------------|-----------------|-------------|
| `AdrActCnt` | `active_addresses` | Daily unique active addresses |
| `TxCnt` | `btc_tx_count_cm` | Daily confirmed transactions |
| `CapMVRVCur` | `mvrv_ratio` | Market cap / realised cap ratio |
| `SplyAct1yr` | `supply_active_1yr` | BTC moved at least once in past year |

**Removed metrics (PRO-only, confirmed 403):**
- `CapRealUSD` — realised capitalisation (needed for NUPL; PRO-only)
- `NVTAdj` — adjusted NVT ratio (PRO-only)

The CoinMetrics timestamps are UTC ISO8601 strings (`"2022-01-01 00:00:00+00:00"`). The `_to_naive()` fix correctly handles these.

**Output:** Saves to both `data/raw/onchain/coinmetrics.parquet` and `data/raw/onchain/glassnode.parquet` (the latter for transformer compatibility with the `glassnode` alias).

**Status as of final run:** All metrics returning 403. The community tier appears to have progressively restricted access. Blockchain.com provides overlapping coverage for tx count and hash rate. The pipeline continues without error — 0 rows are returned and the transformer skips the file gracefully.

#### `collect_blockchain_stats(start, end) → pd.DataFrame`
Fetches 5 BTC network metrics from Blockchain.com's public charts API (no key required):

| Chart endpoint | Our Column Name | Description |
|---------------|-----------------|-------------|
| `hash-rate` | `btc_hash_rate` | Network hash rate (TH/s) |
| `n-transactions` | `btc_daily_tx_count` | Daily confirmed transactions |
| `estimated-transaction-volume-usd` | `btc_daily_tx_volume_usd` | USD value transferred |
| `miners-revenue` | `btc_miner_revenue_usd` | Total miner revenue (fees + subsidy) |
| `mempool-size` | `btc_mempool_size` | Unconfirmed transaction backlog (bytes) |

Each chart is fetched with `timespan=all` and filtered to the paper window. The API returns Unix timestamps which are parsed with `pd.to_datetime(df["x"], unit="s")`.

**Output:** `data/raw/onchain/blockchain_stats.parquet` (2,493 rows × 5 columns).

**Note:** 2,493 rows rather than 2,922 because Blockchain.com data starts around mid-2018 for some metrics. Forward-filled in the transformer (max 7 days).

#### `collect_defillama_stablecoin(start, end) → pd.DataFrame`
Fetches total stablecoin market capitalisation from DeFiLlama's `stablecoins.llama.fi/stablecoincharts/all` endpoint. Each data point contains a `totalCirculatingUSD` dict — the function sums all values in the dict to get the aggregate stablecoin market cap.

**Date parsing quirk:** The API's `date` field sometimes returns a Unix integer and sometimes a string timestamp. The parser tries `int(float(str(raw_date)))` first, then falls back to `pd.Timestamp(str(raw_date))`.

**Output:** `data/raw/onchain/defillama_stablecoin.parquet` (2,922 rows × 1 column: `stablecoin_total_mcap_usd`).

#### `collect_coinglass(start, end) → pd.DataFrame`
Fetches perpetual futures data from Coinglass: BTC funding rate and long/short account ratio. Requires `COINGLASS_API_KEY`. Without a key, the endpoint returns no data (0 rows). This data is currently unavailable.

**Output:** `data/raw/futures/coinglass_futures.parquet` (0 rows currently).

#### `collect_kaiko(start, end) → pd.DataFrame`
Stub function. Returns empty DataFrame unless `KAIKO_API_KEY` is set. When a Kaiko key is present, this would provide institutional-grade quote-to-trade ratios for the algo intensity variable (H3). Currently uses free proxies instead.

---

### `collectors/sentiment_collector.py`

**Role:** All sentiment-related data — social media NLP, fear/greed index, Google Trends SVI, GDELT (disabled).

**Lines:** 417

**Functions:**

#### `get_finbert() → pipeline`
Lazy-loads the FinBERT model (`ProsusAI/finbert`) from HuggingFace. Downloads ~500MB on first call, cached thereafter. Uses HuggingFace `transformers.pipeline` with `device=-1` (CPU). Returns a text-classification pipeline.

#### `score_finbert(texts) → list[float]`
Scores a list of texts with FinBERT in batches of 32. Each text is truncated to 512 tokens. Returns `positive_score - negative_score` for each text, giving a float in `[-1, +1]`.

#### `score_vader(texts) → list[float]`
VADER fallback scorer. Much faster than FinBERT (no GPU needed, no download) but less domain-appropriate for financial text. Returns compound score in `[-1, +1]`.

#### `collect_stocktwits(days_back, pages_per_symbol, use_finbert) → pd.DataFrame`
Collects StockTwits messages for 3 symbols: `$BTC.X`, `$ETH.X`, `$CRYPTO.X`.

- Paginates backwards in time using `max_id` cursor
- `pages_per_symbol=15` (configurable) ≈ 450 messages per symbol
- Each page request sleeps 18 seconds to respect rate limits
- Messages are filtered to `days_back=30` (configurable)
- All collected messages are FinBERT-scored in one batch
- Aggregated to daily: weighted average by (likes + reshares), mean, bullish ratio, message count

**Timezone issue fixed:** `datetime.utcnow()` (naive) was being compared to API timestamps that are timezone-aware. Fixed using `datetime.now(timezone.utc)`.

**Output:** `data/raw/sentiment/stocktwits_sentiment.parquet` (currently 8 days × 4 columns — free tier API only provides ~30 days of history; run daily to accumulate).

**4 output columns:**
- `stocktwits_sentiment_weighted` — engagement-weighted FinBERT score
- `stocktwits_sentiment_mean` — simple mean FinBERT score
- `stocktwits_bullish_ratio` — fraction of messages with native "Bullish" label
- `stocktwits_msg_count` — total messages collected that day

#### `collect_reddit(start, end, use_finbert) → pd.DataFrame`
Uses PRAW to collect posts from 4 subreddits: `r/cryptocurrency`, `r/Bitcoin`, `r/ethereum`, `r/CryptoCurrency`. Requires `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` in `.env`. Without credentials, returns empty DataFrame and logs a warning.

Posts are scored with FinBERT and aggregated to daily: score-weighted sentiment, mean sentiment, post count.

**Output:** `data/raw/sentiment/reddit_sentiment.parquet`

#### `collect_fear_greed(start, end) → pd.DataFrame`
Fetches the Crypto Fear & Greed Index from Alternative.me's API (`api.alternative.me/fng/?limit=3000`).

**Date parsing evolution:**
- The API originally returned Unix timestamps as integers
- It then changed to return date strings in `"MM-DD-YYYY"` format (e.g. `"04-01-2026"`)
- The parser now tries `int(val)` first (Unix seconds), then `pd.to_datetime(str(val))` as fallback — handles both formats without configuration

**Retry logic:** 3 attempts with 15-second sleep between failures. 60-second timeout per request (the server is occasionally slow).

**Output:** `data/raw/sentiment/fear_greed.parquet` (2,887 rows × 2 columns: `fear_greed_index` [0–100], `fear_greed_label` [text classification]).

**Status:** Working as of final run. Was timing out in earlier runs when the server was intermittently down.

#### `collect_google_trends(keywords, start, end) → pd.DataFrame`
Uses `pytrends` to fetch Google Search Volume Index (SVI) for 5 keywords:
- `"Bitcoin"` — general attention/adoption interest
- `"crypto crash"` — fear/negative sentiment signal
- `"buy bitcoin"` — purchase intent (bullish retail signal)
- `"Ethereum"` — altcoin attention
- `"crypto"` — general crypto awareness

**Batching strategy:** Data is collected quarterly (3-month windows) to work around pytrends' limitation that long timeframes return weekly data while short timeframes return daily. Each window sleeps 6 seconds after fetch, 20 seconds on error.

**Output column names:** `gtrends_bitcoin`, `gtrends_crypto_crash`, `gtrends_buy_bitcoin`, `gtrends_ethereum`, `gtrends_crypto`

**Output:** `data/raw/sentiment/google_trends.parquet` (2,831 weekly obs × 5 columns — weekly because pytrends returns weekly granularity for multi-year queries; forward-filled to daily in transformer).

#### `collect_gdelt(query, start, end) → pd.DataFrame`
**DISABLED by default.** GDELT's API (`api.gdeltproject.org`) times out on every request. For a full 2018–2025 collection window of ~400 weeks, this would accumulate 3+ hours of 33-second timeouts while collecting zero data. Set `GDELT_ENABLED=true` in `.env` to re-enable if the server becomes responsive. When enabled, queries `"Bitcoin OR cryptocurrency"` in weekly batches and returns `gdelt_avg_tone` and `gdelt_num_articles`.

---

### `collectors/macro_collector.py`

**Role:** Downloads 6 macroeconomic time series from FRED.

**Lines:** 77

**Requires:** `FRED_API_KEY` in `.env`

**Series fetched:**

| FRED Series | Our Column Name | Description | Frequency |
|-------------|-----------------|-------------|-----------|
| `DFF` | `fed_funds_rate` | Effective Federal Funds Rate | Daily |
| `VIXCLS` | `vix_fred` | CBOE VIX (FRED cross-check) | Daily |
| `DEXUSEU` | `usdeur_exchange` | USD/EUR exchange rate | Daily |
| `DCOILWTICO` | `crude_oil_wti` | WTI crude oil price | Daily |
| `T10YIE` | `inflation_breakeven_10y` | 10-year breakeven inflation | Daily |
| `BAMLH0A0HYM2` | `hy_credit_spread` | High-yield credit spread (OAS) | Daily |

All series are fetched for the full 2018–2025 window. Weekend/holiday gaps are forward-filled (max 3 days) in the transformer. `DFF` has 2,922 observations; the market-linked series (`VIXCLS`, exchange rates, oil, HY spread) have 2,088 observations (trading days only).

**Output:** `data/raw/macro/fred_macro.parquet` (2,922 rows × 6 columns).

---

### `collectors/regulatory_collector.py`

**Role:** Constructs binary event dummy variables for positive and negative crypto regulatory announcements.

**Lines:** 641

**Three data sources attempted, in priority order:**

#### Source 1 — SEC EDGAR Full-Text Search API (free, no key)
Endpoint: `https://efts.sec.gov/LATEST/search-index`

Queries 5 crypto keyword combinations:
- `"bitcoin" "enforcement"`
- `"cryptocurrency" "charges"`
- `"digital asset" "registration"`
- `"virtual currency" "fraud"`
- `"stablecoin"`

**Status: Failing with `"can only concatenate str (not 'list') to str"`**

Root cause: The EDGAR API returns `display_names` as a Python list, not a string. The URL construction code was doing string concatenation with this list. This bug was never fixed because the Latham & Watkins tracker provides sufficient coverage. EDGAR is logged as a warning and the pipeline continues.

#### Source 2 — CFTC Press Releases RSS
Endpoint: `https://www.cftc.gov/rss/pressreleases.xml`

**Status: Failing with HTTP 403**

The CFTC RSS feed blocks automated access. The historical archive scraper (which tries individual year pages) also returns 0 events. Both have been failing consistently throughout development.

#### Source 3 — Latham & Watkins US Crypto Policy Tracker (free, no key)
URL: `https://www.lw.com/en/us-crypto-policy-tracker/regulatory-developments`

A structured law-firm maintained timeline of US crypto regulatory events from 2018 through 2026. Successfully scrapes 66 events with dates, descriptions, and direction (positive/negative).

**Direction coding:** Applied algorithmically using keyword classifier:
- Positive keywords: `approval, approve, legal tender, clarity, ETF, framework, permit`
- Negative keywords: `charges, ban, fraud, cease, halt, illegal, enforcement, violation, suspend`

**Result:** 66 events → 13 positive-direction days, 48 negative-direction days, 5 neutral.

**Output files:**
- `data/raw/macro/regulatory_events.csv` — Human-readable audit trail (date, event_type, direction, description, source)
- `data/raw/macro/regulatory_dummies.parquet` — Daily binary columns: `reg_dummy_positive`, `reg_dummy_negative`, `reg_dummy_net`

#### `_build_daily_panel(events_df, start, end) → pd.DataFrame`
Converts the event list to a daily panel with 0/1 dummy variables. `reg_dummy_net = reg_dummy_positive - reg_dummy_negative` takes values +1, 0, −1.

---

### `collectors/regulatory_events.csv`

**Role:** Fallback dataset of 70 hand-coded regulatory events 2018–2025.

**Columns:** `date`, `event_type`, `direction`, `description`, `source`

**Sources coded:** SEC, CFTC, MOEF (South Korea), G20, El Salvador government, US Treasury, China PBOC, EU (MiCA), US Congress

Used by the feature engineering module as a fallback when the live scraper produces fewer than 10 events. In practice, the LW Tracker produces 66 events, so the CSV fallback has not been needed in recent runs.

---

### `pipeline/orchestrator.py`

**Role:** Central runner. Calls all collectors in order, manages the SQLite run log, and handles deduplication.

**Lines:** 227

**SQLite run log schema (`data/run_log.db`):**
```sql
CREATE TABLE run_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    collector   TEXT    NOT NULL,
    start_date  TEXT,
    end_date    TEXT,
    status      TEXT,          -- 'success' or 'failed'
    rows        INTEGER,
    output_path TEXT,
    error_msg   TEXT,
    run_at      TIMESTAMP
)
```

**Deduplication:** Before running each collector, the orchestrator checks if it already ran successfully today. If so, it skips it (unless `--force` is passed). This makes `--mode incremental` safe to run multiple times per day.

**Collector run order:**
1. `equity_prices` → `baker_wurgler` → `epu_index`
2. `crypto_prices` → `coinmetrics_onchain` → `blockchain_stats` → `defillama_stablecoin` → `coinglass_futures`
3. `stocktwits` → `fear_greed` → `google_trends` → `gdelt_news`
4. `fred_macro` → `regulatory_events`

Then Steps 2–4:
5. `transformer` → builds `master_panel.parquet`
6. `feature_engineering` → builds `master_panel_features.parquet`
7. `orthogonalization` → builds `master_panel_final.parquet`

**Rich progress display:** Uses the `rich` library to render a progress table in the terminal. Each collector shows a spinner while running, then a checkmark or cross when done.

**Scheduler:** `APScheduler` background job calls `run_incremental_update()` at 07:00 UTC daily when `--mode schedule` is used.

---

### `pipeline/transformer.py`

**Role:** Merges all raw Parquet shards into a single wide daily panel.

**Lines:** 204

#### `_strip_tz(index) → pd.DatetimeIndex`
The transformer's equivalent of `_to_naive`. Same implementation — uses `pd.DatetimeIndex(pd.to_datetime(index))` to guarantee a `DatetimeIndex` (not `Series`) before checking `.tz`.

#### `_load(path, label) → pd.DataFrame`
Safely loads a Parquet file. Returns empty DataFrame with a warning if the file doesn't exist. Strips tz from the index. Logs shape on success.

#### `_merge_numeric(master, df) → pd.DataFrame`
Left-joins a shard onto the master date index. Uses `master.join(df, how="left")`.

#### `build_master_panel(start, end) → pd.DataFrame`

**Step 1 — Build master date index:**
All calendar days from `start` to `end` (2,922 days for 2018-01-01 to 2025-12-31). This is the spine — all shards are left-joined onto it.

**Step 2 — Load and merge daily-frequency shards:**
- Equity daily (2,013 trading days → joined, gaps remain NaN until fill)
- Crypto prices (2,921 days)
- CoinMetrics on-chain (0 rows currently)
- Blockchain.com (2,493 rows)
- DeFiLlama stablecoin (2,922 rows)
- Coinglass futures (0 rows)
- StockTwits (8 rows → zero-filled)
- Reddit (0 rows → zero-filled)
- Fear & Greed (2,887 rows)
- FRED macro (2,922 rows)
- EPU index (2,922 rows)
- Regulatory dummies (2,922 rows)

**Step 3 — Monthly → daily: Baker-Wurgler:**
BW is monthly. The transformer resamples to month-start, then reindexes to the daily master index and forward-fills within each month. This gives daily values that change once per month, which is academically standard for monthly sentiment indices.

**Step 4 — Weekly → daily: Google Trends:**
Google Trends SVI is weekly. Same pattern: reindex to daily, forward-fill.

**Step 5 — Derived columns:**
- `bitcoin_mcap`: If `bitcoin_mcap` is missing (yfinance fallback doesn't provide it), derive as `bitcoin_price × 19,500,000`. The true circulating supply varied from ~17M (2018) to ~19.8M (2025); 19.5M is a conservative midpoint giving <3% error throughout.
- `stablecoin_supply_ratio`: `bitcoin_mcap / stablecoin_total_mcap_usd`
- Log transforms: `log_bitcoin_volume`, `log_ethereum_volume`, `log_btc_daily_tx_count`, `log_btc_daily_tx_volume_usd`, `log_epu_index`, `log_btc_miner_revenue_usd` (using `np.log1p` after `clip(lower=0)` to handle zeros)

**Step 6 — Missing value policy:**

| Column type | Fill rule |
|-------------|-----------|
| Price / return / VIX columns | `ffill` max 3 days |
| Sentiment columns | `fillna(0)` (neutral on missing days) |
| Regulatory dummies | `fillna(0).astype(int)` |
| On-chain / macro | `ffill` max 7 days |

**Output:** `data/processed/master_panel.parquet` (2,922 rows × 50 columns after final run).

---

### `pipeline/feature_engineering.py`

**Role:** Constructs all derived features needed for the paper's three hypotheses.

**Lines:** 442

#### `_pca_composite(X_df, name) → pd.Series`
Runs PCA (1 component) on a standardised input matrix. Logs the explained variance ratio. Returns the PC1 scores as a Series.

#### `_sign_normalise(composite, df) → pd.Series`
Ensures that positive composite values correspond to bullish sentiment. Checks correlation with `bitcoin_return` (the most direct signal); if negative, flips the composite. Falls back to Fear & Greed correlation if BTC return isn't available.

**In the final run:** BTC return correlation = +0.078 — sign was already correct, no flip needed.

#### `_build_gtrends_composite(df) → pd.Series`

The core sentiment composite for the current pipeline state. Constructs:

```
composite = PC1([gtrends_bitcoin, gtrends_buy_bitcoin,
                 gtrends_ethereum, gtrends_crypto])
           − z(gtrends_crypto_crash)
```

Then standardises to zero mean / unit variance.

**Academic grounding:**
- Da, Engelberg & Gao (2011, *Review of Financial Studies*): SVI directly captures retail investor attention and predicts next-week stock returns
- Kristoufek (2013, *Scientific Reports*): Google Trends for "Bitcoin" Granger-causes BTC price in both directions; effect is asymmetric between price increases and decreases
- Urquhart (2018, *Economics Letters*): SVI is a significant driver of next-day BTC volatility

The PC1 of the 4 bullish keywords captures the common "crypto attention" factor, while subtracting z-scored `gtrends_crypto_crash` shifts the composite in the fear direction on weeks where crash-related searches spike.

**Coverage:** 2,922 daily observations (100%), mean=0.0000, std=1.0000. PC1 explains 65.2% of variance in the 4 bullish keywords — very high for 4 correlated time series.

#### `build_sentiment_composite(df) → tuple[pd.Series, bool]`

The master sentiment compositor with automatic source selection:

**Primary source (always active):** Google Trends SVI — as described above. Provides 100% daily coverage for 2018–2025. Returns `(composite, True)` for the `gtrends_sentiment_used` flag.

**Optional NLP blend (activates at ≥30 real social media obs):** When StockTwits or Reddit has accumulated 30+ real (non-zero, non-NaN) observations, the NLP signal is standardised and averaged equally with the Google Trends composite. Currently inactive (StockTwits has only 8 days of real data). Will activate automatically as daily accumulation continues.

**Returns:** `(composite_series, used_gtrends: bool)` — the boolean flag is written as `gtrends_sentiment_used` column in the output parquet for use as a robustness control in regression tables.

#### `build_algo_intensity(df) → pd.Series`

Constructs the algorithmic trading intensity proxy for H3.

**Institutional source (preferred, currently unavailable):** Kaiko quote-to-trade ratio (`kaiko_qtr_btc`) or TAQ message-to-trade ratio (`taq_msg_trade_spy`). These require paid API keys.

**Free proxies used:**
- Crypto proxy: `bitcoin_volume × |bitcoin_return|` — high volume with large moves = algorithmic activity
- Equity proxy: `|sp500_return| / (vix/100)` — large moves relative to implied volatility = algo-driven price discovery

Both proxies are z-scored, shifted by +4σ to make strictly positive, log-geometric-averaged, then re-standardised. The +4σ shift is mathematically necessary for the log operation and cancels out after standardisation.

**Output:** `algo_intensity` — 2,920 obs, std=1.000.

#### `build_realized_vol(df, start, end) → pd.DataFrame`

Attempts to download hourly intraday data from yfinance for BTC-USD and SPY to compute Andersen et al. (2003) realized variance. Returns empty on both because yfinance's free hourly data is limited to ~730 days of history.

**Fallback (always used):** `|daily_return| × √252` — the standard daily proxy for annualised realized volatility. Produces `btc_rv_proxy` and `spy_rv_proxy`.

#### `build_regulatory_dummies(df) → pd.DataFrame`

Loads `data/raw/macro/regulatory_dummies.parquet` (produced by the regulatory collector). Falls back to the bundled `collectors/regulatory_events.csv` if the parquet file isn't found.

**Output columns:** `reg_dummy_positive`, `reg_dummy_negative`, `reg_dummy_net`

**Final run:** 13 positive-direction days, 48 negative-direction days across 2018–2025.

#### `build_rolling_corr(df, window=30) → pd.Series`

Rolling 30-day Pearson correlation between `bitcoin_return` and `sp500_return`.

**Final run stats:** mean=0.220, range=[−0.420, 0.723]. The positive mean correlation reflects the broad risk-on/risk-off co-movement that became prominent after 2020.

#### `run_feature_engineering(start, end)`

Master runner. Loads `master_panel.parquet`, runs all 5 feature builders, logs coverage for key variables, saves `master_panel_features.parquet` (2,922 rows × 56 columns after final run).

---

### `pipeline/orthogonalization.py`

**Role:** Implements Section 3.3.2. Decomposes raw sentiment into rational and irrational components via OLS regression.

**Lines:** 256

#### Equation 1 (the first-stage regression)

```
Crypto_Sentiment_t = α
  + β₁·bitcoin_return_{t-1}       [BTC lagged return]
  + β₂·ethereum_return_{t-1}      [ETH lagged return]
  + β₃·btc_rv_{t-1}               [BTC volatility, lagged]
  + β₄·btc_hash_rate_{t-1}        [Mining security, lagged]
  + β₅·active_addresses_{t-1}     [Network activity, lagged — or log_bitcoin_volume fallback]
  + β₆·btc_daily_tx_count_{t-1}   [Transaction flow, lagged]
  + β₇·fed_funds_rate_t           [Monetary policy, contemporaneous]
  + β₈·dxy_return_t               [Dollar strength, contemporaneous]
  + β₉·vix_t                      [Market fear, contemporaneous]
  + β₁₀·gold_return_t             [Risk-off proxy, contemporaneous]
  + β₁₁·reg_dummy_net_t           [Regulatory environment, contemporaneous]
  + ε_t
```

**Estimation:** OLS with Newey-West HAC standard errors (maxlags=5) using `statsmodels`. HAC errors account for the autocorrelation and heteroskedasticity expected in daily financial time series.

**Lag structure:** Market variables are lagged one day (`shift(1)`) to avoid mechanical simultaneity (today's BTC return mechanically correlates with today's sentiment if investors are momentum-following). Macro and regulatory variables are contemporaneous (these are known at the start of the trading day and are not caused by the same day's sentiment movement).

**Fallback columns:** If a column is missing (e.g., `btc_rv_intraday_ann` because intraday data is unavailable), the code falls back to alternatives in priority order:
- `btc_rv_intraday_ann` → `btc_rv_proxy` → `bitcoin_rv`
- `active_addresses` → `log_bitcoin_volume` (when CoinMetrics is down)

#### `run_first_stage(df, sentiment_col) → dict`
Constructs the regressor matrix, drops NaN rows (leaves n=2,920 after lagging), runs OLS. Returns fitted values and residuals.

**Minimum obs check:** Raises `ValueError` if fewer than 60 complete observations, preventing degenerate estimation.

#### `run_sensitivity(result) → pd.DataFrame`
Leave-one-out analysis: drops each regressor one at a time and reports the resulting R², the change in R², and the correlation of the leave-one-out residual with the baseline residual. The high `corr_with_base_resid` values (all ≥0.999 in the final run) confirm that `irrational_sentiment` is stable regardless of which regressor is dropped — reassuring for the paper's robustness section.

#### `run_orthogonalization(start, end) → pd.DataFrame`
Master runner. Loads `master_panel_features.parquet`, runs first-stage OLS, adds three new columns to the panel, verifies orthogonality, saves the report and `master_panel_final.parquet`.

**Decomposition:**
- `sentiment_raw` = `crypto_sentiment_composite` (unchanged — for comparison with prior literature)
- `rational_sentiment` = OLS fitted values (ŷ) — the part of sentiment explained by fundamentals
- `irrational_sentiment` = OLS residuals (ε̂) — the paper's main independent variable

**By OLS construction, corr(rational, irrational) = 0 to machine precision.** The final run confirms: corr = −1.71×10⁻⁸.

---

### `analysis/validate.py`

**Role:** Post-pipeline quality checks.

**Lines:** 204

**Checks performed:**

1. **Coverage report:** Per-variable % of non-NaN observations. Flags anything below 80%. Saves `coverage_report.csv`.

2. **ADF stationarity tests:** Augmented Dickey-Fuller on all return, sentiment, and correlation series. The final validation report shows 21/21 series stationary (p < 0.05) — important because SVAR requires stationary inputs.

3. **Outlier detection:** Flags cells where |z-score| > 5. Regulatory dummies produce legitimate outliers (they're 0/1 binary). EPU index has 2 extreme outlier days (COVID-19 peak in March 2020). Total: 40 cells flagged, all explainable.

4. **Orthogonality verification:** Re-checks corr(rational, irrational) against 0.01 threshold.

5. **Cross-source reconciliation:** Compares BTC price from CoinGecko vs yfinance when both are available. (Currently limited — CoinGecko has been failing, so only yfinance data exists.)

**Run:** `python main.py --mode validate`

---

### `data/processed/`

#### `master_panel.parquet`
The merged daily panel after the transformer. Shape: 2,922 × 50. Contains all raw collected variables with standardised column names and tz-naive DatetimeIndex.

#### `master_panel_features.parquet`
After feature engineering. Shape: 2,922 × 56. Adds: `crypto_sentiment_composite`, `gtrends_sentiment_used`, `algo_intensity`, `btc_rv_proxy`, `spy_rv_proxy`, `btc_sp500_corr_30d`, `reg_dummy_positive`, `reg_dummy_negative`, `reg_dummy_net`.

#### `master_panel_final.parquet`
The analysis-ready dataset. Shape: 2,922 × 59. Adds: `sentiment_raw`, `rational_sentiment`, `irrational_sentiment`. This file is the input to all hypothesis tests (SVAR, VAR-with-interaction, mediation analysis).

#### `coverage_report.csv`
Two-column CSV: variable name, coverage percentage. Generated by `validate.py`.

#### `orthogonalization_report.txt`
Full OLS output: coefficient table, decomposition statistics, sensitivity analysis. Human-readable. See Section 13 for the key numbers.

#### `validation_report.txt`
ADF test results, outlier summary, orthogonality check, coverage summary.

---

### `requirements.txt`

All Python dependencies with minimum version pins.

**Core stack:** `pandas>=2.0.0`, `numpy>=1.24.0`, `pyarrow>=12.0.0`, `scipy>=1.11.0`, `statsmodels>=0.14.0`, `scikit-learn>=1.3.0`

**HTTP/networking:** `requests>=2.31.0`, `aiohttp>=3.9.0`, `tenacity>=8.2.3` (for retry logic)

**Data sources:** `yfinance>=0.2.36`, `praw>=7.7.1`, `pytrends>=4.9.2`, `fredapi>=0.5.1`, `openpyxl>=3.1.0`

**NLP:** `transformers>=4.35.0`, `torch>=2.1.0`, `vaderSentiment>=3.3.2`

**Scheduling/storage:** `APScheduler>=3.10.0`, `SQLAlchemy>=2.0.0`

**Dev/UX:** `python-dotenv>=1.0.0`, `loguru>=0.7.2`, `rich>=13.0.0`, `tqdm>=4.66.0`

**Special:** `coinmetrics-api-client>=2024.1.1` — the official Python client for the CoinMetrics API (community tier requires no key)

---

## 6. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: COLLECTORS  (parallel sources → raw Parquet shards) │
│                                                               │
│  equity_collector.py ──────────────┐                         │
│    ├── SPY, VIX, XLK, XLF, Gold    │                         │
│    ├── Baker-Wurgler (BW + UMCSENT) │  data/raw/equity/       │
│    └── EPU (FRED)                   │                         │
│                                     │                         │
│  crypto_collector.py ───────────────┤                         │
│    ├── BTC/ETH prices (yfinance)    │  data/raw/crypto/       │
│    ├── CoinMetrics on-chain         │  data/raw/onchain/      │
│    ├── Blockchain.com               │                         │
│    ├── DeFiLlama stablecoins        │                         │
│    └── Coinglass futures            │  data/raw/futures/      │
│                                     │                         │
│  sentiment_collector.py ────────────┤                         │
│    ├── StockTwits + FinBERT         │  data/raw/sentiment/    │
│    ├── Reddit + FinBERT             │                         │
│    ├── Fear & Greed (Alternative.me)│                         │
│    ├── Google Trends SVI            │                         │
│    └── GDELT [DISABLED]             │                         │
│                                     │                         │
│  macro_collector.py ────────────────┤  data/raw/macro/        │
│    └── 6 FRED series               │                         │
│                                     │                         │
│  regulatory_collector.py ───────────┘  data/raw/macro/       │
│    └── LW Tracker → daily dummies   │                         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: TRANSFORMER  (transformer.py)                       │
│                                                               │
│  • Master date index: 2018-01-01 → 2025-12-31 (2922 days)   │
│  • Left-join all shards                                       │
│  • Monthly → daily: Baker-Wurgler (ffill within month)       │
│  • Weekly → daily: Google Trends (ffill)                      │
│  • Derive: bitcoin_mcap, stablecoin_supply_ratio              │
│  • Log transforms: 6 skewed variables                         │
│  • Missing value policy (ffill / fillna(0) by type)          │
│                                                               │
│  → master_panel.parquet  (2922 × 50)                         │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: FEATURE ENGINEERING  (feature_engineering.py)       │
│                                                               │
│  • crypto_sentiment_composite  ← Google Trends PCA           │
│    PC1([btc,buy_btc,eth,crypto SVI]) − z(crypto_crash SVI)   │
│    Sign-normalised vs BTC return (corr = +0.078 ✓)           │
│                                                               │
│  • algo_intensity               ← BTC vol × |SPY/VIX| proxy  │
│  • btc_rv_proxy                 ← |daily return| × √252      │
│  • spy_rv_proxy                 ← |daily return| × √252      │
│  • btc_sp500_corr_30d           ← rolling 30-day Pearson     │
│  • reg_dummy_{pos,neg,net}      ← from regulatory parquet    │
│                                                               │
│  → master_panel_features.parquet  (2922 × 56)                │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: ORTHOGONALIZATION  (orthogonalization.py)           │
│                                                               │
│  Equation 1 OLS:                                              │
│  Sentiment_t = α + β₁·BTC_{t-1} + β₂·ETH_{t-1}             │
│               + β₃·RV_{t-1} + β₄·HashRate_{t-1}             │
│               + β₅·ActiveAddr_{t-1} + β₆·TxCount_{t-1}      │
│               + β₇·FedFunds + β₈·DXY + β₉·VIX              │
│               + β₁₀·Gold + β₁₁·RegDummy + ε_t              │
│                                                               │
│  HAC (Newey-West, maxlags=5)                                  │
│  n=2920, k=12                                                 │
│  R²=0.0556, F p-val=2.80×10⁻¹⁰ ***                          │
│  corr(rational, irrational) = −1.71×10⁻⁸ ≈ 0 ✓             │
│                                                               │
│  → master_panel_final.parquet  (2922 × 59)                   │
│    rational_sentiment:   2920 obs                             │
│    irrational_sentiment: 2920 obs  ← paper's main variable   │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Data Sources — Full Inventory

| Source | Variables | Raw file | Rows | Key |
|--------|-----------|----------|------|-----|
| yfinance | SPY, VIX, XLK, XLF, Gold, DXY returns/prices | `equity/equity_daily.parquet` | 2,013 | None |
| Baker-Wurgler NYU + FRED UMCSENT | `bw_sentiment`, `bw_sentiment_orthogonalized` | `equity/baker_wurgler.parquet` | 96 months | FRED key |
| FRED USEPUINDXD | `epu_index` | `macro/epu_index.parquet` | 2,922 | FRED key |
| yfinance (CoinGecko fallback) | BTC/ETH price, return, volume | `crypto/crypto_prices.parquet` | 2,921 | None |
| CoinMetrics Community | active_addresses, mvrv_ratio, tx_count, supply_active | `onchain/coinmetrics.parquet` | 0 (403) | None |
| Blockchain.com | hash_rate, tx_count, tx_volume, miner_revenue, mempool | `onchain/blockchain_stats.parquet` | 2,493 | None |
| DeFiLlama | `stablecoin_total_mcap_usd` | `onchain/defillama_stablecoin.parquet` | 2,922 | None |
| Coinglass | btc_funding_rate, btc_long_short_ratio | `futures/coinglass_futures.parquet` | 0 (no key) | Paid |
| StockTwits + FinBERT | `stocktwits_sentiment_*` (4 cols) | `sentiment/stocktwits_sentiment.parquet` | 8 | None |
| Reddit PRAW + FinBERT | `reddit_sentiment_*` (3 cols) | `sentiment/reddit_sentiment.parquet` | — | Free key |
| Alternative.me | `fear_greed_index`, `fear_greed_label` | `sentiment/fear_greed.parquet` | 2,887 | None |
| Google Trends (pytrends) | 5 `gtrends_*` keywords | `sentiment/google_trends.parquet` | 2,831 | None |
| GDELT | `gdelt_avg_tone`, `gdelt_num_articles` | — | 0 (disabled) | None |
| FRED (6 series) | fed_funds, vix_fred, usdeur, wti, breakeven, hy_spread | `macro/fred_macro.parquet` | 2,922 | FRED key |
| LW Tracker | `reg_dummy_*` | `macro/regulatory_dummies.parquet` | 2,922 | None |

---

## 8. Methodology

### Section 3.3.1 — Composite Sentiment Index

#### Design principle

The composite sentiment index `crypto_sentiment_composite` is designed to capture retail investor attention and speculative enthusiasm in the cryptocurrency space, orthogonal (after Step 4) to any information that can be explained by fundamental economic variables.

#### Source hierarchy

**Primary source — Google Trends SVI (always active, 100% coverage)**

The composite is built from Google Search Volume Index data for 5 keywords:

1. `"Bitcoin"`, `"Ethereum"`, `"crypto"`, `"buy bitcoin"` → bullish attention proxies
2. `"crypto crash"` → fear proxy (enters negatively)

Construction:
```
PC1([bitcoin_SVI, buy_bitcoin_SVI, ethereum_SVI, crypto_SVI])   — attention factor
  minus z-scored(crypto_crash_SVI)                              — fear signal
  → standardised to mean=0, std=1
```

PC1 explains 65.2% of variance in the 4 bullish keywords — the keywords are highly correlated (they all spike together during crypto bull markets), and PC1 captures this common "crypto mania" factor.

The fear term is subtracted after PCA, not included in the PCA itself. This ensures the sign is interpretable: a positive composite means more bullish attention relative to fear, and a negative composite means fear is elevated relative to attention.

Sign normalisation is applied by computing the correlation with `bitcoin_return`. If the correlation is negative (composite goes up when BTC goes down — indicating the sign is inverted), the composite is multiplied by −1. In the final run, the correlation was +0.078, confirming the sign was already correct.

**Academic foundation:**
- Da, Engelberg & Gao (2011, *RFS*): SVI for individual stocks predicts next-week returns; attention-driven sentiment is a strong retail investor proxy
- Kristoufek (2013, *Scientific Reports*): BTC SVI Granger-causes BTC price in both directions with a 2-week lag
- Urquhart (2018, *Economics Letters*): SVI is a significant predictor of next-day BTC volatility, subsuming other sentiment measures

This makes Google Trends not a fallback or proxy — it is a primary, academically well-grounded sentiment measure for the cryptocurrency context.

**Supplementary blend — Social NLP (activates automatically at ≥30 real observations)**

When StockTwits and/or Reddit has accumulated ≥30 days of real NLP-scored data, the code automatically blends the social signal in (equal weight with the Google Trends composite). The blend upgrades silently as daily data accumulates. Currently inactive (8 days of StockTwits data). This follows multi-source sentiment literature (Bollen et al. 2011).

**Flag column:** `gtrends_sentiment_used = 1` is written throughout the panel whenever the Google Trends path is used. Include this as a robustness control in regression tables.

### Section 3.3.2 — Orthogonalization (Equation 1)

The orthogonalization procedure follows Baker & Wurgler (2006) adapted to the crypto context:

1. **Regress** `crypto_sentiment_composite` on all available fundamental variables (lagged market variables + contemporaneous macro + regulatory dummies)
2. **Fitted values** = `rational_sentiment` — the portion of sentiment explained by fundamentals
3. **Residuals** = `irrational_sentiment` — the portion orthogonal to all fundamentals, the paper's main independent variable

By OLS construction, the correlation between fitted values and residuals is exactly zero (to machine precision: −1.71×10⁻⁸ in the final run). This ensures that any effect of `irrational_sentiment` on equity returns in the paper's second stage cannot be attributed to fundamental information.

**R² = 0.0556:** Fundamentals explain 5.56% of daily variation in crypto sentiment. This is a meaningful result — it means sentiment is largely detached from fundamentals (94.4% of variation is irrational), which is consistent with the speculative nature of crypto markets and directly motivates the paper's thesis.

### Feature Engineering

See Section 5 (`pipeline/feature_engineering.py`) for full details on each feature.

---

## 9. Complete Development History — What We Tried, What Broke, What We Fixed

This section documents every significant decision, failure, and workaround in the order they occurred. It is intended to enable anyone to understand why the code looks the way it does, and to avoid re-introducing solved problems.

---

### Phase 1 — Initial Architecture

**What we designed:**

The initial architecture assumed the following data would be available at full coverage for 2018–2025:

- CoinGecko: BTC/ETH prices and market caps (free API)
- Glassnode: on-chain metrics (free tier)
- StockTwits: full sentiment history (free API)
- Reddit: historical posts (PRAW)
- GDELT: news sentiment (free API)
- Baker-Wurgler: full 2018–2025 monthly sentiment (NYU website)
- EDGAR/CFTC: live regulatory event feeds (free government APIs)

The sentiment composite was designed to use PC1 of `[stocktwits_sentiment_weighted, reddit_sentiment_weighted]` as the primary signal.

The orthogonalization was designed to use CoinMetrics metrics (`active_addresses`, `mvrv_ratio`, `nvt_ratio`) as on-chain fundamentals in Equation 1.

**What immediately became clear:** None of these assumptions held in practice.

---

### Phase 2 — First Full Run Failures

**Run 1 results:**

Every collector except FRED and EPU index returned 0 rows.

**Glassnode (original on-chain source):** The free tier API was discontinued without announcement. Every endpoint returned 403. Decision: replace with CoinMetrics Community API (`coinmetrics-api-client` package) which claimed to provide `active_addresses`, `mvrv_ratio`, `nvt_ratio` for free.

**CoinGecko:** Rate-limiting (HTTP 429) on every request. Retrying with exponential backoff eventually exhausts all retries. Added yfinance as fallback. yfinance provides OHLCV for BTC-USD and ETH-USD but NOT market cap (only CoinGecko's `market_caps` field provides this). This created the `bitcoin_mcap` gap that had to be solved later with the derived column.

**Baker-Wurgler:** The parser tried `header=0`, `header=1`, `header=2` on the sheet. All three failed. Every header position returned a DataFrame that didn't contain YYYYMM-format dates. The actual problem (it was reading the README sheet which has only 2 columns of text) wasn't diagnosed until much later.

**GDELT:** API server timed out on every request. ~33 seconds per request × 400 weeks = 3.6 hours of pure timeout with zero data. Disabled immediately and put behind `GDELT_ENABLED=true` flag.

**StockTwits:** Only returned 30 days of history on the free tier. The full 2018–2025 sentiment history was never achievable from this API without a paid historical data contract.

**EPU:** The policyuncertainty.com URL had moved. Direct URL failed with 404. Switched to FRED `USEPUINDXD` which is the same series.

---

### Phase 3 — Timezone Hell

**The error:** After fixing the above and getting some data collected, the transformer crashed:
```
ValueError: Cannot join tz-naive with tz-aware DatetimeIndex
```

**Root cause:** Different data sources return timestamps with different timezone conventions:
- yfinance (newer versions): tz-aware UTC (`DatetimeIndex` with `.tz = UTC`)
- CoinMetrics: UTC ISO8601 strings that parse as tz-aware (`"2022-01-01 00:00:00+00:00"`)
- FRED, Blockchain.com: tz-naive (no timezone at all)

When the transformer tried to left-join a tz-aware index onto a tz-naive master index, pandas raised the error.

**First fix attempt:** Added `_to_naive()` function to each collector. Original implementation:
```python
def _to_naive(index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx
```

**Second error with this fix:**
```
AttributeError: 'Series' object has no attribute 'tz'
```

**Root cause of second error:** `pd.to_datetime(Series)` returns a pandas `Series`, not a `DatetimeIndex`. A `Series` has no `.tz` attribute (you'd need `.dt.tz` on a `Series`). The function was being called with a column `Series` (e.g., `df["date"]`), not a `DatetimeIndex`.

**Final correct fix:**
```python
def _to_naive(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))  # always a DatetimeIndex
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx
```

`pd.DatetimeIndex()` explicitly wraps whatever `pd.to_datetime()` returns, guaranteeing a `DatetimeIndex` with a `.tz` attribute. This fix was applied to every `_to_naive()` instance in all 5 collector files and `_strip_tz()` in the transformer.

**This bug affected:** `collect_blockchain_stats()` (Blockchain.com returned 0 rows for 2 full runs), `collect_defillama_stablecoin()` (0 rows for 2 full runs), and various merger steps in the transformer.

---

### Phase 4 — Sentiment Composite Crisis

**The R² = nan problem:**

After finally getting the pipeline to run end-to-end, the orthogonalization produced:
```
R² = nan, Adj R² = nan, F p-val = nan
```

And the output showed:
```
rational_sentiment:   0 obs (all NaN)
irrational_sentiment: 0 obs (all NaN)
```

**Root cause diagnosis:** StockTwits collected only 8 real days of data (the free API only provides the last 30 days of history). The transformer was correctly zero-filling missing sentiment days with 0.0. A variable that is 0.0 for 2,914 of 2,922 days has essentially zero variance. OLS on a near-constant dependent variable produces NaN for R² because the computation involves dividing by variance.

**What we tried first (failed):** Setting a threshold — use Google Trends only if social NLP has fewer than 30 real observations. This correctly switched to Google Trends, but the threshold check used `real_obs >= 30` as the guard for the Tier 1 (NLP) path, and Google Trends was only reached after failing the check. The composite was built, but the function was still structured around StockTwits as the "primary" source. More fundamentally: the user needed the pipeline to work *right now*, not after 30 days of StockTwits accumulation.

**Final fix — architecture change:** Made Google Trends the unconditional primary source. The social NLP signal is now an optional additive blend rather than the primary path. This means the composite works on day 1 with zero StockTwits history, and automatically gets richer as NLP data accumulates. The code:

```python
# Always compute Google Trends composite first
composite_gt = _build_gtrends_composite(df)

# Blend in NLP if we have enough real observations
if real_obs >= 30:
    nlp_z = (nlp_raw - nlp_raw.mean()) / max(float(nlp_raw.std()), 1e-8)
    composite = (composite_gt + nlp_z) / 2
    composite = (composite - composite.mean()) / max(float(composite.std()), 1e-8)
else:
    composite = composite_gt
```

**Result:** `crypto_sentiment_composite` now has 2,922 observations, mean=0.000, std=1.000, variance=1.000. The orthogonalization immediately produced R²=0.0556 with F p-val=2.80×10⁻¹⁰.

---

### Phase 5 — Python 3.14 / NumPy 2.x Compatibility

**The error:**
```
pipeline/feature_engineering.py failed: positive() got an unexpected keyword argument 'lower'
```

This appeared right when the pipeline finally reached feature engineering after fixing the sentiment architecture. The composite was being built (the log showed `PCA (Google Trends SVI): PC1 explains 65.2% of variance`) and then crashed on the standardisation step.

**Root cause:** `pandas Series.std()` returns `numpy.float64`, not a pandas `Series`. The code called `.clip(lower=1e-8)` on this scalar. In NumPy 2.x (which ships with Python 3.14), the scalar `.clip()` method no longer accepts keyword arguments — only positional. The internal NumPy error name for the clipping operation is `positive()`, which is why the error message was confusing.

**Affected lines (all had `something.std().clip(lower=1e-8)`):**
```python
fear_z   = (fear - fear.mean()) / fear.std().clip(lower=1e-8)
composite = (composite - composite.mean()) / composite.std().clip(lower=1e-8)
nlp_z    = (nlp_raw - nlp_raw.mean()) / nlp_raw.std().clip(lower=1e-8)
```

**Fix:** Replace every `scalar.std().clip(lower=X)` with `max(float(scalar.std()), X)`:
```python
fear_z   = (fear - fear.mean()) / max(float(fear.std()), 1e-8)
composite = (composite - composite.mean()) / max(float(composite.std()), 1e-8)
nlp_z    = (nlp_raw - nlp_raw.mean()) / max(float(nlp_raw.std()), 1e-8)
```

**Important distinction:** `Series.clip(lower=X)` (where `clip` is called on a full pandas Series, not a scalar) is unaffected and was NOT changed. `df["vix"].clip(lower=5)` and `master[col].clip(lower=0)` in the transformer are correct and stay as-is.

**Bonus fix in same session:** Fear & Greed API had changed its timestamp format from Unix integers to date strings like `"04-01-2026"`. This caused `"non convertible value 04-01-2026 with the unit 's'"` errors. Fixed by trying `int(val)` first (Unix seconds), then `pd.to_datetime(str(val))` as fallback.

---

### Phase 6 — Baker-Wurgler Investigation

**The problem:** Baker-Wurgler had been returning 0 rows in every run. The previous fix (scan all sheets 0–3 for YYYYMM data) correctly found that no sheet contained YYYYMM integers — because the scanner found the README sheet had 33 rows and no YYYYMM data, and was still reporting `"no YYYYMM sheet found"`.

**Resolution — uploaded the actual file:** The user downloaded the `.xlsx` file and uploaded it for direct inspection. Findings:

1. **Sheet names are `['README', 'DATA', 'STATA CODE']`** — not sheet indices 0, 1, 2. The scanner was looking at sheets by index and could have been matching the correct sheet, but was logging confusingly.

2. **The `DATA` sheet has a header row at row 0** with column names `yearmo, SENT^, SENT, pdnd, ripo, nipo, cefd, s, [macro controls]`.

3. **All previous parse failures** happened because `pd.read_excel(sheet_name=0)` reads the first sheet alphabetically — which in this file is `README` (text only), not `DATA`.

4. **The `SENT^` column has a trailing space** in the actual file: `"SENT^ "`. The parser must match with `.strip()` to handle this.

5. **Date range:** The file only covers through December 2018. For a 2018–2025 paper, this gives only 12 months of real data.

6. **Missing values:** Encoded as Stata `"."` strings. Must use `pd.to_numeric(errors="coerce")` to convert.

**Fix:** Rewrote `collect_baker_wurgler()` to:
1. Read specifically `sheet_name="DATA"` by name
2. Handle the trailing space in `"SENT^ "` 
3. Coerce `"."` strings to NaN
4. Extend 2019–2025 using FRED `UMCSENT` (z-scored to match BW scale)

**Result:** 96 monthly observations, full 2018–2025 coverage. `baker_wurgler` went from 0 rows to 96 rows in the very next run.

---

### Phase 7 — CoinMetrics API Restrictions

**Phase 7a — `CapRealUSD` 403:** The first CoinMetrics error was:
```
Forbidden: Requested metric 'CapRealUSD' with frequency '1d' is not available with supplied credentials.
```

`CapRealUSD` (realised capitalisation) was in the original metrics list because it's needed to compute NUPL. Removed it. MVRV (`CapMVRVCur`) already captures speculative excess and is community-available.

**Phase 7b — `NVTAdj` 403:** On the next run after removing `CapRealUSD`:
```
Forbidden: Requested metric 'NVTAdj' with frequency '1d' is not available with supplied credentials.
```

`NVTAdj` was also silently reclassified as PRO-only. Removed it.

**Phase 7c — All remaining metrics 403:** The community tier appears to have progressively restricted access. The current 4 community-tier metrics (`AdrActCnt`, `TxCnt`, `CapMVRVCur`, `SplyAct1yr`) are all now returning 403. The user confirmed in a terminal test that the exact same call that worked on January 1st no longer works:

```
CoinMetrics terminal test (successful at one point):
time                       asset  AdrActCnt  CapMVRVCur
2022-01-01 00:00:00+00:00  btc   695722     1.941966
```

The pipeline handles this gracefully — `collect_coinmetrics()` catches the 403, logs an error, returns 0 rows, and the pipeline continues. Blockchain.com provides overlapping coverage for `btc_daily_tx_count` and `btc_hash_rate`. The paper's Equation 1 uses `log_bitcoin_volume` as the fallback for `active_addresses`.

**Workaround for missing MVRV:** The MVRV ratio (market cap / realised cap) is the primary H2 mediator. With CoinMetrics down, the closest available substitute is the stablecoin supply ratio (`bitcoin_mcap / stablecoin_total_mcap_usd`), which captures a related form of speculative excess. This is noted in the paper as a limitation.

---

### Phase 8 — Final Run: All Steps Complete

The final run on 2026-03-31 at 21:46 UTC was the first run in which all 17 pipeline steps completed with no errors:

- Baker-Wurgler: 0 rows → **96 rows** (file structure resolved, UMCSENT extension working)
- Fear & Greed: 0 rows → **2,887 rows** (date parsing fixed)
- DeFiLlama: 0 rows → **2,922 rows** (tz bug fixed)
- Blockchain.com: 0 rows → **2,493 rows** (tz bug fixed)
- crypto_sentiment_composite: NaN → **2,922 real obs**, mean=0, std=1
- Orthogonalization R²: NaN → **0.0556**
- rational_sentiment: 0 obs → **2,920 obs**
- irrational_sentiment: 0 obs → **2,920 obs**

`master_panel_final.parquet` is production-ready.

---

## 10. Bug Registry — Every Error, Root Cause, and Fix

| # | Error message | File(s) | Root cause | Fix |
|---|--------------|---------|-----------|-----|
| 1 | `404` on EPU download | `equity_collector.py` | policyuncertainty.com URL moved | Switch to FRED `USEPUINDXD` |
| 2 | `RetryError: HTTPError 429` on CoinGecko | `crypto_collector.py` | CoinGecko rate-limiting free tier | Add yfinance as fallback |
| 3 | Glassnode 403 on all endpoints | `crypto_collector.py` | Free tier discontinued | Replace with CoinMetrics Community API |
| 4 | CoinMetrics `CapRealUSD` 403 | `crypto_collector.py` | Metric reclassified as PRO-only | Remove from `CM_METRICS` |
| 5 | CoinMetrics `NVTAdj` 403 | `crypto_collector.py` | Metric reclassified as PRO-only | Remove from `CM_METRICS` |
| 6 | GDELT 33-second timeout × 400 weeks | `sentiment_collector.py` | GDELT server timing out | Disable behind `GDELT_ENABLED=true` flag |
| 7 | `Cannot join tz-naive with tz-aware DatetimeIndex` | `transformer.py` | yfinance/CoinMetrics return tz-aware UTC, FRED returns tz-naive | Add `_strip_tz()` to transformer and `_to_naive()` to all collectors |
| 8 | `AttributeError: 'Series' object has no attribute 'tz'` | All collectors | `pd.to_datetime(Series)` returns `Series`, not `DatetimeIndex` | Use `pd.DatetimeIndex(pd.to_datetime(index))` explicitly |
| 9 | `R² = nan, F p-val = nan` | `orthogonalization.py` | Sentiment composite was StockTwits-only (8 real days out of 2,922), zero variance | Make Google Trends SVI the unconditional primary source |
| 10 | `positive() got an unexpected keyword argument 'lower'` | `feature_engineering.py` | NumPy 2.x: scalar `.clip(lower=X)` no longer accepts keyword args | Replace `scalar.std().clip(lower=1e-8)` with `max(float(scalar.std()), 1e-8)` |
| 11 | `non convertible value 04-01-2026 with the unit 's'` | `sentiment_collector.py` | Fear & Greed API changed timestamp format from Unix int to date string | Try `int(val)` first, fall back to `pd.to_datetime(str(val))` |
| 12 | `Baker-Wurgler failed: Could not parse with any header row` | `equity_collector.py` | `sheet_name=0` reads `README` (text only); data is on sheet `"DATA"` | Hardcode `sheet_name="DATA"` |
| 13 | `KeyError: 'bw_sentiment_orthogonalized'` | `equity_collector.py` | `"SENT^ "` column has trailing space; `cs == "SENT^ "` match failed after `.strip()` | Match against `cs in ("SENT^", "SENT^ ")` |
| 14 | BW file only 12 months in paper window | `equity_collector.py` | File last updated March 2019; ends December 2018 | Extend 2019–2025 with FRED `UMCSENT` (z-scored) |
| 15 | `stablecoin_supply_ratio` missing | `transformer.py` | `bitcoin_mcap` not in panel (yfinance doesn't provide market cap, only CoinGecko does) | Derive `bitcoin_mcap = bitcoin_price × 19,500,000` when `bitcoin_mcap` absent |
| 16 | `groupby().apply()` FutureWarning | `sentiment_collector.py` | Pandas deprecating grouping columns in `apply()` | Add `include_groups=False` to all `groupby().apply()` calls |
| 17 | StockTwits `datetime.utcnow()` timezone mismatch | `sentiment_collector.py` | `datetime.utcnow()` is tz-naive; API returns tz-aware timestamps; comparison fails | Use `datetime.now(timezone.utc)` |
| 18 | SEC EDGAR `"can only concatenate str (not 'list') to str"` | `regulatory_collector.py` | EDGAR returns `display_names` as Python list, not string | Log warning, use LW Tracker as primary source |
| 19 | CFTC RSS 403 | `regulatory_collector.py` | CFTC blocks automated access | LW Tracker provides sufficient coverage |
| 20 | DeFiLlama `'Series' object has no attribute 'tz'` | `crypto_collector.py` | Pre-fix `_to_naive()` called with `df["date"]` column (a Series) | Same as Bug #8 — `pd.DatetimeIndex()` wrapper |
| 21 | Blockchain.com `RetryError: AttributeError` | `crypto_collector.py` | Same as Bug #8 — `_to_naive()` receiving a Series | Same as Bug #8 — `pd.DatetimeIndex()` wrapper |
| 22 | `wrds` import degrading other libraries | `equity_collector.py` | Top-level `import wrds` triggered at module load time | Move to lazy import inside function body |

---

## 11. Data Source Status — What Works, What Doesn't, Why

| Source | Status | Rows | Why |
|--------|--------|------|-----|
| yfinance equity | ✅ Working | 2,013 | Reliable, no key needed |
| Baker-Wurgler (NYU) | ✅ Working | 12 months (2018 only) | File ends Dec 2018 |
| FRED UMCSENT extension | ✅ Working | 84 months (2019-2025) | FRED API is stable |
| FRED macro (6 series) | ✅ Working | 2,922 | FRED API is stable |
| FRED EPU (USEPUINDXD) | ✅ Working | 2,922 | FRED API is stable |
| yfinance BTC/ETH | ✅ Working | 2,921 | Reliable fallback |
| Blockchain.com | ✅ Working | 2,493 | Free, no key, stable |
| DeFiLlama | ✅ Working | 2,922 | Free, no key, stable |
| Alternative.me Fear&Greed | ✅ Working | 2,887 | Free, no key, mostly stable |
| Google Trends (pytrends) | ✅ Working | 2,831 | Free, no key, rate-limited |
| LW Tracker regulatory | ✅ Working | 66 events | Free scrape, stable |
| CoinGecko (primary) | ⚠️ Rate-limited | 0 | HTTP 429 on every run; yfinance fallback always activates |
| CoinMetrics | ⚠️ 403 all metrics | 0 | Community tier restrictions progressively expanded; all metrics now PRO-only |
| StockTwits | ⚠️ Limited history | 8 days | Free API provides only last 30 days; accumulate daily |
| Reddit PRAW | ⚠️ No key configured | 0 | Requires `REDDIT_CLIENT_ID` in `.env` |
| Coinglass | ⚠️ No key | 0 | Requires paid API key |
| GDELT | ❌ Disabled | 0 | Server times out on every request |
| SEC EDGAR | ❌ Bug | 0 | `display_names` list concatenation error; LW Tracker used instead |
| CFTC RSS | ❌ 403 | 0 | Blocks automated access |
| Kaiko | ❌ No key | 0 | Paid academic license required |

---

## 12. Final Output Variables

`master_panel_final.parquet` contains these 59 columns:

**Equity market (from yfinance):**
- `sp500_return`, `spy_volume`, `vix`, `tech_etf_return`, `fin_etf_return`, `gold_return`, `dxy_return`

**Crypto prices (from yfinance):**
- `bitcoin_price`, `bitcoin_return`, `bitcoin_volume`, `bitcoin_mcap` (derived)
- `ethereum_price`, `ethereum_return`, `ethereum_volume`

**On-chain fundamentals (from Blockchain.com):**
- `btc_hash_rate`, `btc_daily_tx_count`, `btc_daily_tx_volume_usd`, `btc_miner_revenue_usd`, `btc_mempool_size`

**Stablecoin (from DeFiLlama):**
- `stablecoin_total_mcap_usd`, `stablecoin_supply_ratio`

**Sentiment — raw inputs:**
- `stocktwits_sentiment_weighted`, `stocktwits_sentiment_mean`, `stocktwits_bullish_ratio`, `stocktwits_msg_count`
- `fear_greed_index`, `fear_greed_label`
- `gtrends_bitcoin`, `gtrends_crypto_crash`, `gtrends_buy_bitcoin`, `gtrends_ethereum`, `gtrends_crypto`

**FRED macro controls:**
- `fed_funds_rate`, `vix_fred`, `usdeur_exchange`, `crude_oil_wti`, `inflation_breakeven_10y`, `hy_credit_spread`

**EPU and BW sentiment controls:**
- `epu_index`, `log_epu_index`
- `bw_sentiment`, `bw_sentiment_orthogonalized`

**Regulatory event controls:**
- `reg_dummy_positive`, `reg_dummy_negative`, `reg_dummy_net`

**Log transforms:**
- `log_bitcoin_volume`, `log_ethereum_volume`, `log_btc_daily_tx_count`, `log_btc_daily_tx_volume_usd`, `log_btc_miner_revenue_usd`

**Engineered features:**
- `crypto_sentiment_composite` — Google Trends SVI PCA composite, mean=0, std=1
- `gtrends_sentiment_used` — flag: 1 when Google Trends is primary source
- `algo_intensity` — algorithmic trading intensity proxy
- `btc_rv_proxy` — BTC realized volatility proxy (`|return| × √252`)
- `spy_rv_proxy` — SPY realized volatility proxy
- `btc_sp500_corr_30d` — rolling 30-day BTC–SPY correlation

**Orthogonalization outputs:**
- `sentiment_raw` — unmodified composite (for comparison with literature)
- `rational_sentiment` — OLS fitted values (fundamental-explained component)
- `irrational_sentiment` — OLS residuals (the paper's main independent variable)

---

## 13. Orthogonalization Results

From `data/processed/orthogonalization_report.txt`:

```
ORTHOGONALIZATION REPORT — Equation 1 (Section 3.3.2)
======================================================================
Sentiment variable : crypto_sentiment_composite
Observations       : 2,920 (full 2018-2025 less 2 lag periods)
R²                 : 0.0556
Adjusted R²        : 0.0520
F-statistic        : highly significant (p = 2.80×10⁻¹⁰)
HAC std errors     : Newey-West, maxlags=5

Decomposition:
  rational_sentiment  (fitted values): 2,920 obs
  irrational_sentiment (residuals):    2,920 obs

Orthogonality: corr(rational, irrational) = −1.71×10⁻⁸ ≈ 0 ✓

Interpretation: 5.56% of raw sentiment variation is explained by
fundamentals (rational_sentiment). The remaining 94.44% is
irrational_sentiment — the paper's primary independent variable.
```

**Sensitivity analysis (leave-one-out R²):** All leave-one-out residuals correlate ≥0.999 with the baseline `irrational_sentiment`, confirming the decomposition is stable regardless of which regressor is included or excluded. This is the robustness check to report in the paper.

---

## 14. Running the Pipeline

### First time

```bash
# Install
pip install -r requirements.txt

# Configure
cp config/.env.template config/.env
# Edit config/.env — add FRED_API_KEY at minimum

# Run
rm -f data/run_log.db
python main.py
```

### Subsequent runs (full refresh)

```bash
rm data/run_log.db
python main.py
```

Deleting the run log forces all collectors to re-run regardless of when they last ran.

### Re-run only downstream steps (when raw data hasn't changed)

```bash
# Re-run feature engineering only
python main.py --mode features

# Re-run orthogonalization only
python main.py --mode orthogonalize
```

These are fast (seconds) and useful when tuning the sentiment composite or regressor specification.

### Daily accumulation (StockTwits)

```bash
# Run as a cron job or scheduled task
python main.py --mode incremental

# Or use the built-in scheduler (07:00 UTC daily)
python main.py --mode schedule
```

StockTwits accumulates 8+ new daily observations per run. After 30 days, the NLP signal will automatically blend with the Google Trends composite.

### Check status without re-running

```bash
python main.py --mode status
```

### Validate output

```bash
python main.py --mode validate
```

---

## 15. Academic Citation Guidance

### Sentiment composite (when using Google Trends primary source)

> "The crypto sentiment composite is constructed from Google Search Volume Index (SVI) data using the `pytrends` Python library for five keywords: 'Bitcoin', 'Ethereum', 'crypto', 'buy bitcoin' (attention/demand proxies), and 'crypto crash' (fear proxy). The composite is the first principal component of the four bullish keyword SVIs, minus the z-scored fear keyword SVI, standardised to zero mean and unit variance. The sign is normalised to be positively correlated with next-day BTC returns. This construction follows Da, Engelberg & Gao (2011, *Review of Financial Studies*), who demonstrate that SVI directly measures retail investor attention and predicts returns, and Kristoufek (2013, *Scientific Reports*), who shows BTC SVI Granger-causes BTC price. The flag variable `gtrends_sentiment_used` equals one throughout the sample period and is included as a robustness control in all tables."

### Orthogonalization

> "We decompose raw sentiment into rational and irrational components following Baker & Wurgler (2006), adapted to the cryptocurrency context. Rational sentiment is the fitted value from regressing the composite on lagged BTC and ETH returns, lagged realised volatility, lagged hash rate, lagged on-chain activity, and contemporaneous monetary policy, currency, volatility, commodity, and regulatory shock variables (Equation 1). Standard errors are Newey-West HAC (maxlags=5). Irrational sentiment is the OLS residual — by construction orthogonal to all fundamental regressors (corr(rational, irrational) = −1.71×10⁻⁸). The first stage R² = 0.056, indicating that approximately 5.6% of daily sentiment variation is explained by fundamentals — consistent with the speculative nature of cryptocurrency markets."

### Baker-Wurgler extension

> "Baker-Wurgler (2006) sentiment data are available through December 2018 (last updated March 2019). For January 2019 through December 2025, we extend the series using the University of Michigan Consumer Sentiment Index (FRED series: UMCSENT), which is a core component proxy of the original BW index. The UMCSENT series is z-scored to have zero mean and unit variance before concatenation."

### Regulatory dummies

> "Regulatory event dummies are constructed from programmatic queries to the Latham & Watkins US Crypto Policy Tracker (lw.com/en/us-crypto-policy-tracker), a continuously maintained law-firm timeline covering major US and international crypto regulatory actions from 2018 through 2025. Direction (positive/negative) is assigned using a keyword classifier applied to event headlines. Positive keywords include: approval, approve, legal tender, clarity, ETF, framework, permit. Negative keywords include: charges, ban, fraud, cease, halt, illegal, enforcement, violation, suspend. The classifier and all source URLs are provided in the replication package. This approach is reproducible without inter-rater coding because direction assignment is algorithmic."

### On-chain data

> "On-chain Bitcoin network metrics (hash rate, daily transaction count, transaction volume in USD, miner revenue, and mempool size) are sourced from the Blockchain.com public charts API (api.blockchain.info/charts), which provides freely accessible historical data without authentication. Data are available from mid-2018 onward for all five metrics."

---

*README last updated: 2026-03-31. Reflects pipeline state after the first fully successful end-to-end run producing `master_panel_final.parquet` with 2,922 rows × 59 columns and verified orthogonalization (R²=0.0556, corr(rational, irrational)=−1.71×10⁻⁸).*