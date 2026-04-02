"""
pipeline/transformer.py

Merges all raw Parquet shards into a single master_panel.parquet.

Shard sources (all from data/raw/):
  equity/equity_daily.parquet          yfinance prices/returns
  equity/baker_wurgler.parquet         monthly BW sentiment → ffill daily
  crypto/crypto_prices.parquet         CoinGecko BTC/ETH prices/returns
  onchain/glassnode.parquet            NUPL, MVRV, exchange flows (CoinMetrics alias)
  onchain/blockchain_stats.parquet     hash rate, tx count, mempool
  onchain/defillama_stablecoin.parquet stablecoin market cap
  futures/coinglass_futures.parquet    funding rate, long/short ratio
  sentiment/stocktwits_sentiment.parquet
  sentiment/reddit_sentiment.parquet
  sentiment/fear_greed.parquet
  sentiment/gdelt_news.parquet
  sentiment/google_trends.parquet      weekly → ffill daily
  macro/fred_macro.parquet             Fed funds, credit spreads, etc.
  macro/epu_index.parquet              Economic Policy Uncertainty (FRED)
  macro/regulatory_dummies.parquet     reg_dummy_positive/negative/net

Timezone handling:
  All shard indexes are stripped of timezone info after loading.
  The master panel index is always tz-naive UTC dates.
  This handles shards from yfinance (tz-aware in newer versions),
  CoinMetrics (UTC ISO8601), and FRED (tz-naive) consistently.

Missing value policy:
  return/price/vix columns → ffill max 3 days (weekends/holidays)
  sentiment columns        → fillna(0)  (neutral on missing days)
  on-chain/macro columns   → ffill max 7 days (data delay)

Input:  data/raw/**/*.parquet
Output: data/processed/master_panel.parquet
        data/processed/coverage_report.csv
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, RAW_DIR, START_DATE, END_DATE


def _strip_tz(index) -> pd.DatetimeIndex:
    """
    Strip timezone from any date-like input → always tz-naive DatetimeIndex.
    Handles pd.Series, pd.DatetimeIndex, and list inputs.
    Uses pd.DatetimeIndex() explicitly because pd.to_datetime(Series) returns
    a Series (no .tz attribute), which would cause AttributeError.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


def _load(path: Path, label: str) -> pd.DataFrame:
    """Load one Parquet shard; return empty DataFrame on missing/error.
    Always strips timezone info from the index so all shards are tz-naive."""
    if not path.exists():
        logger.info(f"  not found (skipping): {label}")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df.index = _strip_tz(df.index)
        logger.info(f"  ✓ {label}: {df.shape}")
        return df
    except Exception as e:
        logger.warning(f"  ✗ {label}: {e}")
        return pd.DataFrame()


def _merge_numeric(master: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Join numeric columns from df into master on the date index.
    Strips timezone from df index before joining."""
    if df.empty:
        return master
    df.index = _strip_tz(df.index)
    num = df.select_dtypes(include=[np.number])
    return master.join(num, how="left")


def build_master_panel(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Load every raw shard, merge on date, apply fill policies,
    compute derived columns, and save master_panel.parquet.
    """
    logger.info("=== Transformer: building master panel ===")

    # ── Load all shards ───────────────────────────────────
    equity   = _load(RAW_DIR / "equity"    / "equity_daily.parquet",         "equity")
    crypto   = _load(RAW_DIR / "crypto"    / "crypto_prices.parquet",         "crypto prices")
    gn       = _load(RAW_DIR / "onchain"   / "glassnode.parquet",             "glassnode")
    bc       = _load(RAW_DIR / "onchain"   / "blockchain_stats.parquet",      "blockchain.com")
    stable   = _load(RAW_DIR / "onchain"   / "defillama_stablecoin.parquet",  "defillama")
    futures  = _load(RAW_DIR / "futures"   / "coinglass_futures.parquet",     "coinglass")
    st       = _load(RAW_DIR / "sentiment" / "stocktwits_sentiment.parquet",  "stocktwits")
    reddit   = _load(RAW_DIR / "sentiment" / "reddit_sentiment.parquet",      "reddit")
    fg       = _load(RAW_DIR / "sentiment" / "fear_greed.parquet",            "fear & greed")
    gdelt    = _load(RAW_DIR / "sentiment" / "gdelt_news.parquet",            "gdelt")
    macro    = _load(RAW_DIR / "macro"     / "fred_macro.parquet",            "fred macro")
    epu      = _load(RAW_DIR / "macro"     / "epu_index.parquet",             "epu index")
    reg_dum  = _load(RAW_DIR / "macro"     / "regulatory_dummies.parquet",    "regulatory dummies")
    # Monthly frequency — handled separately below
    bw       = _load(RAW_DIR / "equity"    / "baker_wurgler.parquet",         "baker-wurgler")

    # ── Master date index (always tz-naive) ───────────────
    master = pd.DataFrame(index=pd.date_range(start, end, freq="D", name="date"))

    # ── Merge daily-frequency shards ──────────────────────
    for df, label in [
        (equity,  "equity"),    (crypto,  "crypto"),
        (gn,      "glassnode"), (bc,      "blockchain"),
        (stable,  "defillama"), (futures, "futures"),
        (st,      "stocktwits"),(reddit,  "reddit"),
        (fg,      "fear_greed"),(gdelt,   "gdelt"),
        (macro,   "macro"),     (epu,     "epu"),
        (reg_dum, "reg_dummies"),
    ]:
        master = _merge_numeric(master, df)

    # ── Monthly → daily: Baker-Wurgler ────────────────────
    if not bw.empty:
        bw_daily = bw.resample("D").last().ffill()
        master   = _merge_numeric(master, bw_daily)

    # ── Weekly → daily: Google Trends ─────────────────────
    gt = _load(RAW_DIR / "sentiment" / "google_trends.parquet", "google trends")
    if not gt.empty:
        gt_daily = gt.resample("D").last().ffill()
        master   = _merge_numeric(master, gt_daily)

    logger.info(f"  After merge: {master.shape}")

    # ── Derived columns ───────────────────────────────────

    # NUPL requires CapRealUSD (realised cap) which is CoinMetrics PRO-only.
    # MVRV (mvrv_ratio) is the community-tier equivalent and is already collected.
    # Stablecoin Supply Ratio
    # Ensure bitcoin_mcap exists: CoinGecko provides it directly;
    # yfinance fallback only has bitcoin_price, so we derive mcap using
    # approximate circulating supply (~19.5M BTC, stable over 2018-2025 window).
    if "bitcoin_mcap" not in master.columns and "bitcoin_price" in master.columns:
        APPROX_BTC_SUPPLY = 19_500_000
        master["bitcoin_mcap"] = master["bitcoin_price"] * APPROX_BTC_SUPPLY
        logger.info("  ✓ bitcoin_mcap (derived: price × ~19.5M circulating supply)")
    if "bitcoin_mcap" in master.columns and "stablecoin_total_mcap_usd" in master.columns:
        master["stablecoin_supply_ratio"] = (
            master["bitcoin_mcap"] / master["stablecoin_total_mcap_usd"])
        logger.info("  ✓ stablecoin_supply_ratio")

    # Exchange net flow fallback
    if "exchange_net_flow" not in master.columns:
        if "exchange_inflow_btc" in master.columns and "exchange_outflow_btc" in master.columns:
            master["exchange_net_flow"] = (
                master["exchange_inflow_btc"] - master["exchange_outflow_btc"])
            logger.info("  ✓ exchange_net_flow")

    # Log transforms
    for col in [
        "bitcoin_volume", "ethereum_volume", "bitcoin_mcap", "ethereum_mcap",
        "btc_daily_tx_count", "btc_daily_tx_volume_usd", "epu_index",
        "btc_miner_revenue_usd",
    ]:
        if col in master.columns:
            master[f"log_{col}"] = np.log1p(master[col].clip(lower=0))

    # ── Missing value policy ──────────────────────────────

    return_cols = [c for c in master.columns
                   if any(k in c for k in ("return", "price", "vix"))]
    master[return_cols] = master[return_cols].ffill(limit=3)

    sent_cols = [c for c in master.columns if any(k in c for k in (
        "sentiment", "fear_greed_index", "gdelt", "gtrends",
        "bullish_ratio", "msg_count", "post_count",
    ))]
    master[sent_cols] = master[sent_cols].fillna(0)

    reg_cols = [c for c in master.columns if c.startswith("reg_dummy")]
    master[reg_cols] = master[reg_cols].fillna(0).astype(int)

    other_cols = [c for c in master.columns
                  if c not in return_cols + sent_cols + reg_cols]
    master[other_cols] = master[other_cols].ffill(limit=7)

    # ── Save ──────────────────────────────────────────────
    out = DATA_DIR / "processed" / "master_panel.parquet"
    master.to_parquet(out)

    cov = (master.notna().sum() / len(master) * 100).round(1)
    cov.to_frame("coverage_pct").to_csv(DATA_DIR / "processed" / "coverage_report.csv")

    logger.success(f"master_panel.parquet → {out}  {master.shape}")
    return master


if __name__ == "__main__":
    df = build_master_panel()
    print(df.describe().round(4))