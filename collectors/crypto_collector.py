"""
collectors/crypto_collector.py

Fetches all cryptocurrency market variables required by Section 3.3:

  CoinGecko (free, no key) — prices, market cap, volume; yfinance fallback
  CoinMetrics Community (free, uses official Python client) — on-chain metrics
  Blockchain.com (free, no key) — hash rate, tx count, mempool, miner revenue
  DeFiLlama (free, no key) — total stablecoin market cap
  Coinglass (free) — perpetual futures funding rate, long/short ratio
  [STUB] Kaiko — activated when KAIKO_API_KEY is set in .env

All date indexes are saved as tz-naive UTC to avoid
"Cannot join tz-naive with tz-aware DatetimeIndex" in transformer.py.
"""
import sys
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR, RAW_DIR, START_DATE, END_DATE,
    COINGECKO_API_KEY, COINGLASS_API_KEY, KAIKO_API_KEY,
)

CG_BASE = "https://api.coingecko.com/api/v3"
CG_HDR  = {"x-cg-pro-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}


def _to_naive(index) -> pd.DatetimeIndex:
    """
    Strip timezone from any date-like input → always returns tz-naive DatetimeIndex.
    Uses pd.DatetimeIndex() explicitly (not pd.to_datetime()) because pd.to_datetime()
    on a pandas Series returns a Series (no .tz attr), causing AttributeError.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


# ══════════════════════════════════════════════════════════
# CoinGecko — prices, market cap, volume
# ══════════════════════════════════════════════════════════

@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=4, max=60))
def _cg_get(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, headers=CG_HDR, timeout=30)
    if resp.status_code == 429:
        raise Exception("CoinGecko rate limit")
    resp.raise_for_status()
    return resp.json()


def _fetch_coin_cg(coin_id: str, start: str, end: str) -> pd.DataFrame:
    logger.info(f"  CoinGecko: {coin_id}")
    s = int(pd.Timestamp(start).timestamp())
    e = int(pd.Timestamp(end).timestamp())
    data = _cg_get(
        f"{CG_BASE}/coins/{coin_id}/market_chart/range",
        {"vs_currency": "usd", "from": s, "to": e},
    )
    def _frame(key, col):
        d = pd.DataFrame(data[key], columns=["ts", col])
        d["date"] = pd.to_datetime(d["ts"], unit="ms").dt.normalize()
        d["date"] = _to_naive(d["date"])
        return d.set_index("date")[[col]]
    df = (_frame("prices",        f"{coin_id}_price")
          .join(_frame("market_caps",   f"{coin_id}_mcap"))
          .join(_frame("total_volumes", f"{coin_id}_volume")))
    df[f"{coin_id}_return"] = np.log(
        df[f"{coin_id}_price"] / df[f"{coin_id}_price"].shift(1))
    time.sleep(7)
    return df


def collect_crypto_prices(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    frames = []
    for coin in ["bitcoin", "ethereum"]:
        try:
            frames.append(_fetch_coin_cg(coin, start, end))
        except Exception as e:
            logger.warning(f"  CoinGecko {coin} failed: {e} — yfinance fallback")
            ticker = "BTC-USD" if coin == "bitcoin" else "ETH-USD"
            raw    = yf.download(ticker, start=start, end=end,
                                  progress=False, auto_adjust=True)
            close  = raw["Close"].squeeze()
            idx    = _to_naive(raw.index)
            fb = pd.DataFrame({
                f"{coin}_price":  close.values,
                f"{coin}_return": np.log(close / close.shift(1)).values,
                f"{coin}_volume": raw["Volume"].squeeze().values,
            }, index=idx)
            frames.append(fb)
    df = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    if not df.empty:
        df.index = _to_naive(df.index)
        df.index.name = "date"
        out = RAW_DIR / "crypto" / "crypto_prices.parquet"
        df.to_parquet(out)
        logger.success(f"Crypto prices → {out}  {df.shape}")
    return df


# ══════════════════════════════════════════════════════════
# CoinMetrics Community — on-chain metrics
# ══════════════════════════════════════════════════════════

CM_METRICS = {
    # Community-tier metrics verified working (no API key required)
    "AdrActCnt":  "active_addresses",     # Eq.1 regressor β5 — unique active addresses
    "TxCnt":      "btc_tx_count_cm",      # Eq.1 regressor β6 — daily confirmed txns
    "CapMVRVCur": "mvrv_ratio",           # H2 mediator — market cap / realised cap
    "SplyAct1yr": "supply_active_1yr",    # investor behaviour proxy — coins moved in 1yr
    # REMOVED (PRO-only, returns 403):
    #   CapRealUSD — realised capitalisation
    #   NVTAdj     — adjusted NVT ratio
    # MVRV (CapMVRVCur) serves as the primary H2 speculative excess indicator.
}


def collect_coinmetrics(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Collect on-chain metrics via coinmetrics-api-client (community, no key needed).
    Strips UTC timezone from the index before saving.
    """
    logger.info("  CoinMetrics Community API: BTC on-chain metrics...")
    try:
        from coinmetrics.api_client import CoinMetricsClient
        client = CoinMetricsClient()
        df_raw = client.get_asset_metrics(
            assets=["btc"],
            metrics=list(CM_METRICS.keys()),
            frequency="1d",
            start_time=pd.Timestamp(start).strftime("%Y-%m-%d"),
            end_time=pd.Timestamp(end).strftime("%Y-%m-%d"),
        ).to_dataframe()

        if df_raw.empty:
            logger.warning("  CoinMetrics returned empty dataframe")
            return pd.DataFrame()

        # CoinMetrics "time" column is UTC ISO8601 — strip tz before using as index
        df_raw["date"] = _to_naive(pd.to_datetime(df_raw["time"]).dt.normalize())
        df_raw = df_raw.set_index("date")
        df_raw = df_raw.drop(columns=["asset", "time"], errors="ignore")

        for col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

        df = df_raw.rename(columns={k: v for k, v in CM_METRICS.items()
                                     if k in df_raw.columns})
        df = df.sort_index().loc[start:end]

        # Save to both paths (glassnode.parquet = transformer alias)
        for path in [RAW_DIR / "onchain" / "coinmetrics.parquet",
                     RAW_DIR / "onchain" / "glassnode.parquet"]:
            df.to_parquet(path)
        logger.success(f"CoinMetrics → {len(df)} rows, cols: {list(df.columns)}")
        return df

    except ImportError:
        logger.error(
            "  coinmetrics-api-client not installed. Run: pip install coinmetrics-api-client"
        )
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"  CoinMetrics failed: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════
# Blockchain.com — hash rate, tx count, mempool, etc.
# ══════════════════════════════════════════════════════════

BLOCKCHAIN_CHARTS = {
    "hash-rate":                        "btc_hash_rate",
    "n-transactions":                   "btc_daily_tx_count",
    "estimated-transaction-volume-usd": "btc_daily_tx_volume_usd",
    "miners-revenue":                   "btc_miner_revenue_usd",
    "mempool-size":                     "btc_mempool_size",
}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=30))
def _blockchain(chart: str, start: str, end: str) -> pd.Series:
    resp = requests.get(
        f"https://api.blockchain.info/charts/{chart}",
        params={"timespan": "all", "start": int(pd.Timestamp(start).timestamp()),
                "format": "json", "sampled": "true", "cors": "true"},
        headers={"User-Agent": "crypto_research/1.0"}, timeout=30,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["values"], columns=["x", "y"])
    df["date"] = _to_naive(pd.to_datetime(df["x"], unit="s").dt.normalize())
    s = df.set_index("date")["y"]
    return s[~s.index.duplicated(keep="last")].sort_index().loc[start:end]


def collect_blockchain_stats(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    frames = {}
    for chart, col in BLOCKCHAIN_CHARTS.items():
        try:
            frames[col] = _blockchain(chart, start, end)
            logger.info(f"  Blockchain.com ✓ {col}: {len(frames[col])} obs")
            time.sleep(3)
        except Exception as e:
            logger.warning(f"  Blockchain.com ✗ {chart}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).sort_index()
    df.index = _to_naive(df.index)
    df.index.name = "date"
    path = RAW_DIR / "onchain" / "blockchain_stats.parquet"
    df.to_parquet(path)
    logger.success(f"Blockchain.com → {path}  {df.shape}")
    return df


# ══════════════════════════════════════════════════════════
# DeFiLlama — stablecoin market cap
# ══════════════════════════════════════════════════════════

def collect_defillama_stablecoin(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    logger.info("  DeFiLlama: stablecoin market cap...")
    try:
        resp = requests.get("https://stablecoins.llama.fi/stablecoincharts/all",
                            timeout=30, headers={"User-Agent": "crypto_research/1.0"})
        resp.raise_for_status()
        records = []
        for entry in resp.json():
            raw_date = entry.get("date", entry.get("timestamp"))
            if raw_date is None:
                continue
            try:
                ts = pd.Timestamp(int(float(str(raw_date))), unit="s")
            except (ValueError, TypeError, OSError):
                try:
                    ts = pd.Timestamp(str(raw_date))
                except Exception:
                    continue
            circ = entry.get("totalCirculatingUSD", {})
            records.append({
                "date": ts.normalize(),
                "stablecoin_total_mcap_usd": sum(
                    v for v in circ.values() if isinstance(v, (int, float))),
            })
        df = pd.DataFrame(records)
        df["date"] = _to_naive(pd.to_datetime(df["date"]))
        df = df.set_index("date").sort_index().loc[start:end]
        path = RAW_DIR / "onchain" / "defillama_stablecoin.parquet"
        df.to_parquet(path)
        logger.success(f"DeFiLlama stablecoin → {path}  ({len(df)} days)")
        return df
    except Exception as e:
        logger.error(f"  DeFiLlama failed: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════
# Coinglass — perpetual futures
# ══════════════════════════════════════════════════════════

def collect_coinglass(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    hdr     = {"coinglassSecret": COINGLASS_API_KEY} if COINGLASS_API_KEY else {}
    results = {}
    try:
        resp = requests.get("https://open-api.coinglass.com/public/v2/funding",
                            params={"symbol": "BTC"}, headers=hdr, timeout=20)
        data = resp.json().get("data", [])
        if isinstance(data, list) and data:
            df = pd.DataFrame(data)
            if "t" in df.columns and "h" in df.columns:
                df["date"] = _to_naive(pd.to_datetime(df["t"], unit="ms").dt.normalize())
                results["btc_funding_rate"] = df.set_index("date")["h"]
        time.sleep(3)
    except Exception as e:
        logger.warning(f"  Coinglass funding rate: {e}")
    try:
        resp = requests.get(
            "https://open-api.coinglass.com/public/v2/globalLongShortAccountRatio",
            params={"symbol": "BTC", "interval": "1d", "limit": 3000},
            headers=hdr, timeout=20)
        data = resp.json().get("data", [])
        if data:
            df = pd.DataFrame(data)
            df["date"] = _to_naive(pd.to_datetime(df["time"], unit="ms").dt.normalize())
            df = df.set_index("date")
            results["btc_long_short_ratio"] = (
                df["longAccount"].astype(float) / df["shortAccount"].astype(float))
        time.sleep(3)
    except Exception as e:
        logger.warning(f"  Coinglass long/short: {e}")
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).sort_index().loc[start:end]
    df.index = _to_naive(df.index)
    path = RAW_DIR / "futures" / "coinglass_futures.parquet"
    df.to_parquet(path)
    logger.success(f"Coinglass futures → {path}  {df.shape}")
    return df


# ══════════════════════════════════════════════════════════
# Kaiko stub
# ══════════════════════════════════════════════════════════

def collect_kaiko(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    if not KAIKO_API_KEY:
        logger.info("  KAIKO_API_KEY not set — skipping Kaiko")
        return pd.DataFrame()
    logger.info("  Kaiko: implement REST calls per docs.kaiko.com")
    return pd.DataFrame()


if __name__ == "__main__":
    collect_crypto_prices()
    collect_coinmetrics()
    collect_blockchain_stats()
    collect_defillama_stablecoin()
    collect_coinglass()