"""
collectors/bgeometrics_collector.py

Fetches Bitcoin on-chain mediator variables from the BGeometrics free API
(bitcoin-data.com/v1/) — no API key required.

Closes Gap 2 variables that were unavailable from Glassnode (discontinued)
and CoinMetrics (community tier now fully restricted):

  Variable              Endpoint                    Paper role
  ─────────────────── ─ ─────────────────────────── ─ ────────────────────────
  nupl                  /v1/nupl                     H2 mediator — aggregate
                                                     holder profit/loss state
  mvrv_ratio_bg         /v1/mvrv-ratio               H2 mediator — speculative
                                                     excess (market/realised)
  btc_exchange_netflow  /v1/exchanges-netflow-total  H2 mediator — sell pressure
                                                     (positive = inflow = bearish)
  btc_exchange_inflow   /v1/exchanges-inflow-total   H2 component
  btc_exchange_outflow  /v1/exchanges-outflow-total  H2 component
  sopr                  /v1/sopr                     Spent Output Profit Ratio
  btc_funding_rate_bg   /v1/funding-rate             Perpetual futures sentiment
  active_addresses_bg   /v1/active-addresses         Network activity proxy

Response format (both observed variants handled):
  Variant A: [{"t": unix_timestamp, "v": float}, ...]
  Variant B: [{"date": "YYYY-MM-DD", "value": float}, ...]
  Variant C: {"data": [...], "dates": [...]}

All outputs saved to data/raw/onchain/bgeometrics_onchain.parquet
and to data/raw/onchain/bgeometrics_{metric}.parquet individually.

Historical depth: covers back to ~2010 for most metrics (full BTC history).
"""
import sys
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, START_DATE, END_DATE

BASE_URL = "https://bitcoin-data.com/v1"

# Endpoint → output column name
BG_METRICS = {
    "nupl":                    "nupl",
    "mvrv-ratio":              "mvrv_ratio_bg",
    "exchanges-netflow-total": "btc_exchange_netflow",
    "exchanges-inflow-total":  "btc_exchange_inflow",
    "exchanges-outflow-total": "btc_exchange_outflow",
    "sopr":                    "sopr",
    "funding-rate":            "btc_funding_rate_bg",
    "active-addresses":        "active_addresses_bg",
}


def _to_naive(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=30))
def _fetch_endpoint(endpoint: str) -> list:
    url = f"{BASE_URL}/{endpoint}"
    resp = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "crypto_research/1.0"},
    )
    resp.raise_for_status()
    return resp.json()


def _parse_response(raw, col_name: str) -> pd.Series:
    """
    Parse any BGeometrics response format into a dated pandas Series.
    Handles all three known response variants.
    """
    if isinstance(raw, dict):
        # Variant C: {"data": [...], "dates": [...]} or nested structure
        if "data" in raw and "dates" in raw:
            dates  = pd.to_datetime(raw["dates"])
            values = [float(v) if v is not None else np.nan for v in raw["data"]]
            s = pd.Series(values, index=_to_naive(dates), name=col_name)
            return s
        # Try to find any list values
        for key, val in raw.items():
            if isinstance(val, list) and val:
                raw = val
                break

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Unexpected response format: {type(raw)}")

    # Variant A: [{"t": unix_ts, "v": float}, ...]
    if "t" in raw[0]:
        dates  = pd.to_datetime([r["t"] for r in raw], unit="s").normalize()
        values = [float(r.get("v", np.nan)) for r in raw]

    # Variant B: [{"date": "YYYY-MM-DD", "value": float}, ...]
    elif "date" in raw[0]:
        dates  = pd.to_datetime([r["date"] for r in raw])
        values = [float(r.get("value", r.get("v", np.nan))) for r in raw]

    # Variant D: [[unix_ts, value], ...]
    elif isinstance(raw[0], list) and len(raw[0]) == 2:
        dates  = pd.to_datetime([r[0] for r in raw], unit="s").normalize()
        values = [float(r[1]) for r in raw]

    # Variant E: BGeometrics format {"d": "YYYY-MM-DD", "unixTs": "...", "<metric>": "0.38"}
    elif "d" in raw[0]:
        dates = pd.to_datetime([r["d"] for r in raw])
        # Auto-detect value key: exclude known non-value keys
        non_val = {"d", "unixTs", "t", "date", "timestamp"}
        sample  = raw[0]
        v_key   = next((k for k in sample if k not in non_val), None)
        if not v_key:
            raise ValueError(f"Cannot find value key in BGeometrics response: {list(sample.keys())}")
        values = [float(r.get(v_key, np.nan)) for r in raw]

    else:
        # Last resort: try every key for timestamp and value
        sample = raw[0]
        ts_key = next((k for k in sample if "time" in k.lower() or k == "t"), None)
        v_key  = next((k for k in sample if "val" in k.lower() or k == "v"), None)
        if not ts_key or not v_key:
            raise ValueError(f"Cannot parse response keys: {list(sample.keys())}")
        try:
            dates = pd.to_datetime(
                [r[ts_key] for r in raw], unit="s").normalize()
        except Exception:
            dates = pd.to_datetime([r[ts_key] for r in raw])
        values = [float(r.get(v_key, np.nan)) for r in raw]

    idx = _to_naive(dates)
    s   = pd.Series(values, index=idx, name=col_name)
    return s[~s.index.duplicated(keep="last")].sort_index()


def collect_bgeometrics(
    start: str = START_DATE,
    end:   str = END_DATE,
) -> pd.DataFrame:
    """
    Fetch all BGeometrics on-chain metrics and return as a single DataFrame.

    On first successful call the full history is returned — typically 2010-present.
    Filtered to [start, end] before saving.

    Output: data/raw/onchain/bgeometrics_onchain.parquet
    """
    logger.info("=== BGeometrics On-Chain Collector ===")
    logger.info(f"  Base URL: {BASE_URL}  (no key required)")

    frames = {}
    for endpoint, col_name in BG_METRICS.items():
        try:
            raw  = _fetch_endpoint(endpoint)
            s    = _parse_response(raw, col_name)
            clipped = s.loc[start:end].dropna()

            if clipped.empty:
                logger.warning(f"  ✗ {endpoint}: 0 rows in [{start}, {end}] "
                               f"(full range: {s.index[0].date()} – {s.index[-1].date()})")
            else:
                frames[col_name] = clipped
                logger.info(
                    f"  ✓ {col_name:<30} {len(clipped):>5} obs  "
                    f"[{clipped.index[0].date()} – {clipped.index[-1].date()}]"
                )
                # Save individual parquet per metric
                ind_path = RAW_DIR / "onchain" / f"bgeometrics_{col_name}.parquet"
                clipped.to_frame().to_parquet(ind_path)

            time.sleep(2)   # Be polite to the free API

        except Exception as e:
            logger.error(f"  ✗ {endpoint}: {e}")
            time.sleep(5)

    if not frames:
        logger.error("  BGeometrics: no metrics collected")
        return pd.DataFrame()

    # Merge all into one daily panel
    df = pd.DataFrame(frames).sort_index()
    df.index.name = "date"
    df.index = _to_naive(df.index)

    out = RAW_DIR / "onchain" / "bgeometrics_onchain.parquet"
    df.to_parquet(out)
    logger.success(
        f"BGeometrics → {out}  {df.shape}\n"
        f"  Columns: {list(df.columns)}"
    )
    return df


if __name__ == "__main__":
    df = collect_bgeometrics()
    if not df.empty:
        print(df.describe().round(4))
        print(f"\nDate range: {df.index[0].date()} → {df.index[-1].date()}")
        print(f"Nulls per column:\n{df.isna().sum()}")