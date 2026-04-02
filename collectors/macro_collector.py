"""
collectors/macro_collector.py

Fetches macroeconomic control variables via FRED (Section 3.3 Table 2).
All indexes saved as tz-naive dates.
Requires FRED_API_KEY in config/.env (free: fred.stlouisfed.org/docs/api/api_key.html)

Output: data/raw/macro/fred_macro.parquet
"""
import sys
import pandas as pd
from pathlib import Path
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, START_DATE, END_DATE, FRED_API_KEY

FRED_SERIES = {
    "DFF":          "fed_funds_rate",
    "VIXCLS":       "vix_fred",
    "DEXUSEU":      "usdeur_exchange",
    "DCOILWTICO":   "crude_oil_wti",
    "T10YIE":       "inflation_breakeven_10y",
    "BAMLH0A0HYM2": "hy_credit_spread",
}


def _to_naive(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=3, max=30))
def _fred(series_id: str, start: str, end: str) -> pd.Series:
    from fredapi import Fred
    s = Fred(api_key=FRED_API_KEY).get_series(
        series_id, observation_start=start, observation_end=end)
    s.index = _to_naive(pd.to_datetime(s.index))
    s.name  = series_id
    return s


def collect_macro(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    if not FRED_API_KEY:
        logger.warning("FRED_API_KEY not set — skipping macro collection")
        return pd.DataFrame()

    frames = {}
    for sid, col in FRED_SERIES.items():
        try:
            s = _fred(sid, start, end)
            frames[col] = s
            logger.info(f"  FRED ✓ {sid} → {col}: {len(s)} obs")
        except Exception as e:
            logger.warning(f"  FRED ✗ {sid}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = _to_naive(df.index)
    df = df.resample("D").last().ffill()
    df = df.loc[start:end]
    df.index.name = "date"

    out = RAW_DIR / "macro" / "fred_macro.parquet"
    df.to_parquet(out)
    logger.success(f"FRED macro → {out}  {df.shape}")
    return df


if __name__ == "__main__":
    df = collect_macro()
    print(df.tail())