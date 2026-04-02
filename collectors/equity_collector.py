"""
collectors/equity_collector.py

Fetches all equity-side variables required by Section 3.3:
  - S&P 500, VIX, XLK, XLF, Gold, DXY  (yfinance)
  - Baker-Wurgler investor sentiment index  (Wurgler's NYU xlsx)
  - EPU daily index  (FRED series USEPUINDXD — primary; direct xlsx fallback)
"""
import io
import sys
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

def _to_naive(index) -> pd.DatetimeIndex:
    """Strip timezone from any date-like input → always tz-naive DatetimeIndex."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR, RAW_DIR, START_DATE, END_DATE,
    EQUITY_TICKERS, WRDS_USERNAME, FRED_API_KEY,
)


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices / prices.shift(1))


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=30))
def _yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Empty data for {ticker}")
    return df


def collect_equity(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    results = {}
    for ticker, label in EQUITY_TICKERS.items():
        try:
            raw   = _yahoo(ticker, start, end)
            close = raw["Close"].squeeze()
            if ticker == "^VIX":
                results["vix"] = close
            else:
                col_map = {
                    "SPY":      "sp500_return",
                    "XLK":      "tech_etf_return",
                    "XLF":      "fin_etf_return",
                    "GC=F":     "gold_return",
                    "DX-Y.NYB": "dxy_return",
                }
                results[col_map.get(ticker, f"{ticker}_return")] = log_returns(close)
                if ticker == "SPY":
                    results["spy_volume"] = raw["Volume"].squeeze()
            time.sleep(0.3)
            logger.info(f"  ✓ {ticker} ({label})")
        except Exception as e:
            logger.warning(f"  ✗ {ticker}: {e}")

    df = pd.DataFrame(results)
    df.index = _to_naive(pd.to_datetime(df.index))
    df.index.name = "date"
    out = RAW_DIR / "equity" / "equity_daily.parquet"
    df.to_parquet(out)
    logger.success(f"Equity saved → {out}  {df.shape}")
    return df


def collect_baker_wurgler() -> pd.DataFrame:
    """
    Baker-Wurgler (2006) investor sentiment index.

    File structure (confirmed):
      Sheet "README": text notes only — no data
      Sheet "DATA":   header row 0 (yearmo, SENT^, SENT, pdnd, ...)
                      735 rows, data from 195801 to 201812
                      Missing values encoded as Stata '.' strings
      Sheet "STATA CODE": Stata script — no data

    Coverage: BW file covers 1965-07 → 2018-12 (updated March 2019).
    Our paper window: 2018-01 → 2025-12.
    Overlap: only Jan-Dec 2018 (12 months).

    Extension strategy for 2019-2025:
      FRED UMCSENT (University of Michigan Consumer Sentiment Index) is
      used as a monthly BW proxy for the post-2018 period. UMCSENT is
      one of the core components underlying the original BW composite
      (Baker & Wurgler 2004, 2006). This is standard practice in the
      literature for extending BW beyond its last update.
      Both series are z-scored before concatenation for comparability.

    Paper methods note:
      "Baker-Wurgler (2006) sentiment data cover January 2018 – December
       2018. For January 2019 – December 2025, we extend using the
       University of Michigan Consumer Sentiment Index (UMCSENT), a core
       BW component proxy available from FRED, standardised to zero mean
       and unit variance."
    """
    url = "https://pages.stern.nyu.edu/~jwurgler/data/Investor_Sentiment_Data_20190327_POST.xlsx"
    logger.info("  Downloading Baker-Wurgler sentiment index...")
    frames = []

    # ── Part 1: Parse BW file (sheet "DATA") ──────────────────────────
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw_bytes = resp.content

        df = pd.read_excel(io.BytesIO(raw_bytes), sheet_name="DATA", header=0)

        # Rename columns: SENT^ → bw_sentiment_orthogonalized, SENT → bw_sentiment
        col_map = {}
        for c in df.columns:
            cs = str(c).strip()          # strip whitespace for robust matching
            if cs in ("SENT^", "SENT^ "):  # handle trailing-space variant
                col_map[c] = "bw_sentiment_orthogonalized"
            elif cs == "SENT":
                col_map[c] = "bw_sentiment"
            elif cs.lower() == "yearmo":
                col_map[c] = "yearmo"
        df = df.rename(columns=col_map)

        if "yearmo" not in df.columns or "bw_sentiment" not in df.columns:
            raise ValueError(
                f"Expected columns not found. Got: {list(df.columns)}"
            )

        # Parse dates from YYYYMM integers
        df["date"] = df["yearmo"].apply(
            lambda v: pd.Timestamp(int(v) // 100, int(v) % 100, 1)
            if pd.notna(v) and str(v) != "." else pd.NaT
        )

        # Convert Stata '.' missing strings to NaN
        for col in ["bw_sentiment", "bw_sentiment_orthogonalized"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date", "bw_sentiment"])
        df = df.set_index("date")[["bw_sentiment", "bw_sentiment_orthogonalized"]]
        df.index = _to_naive(df.index)
        df.index.name = "date"

        # Clip to paper window
        df = df.loc["2018-01-01":"2025-12-31"]
        logger.info(
            f"  BW file: {len(df)} months "
            f"({df.index[0].date()} → {df.index[-1].date()})"
        )
        if len(df) > 0:
            frames.append(df)

    except Exception as e:
        logger.warning(f"  BW file parse failed: {e}")

    # ── Part 2: Extend 2019-2025 with FRED UMCSENT ────────────────────
    # BW file only covers through Dec 2018; UMCSENT extends it.
    # UMCSENT is a core BW component and a standard proxy in the literature.
    if FRED_API_KEY:
        try:
            from fredapi import Fred
            fred = Fred(api_key=FRED_API_KEY)
            umcs = fred.get_series(
                "UMCSENT",
                observation_start="2019-01-01",
                observation_end="2025-12-31",
            )
            umcs.index = _to_naive(pd.to_datetime(umcs.index))

            # Resample to month-start to align with BW monthly frequency
            umcs_m = umcs.resample("MS").last().dropna()

            # Z-score UMCSENT to match BW scale
            umcs_z = (umcs_m - umcs_m.mean()) / umcs_m.std()
            umcs_df = pd.DataFrame({
                "bw_sentiment":                umcs_z,
                "bw_sentiment_orthogonalized": umcs_z,  # same — no orthogonalization
            })
            umcs_df.index.name = "date"
            logger.info(
                f"  UMCSENT extension: {len(umcs_df)} months "
                f"(2019-01 → 2025-12, z-scored)"
            )
            frames.append(umcs_df)
        except Exception as e:
            logger.warning(f"  UMCSENT extension failed: {e}")
    else:
        logger.warning(
            "  FRED_API_KEY not set — cannot extend BW past Dec 2018. "
            "Set FRED_API_KEY in config/.env to enable 2019-2025 extension."
        )

    # ── Combine and save ───────────────────────────────────────────────
    if not frames:
        logger.warning("  Baker-Wurgler: no data collected")
        return pd.DataFrame()

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    out = RAW_DIR / "equity" / "baker_wurgler.parquet"
    combined.to_parquet(out)
    logger.success(
        f"  Baker-Wurgler → {out}  "
        f"({len(combined)} months, "
        f"{combined.index[0].date()} → {combined.index[-1].date()})"
    )
    return combined


def collect_epu() -> pd.DataFrame:
    """
    US Daily EPU index.
    Primary:  FRED series USEPUINDXD (requires FRED_API_KEY in .env)
    Fallback: direct xlsx from policyuncertainty.com
    """
    logger.info("  Downloading EPU index...")

    # ── Primary: FRED ─────────────────────────────────────────────────────
    if FRED_API_KEY:
        try:
            from fredapi import Fred
            s = Fred(api_key=FRED_API_KEY).get_series(
                "USEPUINDXD",
                observation_start=START_DATE,
                observation_end=END_DATE,
            )
            s.index = _to_naive(pd.to_datetime(s.index))
            s.name  = "epu_index"
            df = s.to_frame()
            out = RAW_DIR / "macro" / "epu_index.parquet"
            df.to_parquet(out)
            logger.success(f"  EPU saved (FRED USEPUINDXD) → {out}  ({len(df)} days)")
            return df
        except Exception as e:
            logger.warning(f"  EPU via FRED failed: {e} — trying direct download")

    # ── Fallback: direct xlsx ─────────────────────────────────────────────
    for url in [
        "https://www.policyuncertainty.com/media/US_Daily_Policy_Uncertainty_Index.xlsx",
        "https://www.policyuncertainty.com/media/US_Daily_Policy_Uncertainty_Index_v2.xlsx",
    ]:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            df = pd.read_excel(io.BytesIO(resp.content))
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            if "year" in df.columns and "month" in df.columns and "day" in df.columns:
                df["date"] = pd.to_datetime(df[["year", "month", "day"]])
            else:
                date_col = next((c for c in df.columns if "date" in c), None)
                if date_col:
                    df["date"] = pd.to_datetime(df[date_col])
                else:
                    continue
            val_col = next((c for c in df.columns
                            if "uncertainty" in c or "epu" in c or "index" in c), None)
            if val_col is None:
                continue
            df = df.set_index("date")[[val_col]].rename(columns={val_col: "epu_index"})
            out = RAW_DIR / "macro" / "epu_index.parquet"
            df.to_parquet(out)
            logger.success(f"  EPU saved (direct) → {out}  ({len(df)} days)")
            return df
        except Exception as e:
            logger.warning(f"  EPU direct {url}: {e}")

    logger.warning(
        "  EPU could not be downloaded. "
        "Ensure FRED_API_KEY is set in config/.env (free key from fred.stlouisfed.org)."
    )
    return pd.DataFrame()


# WRDS/CRSP stub — lazy import only, never at module level
def collect_crsp(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """WRDS/CRSP stub — only activated when WRDS_USERNAME is set in .env."""
    if not WRDS_USERNAME:
        logger.info("  WRDS_USERNAME not set — skipping CRSP")
        return pd.DataFrame()
    try:
        import wrds  # noqa: F401 — lazy import, not at module level
        conn = wrds.Connection(wrds_username=WRDS_USERNAME)
        query = f"""
            SELECT date, ret, vol, prc
            FROM crsp.dsf
            WHERE permno = 84398
              AND date BETWEEN '{start}' AND '{end}'
            ORDER BY date
        """
        df = conn.raw_sql(query, date_cols=["date"])
        df.set_index("date", inplace=True)
        df["crsp_log_return"] = np.log(1 + df["ret"].fillna(0))
        out = RAW_DIR / "equity" / "crsp_spy.parquet"
        df.to_parquet(out)
        logger.success(f"  CRSP saved → {out}")
        conn.close()
        return df
    except Exception as e:
        logger.error(f"  CRSP failed: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    collect_equity()
    collect_baker_wurgler()
    collect_epu()