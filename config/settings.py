"""
config/settings.py — single source of truth for all pipeline configuration.
All other modules import from here; nothing else reads .env directly.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)

# ── Date range (paper: Jan 2018 – Dec 2025) ───────────────
START_DATE: str = os.getenv("START_DATE", "2018-01-01")
END_DATE:   str = os.getenv("END_DATE",   "2025-12-31")

# ── Paths ─────────────────────────────────────────────────
_root    = Path(__file__).parent.parent
DATA_DIR = _root / Path(os.getenv("DATA_DIR", "data"))
LOG_DIR  = _root / Path(os.getenv("LOG_DIR",  "logs"))
DB_PATH  = _root / Path(os.getenv("DB_PATH",  "data/run_log.db"))
RAW_DIR  = DATA_DIR / "raw"

for _d in [
    RAW_DIR / "equity",   RAW_DIR / "crypto",  RAW_DIR / "sentiment",
    RAW_DIR / "onchain",  RAW_DIR / "futures",  RAW_DIR / "macro",
    DATA_DIR / "processed", LOG_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

# ── API credentials ───────────────────────────────────────
FRED_API_KEY         = os.getenv("FRED_API_KEY", "")
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "crypto_research/1.0")
# Glassnode removed — replaced by CoinMetrics Community API (free, no key)
COINGECKO_API_KEY    = os.getenv("COINGECKO_API_KEY", "")
COINMETRICS_API_KEY  = os.getenv("COINMETRICS_API_KEY", "")   # optional free key
COINGLASS_API_KEY    = os.getenv("COINGLASS_API_KEY", "")
KAIKO_API_KEY        = os.getenv("KAIKO_API_KEY", "")
WRDS_USERNAME        = os.getenv("WRDS_USERNAME", "")

# ── Tickers ───────────────────────────────────────────────
EQUITY_TICKERS = {
    "SPY":       "S&P 500 ETF",
    "^VIX":      "CBOE VIX",
    "XLK":       "Tech sector ETF",
    "XLF":       "Financials sector ETF",
    "GC=F":      "Gold futures",
    "DX-Y.NYB":  "Dollar Index DXY",
}
CRYPTO_IDS_CG     = ["bitcoin", "ethereum"]
REDDIT_SUBREDDITS = ["cryptocurrency", "Bitcoin", "ethereum", "CryptoCurrency"]
