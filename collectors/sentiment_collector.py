"""
collectors/sentiment_collector.py

Fetches sentiment variables required by Section 3.3.1:
  StockTwits  (free, no key)   — $BTC.X, $ETH.X, $CRYPTO.X, FinBERT-scored
  Reddit PRAW (free key)       — r/cryptocurrency, r/Bitcoin, r/ethereum
  Fear & Greed (Alternative.me)— composite 0-100 index
  Google Trends (pytrends)     — weekly SVI, batched quarterly
  GDELT                        — DISABLED (server timing out on every request)
                                  Set GDELT_ENABLED=true in .env to re-enable.

All date indexes saved as tz-naive UTC.
Pandas DeprecationWarnings suppressed via include_groups=False in groupby.apply().
"""
import sys
import os
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR, RAW_DIR, START_DATE, END_DATE,
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    REDDIT_SUBREDDITS,
)

GDELT_ENABLED = os.getenv("GDELT_ENABLED", "false").lower() == "true"


def _to_naive(index) -> pd.DatetimeIndex:
    """Strip timezone from any date-like input → always tz-naive DatetimeIndex."""
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


# ── FinBERT / VADER ───────────────────────────────────────

_finbert = None


def get_finbert():
    global _finbert
    if _finbert is None:
        logger.info("Loading FinBERT (~500MB download on first run)...")
        from transformers import pipeline as hf_pipeline
        _finbert = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            device=-1,
        )
        logger.success("FinBERT ready.")
    return _finbert


def score_finbert(texts: list) -> list:
    pipe = get_finbert()
    out, batch = [], 32
    for i in range(0, len(texts), batch):
        chunk = [t[:512] for t in texts[i:i + batch]]
        try:
            for res in pipe(chunk, truncation=True, max_length=512):
                lbl = {r["label"]: r["score"] for r in res}
                out.append(lbl.get("positive", 0) - lbl.get("negative", 0))
        except Exception as e:
            logger.warning(f"FinBERT batch error: {e}")
            out.extend([0.0] * len(chunk))
    return out


def score_vader(texts: list) -> list:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(t)["compound"] for t in texts]


def _scorer(use_finbert: bool = True):
    try:
        return (score_finbert, "FinBERT") if use_finbert else (score_vader, "VADER")
    except Exception:
        return (score_vader, "VADER")


# ── StockTwits ────────────────────────────────────────────

ST_BASE    = "https://api.stocktwits.com/api/2"
ST_SYMBOLS = ["BTC.X", "ETH.X", "CRYPTO.X"]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=20, max=120))
def _st_fetch(symbol: str, max_id: int = None) -> dict:
    params = {"limit": 30}
    if max_id:
        params["max"] = max_id
    resp = requests.get(
        f"{ST_BASE}/streams/symbol/{symbol}.json",
        params=params, timeout=20,
        headers={"User-Agent": "crypto_research/1.0"},
    )
    if resp.status_code == 429:
        raise Exception("StockTwits rate limited")
    resp.raise_for_status()
    return resp.json()


def collect_stocktwits(
    days_back: int = 30,
    pages_per_symbol: int = 15,
    use_finbert: bool = True,
) -> pd.DataFrame:
    """
    Collect StockTwits messages and score with FinBERT.
    pages_per_symbol=15 takes ~8 minutes. Increase for more history.
    """
    fn, fn_name = _scorer(use_finbert)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    logger.info(f"  StockTwits: {days_back}d back, {pages_per_symbol} pages/symbol, scorer={fn_name}")

    records = []
    for sym in ST_SYMBOLS:
        max_id, pages = None, 0
        for _ in range(pages_per_symbol):
            try:
                data = _st_fetch(sym, max_id)
                msgs = data.get("messages", [])
                if not msgs:
                    break
                stop = False
                for m in msgs:
                    ts = pd.Timestamp(m["created_at"]).to_pydatetime()
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts < cutoff:
                        stop = True
                        break
                    sentiment_label = None
                    if m.get("sentiment"):
                        sentiment_label = m["sentiment"].get("basic")
                    records.append({
                        "date":     pd.Timestamp(m["created_at"]).normalize(),
                        "symbol":   sym,
                        "text":     m.get("body", "")[:600],
                        "likes":    m.get("likes", {}).get("total", 0),
                        "reshares": m.get("reshares", {}).get("reshared_count", 0),
                        "st_label": sentiment_label,
                    })
                max_id = min(m["id"] for m in msgs) - 1
                pages += 1
                time.sleep(18)
                if stop:
                    break
            except Exception as e:
                logger.warning(f"  StockTwits ${sym}: {e}")
                time.sleep(30)
                break
        sym_count = sum(1 for r in records if r["symbol"] == sym)
        logger.info(f"  ${sym}: {pages} pages, {sym_count} msgs")

    if not records:
        logger.warning("  No StockTwits messages collected.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"]     = _to_naive(pd.to_datetime(df["date"]))
    df["nlp_score"] = fn(df["text"].tolist())
    df["weight"]    = (df["likes"] + df["reshares"]).clip(lower=1)

    # include_groups=False suppresses pandas FutureWarning about grouping columns
    daily = (
        df.groupby("date")[["nlp_score", "weight", "st_label"]]
        .apply(
            lambda g: pd.Series({
                "stocktwits_sentiment_weighted": np.average(
                    g["nlp_score"], weights=g["weight"]),
                "stocktwits_sentiment_mean": g["nlp_score"].mean(),
                "stocktwits_bullish_ratio":  (g["st_label"] == "Bullish").sum()
                                              / max(g["st_label"].notna().sum(), 1),
                "stocktwits_msg_count": len(g),
            }),
            include_groups=False,
        )
    )
    daily.index = _to_naive(daily.index)
    daily.index.name = "date"
    out = RAW_DIR / "sentiment" / "stocktwits_sentiment.parquet"
    daily.to_parquet(out)
    logger.success(f"StockTwits → {out}  {daily.shape}")
    return daily


# ── Reddit PRAW ───────────────────────────────────────────

def collect_reddit(
    start: str = START_DATE,
    end: str = END_DATE,
    use_finbert: bool = True,
) -> pd.DataFrame:
    if not REDDIT_CLIENT_ID:
        logger.warning("  REDDIT_CLIENT_ID not set — skipping Reddit")
        return pd.DataFrame()
    import praw
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        read_only=True,
    )
    fn, fn_name = _scorer(use_finbert)
    start_ts = pd.Timestamp(start).timestamp()
    end_ts   = pd.Timestamp(end).timestamp()
    records  = []
    for sub in REDDIT_SUBREDDITS:
        logger.info(f"  Reddit: r/{sub}")
        try:
            for post in reddit.subreddit(sub).new(limit=1000):
                if not (start_ts <= post.created_utc <= end_ts):
                    continue
                records.append({
                    "date":  _to_naive(pd.to_datetime(
                        pd.Timestamp(post.created_utc, unit="s").normalize())),
                    "text":  f"{post.title}. {post.selftext or ''}".strip()[:800],
                    "score": max(post.score, 1),
                })
            time.sleep(2)
        except Exception as e:
            logger.warning(f"  r/{sub}: {e}")

    if not records:
        logger.warning("  No Reddit posts in date range.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    logger.info(f"  Scoring {len(df)} posts with {fn_name}...")
    df["sentiment"] = fn(df["text"].tolist())

    daily = (
        df.groupby("date")[["sentiment", "score"]]
        .apply(
            lambda g: pd.Series({
                "reddit_sentiment_weighted": np.average(
                    g["sentiment"], weights=g["score"]),
                "reddit_sentiment_mean": g["sentiment"].mean(),
                "reddit_post_count":     len(g),
            }),
            include_groups=False,
        )
    )
    daily.index = _to_naive(daily.index)
    daily.index.name = "date"
    out = RAW_DIR / "sentiment" / "reddit_sentiment.parquet"
    daily.to_parquet(out)
    logger.success(f"Reddit → {out}  {daily.shape}")
    return daily


# ── Fear & Greed ──────────────────────────────────────────

def collect_fear_greed(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    logger.info("  Fear & Greed Index (Alternative.me)...")
    for attempt in range(3):
        try:
            resp = requests.get(
                "https://api.alternative.me/fng/?limit=3000&format=json&date_format=us",
                timeout=60,
                headers={"User-Agent": "crypto_research/1.0"},
            )
            resp.raise_for_status()
            df = pd.DataFrame(resp.json()["data"])
            # API returns timestamp as Unix seconds OR date string (e.g. "04-01-2026")
            # Handle both formats
            def _parse_fg_date(val):
                try:
                    return pd.Timestamp(int(val), unit="s").normalize()
                except (ValueError, TypeError):
                    try:
                        return pd.Timestamp(pd.to_datetime(str(val))).normalize()
                    except Exception:
                        return pd.NaT
            df["date"] = _to_naive(df["timestamp"].apply(_parse_fg_date))
            df = df.set_index("date")[["value", "value_classification"]].rename(columns={
                "value": "fear_greed_index",
                "value_classification": "fear_greed_label",
            })
            df["fear_greed_index"] = pd.to_numeric(df["fear_greed_index"], errors="coerce")
            df = df.sort_index().loc[start:end]
            out = RAW_DIR / "sentiment" / "fear_greed.parquet"
            df.to_parquet(out)
            logger.success(f"Fear & Greed → {out}  ({len(df)} days)")
            return df
        except Exception as e:
            logger.warning(f"  Fear & Greed attempt {attempt+1}/3: {e}")
            if attempt < 2:
                time.sleep(15)
    logger.error("  Fear & Greed failed after 3 attempts")
    return pd.DataFrame()


# ── Google Trends ─────────────────────────────────────────

def collect_google_trends(
    keywords: list = None,
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    from pytrends.request import TrendReq
    if keywords is None:
        keywords = ["Bitcoin", "crypto crash", "buy bitcoin", "Ethereum", "crypto"]
    logger.info(f"  Google Trends: {keywords}")
    pt      = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
    periods = pd.date_range(start=start, end=end, freq="QS")
    frames  = []
    for i in range(len(periods) - 1):
        tf = f"{periods[i].strftime('%Y-%m-%d')} {periods[i+1].strftime('%Y-%m-%d')}"
        try:
            pt.build_payload(keywords, timeframe=tf, geo="")
            df = pt.interest_over_time()
            if not df.empty:
                df.drop(columns=["isPartial"], errors="ignore", inplace=True)
                frames.append(df)
            time.sleep(6)
        except Exception as e:
            logger.warning(f"  Google Trends {tf}: {e}")
            time.sleep(20)
    if not frames:
        return pd.DataFrame()
    out_df = pd.concat(frames).sort_index()
    out_df = out_df[~out_df.index.duplicated(keep="last")]
    out_df.index = _to_naive(out_df.index)
    out_df.columns = [f"gtrends_{c.lower().replace(' ', '_')}" for c in out_df.columns]
    out_df.index.name = "date"
    path = RAW_DIR / "sentiment" / "google_trends.parquet"
    out_df.to_parquet(path)
    logger.success(f"Google Trends → {path}  {out_df.shape}")
    return out_df


# ── GDELT — disabled ──────────────────────────────────────

def collect_gdelt(
    query: str = "Bitcoin OR cryptocurrency",
    start: str = START_DATE,
    end: str = END_DATE,
) -> pd.DataFrame:
    """
    GDELT news sentiment — DISABLED by default.
    api.gdeltproject.org times out on every request (~33s per week × 400 weeks
    = 3+ hours of timeouts for 2018-2025 with zero data collected).
    Set GDELT_ENABLED=true in config/.env to re-enable.
    """
    if not GDELT_ENABLED:
        logger.info(
            "  GDELT: skipped (server timing out). "
            "Set GDELT_ENABLED=true in .env to retry."
        )
        return pd.DataFrame()

    logger.info(f"  GDELT: '{query}'...")
    url     = "https://api.gdeltproject.org/api/v2/doc/doc"
    results = []
    for d in pd.date_range(start=start, end=end, freq="7D"):
        d_end = min(d + pd.Timedelta("7D"), pd.Timestamp(end))
        try:
            resp = requests.get(url, params={
                "query":  f'"{query}" sourcelang:eng',
                "mode":   "timelinevol",
                "format": "json",
                "startdatetime": d.strftime("%Y%m%d%H%M%S"),
                "enddatetime":   d_end.strftime("%Y%m%d%H%M%S"),
            }, timeout=15, headers={"User-Agent": "crypto_research/1.0"})
            if resp.status_code != 200 or not resp.text.strip():
                time.sleep(2)
                continue
            try:
                data = resp.json()
            except Exception:
                time.sleep(2)
                continue
            for item in data.get("timeline", [{}])[0].get("data", []):
                raw_date = item.get("date", "")
                if raw_date and len(raw_date) >= 8:
                    try:
                        results.append({
                            "date": pd.Timestamp(raw_date[:8], format="%Y%m%d").normalize(),
                            "gdelt_avg_tone":     0.0,
                            "gdelt_num_articles": float(item.get("value", 0)),
                        })
                    except Exception:
                        pass
            time.sleep(2)
        except Exception as e:
            logger.warning(f"  GDELT {d.date()}: {e}")
            time.sleep(2)

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df["date"] = _to_naive(pd.to_datetime(df["date"]))
    df = df.set_index("date").groupby("date").mean().sort_index().loc[start:end]
    path = RAW_DIR / "sentiment" / "gdelt_news.parquet"
    df.to_parquet(path)
    logger.success(f"GDELT → {path}  {df.shape}")
    return df


if __name__ == "__main__":
    collect_stocktwits(days_back=7, pages_per_symbol=3, use_finbert=False)
    collect_fear_greed()
    collect_google_trends()