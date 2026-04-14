"""
collectors/reddit_2025_gap_fill.py

Fetches Reddit posts and comments for January 2025 – December 2025
using the Arctic Shift API (arctic-shift.photon-reddit.com) and merges
with the existing reddit_sentiment.parquet produced by pushshift_loader.py.

Arctic Shift is a free, no-key-required archive of Reddit data maintained
by Arthur Heitmann. It covers data from the Pushshift archive plus ongoing
collection. No authentication needed.

API base: https://arctic-shift.photon-reddit.com
Endpoints used:
  GET /api/posts/search?subreddit=Bitcoin&after=2025-01-01&before=2025-02-01&limit=100
  GET /api/comments/search?subreddit=Bitcoin&after=2025-01-01&before=2025-02-01&limit=100

Parameters:
  subreddit : str   subreddit name (without r/)
  after     : str   ISO date, inclusive start
  before    : str   ISO date, exclusive end
  limit     : int   max results per page (max 100)
  sort      : str   "asc" to paginate forward in time

Rate limits: no key required; be polite — 1-2 req/sec max.
The script sleeps between requests and checks X-RateLimit-Remaining.

Usage:
  python collectors/reddit_2025_gap_fill.py

Output:
  Updates data/raw/sentiment/reddit_sentiment.parquet in-place,
  appending 2025 rows so the file covers 2018-01-01 → 2025-12-31.
"""
import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, START_DATE

# ── Configuration ──────────────────────────────────────────────────────────────

AS_BASE    = "https://arctic-shift.photon-reddit.com"
SUBREDDITS = ["Bitcoin", "CryptoCurrency", "ethereum"]

# Only fetch 2025 — the gap we need to fill
GAP_START = "2025-01-01"
GAP_END   = "2025-12-31"

# Batch monthly to keep requests manageable and avoid timeouts
BATCH_MONTHS = 1        # fetch one month at a time per subreddit
LIMIT        = 100      # max posts per API request (Arctic Shift max = 100)
SLEEP_BETWEEN_REQUESTS = 2.5   # seconds — increased to reduce connection resets
MAX_RETRIES_ON_RESET   = 5     # retry on ConnectionResetError before giving up
RETRY_SLEEP_ON_RESET   = 30    # seconds to wait after a connection reset

# NLP scorer
NLP_SCORER = "vader"   # fast; switch to "finbert" for higher quality


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_naive(index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def _check_rate_limit(resp: requests.Response) -> None:
    """Respect X-RateLimit headers — sleep if running low."""
    remaining = int(resp.headers.get("X-RateLimit-Remaining", 100))
    reset_in  = int(resp.headers.get("X-RateLimit-Reset", 0))
    if remaining < 5:
        logger.warning(f"  Rate limit low ({remaining} remaining) — sleeping {reset_in}s")
        time.sleep(max(reset_in, 10))


def _fetch_page(endpoint: str, params: dict) -> list:
    """
    Fetch one page from Arctic Shift API.
    Returns list of post/comment objects, or [] on error.
    """
    url = f"{AS_BASE}{endpoint}"
    for attempt in range(MAX_RETRIES_ON_RESET):
        try:
            resp = requests.get(url, params=params, timeout=45,
                                headers={"User-Agent": "crypto_research/1.0"})
            _check_rate_limit(resp)
            if resp.status_code == 429:
                logger.warning("  429 rate limited — sleeping 60s")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                logger.warning(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                return []
            data = resp.json()
            # Arctic Shift wraps results in {"data": [...]}
            if isinstance(data, dict):
                return data.get("data", [])
            return data if isinstance(data, list) else []
        except (ConnectionResetError, ConnectionError) as e:
            logger.warning(
                f"  Connection reset (attempt {attempt+1}/{MAX_RETRIES_ON_RESET}): {e}"
                f" — sleeping {RETRY_SLEEP_ON_RESET}s then retrying"
            )
            time.sleep(RETRY_SLEEP_ON_RESET)
        except Exception as e:
            logger.warning(f"  Request error: {e}")
            return []
    logger.error(f"  Gave up after {MAX_RETRIES_ON_RESET} retries")
    return []


def _paginate(endpoint: str, subreddit: str,
              after: str, before: str) -> list[dict]:
    """
    Paginate through all posts/comments in [after, before) for a subreddit.
    Arctic Shift supports cursor-based pagination via the 'after' parameter
    which accepts an ISO timestamp — we advance it using the last item's
    created_utc on each page.
    """
    records = []
    cursor  = after
    page    = 0

    while True:
        params = {
            "subreddit": subreddit,
            "after":     cursor,
            "before":    before,
            "limit":     LIMIT,
            "sort":      "asc",        # oldest first → enables forward pagination
        }
        items = _fetch_page(endpoint, params)
        if not items:
            break

        records.extend(items)
        page += 1

        # Advance cursor to just after the last item's timestamp
        last_ts = items[-1].get("created_utc", 0)
        try:
            last_ts = int(float(str(last_ts)))
        except (ValueError, TypeError):
            break

        # Convert to ISO for next request
        cursor = pd.Timestamp(last_ts, unit="s").strftime("%Y-%m-%dT%H:%M:%S")

        if page % 5 == 0:
            logger.info(f"    page {page}: {len(records):,} records so far "
                        f"(cursor: {cursor})")

        # If we got fewer than limit, we've reached the end
        if len(items) < LIMIT:
            break

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    return records


def _extract_text(obj: dict, obj_type: str) -> str:
    """Extract scoreable text from a post or comment."""
    if obj_type == "post":
        title = obj.get("title", "") or ""
        body  = obj.get("selftext", "") or ""
        body  = "" if body in ("[removed]", "[deleted]") else body
        return f"{title}. {body}".strip(". ").strip()[:800]
    else:  # comment
        body = obj.get("body", "") or ""
        body = "" if body in ("[removed]", "[deleted]") else body
        return body[:800]


# ── NLP Scoring ───────────────────────────────────────────────────────────────

def _score_vader(texts: list[str]) -> list[float]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(t)["compound"] for t in texts]


def _score_finbert(texts: list[str]) -> list[float]:
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text-classification", model="ProsusAI/finbert",
                       top_k=None, device=-1)
    out = []
    for i in range(0, len(texts), 32):
        chunk = [t[:512] for t in texts[i:i + 32]]
        for res in pipe(chunk, truncation=True, max_length=512):
            lbl = {r["label"]: r["score"] for r in res}
            out.append(lbl.get("positive", 0) - lbl.get("negative", 0))
    return out


def _score(texts: list[str]) -> list[float]:
    return _score_finbert(texts) if NLP_SCORER == "finbert" else _score_vader(texts)


# ── Main collector ────────────────────────────────────────────────────────────

def _save_checkpoint(records: list, path: Path) -> None:
    """Aggregate and save what we have so far to a checkpoint parquet."""
    try:
        df = pd.DataFrame(records)
        df.loc[:, "date"] = _to_naive(pd.to_datetime(df["date"]))
        df["nlp_score"] = _score(df["text"].tolist())
        daily = (
            df.groupby("date")[["nlp_score", "score"]]
            .apply(
                lambda g: pd.Series({
                    "reddit_sentiment_weighted": np.average(
                        g["nlp_score"], weights=g["score"]),
                    "reddit_sentiment_mean":     g["nlp_score"].mean(),
                    "reddit_post_count":         len(g),
                }),
                include_groups=False,
            )
            .sort_index()
        )
        daily.index = _to_naive(daily.index)
        daily.index.name = "date"
        daily.to_parquet(path)
        logger.info(f"  Checkpoint saved → {path}  ({len(daily)} days)")
    except Exception as e:
        logger.warning(f"  Checkpoint save failed: {e}")


def collect_reddit_2025(
    gap_start: str = GAP_START,
    gap_end:   str = GAP_END,
) -> pd.DataFrame:
    """
    Fetch 2025 Reddit data from Arctic Shift API, score with NLP,
    aggregate to daily, and return as a DataFrame with the same
    schema as reddit_sentiment.parquet.
    """
    logger.info("=== Arctic Shift Reddit 2025 Gap-Fill ===")
    logger.info(f"  API: {AS_BASE}")
    logger.info(f"  Range: {gap_start} → {gap_end}")
    logger.info(f"  Subreddits: {SUBREDDITS}")
    logger.info(f"  Scorer: {NLP_SCORER}")

    all_records = []

    # Generate monthly batches
    months = pd.date_range(start=gap_start, end=gap_end, freq="MS")

    # Resume support: skip subreddits already saved in a partial checkpoint file
    # Set RESUME_FROM = "CryptoCurrency" to skip Bitcoin and start at CryptoCurrency
    RESUME_FROM = os.getenv("RESUME_FROM", "")
    skipping   = bool(RESUME_FROM)

    # Load partial checkpoint if exists (in case of crash mid-run)
    checkpoint_path = RAW_DIR / "sentiment" / "reddit_2025_checkpoint.parquet"
    if checkpoint_path.exists():
        existing_checkpoint = pd.read_parquet(checkpoint_path)
        existing_checkpoint.index = _to_naive(pd.to_datetime(existing_checkpoint.index))
        all_records_df = existing_checkpoint  # will be merged at end
        logger.info(f"  Loaded checkpoint: {len(existing_checkpoint)} days already processed")
    else:
        all_records_df = None

    for subreddit in SUBREDDITS:
        # Resume: skip subreddits before RESUME_FROM
        if skipping:
            if subreddit == RESUME_FROM:
                skipping = False
                logger.info(f"  Resuming from r/{subreddit}")
            else:
                logger.info(f"  Skipping r/{subreddit} (already collected)")
                continue

        logger.info(f"\n  Processing r/{subreddit}...")
        sub_total = 0

        for month_start in months:
            month_end = (month_start + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
            after_str = month_start.strftime("%Y-%m-%d")
            # Don't go past gap_end
            if month_end > gap_end:
                month_end = gap_end

            # Fetch submissions
            posts = _paginate("/api/posts/search", subreddit, after_str, month_end)
            for p in posts:
                text = _extract_text(p, "post")
                if len(text) < 10:
                    continue
                try:
                    ts = float(str(p.get("created_utc", 0)))
                except (ValueError, TypeError):
                    continue
                all_records.append({
                    "date":      pd.Timestamp(ts, unit="s").normalize(),
                    "subreddit": subreddit,
                    "text":      text,
                    "score":     max(int(p.get("score", 1)), 1),
                })

            time.sleep(SLEEP_BETWEEN_REQUESTS)

            # Fetch comments
            comments = _paginate("/api/comments/search", subreddit, after_str, month_end)
            for c in comments:
                text = _extract_text(c, "comment")
                if len(text) < 10:
                    continue
                try:
                    ts = float(str(c.get("created_utc", 0)))
                except (ValueError, TypeError):
                    continue
                all_records.append({
                    "date":      pd.Timestamp(ts, unit="s").normalize(),
                    "subreddit": subreddit,
                    "text":      text,
                    "score":     max(int(c.get("score", 1)), 1),
                })

            month_count = len(posts) + len(comments)
            sub_total  += month_count
            logger.info(f"    {after_str}: {len(posts):,} posts + "
                        f"{len(comments):,} comments = {month_count:,}")
            time.sleep(SLEEP_BETWEEN_REQUESTS)

        logger.info(f"  r/{subreddit} total: {sub_total:,} records")

        # Save partial checkpoint after each subreddit so a crash doesn't lose everything
        if all_records:
            _save_checkpoint(all_records, checkpoint_path)

    checkpoint_path = RAW_DIR / "sentiment" / "reddit_2025_checkpoint.parquet"
    if not all_records:
        if checkpoint_path.exists():
            logger.warning("  No new records — using checkpoint data")
            daily = pd.read_parquet(checkpoint_path)
            daily.index = _to_naive(pd.to_datetime(daily.index))
            return daily
        logger.error("  No 2025 records collected. Check API availability.")
        return pd.DataFrame()

    logger.info(f"\n  Total 2025 records: {len(all_records):,}")
    logger.info(f"  Scoring with {NLP_SCORER}...")

    df = pd.DataFrame(all_records)
    df.loc[:, "date"] = _to_naive(pd.to_datetime(df["date"]))
    df["nlp_score"] = _score(df["text"].tolist())

    logger.info(f"  Score stats: mean={df['nlp_score'].mean():.4f}, "
                f"std={df['nlp_score'].std():.4f}")

    # Aggregate to daily
    daily = (
        df.groupby("date")[["nlp_score", "score"]]
        .apply(
            lambda g: pd.Series({
                "reddit_sentiment_weighted": np.average(
                    g["nlp_score"], weights=g["score"]),
                "reddit_sentiment_mean":     g["nlp_score"].mean(),
                "reddit_post_count":         len(g),
            }),
            include_groups=False,
        )
        .sort_index()
    )
    daily.index = _to_naive(daily.index)
    daily.index.name = "date"

    logger.success(
        f"  2025 daily sentiment: {len(daily)} days, "
        f"mean={daily['reddit_sentiment_weighted'].mean():.4f}"
    )
    return daily


def merge_and_save(daily_2025: pd.DataFrame) -> None:
    """
    Load existing reddit_sentiment.parquet (2018-2024),
    append 2025 rows, deduplicate, and save back.
    """
    out_path = RAW_DIR / "sentiment" / "reddit_sentiment.parquet"

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing.index = _to_naive(pd.to_datetime(existing.index))
        logger.info(f"  Existing file: {len(existing)} rows "
                    f"({existing.index[0].date()} → {existing.index[-1].date()})")

        combined = pd.concat([existing, daily_2025]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
    else:
        logger.warning("  No existing reddit_sentiment.parquet — saving 2025 only")
        combined = daily_2025

    combined.to_parquet(out_path)
    logger.success(
        f"  Saved → {out_path}\n"
        f"  Final: {len(combined)} rows "
        f"({combined.index[0].date()} → {combined.index[-1].date()})\n"
        f"  Mean daily sentiment: {combined['reddit_sentiment_weighted'].mean():.4f}"
    )
    print(combined.tail(10).to_string())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    daily_2025 = collect_reddit_2025()
    if not daily_2025.empty:
        merge_and_save(daily_2025)
        print(f"\n✓ Done. reddit_sentiment.parquet now covers 2018 → 2025.")
        print("  Run: python main.py  (or python main.py --mode features)")
    else:
        print("✗ No data collected. Check your internet connection and try again.")