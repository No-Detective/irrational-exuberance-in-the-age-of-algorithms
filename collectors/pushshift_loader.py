"""
collectors/pushshift_loader.py

Loads historical Reddit posts from Pushshift zstandard dump files and produces
a daily NLP-scored sentiment series for r/Bitcoin, r/CryptoCurrency, r/ethereum.

This closes Gap 1: the paper calls for social media NLP as the primary sentiment
signal, but StockTwits free API only provides the last 30 days. The Pushshift
archive covers 2005–2024 and is freely available on Academic Torrents.

═══════════════════════════════════════════════════════════════════════
  STEP 1 — DOWNLOAD THE DUMP FILES  (one-time, do this manually)
═══════════════════════════════════════════════════════════════════════

Install a BitTorrent client (e.g. qBittorrent) and open this magnet/page:

  https://academictorrents.com/details/1614740ac8c94505e4ecb9d88be8bed7b6afddd4

In the torrent file browser, ONLY select these files (saves ~8–15 GB
instead of the full 3.4 TB):

  submissions/
    r_Bitcoin_submissions.zst
    r_CryptoCurrency_submissions.zst
    r_ethereum_submissions.zst
    r_cryptocurrency_submissions.zst
  comments/
    r_Bitcoin_comments.zst
    r_CryptoCurrency_comments.zst
    r_ethereum_comments.zst

Place all downloaded .zst files into a directory and set PUSHSHIFT_DIR
in config/.env:

  PUSHSHIFT_DIR=/path/to/your/pushshift_dumps

Coverage: 2005 – December 2024 (essentially complete for 2018–2022;
Reddit API restrictions from mid-2023 create some gaps in 2023–2024).

═══════════════════════════════════════════════════════════════════════
  STEP 2 — RUN THIS SCRIPT
═══════════════════════════════════════════════════════════════════════

  python collectors/pushshift_loader.py

Or integrate into the main pipeline by calling collect_pushshift_reddit()
from the orchestrator after the StockTwits step.

Runtime: 2–6 hours depending on disk speed and whether you use FinBERT
(GPU recommended) or VADER (CPU, ~10× faster, less accurate).

Output: data/raw/sentiment/reddit_sentiment.parquet
        (same schema as the PRAW Reddit collector — slots directly into
        the existing transformer and feature engineering pipeline)

═══════════════════════════════════════════════════════════════════════
  ACADEMIC CITATION
═══════════════════════════════════════════════════════════════════════

  Baumgartner, J., Zannettou, S., Keegan, B., Squire, M., & Blackburn, J.
  (2020). The Pushshift Reddit Dataset. Proceedings of the International
  AAAI Conference on Web and Social Media, 14(1), 830–839.
  https://doi.org/10.1609/icwsm.v14i1.7347

  Cite this when describing the Reddit data source in your methods section.
"""
import os
import sys
import json
import time
import zstandard as zstd
import numpy as np
import pandas as pd
from pathlib import Path
from io import TextIOWrapper
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import RAW_DIR, START_DATE, END_DATE

# ── Configuration ──────────────────────────────────────────────────────────────

# Set PUSHSHIFT_DIR in config/.env, or override here
PUSHSHIFT_DIR = Path(os.getenv("PUSHSHIFT_DIR", "~/pushshift_dumps")).expanduser()

# Subreddits to process — must match downloaded file names
SUBREDDITS = ["Bitcoin", "CryptoCurrency", "ethereum", "cryptocurrency"]

# File types to process
FILE_TYPES = ["submissions", "comments"]

# Date range from settings (override here if needed)
START_TS = pd.Timestamp(START_DATE).timestamp()
END_TS   = pd.Timestamp(END_DATE).timestamp()

# NLP scorer: "finbert" (better, slower, GPU recommended) or "vader" (fast, CPU)
NLP_SCORER = os.getenv("PUSHSHIFT_SCORER", "finbert")

# Minimum score (Reddit upvotes) to include a post — filters low-signal content
MIN_SCORE = int(os.getenv("PUSHSHIFT_MIN_SCORE", "1"))

# Batch size for FinBERT inference
FINBERT_BATCH = 32


# ── NLP Scorers ───────────────────────────────────────────────────────────────

_finbert = None

def _get_finbert():
    global _finbert
    if _finbert is None:
        logger.info("  Loading FinBERT (~500MB, cached after first run)...")
        from transformers import pipeline as hf_pipeline
        _finbert = hf_pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            device=-1,          # CPU; change to 0 for GPU
        )
        logger.success("  FinBERT ready.")
    return _finbert


def _score_finbert(texts: list[str]) -> list[float]:
    pipe = _get_finbert()
    scores = []
    for i in range(0, len(texts), FINBERT_BATCH):
        chunk = [t[:512] for t in texts[i:i + FINBERT_BATCH]]
        try:
            for res in pipe(chunk, truncation=True, max_length=512):
                lbl = {r["label"]: r["score"] for r in res}
                scores.append(lbl.get("positive", 0) - lbl.get("negative", 0))
        except Exception as e:
            logger.warning(f"    FinBERT batch error: {e}")
            scores.extend([0.0] * len(chunk))
    return scores


def _score_vader(texts: list[str]) -> list[float]:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return [sia.polarity_scores(t)["compound"] for t in texts]


def _score(texts: list[str]) -> list[float]:
    if NLP_SCORER == "finbert":
        return _score_finbert(texts)
    return _score_vader(texts)


# ── Pushshift File Reader ─────────────────────────────────────────────────────

def _read_zst_lines(path: Path):
    """
    Generator that yields decoded JSON objects from a .zst compressed NDJSON file.
    Memory-efficient — streams line by line without loading the full file.
    """
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _extract_text(obj: dict, file_type: str) -> str:
    """Extract the text content from a Reddit post or comment object."""
    if file_type == "submissions":
        title = obj.get("title", "") or ""
        body  = obj.get("selftext", "") or ""
        # [removed] and [deleted] are placeholder strings, not real content
        body  = "" if body in ("[removed]", "[deleted]") else body
        return f"{title}. {body}".strip(". ").strip()[:800]
    else:   # comments
        body = obj.get("body", "") or ""
        body = "" if body in ("[removed]", "[deleted]") else body
        return body[:800]


def _find_dump_files(subreddit: str, file_type: str) -> list[Path]:
    """
    Find all .zst files for a given subreddit and file type.
    Supports multiple naming conventions used across Pushshift dump releases:
      r_{sub}_submissions.zst
      {sub}_submissions.zst
      RS_YYYY-MM.zst  (monthly global dumps — not subreddit-specific)
    """
    patterns = [
        f"r_{subreddit}_{file_type}.zst",
        f"{subreddit}_{file_type}.zst",
        f"r_{subreddit.lower()}_{file_type}.zst",
        f"{subreddit.lower()}_{file_type}.zst",
    ]
    found = []
    for p in patterns:
        candidate = PUSHSHIFT_DIR / p
        if candidate.exists():
            found.append(candidate)

    if not found:
        logger.warning(f"  No dump file found for r/{subreddit} {file_type} in {PUSHSHIFT_DIR}")
        logger.warning(f"  Looked for: {patterns}")
    return found


# ── Main Collector ────────────────────────────────────────────────────────────

def collect_pushshift_reddit(
    start: str = START_DATE,
    end:   str = END_DATE,
    scorer: str = NLP_SCORER,
) -> pd.DataFrame:
    """
    Load Pushshift dump files for crypto subreddits, score with NLP,
    and produce a daily sentiment DataFrame matching the PRAW Reddit collector schema.

    Parameters
    ----------
    start, end : str  YYYY-MM-DD date range filter
    scorer     : str  "finbert" or "vader"

    Returns
    -------
    pd.DataFrame with columns:
      reddit_sentiment_weighted  — score-weighted daily NLP sentiment
      reddit_sentiment_mean      — simple daily mean NLP sentiment
      reddit_post_count          — total posts/comments that day
    """
    global NLP_SCORER
    NLP_SCORER = scorer

    logger.info("=== Pushshift Reddit Collector ===")
    logger.info(f"  Dump directory: {PUSHSHIFT_DIR}")
    logger.info(f"  Subreddits: {SUBREDDITS}")
    logger.info(f"  Date range: {start} → {end}")
    logger.info(f"  Scorer: {scorer}")

    if not PUSHSHIFT_DIR.exists():
        logger.error(
            f"  PUSHSHIFT_DIR does not exist: {PUSHSHIFT_DIR}\n"
            f"  Set PUSHSHIFT_DIR in config/.env to the folder containing "
            f"your downloaded .zst files."
        )
        return pd.DataFrame()

    start_ts = pd.Timestamp(start).timestamp()
    end_ts   = pd.Timestamp(end).timestamp()

    all_records = []
    seen_paths  = set()   # prevent processing the same .zst file twice

    for subreddit in SUBREDDITS:
        for file_type in FILE_TYPES:
            dump_files = _find_dump_files(subreddit, file_type)
            if not dump_files:
                continue

            for dump_path in dump_files:
                if dump_path.resolve() in seen_paths:
                    logger.info(f"  Skipping {dump_path.name} (already processed)")
                    continue
                seen_paths.add(dump_path.resolve())
                logger.info(f"  Reading {dump_path.name}...")
                count = 0
                skipped_date = 0
                skipped_text = 0

                for obj in _read_zst_lines(dump_path):
                    # Filter by date
                    # Newer Pushshift dumps store created_utc as a string, not int
                    try:
                        created = float(obj.get("created_utc", 0))
                    except (ValueError, TypeError):
                        created = 0.0
                    if not (start_ts <= created <= end_ts):
                        skipped_date += 1
                        continue

                    # Extract text
                    text = _extract_text(obj, file_type)
                    if len(text) < 10:
                        skipped_text += 1
                        continue

                    # Extract engagement score (upvotes/score)
                    engagement = max(int(obj.get("score", 1)), 1)

                    all_records.append({
                        "date":       pd.Timestamp(created, unit="s").normalize(),
                        "subreddit":  subreddit,
                        "text":       text,
                        "score":      engagement,
                        "file_type":  file_type,
                    })
                    count += 1

                    # Progress log every 100k records
                    if count % 100_000 == 0:
                        logger.info(f"    {count:,} records read from {dump_path.name}...")

                logger.info(
                    f"  ✓ r/{subreddit} {file_type}: {count:,} posts "
                    f"(skipped {skipped_date:,} out of range, "
                    f"{skipped_text:,} too short)"
                )

    if not all_records:
        logger.error(
            "  No records collected. Check:\n"
            "  1. PUSHSHIFT_DIR points to the correct folder\n"
            "  2. The .zst files are downloaded and named correctly\n"
            "  3. The date range overlaps with the dump file coverage"
        )
        return pd.DataFrame()

    logger.info(f"  Total records before NLP: {len(all_records):,}")
    logger.info(f"  Running NLP scoring ({scorer})...")

    df_raw = pd.DataFrame(all_records)
    df_raw.loc[:, "date"] = pd.DatetimeIndex(pd.to_datetime(df_raw["date"]))

    # NLP scoring in one batch pass
    t0 = time.time()
    df_raw["nlp_score"] = _score(df_raw["text"].tolist())
    elapsed = time.time() - t0
    logger.info(f"  NLP scoring done in {elapsed/60:.1f} minutes")
    logger.info(
        f"  Score stats: mean={df_raw['nlp_score'].mean():.4f}, "
        f"std={df_raw['nlp_score'].std():.4f}"
    )

    # Aggregate to daily — engagement-weighted mean
    logger.info("  Aggregating to daily...")
    daily = (
        df_raw.groupby("date")[["nlp_score", "score"]]
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

    daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index))
    daily.index.name = "date"

    out = RAW_DIR / "sentiment" / "reddit_sentiment.parquet"
    daily.to_parquet(out)

    logger.success(
        f"Pushshift Reddit → {out}  {daily.shape}\n"
        f"  Date range: {daily.index[0].date()} → {daily.index[-1].date()}\n"
        f"  Total posts scored: {int(daily['reddit_post_count'].sum()):,}\n"
        f"  Mean daily sentiment: {daily['reddit_sentiment_weighted'].mean():.4f}"
    )
    return daily


# ── Validation helper ─────────────────────────────────────────────────────────

def validate_dump_directory():
    """
    Print what .zst files are available and which subreddits have coverage.
    Run this first to confirm your downloads are in the right place.
    """
    print(f"\nDump directory: {PUSHSHIFT_DIR}")
    if not PUSHSHIFT_DIR.exists():
        print("  ✗ Directory does not exist!")
        print(f"  Create it and add your .zst files, then set:")
        print(f"  PUSHSHIFT_DIR={PUSHSHIFT_DIR} in config/.env")
        return

    zst_files = list(PUSHSHIFT_DIR.glob("*.zst"))
    if not zst_files:
        print("  ✗ No .zst files found in directory")
        return

    print(f"  Found {len(zst_files)} .zst files:")
    for f in sorted(zst_files):
        size_mb = f.stat().st_size / 1024**2
        print(f"    {f.name}  ({size_mb:.0f} MB)")

    print(f"\nSubreddit coverage check:")
    for sub in SUBREDDITS:
        for ft in FILE_TYPES:
            files = _find_dump_files(sub, ft)
            status = "✓" if files else "✗ MISSING"
            print(f"  {status}  r/{sub} {ft}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pushshift Reddit NLP Collector")
    parser.add_argument("--validate", action="store_true",
                        help="Check dump directory and file availability only")
    parser.add_argument("--scorer", default=NLP_SCORER,
                        choices=["finbert", "vader"],
                        help="NLP scorer (default: finbert)")
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end",   default=END_DATE)
    args = parser.parse_args()

    if args.validate:
        validate_dump_directory()
    else:
        df = collect_pushshift_reddit(
            start=args.start, end=args.end, scorer=args.scorer
        )
        if not df.empty:
            print(df.describe().round(4))