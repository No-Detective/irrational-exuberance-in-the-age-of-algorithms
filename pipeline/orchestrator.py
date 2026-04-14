"""
pipeline/orchestrator.py

Central runner. Manages:
  - SQLite run log with per-collector status and deduplication
  - Full historical collection (Steps 1–5)
  - Daily incremental updates
  - APScheduler background job (07:00 UTC)
  - Rich progress display

Pipeline order:
  Step 1  Collectors → raw Parquet shards
  Step 2  Transformer → master_panel.parquet
  Step 3  Feature engineering → master_panel_features.parquet
  Step 4  Orthogonalization → master_panel_final.parquet
  Step 5  Validation report
"""
import sys
import sqlite3
import datetime
from pathlib import Path
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, LOG_DIR, DB_PATH, START_DATE, END_DATE

console = Console()

# ── SQLite run log ────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            collector   TEXT    NOT NULL,
            start_date  TEXT,
            end_date    TEXT,
            status      TEXT,
            rows        INTEGER DEFAULT 0,
            output_path TEXT,
            error_msg   TEXT,
            run_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def log_run(collector: str, status: str, rows: int = 0,
            output_path: str = "", error_msg: str = ""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO run_log (collector,start_date,end_date,status,rows,output_path,error_msg) "
        "VALUES (?,?,?,?,?,?,?)",
        (collector, START_DATE, END_DATE, status, rows, output_path, error_msg),
    )
    conn.commit()
    conn.close()


def already_ran_today(name: str) -> bool:
    today = datetime.date.today().isoformat()
    conn  = sqlite3.connect(DB_PATH)
    row   = conn.execute(
        "SELECT id FROM run_log WHERE collector=? AND status='success' "
        "AND DATE(run_at)=? LIMIT 1", (name, today)).fetchone()
    conn.close()
    return row is not None


def run_step(name: str, fn, force: bool = False) -> bool:
    if not force and already_ran_today(name):
        logger.info(f"  ↷ {name} already ran today — skipping")
        return True
    logger.info(f"▶ {name}")
    try:
        result = fn()
        rows   = len(result) if hasattr(result, "__len__") else 0
        log_run(name, "success", rows=rows)
        logger.success(f"  ✓ {name} ({rows} rows)")
        return True
    except Exception as e:
        logger.error(f"  ✗ {name}: {e}")
        log_run(name, "failed", error_msg=str(e))
        return False


# ── Full pipeline ─────────────────────────────────────────

def run_full_collection(start: str = START_DATE, end: str = END_DATE, force: bool = False):
    """
    Run all collectors, then the three-step pipeline.
    Order is fixed — pipeline steps depend on earlier outputs.
    """
    init_db()
    logger.info(f"=== Full collection: {start} → {end} ===")

    # Lazy imports (avoid circular import at module level)
    from collectors.equity_collector     import collect_equity, collect_baker_wurgler, collect_epu
    from collectors.bgeometrics_collector import collect_bgeometrics
    from collectors.crypto_collector     import (collect_crypto_prices, collect_coinmetrics,
                                                  collect_blockchain_stats,
                                                  collect_defillama_stablecoin, collect_coinglass)
    from collectors.sentiment_collector  import (collect_stocktwits, collect_reddit,
                                                  collect_fear_greed, collect_google_trends,
                                                  collect_gdelt)
    from collectors.macro_collector       import collect_macro
    from collectors.regulatory_collector  import collect_regulatory_events
    from pipeline.transformer            import build_master_panel
    from pipeline.feature_engineering   import run_feature_engineering
    from pipeline.orthogonalization      import run_orthogonalization

    # ── Step 1: Data collection ───────────────────────────
    collector_steps = [
        # Equity
        ("equity_prices",       lambda: collect_equity(start, end)),
        ("baker_wurgler",       collect_baker_wurgler),
        ("epu_index",           collect_epu),
        # Crypto prices
        ("crypto_prices",       lambda: collect_crypto_prices(start, end)),
        # On-chain
        ("coinmetrics_onchain", lambda: collect_coinmetrics(start, end)),
        ("blockchain_stats",    lambda: collect_blockchain_stats(start, end)),
        ("bgeometrics_onchain", lambda: collect_bgeometrics(start, end)),
        ("defillama_stablecoin",lambda: collect_defillama_stablecoin(start, end)),
        # Futures
        ("coinglass_futures",   lambda: collect_coinglass(start, end)),
        # Sentiment — StockTwits (incremental; ~30 days per run)
        ("stocktwits",          lambda: collect_stocktwits(days_back=30, pages_per_symbol=15, use_finbert=True)),
        # Sentiment — Reddit (uncomment once REDDIT_CLIENT_ID is set)
        # ("reddit",            lambda: collect_reddit(start, end)),
        # Sentiment — no-key sources
        ("fear_greed",          lambda: collect_fear_greed(start, end)),
        ("google_trends",       lambda: collect_google_trends(start=start, end=end)),
        ("gdelt_news",          lambda: collect_gdelt(start=start, end=end)),
        # Macro
        ("fred_macro",          lambda: collect_macro(start, end)),
        # Regulatory events — fetched from SEC EDGAR + CFTC + LW Tracker
        ("regulatory_events",   lambda: collect_regulatory_events(start, end)),
    ]

    results = {}
    with Progress(SpinnerColumn(),
                  TextColumn("[progress.description]{task.description}"),
                  BarColumn(), console=console) as prog:
        task = prog.add_task("Collecting...", total=len(collector_steps))
        for name, fn in collector_steps:
            ok = run_step(name, fn, force=force)
            results[name] = "✓" if ok else "✗"
            prog.advance(task)

    # ── Step 2: Merge raw shards ──────────────────────────
    logger.info("=== Step 2/4: Building master panel ===")
    try:
        panel = build_master_panel(start, end)
        log_run("master_panel", "success", rows=len(panel),
                output_path=str(DATA_DIR / "processed" / "master_panel.parquet"))
        results["master_panel"] = "✓"
    except Exception as e:
        logger.error(f"master_panel failed: {e}")
        log_run("master_panel", "failed", error_msg=str(e))
        results["master_panel"] = "✗"

    # ── Step 3: Feature engineering ───────────────────────
    if results.get("master_panel") == "✓":
        logger.info("=== Step 3/4: Feature engineering ===")
        try:
            run_feature_engineering(start, end)
            log_run("feature_engineering", "success",
                    output_path=str(DATA_DIR / "processed" / "master_panel_features.parquet"))
            results["feature_engineering"] = "✓"
        except Exception as e:
            logger.error(f"feature_engineering failed: {e}")
            log_run("feature_engineering", "failed", error_msg=str(e))
            results["feature_engineering"] = "✗"
    else:
        results["feature_engineering"] = "↷ skipped"

    # ── Step 4: Orthogonalization ─────────────────────────
    if results.get("feature_engineering") == "✓":
        logger.info("=== Step 4/4: Orthogonalization ===")
        try:
            run_orthogonalization(start, end)
            log_run("orthogonalization", "success",
                    output_path=str(DATA_DIR / "processed" / "master_panel_final.parquet"))
            results["orthogonalization"] = "✓"
        except Exception as e:
            logger.error(f"orthogonalization failed: {e}")
            log_run("orthogonalization", "failed", error_msg=str(e))
            results["orthogonalization"] = "✗"
    else:
        results["orthogonalization"] = "↷ skipped"

    # ── Summary table ─────────────────────────────────────
    table = Table(title="Pipeline Summary", show_header=True)
    table.add_column("Step", style="cyan")
    table.add_column("Status")
    for name, status in results.items():
        color = "green" if status == "✓" else ("yellow" if "skipped" in status else "red")
        table.add_row(name, f"[{color}]{status}[/{color}]")
    console.print(table)


# ── Incremental update ────────────────────────────────────

def run_incremental_update():
    """Collect yesterday's data and re-run all pipeline steps."""
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    today     = datetime.date.today().isoformat()
    logger.info(f"=== Incremental update: {yesterday} → {today} ===")
    run_full_collection(start=yesterday, end=today, force=True)


# ── Daily scheduler ───────────────────────────────────────

def start_scheduler():
    """Run incremental updates every day at 07:00 UTC."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(run_incremental_update, "cron", hour=7, minute=0)
    logger.info("Scheduler started — daily updates at 07:00 UTC")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")