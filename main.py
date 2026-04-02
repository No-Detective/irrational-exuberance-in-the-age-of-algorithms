"""
main.py — Entry point for the crypto-equity sentiment research pipeline.

Usage:
  python main.py                                    # full run (2018–2025)
  python main.py --mode full --start 2020-01-01     # custom date range
  python main.py --mode incremental                 # yesterday's data only
  python main.py --mode features                    # re-run PCA + algo intensity
  python main.py --mode orthogonalize               # re-run rational/irrational split
  python main.py --mode validate                    # validate existing data
  python main.py --mode schedule                    # start daily 07:00 UTC cron
  python main.py --mode status                      # print recent run log
  python main.py --force                            # skip deduplication check
"""
import sys
import argparse
import sqlite3
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import DATA_DIR, LOG_DIR, DB_PATH, START_DATE, END_DATE

# Configure file logging
LOG_FILE = LOG_DIR / "pipeline_{time:YYYY-MM-DD}.log"
logger.add(str(LOG_FILE), rotation="1 day", retention="30 days", level="INFO")


def print_status():
    if not DB_PATH.exists():
        print("No run log found. Run the pipeline first.")
        return
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(
        "SELECT collector, status, rows, run_at, error_msg "
        "FROM run_log ORDER BY run_at DESC LIMIT 60", conn)
    conn.close()
    if df.empty:
        print("No runs recorded yet.")
    else:
        print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Crypto-equity sentiment research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "incremental", "features", "orthogonalize",
                 "validate", "schedule", "status"],
        default="full",
    )
    parser.add_argument("--start",  default=START_DATE, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=END_DATE,   help="End date   YYYY-MM-DD")
    parser.add_argument("--force",  action="store_true",
                        help="Re-run collectors even if they ran today")
    args = parser.parse_args()

    if args.mode == "status":
        print_status()

    elif args.mode == "validate":
        from analysis.validate import run_full_validation
        run_full_validation()

    elif args.mode == "features":
        from pipeline.feature_engineering import run_feature_engineering
        run_feature_engineering(args.start, args.end)

    elif args.mode == "orthogonalize":
        from pipeline.orthogonalization import run_orthogonalization
        run_orthogonalization(args.start, args.end)

    elif args.mode == "incremental":
        from pipeline.orchestrator import run_incremental_update
        run_incremental_update()

    elif args.mode == "schedule":
        from pipeline.orchestrator import start_scheduler
        start_scheduler()

    else:  # full
        from pipeline.orchestrator import run_full_collection
        run_full_collection(args.start, args.end, force=args.force)


if __name__ == "__main__":
    main()
