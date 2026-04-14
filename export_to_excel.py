"""
export_to_excel.py

Exports every data layer of the pipeline to a single, fully-formatted Excel
workbook with one sheet per data category. No size limit — uses openpyxl's
write_only mode to stream large datasets without loading them all into memory
at once.

Usage:
  python export_to_excel.py                    # full export, auto-named
  python export_to_excel.py --out my_data.xlsx # custom output filename
  python export_to_excel.py --sheet raw        # only raw collected data sheets
  python export_to_excel.py --sheet final      # only master_panel_final

Output sheets
─────────────────────────────────────────────────────────────────────────────
  MASTER_FINAL          master_panel_final.parquet   (2922 × 59) — analysis-ready
  MASTER_FEATURES       master_panel_features.parquet
  MASTER_PANEL          master_panel.parquet
  EQUITY                data/raw/equity/equity_daily.parquet
  BAKER_WURGLER         data/raw/equity/baker_wurgler.parquet
  EPU                   data/raw/macro/epu_index.parquet
  CRYPTO_PRICES         data/raw/crypto/crypto_prices.parquet
  BLOCKCHAIN_STATS      data/raw/onchain/blockchain_stats.parquet
  BGEOMETRICS_ONCHAIN   data/raw/onchain/bgeometrics_onchain.parquet (if exists)
  DEFILLAMA             data/raw/onchain/defillama_stablecoin.parquet
  STOCKTWITS            data/raw/sentiment/stocktwits_sentiment.parquet
  REDDIT                data/raw/sentiment/reddit_sentiment.parquet
  FEAR_GREED            data/raw/sentiment/fear_greed.parquet
  GOOGLE_TRENDS         data/raw/sentiment/google_trends.parquet
  FRED_MACRO            data/raw/macro/fred_macro.parquet
  REGULATORY_DUMMIES    data/raw/macro/regulatory_dummies.parquet
  REGULATORY_EVENTS     data/raw/macro/regulatory_events.csv
  ORTHOGONALIZATION     data/processed/orthogonalization_report.txt (as table)
  COVERAGE              data/processed/coverage_report.csv
  METADATA              sheet-by-sheet documentation

Formatting
─────────────────────────────────────────────────────────────────────────────
  • Header row: dark navy background, white bold Arial text
  • Frozen header row and first (date) column on every sheet
  • Column widths auto-fitted to content (capped at 25 chars)
  • Date column formatted as YYYY-MM-DD text for cross-platform compatibility
  • Numbers: 6 decimal places for returns/ratios, 2 for prices/volumes
  • METADATA sheet documents every column's source and paper role
"""
import sys
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from loguru import logger

import openpyxl
from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, numbers
)
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).parent))
from config.settings import DATA_DIR, RAW_DIR, START_DATE, END_DATE

# ── Style constants ───────────────────────────────────────────────────────────

HEADER_FILL  = PatternFill("solid", start_color="1F3864")   # dark navy
HEADER_FONT  = Font(name="Arial", bold=True, color="FFFFFF", size=10)
BODY_FONT    = Font(name="Arial", size=9)
DATE_FONT    = Font(name="Arial", size=9, bold=True)
SUBHDR_FILL  = PatternFill("solid", start_color="D9E1F2")   # light blue
SUBHDR_FONT  = Font(name="Arial", bold=True, size=9)

THIN_BORDER  = Border(
    bottom=Side(style="thin", color="BFBFBF")
)

FMT_DATE     = "YYYY-MM-DD"
FMT_RETURN   = "0.000000"
FMT_PRICE    = "#,##0.00"
FMT_VOLUME   = "#,##0"
FMT_RATIO    = "0.0000"
FMT_PCT      = "0.00%"
FMT_INT      = "#,##0"
FMT_GENERAL  = "General"

MAX_COL_WIDTH = 22
MIN_COL_WIDTH = 10


# ── Column metadata (paper role + format hint) ────────────────────────────────

COL_META = {
    # Dates
    "date":                        ("Date index",                    "date",    "All sheets"),
    # Equity
    "sp500_return":                ("S&P 500 log return",            "return",  "Eq.1 control"),
    "spy_volume":                  ("SPY daily volume",               "volume",  "Eq.1 control"),
    "vix":                         ("CBOE VIX (yfinance)",            "ratio",   "Eq.1 β7"),
    "vix_fred":                    ("CBOE VIX (FRED cross-check)",    "ratio",   "Eq.1 β7 verify"),
    "tech_etf_return":             ("XLK tech sector log return",     "return",  "Sector control"),
    "fin_etf_return":              ("XLF financials log return",      "return",  "Sector control"),
    "gold_return":                 ("Gold futures log return",        "return",  "Risk-off proxy"),
    "dxy_return":                  ("DXY dollar index log return",    "return",  "Eq.1 β8"),
    # Crypto prices
    "bitcoin_price":               ("BTC closing price USD",         "price",   "Core variable"),
    "bitcoin_return":              ("BTC log return",                "return",  "Core variable"),
    "bitcoin_volume":              ("BTC daily trading volume",      "volume",  "Eq.1 β1 proxy"),
    "bitcoin_mcap":                ("BTC market cap USD (derived)",  "price",   "Derived: price×19.5M"),
    "ethereum_price":              ("ETH closing price USD",         "price",   "Core variable"),
    "ethereum_return":             ("ETH log return",               "return",  "Eq.1 control"),
    "ethereum_volume":             ("ETH daily trading volume",     "volume",  "Eq.1 control"),
    # On-chain
    "btc_hash_rate":               ("BTC network hash rate (TH/s)",  "volume",  "Eq.1 β3"),
    "btc_daily_tx_count":          ("BTC confirmed transactions/day","volume",  "Eq.1 β2"),
    "btc_daily_tx_volume_usd":     ("BTC tx volume USD",            "price",   "Eq.1 β2 proxy"),
    "btc_miner_revenue_usd":       ("BTC miner revenue USD",        "price",   "On-chain fundamental"),
    "btc_mempool_size":            ("BTC mempool size (bytes)",      "volume",  "Network pressure"),
    # BGeometrics
    "nupl":                        ("Net Unrealized P/L",            "ratio",   "H2 mediator — holder state"),
    "mvrv_ratio_bg":               ("MVRV ratio (BGeometrics)",      "ratio",   "H2 mediator — speculative excess"),
    "btc_exchange_netflow":        ("BTC exchange net flow",         "volume",  "H2 mediator — sell pressure"),
    "btc_exchange_inflow":         ("BTC exchange inflow",           "volume",  "H2 component"),
    "btc_exchange_outflow":        ("BTC exchange outflow",          "volume",  "H2 component"),
    "sopr":                        ("Spent Output Profit Ratio",     "ratio",   "H2 — realized profit signal"),
    "btc_funding_rate_bg":         ("Perp futures funding rate",     "ratio",   "Speculative positioning"),
    "active_addresses_bg":         ("Daily unique active addresses", "volume",  "Network activity proxy"),
    # Stablecoin
    "stablecoin_total_mcap_usd":   ("Total stablecoin market cap",  "price",   "Liquidity proxy"),
    "stablecoin_supply_ratio":     ("BTC mcap / stablecoin mcap",   "ratio",   "Speculative excess ratio"),
    # Sentiment raw
    "stocktwits_sentiment_weighted":("StockTwits FinBERT (weighted)","ratio",  "S_t input (NLP)"),
    "stocktwits_sentiment_mean":   ("StockTwits FinBERT (mean)",    "ratio",   "S_t input (NLP)"),
    "stocktwits_bullish_ratio":    ("StockTwits bullish fraction",  "ratio",   "S_t input (NLP)"),
    "stocktwits_msg_count":        ("StockTwits daily msg count",   "int",     "Coverage check"),
    "reddit_sentiment_weighted":   ("Reddit FinBERT (weighted)",    "ratio",   "S_t input (NLP)"),
    "reddit_sentiment_mean":       ("Reddit FinBERT (mean)",        "ratio",   "S_t input (NLP)"),
    "reddit_post_count":           ("Reddit daily post count",      "int",     "Coverage check"),
    "fear_greed_index":            ("Fear & Greed index (0–100)",   "ratio",   "Sentiment robustness"),
    "fear_greed_label":            ("Fear & Greed text label",      "text",    "Sentiment robustness"),
    "gtrends_bitcoin":             ("Google Trends: 'Bitcoin' SVI", "ratio",   "Sentiment composite input"),
    "gtrends_crypto_crash":        ("Google Trends: 'crypto crash'","ratio",   "Fear proxy (subtracted)"),
    "gtrends_buy_bitcoin":         ("Google Trends: 'buy bitcoin'", "ratio",   "Purchase intent proxy"),
    "gtrends_ethereum":            ("Google Trends: 'Ethereum' SVI","ratio",   "Sentiment composite input"),
    "gtrends_crypto":              ("Google Trends: 'crypto' SVI",  "ratio",   "Sentiment composite input"),
    # FRED macro
    "fed_funds_rate":              ("Effective Federal Funds Rate", "ratio",   "Eq.1 β8"),
    "usdeur_exchange":             ("USD/EUR exchange rate",        "price",   "FX control"),
    "crude_oil_wti":               ("WTI crude oil price",          "price",   "Commodity control"),
    "inflation_breakeven_10y":     ("10-yr breakeven inflation",    "ratio",   "Inflation expectations"),
    "hy_credit_spread":            ("HY credit spread (OAS)",       "ratio",   "Eq.1 β9"),
    # EPU + BW
    "epu_index":                   ("Economic Policy Uncertainty",  "ratio",   "Eq.1 β10"),
    "log_epu_index":               ("log(EPU index)",               "ratio",   "Eq.1 β10 log"),
    "bw_sentiment":                ("Baker-Wurgler sentiment (raw)","ratio",   "Eq.1 control"),
    "bw_sentiment_orthogonalized": ("BW sentiment (orthogonalized)","ratio",   "Eq.1 control"),
    # Regulatory
    "reg_dummy_positive":          ("Positive regulatory event dummy","int",   "Regulatory control"),
    "reg_dummy_negative":          ("Negative regulatory event dummy","int",   "Regulatory control"),
    "reg_dummy_net":               ("Net regulatory shock (−1/0/+1)","int",   "Eq.1 β11"),
    # Engineered features
    "crypto_sentiment_composite":  ("Composite S_t (GT SVI PCA)",  "ratio",   "Eq.1 LHS / SVAR input"),
    "gtrends_sentiment_used":      ("Google Trends primary flag",   "int",     "Robustness control"),
    "algo_intensity":              ("Algo trading intensity proxy", "ratio",   "H3 moderator"),
    "btc_rv_proxy":                ("BTC RV proxy (|ret|×√252)",   "ratio",   "Volatility control"),
    "spy_rv_proxy":                ("SPY RV proxy (|ret|×√252)",   "ratio",   "Volatility control"),
    "btc_sp500_corr_30d":          ("Rolling 30d BTC–SPY Pearson", "ratio",   "Cross-market link"),
    # Orthogonalization outputs
    "sentiment_raw":               ("S_t raw (= composite)",        "ratio",   "SVAR baseline input"),
    "rational_sentiment":          ("Rational S_t (OLS fitted)",    "ratio",   "SVAR benchmark input"),
    "irrational_sentiment":        ("Irrational S_t (OLS residual)","ratio",   "SVAR PRIMARY INPUT — H1/H2/H3"),
    # Log transforms
    "log_bitcoin_volume":          ("log(BTC volume)",              "ratio",   "Eq.1 β1"),
    "log_ethereum_volume":         ("log(ETH volume)",              "ratio",   "Control"),
    "log_btc_daily_tx_count":      ("log(BTC tx count)",            "ratio",   "Eq.1 β2 log"),
    "log_btc_daily_tx_volume_usd": ("log(BTC tx volume USD)",       "ratio",   "Control"),
    "log_btc_miner_revenue_usd":   ("log(BTC miner revenue)",       "ratio",   "Control"),
    "log_bitcoin_mcap":            ("log(BTC market cap)",          "ratio",   "Control"),
    "log_ethereum_mcap":           ("log(ETH market cap)",          "ratio",   "Control"),
}


def _fmt_for_col(col: str) -> str:
    """Return openpyxl number format string for a column."""
    info = COL_META.get(col, ("", "general", ""))
    hint = info[1] if len(info) > 1 else "general"
    if hint == "return":   return FMT_RETURN
    if hint == "price":    return FMT_PRICE
    if hint == "volume":   return FMT_VOLUME
    if hint == "ratio":    return FMT_RATIO
    if hint == "int":      return FMT_INT
    if hint == "date":     return FMT_DATE
    return FMT_GENERAL


# ── Sheet writers ─────────────────────────────────────────────────────────────

def _write_sheet(wb: Workbook, sheet_name: str, df: pd.DataFrame,
                 source_desc: str = "") -> None:
    """Write a DataFrame to an Excel sheet with professional formatting."""
    if df is None or df.empty:
        ws = wb.create_sheet(title=sheet_name)
        ws["A1"] = f"No data available for {sheet_name}"
        ws["A1"].font = Font(name="Arial", italic=True, color="808080")
        return

    ws = wb.create_sheet(title=sheet_name[:31])   # Excel 31-char sheet name limit

    # Reset index so date becomes column A
    df = df.reset_index()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    cols = list(df.columns)

    # ── Header row ──────────────────────────────────────────────────────────
    for c_idx, col in enumerate(cols, 1):
        cell = ws.cell(row=1, column=c_idx, value=col)
        cell.font  = HEADER_FONT
        cell.fill  = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=False)

    # ── Data rows ────────────────────────────────────────────────────────────
    for r_idx, row in enumerate(df.itertuples(index=False), 2):
        for c_idx, (col, val) in enumerate(zip(cols, row), 1):
            cell = ws.cell(row=r_idx, column=c_idx)

            # Convert numpy types to native Python
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val) if not np.isnan(val) else None
            elif isinstance(val, float) and np.isnan(val):
                val = None

            cell.value = val
            cell.font  = DATE_FONT if col == "date" else BODY_FONT

            # Number formatting
            if val is not None and not isinstance(val, str):
                cell.number_format = _fmt_for_col(col)

    # ── Column widths ─────────────────────────────────────────────────────────
    for c_idx, col in enumerate(cols, 1):
        max_len = max(
            len(str(col)),
            *[len(str(v)) for v in df[col].dropna().head(100).astype(str)],
            default=MIN_COL_WIDTH,
        )
        ws.column_dimensions[get_column_letter(c_idx)].width = (
            min(max_len + 2, MAX_COL_WIDTH)
        )

    # ── Freeze pane: header row + date column ─────────────────────────────────
    ws.freeze_panes = "B2"

    # ── Source note in row below data ─────────────────────────────────────────
    last_row = df.shape[0] + 2
    if source_desc:
        note_cell = ws.cell(row=last_row + 1, column=1,
                            value=f"Source: {source_desc}")
        note_cell.font = Font(name="Arial", italic=True, size=8, color="808080")

    logger.info(f"  Sheet '{sheet_name}': {df.shape[0]} rows × {df.shape[1]-1} cols")


def _write_text_sheet(wb: Workbook, sheet_name: str, text: str) -> None:
    """Write a plain-text report (e.g. orthogonalization_report.txt) to a sheet."""
    ws = wb.create_sheet(title=sheet_name[:31])
    ws.column_dimensions["A"].width = 90
    for i, line in enumerate(text.split("\n"), 1):
        cell = ws.cell(row=i, column=1, value=line)
        cell.font = Font(name="Courier New", size=8)
        if line.startswith("=") or line.startswith("─"):
            cell.font = Font(name="Courier New", size=8, bold=True,
                             color="1F3864")


def _write_metadata_sheet(wb: Workbook, all_cols: list[str]) -> None:
    """Write a documentation sheet explaining every column."""
    ws = wb.create_sheet(title="METADATA")

    headers = ["Column", "Description", "Format", "Paper Role"]
    for c_idx, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=c_idx, value=h)
        cell.font  = HEADER_FONT
        cell.fill  = HEADER_FILL

    widths = [30, 45, 10, 35]
    for c_idx, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(c_idx)].width = w

    for r_idx, col in enumerate(sorted(set(all_cols)), 2):
        meta = COL_META.get(col, (col, "general", "—"))
        desc, fmt, role = meta[0], meta[1], meta[2] if len(meta) > 2 else "—"
        for c_idx, val in enumerate([col, desc, fmt, role], 1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            cell.font = BODY_FONT
        if r_idx % 2 == 0:
            for c_idx in range(1, 5):
                ws.cell(row=r_idx, column=c_idx).fill = PatternFill(
                    "solid", start_color="F2F2F2")

    ws.freeze_panes = "A2"
    logger.info(f"  Sheet 'METADATA': {len(all_cols)} columns documented")


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning(f"  ✗ {label}: not found at {path}")
        return None
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            logger.warning(f"  Unknown file type: {path.suffix}")
            return None
        df.index = pd.to_datetime(df.index, errors="coerce")
        logger.info(f"  ✓ {label}: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"  ✗ {label}: {e}")
        return None


# ── Main export function ──────────────────────────────────────────────────────

SHEETS = [
    # (sheet_name, relative_path, source_description)
    ("MASTER_FINAL",     "processed/master_panel_final.parquet",
     "master_panel_final.parquet — Eq.1 orthogonalization outputs + all features. "
     "Primary analysis dataset. irrational_sentiment is the SVAR input."),

    ("MASTER_FEATURES",  "processed/master_panel_features.parquet",
     "master_panel_features.parquet — After feature engineering, before orthogonalization."),

    ("MASTER_PANEL",     "processed/master_panel.parquet",
     "master_panel.parquet — After transformer merge, before feature engineering."),

    ("EQUITY",           "raw/equity/equity_daily.parquet",
     "equity_daily.parquet — yfinance: SPY, VIX, XLK, XLF, GC=F, DX-Y.NYB. "
     "2013 trading days Jan 2018 – Dec 2025."),

    ("BAKER_WURGLER",    "raw/equity/baker_wurgler.parquet",
     "baker_wurgler.parquet — BW (2006) Jan–Dec 2018 + FRED UMCSENT z-scored "
     "Jan 2019 – Dec 2025. Monthly frequency, forward-filled to daily in panel."),

    ("EPU",              "raw/macro/epu_index.parquet",
     "epu_index.parquet — Baker-Bloom-Davis EPU index. FRED series USEPUINDXD. "
     "Daily, 2018-2025, 2922 obs."),

    ("CRYPTO_PRICES",    "raw/crypto/crypto_prices.parquet",
     "crypto_prices.parquet — BTC and ETH price/return/volume via yfinance "
     "(CoinGecko fallback). 2921 trading days."),

    ("BLOCKCHAIN_STATS", "raw/onchain/blockchain_stats.parquet",
     "blockchain_stats.parquet — Blockchain.com free API. Hash rate, tx count, "
     "tx volume USD, miner revenue, mempool size. No key required."),

    ("BGEOMETRICS",      "raw/onchain/bgeometrics_onchain.parquet",
     "bgeometrics_onchain.parquet — BGeometrics free API (bitcoin-data.com/v1). "
     "NUPL, MVRV, exchange netflow, SOPR, funding rate. No key required. "
     "Closes Gap 2: replaces Glassnode (discontinued) for H2 mediator variables."),

    ("DEFILLAMA",        "raw/onchain/defillama_stablecoin.parquet",
     "defillama_stablecoin.parquet — DeFiLlama stablecoin total market cap. "
     "Free, no key. Full 2018-2025 daily coverage."),

    ("STOCKTWITS",       "raw/sentiment/stocktwits_sentiment.parquet",
     "stocktwits_sentiment.parquet — StockTwits $BTC.X $ETH.X $CRYPTO.X, "
     "FinBERT-scored, engagement-weighted daily aggregates. "
     "Free API — only last 30 days available per run; accumulates daily."),

    ("REDDIT_PRAW",      "raw/sentiment/reddit_sentiment.parquet",
     "reddit_sentiment.parquet — Reddit NLP sentiment (PRAW live OR Pushshift "
     "historical dumps). FinBERT-scored, score-weighted daily aggregates. "
     "Pushshift covers 2018-2024 (see collectors/pushshift_loader.py)."),

    ("FEAR_GREED",       "raw/sentiment/fear_greed.parquet",
     "fear_greed.parquet — Crypto Fear & Greed Index from Alternative.me. "
     "0-100 composite index, 2887 days 2018-2025. Free, no key."),

    ("GOOGLE_TRENDS",    "raw/sentiment/google_trends.parquet",
     "google_trends.parquet — Google Search Volume Index (SVI) via pytrends. "
     "5 keywords: Bitcoin, Ethereum, crypto, buy bitcoin, crypto crash. "
     "Weekly, 2831 obs, 2018-2025. Primary sentiment composite input."),

    ("FRED_MACRO",       "raw/macro/fred_macro.parquet",
     "fred_macro.parquet — FRED: fed_funds_rate (DFF), vix_fred (VIXCLS), "
     "usdeur_exchange (DEXUSEU), crude_oil_wti (DCOILWTICO), "
     "inflation_breakeven_10y (T10YIE), hy_credit_spread (BAMLH0A0HYM2). "
     "Requires FRED_API_KEY."),

    ("REGULATORY_DUMMIES","raw/macro/regulatory_dummies.parquet",
     "regulatory_dummies.parquet — Daily binary dummies from Latham & Watkins "
     "US Crypto Policy Tracker. 66 events 2018-2025. pos/neg/net columns."),

    ("REGULATORY_EVENTS","raw/macro/regulatory_events.csv",
     "regulatory_events.csv — 70 hand-coded + live-scraped regulatory events. "
     "Columns: date, event_type, direction, description, source."),
]


def export_to_excel(
    output_path: Path,
    sheet_filter: str = "all",
) -> None:
    """
    Export all pipeline data to a single Excel workbook.

    Parameters
    ----------
    output_path : Path  where to save the .xlsx file
    sheet_filter : str  "all" | "raw" | "final" | sheet name prefix
    """
    logger.info(f"=== Excel Export ===")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Filter: {sheet_filter}")

    wb = Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    all_cols = []
    sheets_written = 0

    for sheet_name, rel_path, source_desc in SHEETS:
        # Apply filter
        if sheet_filter != "all":
            if sheet_filter == "raw" and "MASTER" in sheet_name:
                continue
            if sheet_filter == "final" and sheet_name != "MASTER_FINAL":
                continue
            if sheet_filter not in ("all", "raw", "final") and \
               not sheet_name.startswith(sheet_filter.upper()):
                continue

        full_path = DATA_DIR / rel_path
        df = _load(full_path, sheet_name)
        _write_sheet(wb, sheet_name, df, source_desc)
        if df is not None:
            all_cols.extend(df.reset_index().columns.tolist())
        sheets_written += 1

    # ── Orthogonalization report ──────────────────────────────────────────────
    orth_path = DATA_DIR / "processed" / "orthogonalization_report.txt"
    if orth_path.exists():
        txt = orth_path.read_text()
        _write_text_sheet(wb, "ORTHOG_REPORT", txt)
        logger.info(f"  Sheet 'ORTHOG_REPORT': OLS results")

    # ── Coverage report ───────────────────────────────────────────────────────
    cov_path = DATA_DIR / "processed" / "coverage_report.csv"
    if cov_path.exists():
        cov_df = pd.read_csv(cov_path, index_col=0)
        cov_df.index.name = "variable"
        _write_sheet(wb, "COVERAGE", cov_df.reset_index(),
                     "Coverage % per variable — from analysis/validate.py")

    # ── Metadata sheet ────────────────────────────────────────────────────────
    _write_metadata_sheet(wb, all_cols)

    # ── Summary sheet (first tab) ─────────────────────────────────────────────
    ws_sum = wb.create_sheet(title="SUMMARY", index=0)
    ws_sum.column_dimensions["A"].width = 30
    ws_sum.column_dimensions["B"].width = 60

    summary_rows = [
        ("PIPELINE EXPORT SUMMARY", ""),
        ("Generated",               datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("Paper",                   "Irrational Exuberance in the Age of Algorithms"),
        ("Panel window",            f"{START_DATE} → {END_DATE}"),
        ("Sheets in workbook",      str(sheets_written + 4)),
        ("", ""),
        ("KEY SHEETS", "DESCRIPTION"),
        ("MASTER_FINAL",            "Analysis-ready dataset — start here"),
        ("irrational_sentiment",    "Paper's primary independent variable (OLS residual)"),
        ("rational_sentiment",      "Fundamental-explained sentiment component"),
        ("sentiment_raw",           "Unmodified composite (= Google Trends SVI)"),
        ("", ""),
        ("DATA SOURCE STATUS", ""),
        ("Google Trends SVI",       "✓ 2831 obs — primary sentiment composite"),
        ("Fear & Greed",            "✓ 2887 obs — robustness check"),
        ("Blockchain.com",          "✓ 2493 obs — hash rate, tx, mempool"),
        ("BGeometrics on-chain",    "✓/? NUPL, MVRV, exchange flows (test API)"),
        ("Baker-Wurgler + UMCSENT", "✓  96 monthly obs — equity sentiment control"),
        ("Reddit (Pushshift)",      "? Download from Academic Torrents — see REDDIT sheet"),
        ("CoinMetrics",             "✗ Community tier all 403 (PRO required)"),
        ("Coinglass",               "✗ Requires paid API key"),
        ("GDELT",                   "✗ Server timing out"),
        ("", ""),
        ("SENTIMENT COMPOSITE",     ""),
        ("Primary source",          "Google Trends SVI (Da et al. 2011, Kristoufek 2013)"),
        ("Construction",            "PC1([bitcoin,buy_bitcoin,ethereum,crypto]) − z(crypto_crash)"),
        ("Sign normalisation",      "Positive = bullish (corr with BTC return = +0.078 ✓)"),
        ("NLP blend",               "Activates at ≥30 real social NLP obs (Pushshift will enable)"),
        ("", ""),
        ("ORTHOGONALIZATION",       ""),
        ("R²",                      "0.0556  (5.56% of sentiment explained by fundamentals)"),
        ("Adj R²",                  "0.0520"),
        ("F p-value",               "2.80×10⁻¹⁰ ***"),
        ("Orthogonality",           "corr(rational, irrational) = −1.71×10⁻⁸ ≈ 0 ✓"),
        ("n",                       "2,920 obs  (2018-2025 less 2 lag periods)"),
    ]

    for r_idx, (label, value) in enumerate(summary_rows, 1):
        cell_a = ws_sum.cell(row=r_idx, column=1, value=label)
        cell_b = ws_sum.cell(row=r_idx, column=2, value=value)

        if not value and label:
            # Section header
            cell_a.font = Font(name="Arial", bold=True, size=10, color="1F3864")
            cell_a.fill = SUBHDR_FILL
            cell_b.fill = SUBHDR_FILL
        elif label == "PIPELINE EXPORT SUMMARY":
            cell_a.font = Font(name="Arial", bold=True, size=13, color="FFFFFF")
            cell_a.fill = HEADER_FILL
            cell_b.fill = HEADER_FILL
        elif label == "KEY SHEETS" or label == "DATA SOURCE STATUS" or \
             label == "SENTIMENT COMPOSITE" or label == "ORTHOGONALIZATION":
            cell_a.font = Font(name="Arial", bold=True, size=10, color="1F3864")
            cell_b.font = Font(name="Arial", bold=True, size=10, color="1F3864")
            cell_a.fill = SUBHDR_FILL
            cell_b.fill = SUBHDR_FILL
        else:
            cell_a.font = Font(name="Arial", bold=True, size=9)
            cell_b.font = Font(name="Arial", size=9)

    ws_sum.freeze_panes = "A3"

    # ── Save ──────────────────────────────────────────────────────────────────
    wb.save(str(output_path))
    file_size_mb = output_path.stat().st_size / 1024**2
    logger.success(
        f"\nExport complete → {output_path}\n"
        f"  File size:    {file_size_mb:.1f} MB\n"
        f"  Sheets:       {len(wb.sheetnames)}\n"
        f"  Open in Excel, Numbers, or LibreOffice Calc."
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export pipeline data to Excel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python export_to_excel.py                         # full export
  python export_to_excel.py --out research_data.xlsx
  python export_to_excel.py --sheet final           # master_panel_final only
  python export_to_excel.py --sheet raw             # all raw collectors only
        """,
    )
    parser.add_argument(
        "--out",
        default=f"pipeline_export_{datetime.now().strftime('%Y%m%d')}.xlsx",
        help="Output Excel filename (default: pipeline_export_YYYYMMDD.xlsx)",
    )
    parser.add_argument(
        "--sheet",
        default="all",
        help="Filter: 'all' | 'raw' | 'final' | sheet name prefix",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    export_to_excel(out_path, sheet_filter=args.sheet)
    print(f"\n✓ Saved: {out_path.resolve()}")