"""
analysis/validate.py

Post-pipeline validation suite — run on master_panel_final.parquet.

Checks:
  1. Coverage report (% non-NaN per variable)
  2. ADF stationarity tests on return and sentiment series
  3. Outlier detection (|z-score| > 5)
  4. Orthogonality verification (corr(rational, irrational) ≈ 0)
  5. Sentiment decomposition quality (R², coverage of each component)
  6. Cross-source reconciliation (CoinGecko vs yfinance BTC price)

Output: data/processed/validation_report.txt
        Printed to console
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR


def _load_best_panel() -> tuple[pd.DataFrame, str]:
    """Load the most complete available panel."""
    for fname in ["master_panel_final.parquet",
                  "master_panel_features.parquet",
                  "master_panel.parquet"]:
        path = DATA_DIR / "processed" / fname
        if path.exists():
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            return df, fname
    raise FileNotFoundError("No panel found. Run: python main.py --mode full")


# ── Check functions ───────────────────────────────────────

def check_coverage(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    cov = (df.notna().sum() / len(df) * 100).round(2).to_frame("coverage_pct")
    cov["n_obs"]     = df.notna().sum()
    cov["n_missing"] = df.isna().sum()
    cov = cov.sort_values("coverage_pct", ascending=False)
    low = cov[cov["coverage_pct"] < 80]
    lines = ["── COVERAGE ──────────────────────────────────────────────"]
    if low.empty:
        lines.append("  All variables ≥80% coverage ✓")
    else:
        lines.append(f"  Variables <80% ({len(low)}):")
        for col, row in low.iterrows():
            lines.append(f"    {col}: {row['coverage_pct']}%")
    return cov, lines


def check_stationarity(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    from statsmodels.tsa.stattools import adfuller
    target_cols = [c for c in df.columns if any(
        k in c for k in ["return", "sentiment", "corr", "fear_greed_index",
                          "gdelt_avg_tone", "algo_intensity"])]
    rows  = []
    lines = ["", "── STATIONARITY (ADF tests) ──────────────────────────"]
    for col in target_cols:
        s = df[col].dropna()
        if len(s) < 30:
            continue
        try:
            stat, pval, _, _, crit, _ = adfuller(s, autolag="AIC")
            rows.append({"column": col, "adf": round(stat, 3),
                         "p_value": round(pval, 4),
                         "stationary": pval < 0.05, "n": len(s)})
        except Exception:
            pass
    adf_df  = pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()
    if adf_df.empty:
        lines.append("  No series to test (data not yet collected?)")
    else:
        n_stat = adf_df["stationary"].sum()
        lines.append(f"  {n_stat}/{len(adf_df)} series are stationary (p<0.05)")
        non_stat = adf_df[~adf_df["stationary"]]
        if not non_stat.empty:
            lines.append("  Non-stationary (consider differencing for SVAR):")
            for col, row in non_stat.iterrows():
                lines.append(f"    {col}: p={row['p_value']}")
    return adf_df, lines


def check_outliers(df: pd.DataFrame) -> list:
    from scipy import stats
    num  = df.select_dtypes(include=[np.number])
    z    = np.abs(stats.zscore(num, nan_policy="omit"))
    mask = pd.DataFrame(z > 5, columns=num.columns, index=df.index)
    counts = mask.sum().sort_values(ascending=False)
    counts = counts[counts > 0]
    lines  = ["", "── OUTLIERS (|z| > 5) ────────────────────────────────"]
    if counts.empty:
        lines.append("  No extreme outliers detected ✓")
    else:
        lines.append(f"  Total outlier cells: {mask.sum().sum()}")
        for col, n in counts.head(10).items():
            lines.append(f"    {col}: {n} cells")
    return lines


def check_orthogonality(df: pd.DataFrame) -> list:
    lines = ["", "── ORTHOGONALIZATION ─────────────────────────────────"]
    for col in ["sentiment_raw", "rational_sentiment", "irrational_sentiment",
                "crypto_sentiment_composite", "algo_intensity"]:
        if col in df.columns:
            s   = df[col].dropna()
            pct = len(s) / len(df) * 100
            lines.append(f"  {col}: {len(s)} obs ({pct:.1f}%), "
                         f"mean={s.mean():.4f}, std={s.std():.4f}")
        else:
            lines.append(f"  {col}: NOT PRESENT")

    if "rational_sentiment" in df.columns and "irrational_sentiment" in df.columns:
        pair = df[["rational_sentiment", "irrational_sentiment"]].dropna()
        if len(pair) > 10:
            corr = pair["rational_sentiment"].corr(pair["irrational_sentiment"])
            lines.append(f"\n  Orthogonality: corr(rational, irrational) = {corr:.10f}")
            lines.append(f"  {'✓ PASS' if abs(corr) < 0.01 else '⚠ FAIL'} "
                         f"(threshold |corr| < 0.01)")

            rpt = DATA_DIR / "processed" / "orthogonalization_report.txt"
            if rpt.exists():
                r2_line = next((l for l in rpt.read_text().split("\n") if "R²" in l), None)
                if r2_line:
                    lines.append(f"  First-stage {r2_line.strip()}")
    return lines


def check_reconcile(df: pd.DataFrame) -> list:
    lines = ["", "── CROSS-SOURCE RECONCILIATION ───────────────────────"]
    if "bitcoin_price" not in df.columns:
        lines.append("  bitcoin_price not in panel — skipping")
        return lines
    try:
        import yfinance as yf
        raw = yf.download("BTC-USD",
                          start=str(df.index.min().date()),
                          end=str(df.index.max().date()),
                          progress=False, auto_adjust=True)
        yf_price = raw["Close"].squeeze()
        yf_price.index = pd.to_datetime(yf_price.index)
        combined = pd.DataFrame({"cg": df["bitcoin_price"], "yf": yf_price}).dropna()
        if len(combined) < 10:
            lines.append("  Insufficient overlap for reconciliation")
            return lines
        corr     = combined["cg"].corr(combined["yf"])
        pct_diff = ((combined["cg"] - combined["yf"]) / combined["yf"] * 100).abs()
        lines.append(f"  CoinGecko vs yfinance BTC price:")
        lines.append(f"    Overlap days:      {len(combined)}")
        lines.append(f"    Pearson corr:      {corr:.6f}")
        lines.append(f"    Mean abs % diff:   {pct_diff.mean():.4f}%")
        lines.append(f"    Max abs % diff:    {pct_diff.max():.4f}%")
        if corr > 0.999:
            lines.append("    ✓ Sources are highly consistent")
        else:
            lines.append("    ⚠ Notable discrepancy — review data")
    except Exception as e:
        lines.append(f"  Reconciliation skipped: {e}")
    return lines


# ── Master runner ─────────────────────────────────────────

def run_full_validation() -> str:
    logger.info("=== Validation Suite ===")
    df, fname = _load_best_panel()

    header = [
        "=" * 70,
        "DATA VALIDATION REPORT",
        f"Panel:      {fname}",
        f"Shape:      {df.shape}",
        f"Date range: {df.index.min().date()} → {df.index.max().date()}",
        "=" * 70,
    ]

    cov_df, cov_lines  = check_coverage(df)
    _,      adf_lines  = check_stationarity(df)
    out_lines          = check_outliers(df)
    orth_lines         = check_orthogonality(df)
    rec_lines          = check_reconcile(df)

    # Save coverage as CSV
    cov_path = DATA_DIR / "processed" / "coverage_report.csv"
    cov_df.to_csv(cov_path)

    all_lines = header + cov_lines + adf_lines + out_lines + orth_lines + rec_lines
    report    = "\n".join(all_lines)

    out_path = DATA_DIR / "processed" / "validation_report.txt"
    out_path.write_text(report)
    print(report)
    logger.success(f"Validation report → {out_path}")
    return report


if __name__ == "__main__":
    run_full_validation()
