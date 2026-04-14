"""
pipeline/orthogonalization.py

Implements Section 3.3.2: "Distinguishing Irrational from Rational Sentiment"

Stage 1 — OLS regression (Equation 1):
    Crypto_Sentiment_t = α
        + β₁·btc_return_{t-1}       + β₂·eth_return_{t-1}
        + β₃·btc_volatility_{t-1}   + β₄·hash_rate_{t-1}
        + β₅·active_addresses_{t-1} + β₆·tx_count_{t-1}
        + β₇·fed_funds_rate_t        + β₈·dxy_return_t
        + β₉·vix_t                   + β₁₀·gold_return_t
        + β₁₁·reg_dummy_net_t
        + ε_t

Stage 2 — Decomposition:
    rational_sentiment   = fitted values (ŷ)   ← explained by fundamentals
    irrational_sentiment = residuals (ε̂)       ← orthogonal to all fundamentals

By OLS construction, corr(rational, irrational) = 0 to machine precision.

Also produces:
  - Sensitivity analysis: leave-one-out R² table
  - Full regression report: data/processed/orthogonalization_report.txt

Input:  data/processed/master_panel_features.parquet
Output: data/processed/master_panel_final.parquet   ← analysis-ready for SVAR
        data/processed/orthogonalization_report.txt
"""
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, START_DATE, END_DATE


# ── Regressor specification ───────────────────────────────
# Maps regressor name → (panel_column, "lag1" | "current")
# Edit fallback list if your collected columns have different names.

REGRESSORS = {
    # t-1 market variables (lagged to avoid mechanical simultaneity)
    "btc_return_lag1":       ("bitcoin_return",        "lag1"),
    "eth_return_lag1":       ("ethereum_return",       "lag1"),
    "btc_vol_lag1":          ("btc_rv_intraday_ann",   "lag1"),   # or btc_rv_proxy
    "hash_rate_lag1":        ("btc_hash_rate",         "lag1"),
    "active_addr_lag1":      ("active_addresses",      "lag1"),   # Glassnode
    "tx_count_lag1":         ("btc_daily_tx_count",    "lag1"),

    # Contemporaneous macro (known at start of day)
    "fed_funds_rate":        ("fed_funds_rate",        "current"),
    "dxy_return":            ("dxy_return",            "current"),
    "vix":                   ("vix",                   "current"),
    "gold_return":           ("gold_return",            "current"),
    "reg_dummy_net":         ("reg_dummy_net",          "current"),
}

# Column fallbacks in priority order
FALLBACKS = {
    "btc_rv_intraday_ann": ["btc_rv_proxy", "bitcoin_rv"],
    "active_addresses":    ["log_bitcoin_volume", "active_addresses", "nupl"],  # log_bitcoin_volume covers full 2018-2025
    "btc_daily_tx_count":  ["btc_daily_tx_count", "log_bitcoin_volume"],
    "btc_hash_rate":       ["btc_hash_rate", "log_bitcoin_volume"],
}


def _resolve(col: str, df: pd.DataFrame) -> str | None:
    if col in df.columns:
        return col
    for fb in FALLBACKS.get(col, []):
        if fb in df.columns:
            logger.info(f"  fallback: '{col}' → '{fb}'")
            return fb
    return None


def _build_X(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the regressor matrix for Equation 1."""
    cols = {}
    for name, (col, timing) in REGRESSORS.items():
        resolved = _resolve(col, df)
        if resolved is None:
            logger.warning(f"  ✗ '{name}' ({col}) missing — omitted from first stage")
            continue
        cols[name] = df[resolved].shift(1) if timing == "lag1" else df[resolved]
    return pd.DataFrame(cols, index=df.index)


def run_first_stage(df: pd.DataFrame,
                    sentiment_col: str = "crypto_sentiment_composite") -> dict:
    """
    Estimate Equation 1 with Newey-West HAC standard errors.
    Returns fitted values (rational) and residuals (irrational).
    """
    if sentiment_col not in df.columns:
        raise ValueError(f"'{sentiment_col}' not in panel. Run feature_engineering first.")

    y    = df[sentiment_col]
    X    = _build_X(df)
    data = pd.concat([y, X], axis=1).dropna()
    y_c  = data[sentiment_col]
    X_c  = sm.add_constant(data.drop(columns=[sentiment_col]))

    n = len(y_c)
    logger.info(f"  First-stage OLS: n={n}, k={X_c.shape[1]} regressors")
    if n < 60:
        raise ValueError(f"Only {n} complete obs — need ≥60 for reliable estimation.")

    model = sm.OLS(y_c, X_c).fit(
        cov_type="HAC", cov_kwds={"maxlags": 5})  # Newey-West

    fitted    = model.fittedvalues.reindex(df.index)
    residuals = model.resid.reindex(df.index)

    logger.info(f"  R² = {model.rsquared:.4f}, Adj R² = {model.rsquared_adj:.4f}, "
                f"F p-val = {model.f_pvalue:.4e}")
    return {
        "model":      model,
        "y":          y_c,
        "X":          data.drop(columns=[sentiment_col]),
        "fitted":     fitted,
        "residuals":  residuals,
        "r2":         model.rsquared,
        "n":          n,
        "regressors": list(data.drop(columns=[sentiment_col]).columns),
    }


def run_sensitivity(result: dict) -> pd.DataFrame:
    """Leave-one-out sensitivity: drop each regressor and report R² change."""
    logger.info("  Sensitivity analysis (leave-one-out)...")
    y, X = result["y"], result["X"]
    base_resid = result["residuals"].reindex(y.index).dropna()
    rows = []
    for drop in result["regressors"]:
        try:
            Xr   = sm.add_constant(X.drop(columns=[drop]))
            data = pd.concat([y, Xr], axis=1).dropna()
            m    = sm.OLS(data.iloc[:, 0], data.iloc[:, 1:]).fit()
            resid = m.resid.reindex(y.index).dropna()
            rows.append({
                "dropped":             drop,
                "r2":                  round(m.rsquared, 4),
                "r2_change":           round(m.rsquared - result["r2"], 4),
                "corr_with_base_resid":round(resid.corr(base_resid.reindex(resid.index)), 4),
            })
        except Exception as e:
            logger.warning(f"  sensitivity skipped {drop}: {e}")
    return pd.DataFrame(rows).set_index("dropped") if rows else pd.DataFrame()


def _report(result: dict, sensitivity: pd.DataFrame,
            rational: pd.Series, irrational: pd.Series) -> str:
    m = result["model"]
    lines = [
        "=" * 70,
        "ORTHOGONALIZATION REPORT — Equation 1 (Section 3.3.2)",
        "=" * 70,
        f"Sentiment variable : crypto_sentiment_composite",
        f"Observations       : {result['n']}",
        f"R²                 : {result['r2']:.4f}",
        f"Adjusted R²        : {m.rsquared_adj:.4f}",
        f"F-statistic        : {m.fvalue:.2f}  (p = {m.f_pvalue:.4e})",
        f"HAC std errors     : Newey-West, maxlags=5",
        "",
        "── Coefficient Table ──────────────────────────────────────────────",
        m.summary().tables[1].as_text(),
        "",
        "── Decomposition ──────────────────────────────────────────────────",
        f"rational_sentiment  (fitted):   mean={rational.mean():.4f}  "
        f"std={rational.std():.4f}",
        f"irrational_sentiment (resid):   mean={irrational.mean():.6f}  "
        f"std={irrational.std():.4f}",
        f"Orthogonality check: corr(rational, irrational) = "
        f"{rational.corr(irrational):.10f}  (target: ~0)",
        "",
        f"Interpretation: {result['r2']:.1%} of raw sentiment is explained by",
        f"fundamentals (rational_sentiment). The remaining {1-result['r2']:.1%}",
        f"is irrational_sentiment — the paper's primary independent variable.",
        "",
        "── Sensitivity Analysis (leave-one-out) ────────────────────────",
    ]
    if not sensitivity.empty:
        lines.append(sensitivity.to_string())
        lines.append("")
        lines.append("  corr_with_base_resid ≥ 0.95 = residual is stable.")
    lines += ["", "=" * 70]
    return "\n".join(lines)


# ── Master runner ─────────────────────────────────────────

def run_orthogonalization(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Load master_panel_features.parquet, decompose composite sentiment into
    rational + irrational components, save master_panel_final.parquet.

    Final panel contains three sentiment variants for dual-analysis design:
      sentiment_raw          → baseline (comparable to prior literature)
      rational_sentiment     → fitted values from Eq. 1
      irrational_sentiment   → residuals (main variable, orthogonal to fundamentals)
    """
    in_path = DATA_DIR / "processed" / "master_panel_features.parquet"
    if not in_path.exists():
        raise FileNotFoundError("master_panel_features.parquet not found. "
                                "Run feature_engineering first.")

    logger.info("=== Orthogonalization ===")
    df = pd.read_parquet(in_path)
    df.index = pd.to_datetime(df.index)
    logger.info(f"  Loaded: {df.shape}")

    result = run_first_stage(df)

    df["sentiment_raw"]        = df["crypto_sentiment_composite"]
    df["rational_sentiment"]   = result["fitted"].rename("rational_sentiment")
    df["irrational_sentiment"] = result["residuals"].rename("irrational_sentiment")

    # Verify orthogonality
    aligned = df[["rational_sentiment", "irrational_sentiment"]].dropna()
    corr    = aligned["rational_sentiment"].corr(aligned["irrational_sentiment"])
    if abs(corr) > 0.01:
        logger.warning(f"  Orthogonality concern: corr = {corr:.8f}")
    else:
        logger.success(f"  Orthogonality confirmed: corr = {corr:.10f}")

    sensitivity = run_sensitivity(result)

    report_txt = _report(result, sensitivity,
                         df["rational_sentiment"], df["irrational_sentiment"])
    rpt_path   = DATA_DIR / "processed" / "orthogonalization_report.txt"
    rpt_path.write_text(report_txt)
    logger.info(f"  Report → {rpt_path}")

    out = DATA_DIR / "processed" / "master_panel_final.parquet"
    df.to_parquet(out)
    logger.success(
        f"master_panel_final.parquet → {out}  {df.shape}\n"
        f"  sentiment_raw:        {df['sentiment_raw'].notna().sum()} obs\n"
        f"  rational_sentiment:   {df['rational_sentiment'].notna().sum()} obs\n"
        f"  irrational_sentiment: {df['irrational_sentiment'].notna().sum()} obs"
    )
    return df


if __name__ == "__main__":
    df = run_orthogonalization()
    cols = ["sentiment_raw", "rational_sentiment", "irrational_sentiment"]
    print(df[[c for c in cols if c in df.columns]].describe().round(4))
    if "rational_sentiment" in df.columns:
        corr = df["rational_sentiment"].corr(df["irrational_sentiment"])
        print(f"\nOrthogonality: corr(rational, irrational) = {corr:.10f}")