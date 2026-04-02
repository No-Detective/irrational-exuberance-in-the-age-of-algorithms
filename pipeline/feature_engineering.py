"""
pipeline/feature_engineering.py

Implements Section 3.3 feature construction — run AFTER transformer.py.

  1. Composite Sentiment Index  (Section 3.3.1)
       Primary source hierarchy (first available wins):
         Primary: Google Trends SVI — always active, 100% daily 2018-2025.
                   PC1([gtrends_bitcoin, gtrends_buy_bitcoin,
                        gtrends_ethereum, gtrends_crypto])
                   minus z(gtrends_crypto_crash)   ← fear proxy subtracted
                   Academic basis: Da, Engelberg & Gao (2011 RFS),
                   Kristoufek (2013), Urquhart (2018).
         Blend:    Social media NLP averaged in when ≥ 30 real obs available.
         Output → crypto_sentiment_composite
       Sign normalised so positive = bullish (vs BTC return; F&G fallback).
       Flag column gtrends_sentiment_used=1 written for transparency.

  2. Algo Intensity Moderator  (Section 3.3.1, H3)
       algo_intensity = geometric_mean(
           z(crypto_algo_proxy), z(equity_algo_proxy))
       Free proxies used; Kaiko/TAQ columns auto-detected if present.

  3. Realized Volatility  (Section 3.3.1, Andersen et al. 2003)
       Yahoo Finance 1-hour intraday data (best free approximation of 5-min).
       Falls back to |daily_return| × √252 if intraday unavailable.

  4. Regulatory Dummies  (Table 2 — orthogonalization regressors)
       Loaded from collectors/regulatory_events.csv
       → reg_dummy_positive, reg_dummy_negative, reg_dummy_net

  5. Rolling BTC–S&P Correlation  (30-day window)

Input:  data/processed/master_panel.parquet
Output: data/processed/master_panel_features.parquet
"""
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import DATA_DIR, START_DATE, END_DATE


# ── 1. PCA Composite Sentiment ────────────────────────────

def _pca_composite(X_df: pd.DataFrame, name: str = "composite") -> pd.Series:
    """Run PCA, return PC1 as a named Series aligned to X_df.index."""
    X_sc = StandardScaler().fit_transform(X_df)
    pca  = PCA(n_components=1, random_state=42)
    pc1  = pca.fit_transform(X_sc).flatten()
    expl = pca.explained_variance_ratio_[0]
    logger.info(f"  PCA ({name}): PC1 explains {expl:.1%} of variance")
    return pd.Series(pc1, index=X_df.index)


def _sign_normalise(composite: pd.Series, df: pd.DataFrame) -> pd.Series:
    """
    Ensure positive composite = bullish.
    Checks against BTC return: higher sentiment should co-move with higher return.
    Falls back to Fear & Greed if return not available.
    """
    # Primary: correlate with BTC forward return (sign should be positive)
    if "bitcoin_return" in df.columns:
        ret = df["bitcoin_return"].reindex(composite.dropna().index).dropna()
        corr = composite.reindex(ret.index).corr(ret)
        if corr < 0:
            composite = -composite
            logger.info(f"  Sign flipped (BTC return corr was {corr:.3f})")
        else:
            logger.info(f"  Sign OK (BTC return corr = {corr:.3f})")
        return composite
    # Fallback: Fear & Greed Index
    if "fear_greed_index" in df.columns:
        fg = df["fear_greed_index"].reindex(composite.dropna().index).dropna()
        corr = composite.reindex(fg.index).corr(fg)
        if corr < 0:
            composite = -composite
            logger.info(f"  Sign flipped (F&G corr was {corr:.3f})")
        else:
            logger.info(f"  Sign OK (F&G corr = {corr:.3f})")
    return composite


def _build_gtrends_composite(df: pd.DataFrame) -> pd.Series:
    """
    Google Trends SVI composite sentiment.

    Academic basis:
      - Da, Engelberg & Gao (2011, RFS): SVI directly captures retail
        investor attention and predicts next-week returns.
      - Kristoufek (2013, Scientific Reports): Google Trends for "Bitcoin"
        Granger-causes BTC price in both directions.
      - Urquhart (2018, Economics Letters): SVI is a significant driver
        of next-day BTC volatility.

    Construction:
      PC1([gtrends_bitcoin, gtrends_buy_bitcoin, gtrends_ethereum,
           gtrends_crypto])  ← attention/demand proxies (bullish)
      minus z(gtrends_crypto_crash)              ← fear proxy (bearish)

    Weekly SVI values are already forward-filled to daily in the transformer,
    so this composite has 100% daily coverage for the full 2018-2025 window.

    When this path is used, the column 'gtrends_sentiment_used' is set to 1
    throughout the panel for transparency in robustness tables.
    """
    gt_bullish = ["gtrends_bitcoin", "gtrends_buy_bitcoin",
                  "gtrends_ethereum", "gtrends_crypto"]
    gt_fear    = "gtrends_crypto_crash"

    available_bullish = [c for c in gt_bullish if c in df.columns]
    if len(available_bullish) < 2:
        logger.warning("  Google Trends: insufficient bullish columns")
        return pd.Series(np.nan, index=df.index)

    X = df[available_bullish].dropna()
    if len(X) < 30:
        logger.warning("  Google Trends: < 30 obs")
        return pd.Series(np.nan, index=df.index)

    composite = _pca_composite(X, "Google Trends SVI")

    # Subtract fear proxy (z-scored) if available
    if gt_fear in df.columns:
        fear = df[gt_fear].reindex(X.index).dropna()
        fear_z = (fear - fear.mean()) / max(float(fear.std()), 1e-8)
        composite = composite.sub(fear_z, fill_value=0)
        logger.info(f"  Google Trends: subtracted fear proxy ({gt_fear})")

    # Standardise to zero mean / unit variance
    composite = (composite - composite.mean()) / max(float(composite.std()), 1e-8)
    logger.info(
        f"  Google Trends composite: {len(composite)} obs, "
        f"mean={composite.mean():.4f}, std={composite.std():.4f}"
    )
    return composite.reindex(df.index)


def build_sentiment_composite(df: pd.DataFrame) -> tuple:
    """
    Build crypto_sentiment_composite using the best available source.

    Source architecture:
      Primary: Google Trends SVI — always used. Full 2018-2025 daily coverage.
               PC1([bitcoin, buy_bitcoin, ethereum, crypto] SVI) − z(crypto_crash SVI).
               Academically grounded: Da et al. (2011 RFS), Kristoufek (2013),
               Urquhart (2018 Economics Letters).
      Blend:   Social media NLP (StockTwits + Reddit) averaged in when ≥ 30
               real observations are available. Accumulates as you run the
               pipeline daily. Zero StockTwits history → Google Trends only.

    Returns:
      (composite: pd.Series, used_gtrends: bool)
    """
    # ── Primary: Google Trends SVI (full 2018-2025 coverage) ───────────────
    # Google Trends is the primary source: 100% daily coverage, 2018-2025.
    # Academic basis: Da, Engelberg & Gao (2011 RFS), Kristoufek (2013),
    # Urquhart (2018). This is not a fallback — it IS the composite.
    # 
    # Social media NLP (StockTwits/Reddit) is blended IN when available:
    # if ≥ 30 real NLP observations exist, they are averaged with the
    # Google Trends signal to produce a richer composite. This follows
    # the multi-source sentiment literature (e.g. Bollen et al. 2011).
    composite_gt = _build_gtrends_composite(df)
    used_gtrends = True

    if composite_gt.isna().all():
        logger.error("  Google Trends composite is empty — check gtrends_* columns")
        return pd.Series(np.nan, index=df.index, name="crypto_sentiment_composite"), False

    # ── Optional blend: Social media NLP ─────────────────────────────────
    social_cols = {
        "stocktwits_sentiment_weighted": "stocktwits",
        "reddit_sentiment_weighted":     "reddit",
    }
    available = {k: v for k, v in social_cols.items() if k in df.columns}
    real_obs   = 0
    for col in available:
        real_obs = max(real_obs, int((df[col].notna() & (df[col] != 0)).sum()))

    if real_obs >= 30:
        logger.info(
            f"  Blending Google Trends with social NLP "
            f"({real_obs} real NLP obs across {len(available)} source(s))"
        )
        if len(available) == 1:
            col      = list(available.keys())[0]
            nlp_raw  = df[col].dropna()
        else:
            X       = df[list(available.keys())].dropna()
            nlp_raw = _pca_composite(X, "social NLP").reindex(df.index)

        # Standardise NLP signal then average with GT signal (equal weight)
        nlp_z = (nlp_raw - nlp_raw.mean()) / max(float(nlp_raw.std()), 1e-8)
        composite = composite_gt.add(nlp_z, fill_value=0) / 2
        composite = (composite - composite.mean()) / max(float(composite.std()), 1e-8)
        logger.info("  NLP blend applied (GT + NLP, equal weight)")
    else:
        logger.info(
            f"  Google Trends only (social NLP has {real_obs} real obs — "
            f"accumulate StockTwits daily to unlock NLP blending at 30+ obs)"
        )
        composite = composite_gt

    composite = _sign_normalise(composite, df)
    composite.name = "crypto_sentiment_composite"
    return composite, used_gtrends


# ── 2. Algo Intensity Moderator ───────────────────────────

def build_algo_intensity(df: pd.DataFrame) -> pd.Series:
    """
    Geometric mean of z-scored crypto and equity algo trading proxies.

    Auto-detects Kaiko/TAQ columns if present; otherwise uses free proxies:
      - Crypto:  BTC volume × price pct-change as activity proxy
      - Equity:  |SPY return| / VIX (mechanical activity vs implied vol)

    The geometric mean requires strictly positive inputs, so both
    z-scores are shifted by +4σ before the log-mean, then re-standardised.
    """
    # Priority: institutional data first, then free proxy
    crypto_col = next(
        (c for c in ["kaiko_qtr_btc", "quote_to_trade_ratio", "crypto_algo_proxy"]
         if c in df.columns), None)
    equity_col = next(
        (c for c in ["taq_msg_trade_spy", "msg_to_trade_ratio", "equity_algo_proxy"]
         if c in df.columns), None)

    proxies = {}

    # Build crypto proxy if no institutional column
    if crypto_col is None:
        if "bitcoin_volume" in df.columns and "bitcoin_price" in df.columns:
            btc_usd_vol = (df["bitcoin_volume"] * df["bitcoin_price"]).pct_change().abs()
            proxies["crypto_algo_proxy"] = btc_usd_vol
            crypto_col = "crypto_algo_proxy"
            logger.info("  algo_intensity: BTC volume activity proxy (crypto)")
    else:
        proxies[crypto_col] = df[crypto_col]

    if equity_col is None:
        if "sp500_return" in df.columns and "vix" in df.columns:
            spy_proxy = df["sp500_return"].abs() / df["vix"].clip(lower=5).div(100)
            proxies["equity_algo_proxy"] = spy_proxy
            equity_col = "equity_algo_proxy"
            logger.info("  algo_intensity: |SPY return|/VIX proxy (equity)")
    else:
        proxies[equity_col] = df[equity_col]

    if crypto_col is None or equity_col is None:
        logger.warning("  Cannot build algo_intensity — missing one or both proxies")
        return pd.Series(np.nan, index=df.index, name="algo_intensity")

    X  = pd.DataFrame({k: proxies[k] for k in [crypto_col, equity_col]}).dropna()
    zs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    z_pos = zs + 4.0   # shift to strictly positive
    geo   = np.exp(np.log(z_pos).mean(axis=1))
    geo_z = (geo - geo.mean()) / geo.std()
    geo_z.name = "algo_intensity"
    logger.info(f"  algo_intensity: {len(geo_z)} obs, std={geo_z.std():.3f}")
    return geo_z.reindex(df.index)


# ── 3. Realized Volatility ────────────────────────────────

def _intraday_rv(ticker: str, start: str, end: str) -> pd.Series:
    """
    Andersen et al. (2003): RV_t = Σ r²_{t,i} over intra-day intervals.
    Uses Yahoo Finance 1-hour data (best free approximation).
    Yahoo only provides ~730 days of intraday history.
    """
    try:
        raw = yf.download(ticker, start=start, end=end,
                          interval="1h", progress=False, auto_adjust=True)
        if raw.empty:
            raise ValueError("empty")
        ret = np.log(raw["Close"].squeeze() / raw["Close"].squeeze().shift(1)).dropna()
        rv  = ret.groupby(ret.index.normalize()).apply(lambda r: (r**2).sum())
        # Annualise: √(252 × 24 intervals/day)
        rv_ann = np.sqrt(rv * 252 * 24)
        rv_ann.name = f"rv_{ticker.lower().replace('-','').replace('^','')}_1h_ann"
        logger.info(f"  Intraday RV ✓ {ticker}: {len(rv_ann)} obs")
        return rv_ann
    except Exception as e:
        logger.warning(f"  Intraday RV ✗ {ticker}: {e} — using daily proxy")
        return pd.Series(dtype=float)


def build_realized_vol(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Compute intraday RV for BTC and SPY; fall back to |daily return| × √252."""
    cols = {}
    for ticker, ret_col, prefix in [
        ("BTC-USD", "bitcoin_return", "btc"),
        ("SPY",     "sp500_return",   "spy"),
    ]:
        rv = _intraday_rv(ticker, start, end)
        if not rv.empty:
            cols[f"{prefix}_rv_intraday_ann"] = rv.reindex(df.index)
        elif ret_col in df.columns:
            proxy = df[ret_col].abs() * np.sqrt(252)
            cols[f"{prefix}_rv_proxy"]        = proxy
            logger.info(f"  RV proxy: |{ret_col}| × √252 for {prefix}")
    return pd.DataFrame(cols, index=df.index)


# ── 4. Regulatory Dummies ─────────────────────────────────

def build_regulatory_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load regulatory dummies from data/raw/macro/regulatory_dummies.parquet,
    which is produced by collectors/regulatory_collector.py.

    If the parquet is not yet available (first run before regulatory_collector
    has run), falls back to the bundled collectors/regulatory_events.csv.

    Column definitions:
        reg_dummy_positive : 1 if any positive regulatory event that day
        reg_dummy_negative : 1 if any negative regulatory event that day
        reg_dummy_net      : positive - negative  (+1 / 0 / -1)
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import RAW_DIR

    zeros = pd.DataFrame({
        "reg_dummy_positive": 0,
        "reg_dummy_negative": 0,
        "reg_dummy_net":      0,
    }, index=df.index, dtype=int)

    # Prefer fetched parquet (produced by regulatory_collector.py)
    parquet_path = RAW_DIR / "macro" / "regulatory_dummies.parquet"
    if parquet_path.exists():
        reg = pd.read_parquet(parquet_path)
        reg.index = pd.to_datetime(reg.index)
        # Strip tz if present (some parquet files may have tz-aware index)
        if reg.index.tz is not None:
            reg.index = reg.index.tz_convert("UTC").tz_localize(None)
        for col in ["reg_dummy_positive", "reg_dummy_negative", "reg_dummy_net"]:
            zeros[col] = reg[col].reindex(df.index).fillna(0).astype(int)
        logger.info(f"  Reg dummies (fetched): {zeros['reg_dummy_positive'].sum()} pos, "
                    f"{zeros['reg_dummy_negative'].sum()} neg days")
        return zeros

    # Fallback: bundled CSV
    csv_path = Path(__file__).parent.parent / "collectors" / "regulatory_events.csv"
    if not csv_path.exists():
        logger.warning("  No regulatory data found — dummies will be zero")
        return zeros

    logger.warning("  Using bundled regulatory_events.csv (fallback). "
                   "Run collectors/regulatory_collector.py for source-verified data.")
    events = pd.read_csv(csv_path, parse_dates=["date"])
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()
    pos = events[events["direction"] == "positive"].groupby("date").size().gt(0).astype(int)
    neg = events[events["direction"] == "negative"].groupby("date").size().gt(0).astype(int)
    zeros["reg_dummy_positive"] = pos.reindex(df.index).fillna(0).astype(int)
    zeros["reg_dummy_negative"] = neg.reindex(df.index).fillna(0).astype(int)
    zeros["reg_dummy_net"]      = zeros["reg_dummy_positive"] - zeros["reg_dummy_negative"]
    logger.info(f"  Reg dummies (fallback CSV): {zeros['reg_dummy_positive'].sum()} pos, "
                f"{zeros['reg_dummy_negative'].sum()} neg days")
    return zeros


# ── 5. Rolling BTC–S&P Correlation ───────────────────────

def build_rolling_corr(df: pd.DataFrame, window: int = 30) -> pd.Series:
    if "bitcoin_return" not in df.columns or "sp500_return" not in df.columns:
        logger.warning("  Rolling corr: missing return columns")
        return pd.Series(np.nan, index=df.index, name="btc_sp500_corr_30d")
    corr = (df["bitcoin_return"]
            .rolling(window, min_periods=max(10, window // 3))
            .corr(df["sp500_return"]))
    corr.name = "btc_sp500_corr_30d"
    logger.info(f"  Rolling {window}d BTC–SPY corr: "
                f"mean={corr.mean():.3f}  range=[{corr.min():.3f}, {corr.max():.3f}]")
    return corr


# ── Master runner ─────────────────────────────────────────

def run_feature_engineering(start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    Load master_panel.parquet, compute all engineered features,
    save master_panel_features.parquet.
    """
    in_path = DATA_DIR / "processed" / "master_panel.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"master_panel.parquet not found. Run transformer first.")

    logger.info("=== Feature Engineering ===")
    df = pd.read_parquet(in_path)
    df.index = pd.to_datetime(df.index)
    logger.info(f"  Loaded: {df.shape}")

    composite, used_gtrends = build_sentiment_composite(df)
    df["crypto_sentiment_composite"]  = composite
    df["gtrends_sentiment_used"]      = int(used_gtrends)
    if used_gtrends:
        logger.info(
            "  Note: crypto_sentiment_composite built from Google Trends SVI "
            "(Da et al. 2011; Kristoufek 2013). Add 'gtrends_sentiment_used' "
            "as a robustness flag in your regression tables."
        )
    df["algo_intensity"]             = build_algo_intensity(df)

    rv = build_realized_vol(df, start, end)
    for col in rv.columns:
        df[col] = rv[col]

    reg = build_regulatory_dummies(df)
    for col in reg.columns:
        df[col] = reg[col]

    df["btc_sp500_corr_30d"] = build_rolling_corr(df)

    out = DATA_DIR / "processed" / "master_panel_features.parquet"
    df.to_parquet(out)
    logger.success(f"master_panel_features.parquet → {out}  {df.shape}")

    new_cols = ["crypto_sentiment_composite", "algo_intensity",
                "reg_dummy_net", "btc_sp500_corr_30d"] + \
               [c for c in df.columns if "rv_" in c]
    for c in new_cols:
        if c in df.columns:
            pct = df[c].notna().mean() * 100
            logger.info(f"    {c}: {pct:.1f}% coverage")

    return df


if __name__ == "__main__":
    df = run_feature_engineering()
    print(df[["crypto_sentiment_composite", "algo_intensity",
              "reg_dummy_net", "btc_sp500_corr_30d"]].describe().round(4))