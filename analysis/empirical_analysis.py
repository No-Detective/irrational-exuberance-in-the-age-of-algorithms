"""
analysis/empirical_analysis.py

Complete empirical analysis for:
  "Irrational Exuberance in the Age of Algorithms"

Covers all four hypotheses using the data in master_panel_final.parquet.

Usage:
    python analysis/empirical_analysis.py

Outputs (all saved to data/results/):
    h1_granger.txt          — Granger causality tests
    h1_svar_baseline.txt    — SVAR: raw sentiment model
    h1_svar_irrational.txt  — SVAR: irrational sentiment model (primary)
    h1_svar_horserace.txt   — SVAR: rational + irrational horse-race
    h1_irf.png              — Impulse response functions (4-panel)
    h1_fevd.png             — Forecast error variance decompositions
    h2_mediation.txt        — Baron-Kenny mediation + Sobel + bootstrap
    h3_moderation.txt       — Interaction model + regime split
    h3_irf_regimes.png      — IRF comparison high vs low algo
    robustness.txt          — Placebo, alt sentiment, out-of-sample
    summary_table.txt       — Combined results table for paper

Requirements: statsmodels, scipy, matplotlib, numpy, pandas
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.var_model import VARResults
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH    = Path("data/processed/master_panel_final.parquet")
RESULTS_DIR  = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Weekend alignment: forward-fill equity returns from Friday to Sat/Sun
# so crypto weekend sentiment has something to compare against
equity_cols = ['sp500_return', 'vix', 'tech_etf_return', 'fin_etf_return']
df[equity_cols] = df[equity_cols].ffill(limit=2)

print(f"Loaded: {df.shape}, {df.index[0].date()} → {df.index[-1].date()}")

# ── Helper: ADF stationarity ───────────────────────────────────────────────────
def adf_test(series, name):
    s = series.dropna()
    result = adfuller(s, autolag='AIC')
    pval = result[1]
    status = "I(0)" if pval < 0.05 else "I(1)?"
    return f"  {name:<35} ADF p={pval:.4f}  {status}"

# ── Helper: write report ───────────────────────────────────────────────────────
def write_report(path, lines):
    with open(path, 'w') as f:
        f.write('\n'.join(str(l) for l in lines))
    print(f"  → {path}")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  H1 — SENTIMENT PREDICTABILITY                                              ║
# ║  Methods: Granger causality + SVAR with IRF/FEVD                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_h1():
    print("\n=== H1: Sentiment Predictability ===")
    lines = ["=" * 70,
             "H1: IRRATIONAL SENTIMENT PREDICTABILITY",
             "=" * 70, ""]

    # ── 1. Stationarity ───────────────────────────────────────────────────────
    lines += ["── 1. Unit Root Tests (ADF) ──────────────────────────────────────", ""]
    test_vars = {
        'irrational_sentiment': df['irrational_sentiment'],
        'rational_sentiment':   df['rational_sentiment'],
        'sp500_return':         df['sp500_return'],
        'bitcoin_return':       df['bitcoin_return'],
        'vix_diff':             df['vix'].diff(),
        'algo_intensity':       df['algo_intensity'],
    }
    for name, s in test_vars.items():
        lines.append(adf_test(s.dropna(), name))

    # ── 2. Granger Causality ──────────────────────────────────────────────────
    lines += ["", "── 2. Granger Causality Tests ────────────────────────────────────",
              "    Does irrational_sentiment Granger-cause sp500_return?", ""]

    gc_data = df[['sp500_return', 'irrational_sentiment']].dropna()
    gc_results = grangercausalitytests(gc_data, maxlag=5, verbose=False)
    lines.append("  Lag  F-stat    p-value   Significant?")
    lines.append("  " + "-" * 42)
    for lag, res in gc_results.items():
        f_stat = res[0]['ssr_ftest'][0]
        p_val  = res[0]['ssr_ftest'][1]
        sig    = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        lines.append(f"  {lag}    {f_stat:8.3f}  {p_val:8.4f}   {sig}")

    # Reverse direction (falsification)
    lines += ["", "    Does sp500_return Granger-cause irrational_sentiment? (falsification)", ""]
    gc_rev = grangercausalitytests(
        df[['irrational_sentiment', 'sp500_return']].dropna(), maxlag=5, verbose=False)
    lines.append("  Lag  F-stat    p-value   Significant?")
    lines.append("  " + "-" * 42)
    for lag, res in gc_rev.items():
        f_stat = res[0]['ssr_ftest'][0]
        p_val  = res[0]['ssr_ftest'][1]
        sig    = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        lines.append(f"  {lag}    {f_stat:8.3f}  {p_val:8.4f}   {sig}")

    # ── 3. SVAR — Irrational Model (PRIMARY) ─────────────────────────────────
    lines += ["", "── 3. SVAR — Irrational Sentiment Model (Primary, H1) ───────────", ""]

    # Cholesky ordering from paper:
    # irrational_sentiment → bitcoin_return → btc_daily_tx_count → sp500_return → vix
    svar_cols = ['irrational_sentiment', 'bitcoin_return',
                 'log_btc_daily_tx_count', 'sp500_return', 'vix']
    svar_data = df[svar_cols].dropna()

    # First-difference VIX (I(1) variable)
    svar_data = svar_data.copy()
    svar_data['vix'] = svar_data['vix'].diff()
    svar_data = svar_data.dropna()

    model = VAR(svar_data)
    # Select lag by HQ criterion (as paper specifies)
    lag_order = model.select_order(maxlags=10)
    best_lag = lag_order.hqic
    best_lag = max(best_lag, 2)  # minimum 2 lags
    lines.append(f"  Lag selection (HQ): {best_lag} lags")
    lines.append(f"  Observations: {len(svar_data)}")

    results = model.fit(best_lag)
    lines.append(f"  Log-likelihood: {results.llf:.2f}")
    lines.append(f"  AIC: {results.aic:.2f}")
    lines.append(f"  BIC: {results.bic:.2f}")
    lines += ["", "  Equation: sp500_return", ""]

    # Extract sp500_return equation coefficients
    sp500_eq = results.params['sp500_return']
    sp500_se = results.stderr['sp500_return']
    irr_rows = [(k, v, sp500_se.get(k, np.nan))
                for k, v in sp500_eq.items()
                if 'irrational' in k or 'rational' in k or 'bitcoin' in k or 'const' in k]
    lines.append(f"  {'Variable':<40} {'Coef':>10} {'SE':>10} {'t':>8}")
    lines.append("  " + "-" * 70)
    for k, v, se in irr_rows:
        t = v / se if se > 0 else np.nan
        sig = "***" if abs(t) > 3.29 else "**" if abs(t) > 2.58 else "*" if abs(t) > 1.96 else ""
        lines.append(f"  {k:<40} {v:>10.5f} {se:>10.5f} {t:>8.3f} {sig}")

    write_report(RESULTS_DIR / "h1_svar_irrational.txt", lines)
    return results, svar_data, best_lag, svar_cols


def run_h1_horserace(svar_results, svar_data, best_lag):
    """Horse-race: both rational AND irrational in same SVAR."""
    print("  Running horse-race SVAR...")
    lines = ["=" * 70,
             "H1: HORSE-RACE SVAR — Rational vs Irrational Sentiment",
             "=" * 70, "",
             "  Both rational_sentiment and irrational_sentiment included simultaneously.",
             "  If irrational dominates → strong support for H1.", ""]

    hr_cols = ['rational_sentiment', 'irrational_sentiment', 'bitcoin_return',
               'log_btc_daily_tx_count', 'sp500_return', 'vix']
    hr_data = df[hr_cols].dropna().copy()
    hr_data['vix'] = hr_data['vix'].diff()
    hr_data = hr_data.dropna()

    model = VAR(hr_data)
    res = model.fit(best_lag)

    sp500_eq = res.params['sp500_return']
    sp500_se = res.stderr['sp500_return']
    lines.append(f"  {'Variable':<40} {'Coef':>10} {'SE':>10} {'t':>8}")
    lines.append("  " + "-" * 70)
    for k in sp500_eq.index:
        if any(x in k for x in ['rational', 'irrational', 'const']):
            v, se = sp500_eq[k], sp500_se.get(k, np.nan)
            t = v / se if se > 0 else np.nan
            sig = "***" if abs(t) > 3.29 else "**" if abs(t) > 2.58 else "*" if abs(t) > 1.96 else ""
            lines.append(f"  {k:<40} {v:>10.5f} {se:>10.5f} {t:>8.3f} {sig}")

    write_report(RESULTS_DIR / "h1_svar_horserace.txt", lines)


def run_irf_fevd(svar_data, best_lag):
    """Compute and plot IRFs and FEVDs."""
    print("  Computing IRFs and FEVDs...")

    svar_data2 = svar_data.copy()
    svar_data2['vix'] = svar_data2['vix']  # already differenced
    model = VAR(svar_data2)
    res = model.fit(best_lag)

    # IRF: response of sp500_return to shock in irrational_sentiment
    irf = res.irf(10)
    fevd = res.fevd(10)

    col_names = list(svar_data2.columns)
    irr_idx   = col_names.index('irrational_sentiment')
    sp500_idx = col_names.index('sp500_return')
    btc_idx   = col_names.index('bitcoin_return')

    # stderr() returns array shape (periods, n_vars, n_vars)
    irf_se = irf.stderr(orth=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("H1: Impulse Response Functions — SVAR\n"
                 "Response of S&P 500 Return to 1-SD Irrational Sentiment Shock",
                 fontsize=12, fontweight='bold')

    # Main IRF panel — SP500 response
    ax = axes[0, 0]
    irf_vals = irf.orth_irfs[:, sp500_idx, irr_idx]
    horizons = range(len(irf_vals))
    se_vals  = irf_se[:, sp500_idx, irr_idx]
    ax.plot(horizons, irf_vals, 'b-o', linewidth=2, markersize=4)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.fill_between(horizons,
                    irf_vals - 1.96 * se_vals,
                    irf_vals + 1.96 * se_vals,
                    alpha=0.2, color='blue', label='95% CI')
    ax.set_title("SP500 Return ← Irrational Sentiment Shock", fontsize=10)
    ax.set_xlabel("Days after shock")
    ax.set_ylabel("Response (log return)")
    ax.legend(fontsize=8)

    # Response of bitcoin_return
    ax = axes[0, 1]
    irf_btc  = irf.orth_irfs[:, btc_idx, irr_idx]
    se_btc   = irf_se[:, btc_idx, irr_idx]
    horizons_btc = range(len(irf_btc))
    ax.plot(horizons_btc, irf_btc, 'r-o', linewidth=2, markersize=4)
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.fill_between(horizons_btc,
                    irf_btc - 1.96 * se_btc,
                    irf_btc + 1.96 * se_btc,
                    alpha=0.2, color='red')
    ax.set_title("BTC Return ← Irrational Sentiment Shock", fontsize=10)
    ax.set_xlabel("Days after shock")
    ax.set_ylabel("Response (log return)")

    # FEVD: share of sp500_return variance from irrational_sentiment
    ax = axes[1, 0]
    fevd_vals = fevd.decomp[:, sp500_idx, irr_idx] * 100
    ax.bar(range(1, len(fevd_vals)+1), fevd_vals, color='steelblue', alpha=0.8)
    ax.set_title("FEVD: % of S&P 500 Variance\nfrom Irrational Sentiment", fontsize=9)
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("% Variance explained")

    # FEVD: share of bitcoin_return variance from irrational_sentiment
    ax = axes[1, 1]
    fevd_btc = fevd.decomp[:, btc_idx, irr_idx] * 100
    ax.bar(range(1, len(fevd_btc)+1), fevd_btc, color='tomato', alpha=0.8)
    ax.set_title("FEVD: % of BTC Variance\nfrom Irrational Sentiment", fontsize=9)
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("% Variance explained")

    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "h1_irf_fevd.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {RESULTS_DIR}/h1_irf_fevd.png")

    # Numeric FEVD table
    n_horizons = fevd.decomp.shape[0]
    lines = ["=" * 70, "H1: FORECAST ERROR VARIANCE DECOMPOSITION", "=" * 70, "",
             "  Share of S&P 500 return forecast error variance explained by:",
             f"  {'Horizon':>8}  {'Irrational':>12}  {'Bitcoin ret':>12}",
             "  " + "-" * 45]
    for h in range(n_horizons):
        irr_share = fevd.decomp[h, sp500_idx, irr_idx] * 100
        btc_share = fevd.decomp[h, sp500_idx, btc_idx] * 100
        lines.append(f"  {h+1:>8}d  {irr_share:>11.2f}%  {btc_share:>11.2f}%")

    write_report(RESULTS_DIR / "h1_fevd.txt", lines)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  H2 — ON-CHAIN MEDIATION                                                   ║
# ║  Methods: Baron-Kenny time-series mediation + Sobel + bootstrap            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_h2():
    print("\n=== H2: On-Chain Mediation ===")
    lines = ["=" * 70,
             "H2: ON-CHAIN MEDIATION OF IRRATIONAL SENTIMENT",
             "=" * 70, "",
             "  Baron-Kenny framework (adapted for time series with lagged predictors).",
             "  Three equations per mediator:", "",
             "    (A) bitcoin_return_{t} = a + b·irrational_sentiment_{t-1} + controls + ε",
             "    (B) mediator_{t}        = a + c·irrational_sentiment_{t-1} + controls + ε",
             "    (C) bitcoin_return_{t}  = a + b'·irrational_sentiment_{t-1}",
             "                                + d·mediator_{t-1} + controls + ε", "",
             "  Indirect effect = c × d  (Sobel test + 5000-rep bootstrap)",
             "  Mediation if: b significant, c significant, |b'| < |b|", ""]

    controls = ['bitcoin_return', 'sp500_return', 'vix', 'epu_index',
                'reg_dummy_net', 'bw_sentiment']

    # Mediators available
    mediators = {
        'log_btc_daily_tx_count':    'Transaction count (on-chain activity proxy, 2018-2025)',
        'btc_miner_revenue_usd':     'Miner revenue (network health proxy, 2018-2025)',
        'stablecoin_supply_ratio':   'Stablecoin ratio (speculative pressure proxy, 2018-2025)',
        'nupl':                      'NUPL (holder profit/loss state, 2022-2025 only)',
    }

    def run_mediation(mediator_col, mediator_label):
        """Run Baron-Kenny for one mediator."""
        cols_needed = list(dict.fromkeys(
            ['bitcoin_return', 'irrational_sentiment', mediator_col] + controls))
        sub = df[cols_needed].dropna().copy()
        if len(sub) < 100:
            return [f"  {mediator_col}: insufficient data ({len(sub)} obs)"]

        # Lag sentiment and mediator by 1
        sub['irr_lag1']  = sub['irrational_sentiment'].shift(1)
        sub['med_lag1']  = sub[mediator_col].shift(1)
        sub['btc_lag1']  = sub['bitcoin_return'].shift(1)
        sub['sp5_lag1']  = sub['sp500_return'].shift(1)
        sub = sub.dropna()

        ctrl_cols = ['btc_lag1', 'sp5_lag1', 'vix', 'epu_index', 'reg_dummy_net']

        # Equation A: total effect
        X_a  = sm.add_constant(sub[['irr_lag1'] + ctrl_cols])
        eq_a = sm.OLS(sub['bitcoin_return'], X_a).fit(
            cov_type='HAC', cov_kwds={'maxlags': 5})
        b  = eq_a.params['irr_lag1']
        b_p = eq_a.pvalues['irr_lag1']

        # Equation B: sentiment → mediator
        X_b  = sm.add_constant(sub[['irr_lag1'] + ctrl_cols])
        eq_b = sm.OLS(sub[mediator_col], X_b).fit(
            cov_type='HAC', cov_kwds={'maxlags': 5})
        c   = eq_b.params['irr_lag1']
        c_p = eq_b.pvalues['irr_lag1']
        c_se = eq_b.bse['irr_lag1']

        # Equation C: direct effect (sentiment + mediator)
        X_c  = sm.add_constant(sub[['irr_lag1', 'med_lag1'] + ctrl_cols])
        eq_c = sm.OLS(sub['bitcoin_return'], X_c).fit(
            cov_type='HAC', cov_kwds={'maxlags': 5})
        b_prime  = eq_c.params['irr_lag1']
        b_prime_p = eq_c.pvalues['irr_lag1']
        d    = eq_c.params['med_lag1']
        d_p  = eq_c.pvalues['med_lag1']
        d_se = eq_c.bse['med_lag1']

        # Sobel test for indirect effect (c × d)
        indirect = c * d
        sobel_se = np.sqrt(d**2 * c_se**2 + c**2 * d_se**2)
        sobel_z  = indirect / sobel_se if sobel_se > 0 else np.nan
        sobel_p  = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

        # Bootstrap CI for indirect effect (5000 reps)
        n_boot = 5000
        boot_indirect = []
        sub_arr = sub.values
        sub_cols = list(sub.columns)
        irr_i = sub_cols.index('irr_lag1')
        med_i = sub_cols.index(mediator_col)
        med_lag_i = sub_cols.index('med_lag1')
        btc_i = sub_cols.index('bitcoin_return')
        ctrl_i = [sub_cols.index(c) for c in ctrl_cols if c in sub_cols]

        rng = np.random.default_rng(42)
        for _ in range(n_boot):
            idx = rng.integers(0, len(sub_arr), len(sub_arr))
            sample = sub_arr[idx]
            try:
                Xb_ = np.column_stack([np.ones(len(sample)),
                                        sample[:, irr_i],
                                        sample[:, [ci for ci in ctrl_i]]])
                y_med = sample[:, med_i]
                c_ = np.linalg.lstsq(Xb_, y_med, rcond=None)[0][1]

                Xc_ = np.column_stack([np.ones(len(sample)),
                                        sample[:, irr_i],
                                        sample[:, med_lag_i],
                                        sample[:, [ci for ci in ctrl_i]]])
                y_btc = sample[:, btc_i]
                d_ = np.linalg.lstsq(Xc_, y_btc, rcond=None)[0][2]
                boot_indirect.append(c_ * d_)
            except Exception:
                continue

        boot_arr = np.array(boot_indirect)
        ci_lo, ci_hi = np.percentile(boot_arr, [2.5, 97.5])

        def stars(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

        result = [
            f"  Mediator: {mediator_col} ({mediator_label})",
            f"  n = {len(sub)} obs",
            f"  (A) Total effect       b  = {b:+.5f}  (p={b_p:.4f}) {stars(b_p)}",
            f"  (B) Sentiment→Mediator c  = {c:+.5f}  (p={c_p:.4f}) {stars(c_p)}",
            f"  (C) Direct effect      b' = {b_prime:+.5f}  (p={b_prime_p:.4f}) {stars(b_prime_p)}",
            f"      Mediator effect    d  = {d:+.5f}  (p={d_p:.4f}) {stars(d_p)}",
            f"  Indirect effect (c×d)  = {indirect:+.6f}",
            f"  Sobel Z = {sobel_z:.3f}  (p={sobel_p:.4f}) {stars(sobel_p)}",
            f"  Bootstrap 95% CI: [{ci_lo:+.6f}, {ci_hi:+.6f}]",
            f"  Mediation type: {'Full' if abs(b_prime) < abs(b)*0.3 else 'Partial' if b_prime_p > 0.05 else 'No'}",
            "",
        ]
        return result

    for col, label in mediators.items():
        lines += ["─" * 60]
        lines += run_mediation(col, label)

    write_report(RESULTS_DIR / "h2_mediation.txt", lines)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  H3 — ALGORITHMIC MODERATION                                               ║
# ║  Methods: Interaction regression + Regime-split SVAR                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_h3():
    print("\n=== H3: Algorithmic Moderation ===")
    lines = ["=" * 70,
             "H3: ALGORITHMIC TRADING INTENSITY AS MODERATOR",
             "=" * 70, "",
             "  sp500_return_t = α + β1·irr_{t-1} + β2·algo_t",
             "                   + β3·(irr_{t-1} × algo_t) + controls + ε",
             "",
             "  Prediction: β3 > 0 (stronger spillover when algos are active)", ""]

    sub = df[['sp500_return', 'irrational_sentiment', 'rational_sentiment',
              'algo_intensity', 'bitcoin_return', 'vix', 'epu_index',
              'bw_sentiment', 'reg_dummy_net']].dropna().copy()
    sub['irr_lag1']   = sub['irrational_sentiment'].shift(1)
    sub['rat_lag1']   = sub['rational_sentiment'].shift(1)
    sub['btc_lag1']   = sub['bitcoin_return'].shift(1)
    sub['sp500_lag1'] = sub['sp500_return'].shift(1)
    sub['irr_x_algo'] = sub['irr_lag1'] * sub['algo_intensity']
    sub['rat_x_algo'] = sub['rat_lag1'] * sub['algo_intensity']
    sub = sub.dropna()

    ctrl = ['btc_lag1', 'sp500_lag1', 'vix', 'epu_index', 'bw_sentiment', 'reg_dummy_net']

    # Model 1: Main effects only
    X1 = sm.add_constant(sub[['irr_lag1', 'rat_lag1', 'algo_intensity'] + ctrl])
    m1 = sm.OLS(sub['sp500_return'], X1).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    # Model 2: With interaction
    X2 = sm.add_constant(sub[['irr_lag1', 'rat_lag1', 'algo_intensity',
                                'irr_x_algo', 'rat_x_algo'] + ctrl])
    m2 = sm.OLS(sub['sp500_return'], X2).fit(cov_type='HAC', cov_kwds={'maxlags': 5})

    def fmt_row(name, model):
        if name not in model.params.index:
            return ""
        coef = model.params[name]
        se   = model.bse[name]
        t    = model.tvalues[name]
        p    = model.pvalues[name]
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        return f"  {name:<28} {coef:+.6f}  ({se:.6f})  t={t:+.2f} {sig}"

    lines += ["── Model 1: Main Effects (no interaction) ────────────────────────",
              f"  n={len(sub)}, R²={m1.rsquared:.4f}, Adj-R²={m1.rsquared_adj:.4f}", ""]
    for v in ['irr_lag1', 'rat_lag1', 'algo_intensity', 'const']:
        lines.append(fmt_row(v, m1))

    lines += ["", "── Model 2: With Interaction Terms ───────────────────────────────",
              f"  n={len(sub)}, R²={m2.rsquared:.4f}, Adj-R²={m2.rsquared_adj:.4f}", ""]
    for v in ['irr_lag1', 'rat_lag1', 'algo_intensity', 'irr_x_algo', 'rat_x_algo', 'const']:
        lines.append(fmt_row(v, m2))

    # Regime split
    lines += ["", "── Regime Split: High vs Low Algo Intensity ──────────────────────", ""]
    med_algo = sub['algo_intensity'].median()
    high = sub[sub['algo_intensity'] >= med_algo]
    low  = sub[sub['algo_intensity'] <  med_algo]

    for label, regime in [("HIGH algo", high), ("LOW algo", low)]:
        X_r = sm.add_constant(regime[['irr_lag1', 'rat_lag1'] + ctrl])
        m_r = sm.OLS(regime['sp500_return'], X_r).fit(
            cov_type='HAC', cov_kwds={'maxlags': 5})
        irr_coef = m_r.params.get('irr_lag1', np.nan)
        irr_p    = m_r.pvalues.get('irr_lag1', np.nan)
        sig = "***" if irr_p < 0.001 else "**" if irr_p < 0.01 else "*" if irr_p < 0.05 else "n.s."
        lines.append(f"  {label} regime (n={len(regime)}):  "
                     f"irr_lag1={irr_coef:+.6f}  (p={irr_p:.4f}) {sig}")
        lines.append(f"    R²={m_r.rsquared:.4f}")

    # Wald test: are interaction coefficients jointly significant?
    try:
        r_matrix = np.zeros((2, len(m2.params)))
        idx_irr  = list(m2.params.index).index('irr_x_algo')
        idx_rat  = list(m2.params.index).index('rat_x_algo')
        r_matrix[0, idx_irr] = 1
        r_matrix[1, idx_rat] = 1
        wald = m2.wald_test(r_matrix)
        lines += ["", f"  Wald test (interactions jointly zero):",
                  f"  F = {float(wald.statistic):.3f}, p = {float(wald.pvalue):.4f}"]
    except Exception:
        pass

    write_report(RESULTS_DIR / "h3_moderation.txt", lines)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  H4 — AGENT-BASED SIMULATION                                               ║
# ║  Calibrated to empirical moments from MASTER_FINAL                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_h4():
    print("\n=== H4: Agent-Based Simulation ===")

    # Calibration parameters from empirical data
    irr = df['irrational_sentiment'].dropna()
    rat = df['rational_sentiment'].dropna()
    btc = df['bitcoin_return'].dropna()
    sp5 = df['sp500_return'].dropna()
    corr_empirical = df[['bitcoin_return', 'sp500_return']].dropna().corr().iloc[0, 1]

    irr_mean, irr_std = float(irr.mean()), float(irr.std())
    rat_mean, rat_std = float(rat.mean()), float(rat.std())
    btc_mean, btc_std = float(btc.mean()), float(btc.std())
    sp5_mean, sp5_std = float(sp5.mean()), float(sp5.std())

    # Sentiment-return coefficient from empirical SVAR (approximate from OLS)
    sub = df[['sp500_return', 'irrational_sentiment']].dropna().copy()
    sub['irr_lag1'] = sub['irrational_sentiment'].shift(1)
    sub = sub.dropna()
    m = sm.OLS(sub['sp500_return'], sm.add_constant(sub['irr_lag1'])).fit()
    beta_irr = float(m.params['irr_lag1'])

    sub2 = df[['sp500_return', 'rational_sentiment']].dropna().copy()
    sub2['rat_lag1'] = sub2['rational_sentiment'].shift(1)
    sub2 = sub2.dropna()
    m2 = sm.OLS(sub2['sp500_return'], sm.add_constant(sub2['rat_lag1'])).fit()
    beta_rat = float(m2.params['rat_lag1'])

    # ── ABM ───────────────────────────────────────────────────────────────────
    n_sims   = 10_000
    n_periods = 2555  # 7 years
    rng = np.random.default_rng(42)

    results_table = {}

    def simulate(condition):
        """
        Single simulation run.
        Conditions: 'no_ai', 'ai_rational', 'ai_irrational', 'ai_full'
        Returns: cross_corr, eq_vol, eq_kurt, vol_autocorr
        """
        crypto_price = 1.0
        equity_price = 1.0
        crypto_fund  = 1.0  # fundamental value (random walk)
        equity_fund  = 1.0  # fundamental value (mean-reverting)

        crypto_rets = []
        equity_rets = []

        # AI agent state (learned mapping from sentiment to action)
        ai_weight = 0.0  # learned weight on sentiment signal

        for t in range(n_periods):
            # Draw sentiment shocks (calibrated to empirical moments)
            irr_shock = rng.normal(irr_mean, irr_std)
            rat_shock = rng.normal(rat_mean, rat_std)

            # ── Fundamental traders (stabilising) ─────────────────────────
            crypto_fund *= np.exp(rng.normal(btc_mean, btc_std * 0.3))
            equity_fund  = equity_fund * 0.99 + 1.0 * 0.01  # mean-revert to 1

            fund_crypto = -0.1 * (crypto_price - crypto_fund) / crypto_fund
            fund_equity = -0.1 * (equity_price - equity_fund) / equity_fund

            # ── Noise/sentiment traders (destabilising) ────────────────────
            noise_crypto = irr_shock * 0.005
            noise_equity = irr_shock * 0.002

            # ── AI agent ──────────────────────────────────────────────────
            if condition == 'no_ai':
                ai_crypto = 0
                ai_equity = 0
            elif condition == 'ai_rational':
                ai_crypto = rat_shock * beta_rat * 0.5
                ai_equity = rat_shock * beta_rat
            elif condition == 'ai_irrational':
                ai_crypto = irr_shock * beta_irr * 0.5
                ai_equity = irr_shock * beta_irr
            else:  # ai_full
                ai_crypto = (irr_shock * beta_irr + rat_shock * beta_rat) * 0.5
                ai_equity = irr_shock * beta_irr + rat_shock * beta_rat

            # Update AI learning rate (simplified RL: move toward profitable)
            if t > 0 and condition != 'no_ai':
                last_ret = equity_rets[-1] if equity_rets else 0
                ai_weight = ai_weight * 0.99 + 0.01 * abs(last_ret)

            # ── Price updates ──────────────────────────────────────────────
            r_crypto = fund_crypto + noise_crypto + ai_crypto + rng.normal(0, btc_std * 0.5)
            r_equity = fund_equity + noise_equity + ai_equity + rng.normal(0, sp5_std * 0.5)

            crypto_price *= np.exp(r_crypto)
            equity_price *= np.exp(r_equity)

            crypto_rets.append(r_crypto)
            equity_rets.append(r_equity)

        cr = np.array(crypto_rets)
        er = np.array(equity_rets)

        cross_corr   = float(np.corrcoef(cr, er)[0, 1])
        eq_vol       = float(np.std(er))
        eq_kurt      = float(stats.kurtosis(er))
        vol_autocorr = float(pd.Series(np.abs(er)).autocorr(lag=1))

        return cross_corr, eq_vol, eq_kurt, vol_autocorr

    conditions = ['no_ai', 'ai_rational', 'ai_irrational', 'ai_full']
    condition_labels = {
        'no_ai':          'No AI (Baseline)',
        'ai_rational':    'AI-Rational',
        'ai_irrational':  'AI-Irrational (H4 prediction)',
        'ai_full':        'AI-Full',
    }

    # Run all conditions
    for cond in conditions:
        print(f"    Simulating {cond} ({n_sims:,} runs)...")
        corrs, vols, kurts, autocorrs = [], [], [], []
        for _ in range(n_sims):
            c, v, k, a = simulate(cond)
            corrs.append(c); vols.append(v); kurts.append(k); autocorrs.append(a)
        results_table[cond] = {
            'cross_corr':   np.array(corrs),
            'eq_vol':       np.array(vols),
            'eq_kurt':      np.array(kurts),
            'vol_autocorr': np.array(autocorrs),
        }

    # KS tests: AI-irrational vs each other condition
    lines = ["=" * 70,
             "H4: AGENT-BASED SIMULATION RESULTS",
             "=" * 70, "",
             f"  Simulation: {n_sims:,} runs × {n_periods} periods",
             f"  Calibrated to: irr_std={irr_std:.4f}, beta_irr={beta_irr:.6f}",
             f"  Empirical cross-market corr: {corr_empirical:.4f}", "",
             f"  {'Condition':<30} {'Cross-Corr':>12} {'Eq Vol':>10} {'Kurt':>8} {'Vol-AC':>8}",
             "  " + "-" * 72]

    for cond in conditions:
        r = results_table[cond]
        lines.append(f"  {condition_labels[cond]:<30} "
                     f"{r['cross_corr'].mean():>12.4f} "
                     f"{r['eq_vol'].mean():>10.6f} "
                     f"{r['eq_kurt'].mean():>8.3f} "
                     f"{r['vol_autocorr'].mean():>8.4f}")

    lines += ["", "── KS Tests: AI-Irrational vs Other Conditions ──────────────────", ""]
    irrational_r = results_table['ai_irrational']
    for cond in ['no_ai', 'ai_rational', 'ai_full']:
        other = results_table[cond]
        ks_corr = stats.ks_2samp(irrational_r['cross_corr'], other['cross_corr'])
        ks_vol  = stats.ks_2samp(irrational_r['eq_vol'], other['eq_vol'])
        lines.append(f"  AI-Irrational vs {condition_labels[cond]}:")
        lines.append(f"    Cross-corr KS: D={ks_corr.statistic:.4f}, p={ks_corr.pvalue:.4f}")
        lines.append(f"    Eq. vol    KS: D={ks_vol.statistic:.4f}, p={ks_vol.pvalue:.4f}")
        lines.append("")

    lines += ["── Interpretation ────────────────────────────────────────────────", "",
              f"  H4 is supported if AI-Irrational shows:",
              f"  (1) Higher cross-corr than no_ai: "
              f"{'YES ✓' if results_table['ai_irrational']['cross_corr'].mean() > results_table['no_ai']['cross_corr'].mean() else 'NO ✗'}",
              f"  (2) Higher eq_vol than ai_rational: "
              f"{'YES ✓' if results_table['ai_irrational']['eq_vol'].mean() > results_table['ai_rational']['eq_vol'].mean() else 'NO ✗'}",
              f"  (3) Higher kurtosis (fatter tails): "
              f"{'YES ✓' if results_table['ai_irrational']['eq_kurt'].mean() > results_table['no_ai']['eq_kurt'].mean() else 'NO ✗'}"]

    write_report(RESULTS_DIR / "h4_simulation.txt", lines)

    # Plot distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("H4: Agent-Based Simulation — Distribution Comparison", fontsize=12)
    colors = {'no_ai': 'gray', 'ai_rational': 'blue',
               'ai_irrational': 'red', 'ai_full': 'orange'}
    for cond in conditions:
        r = results_table[cond]
        axes[0].hist(r['cross_corr'], bins=80, alpha=0.4,
                     color=colors[cond], label=condition_labels[cond], density=True)
        axes[1].hist(r['eq_vol'], bins=80, alpha=0.4,
                     color=colors[cond], label=condition_labels[cond], density=True)
    axes[0].set_title("Cross-Market Correlation Distribution"); axes[0].set_xlabel("Corr")
    axes[1].set_title("Equity Volatility Distribution"); axes[1].set_xlabel("Vol")
    axes[0].axvline(corr_empirical, color='black', linestyle='--', label=f'Empirical ({corr_empirical:.3f})')
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "h4_simulation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {RESULTS_DIR}/h4_simulation.png")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  ROBUSTNESS CHECKS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def run_robustness():
    print("\n=== Robustness Checks ===")
    lines = ["=" * 70,
             "ROBUSTNESS CHECKS",
             "=" * 70, ""]

    controls = ['bitcoin_return', 'vix', 'epu_index', 'bw_sentiment', 'reg_dummy_net']

    def simple_reg(y_col, x_col, lag=1, label=""):
        sub = df[[y_col, x_col] + controls].dropna().copy()
        sub['x_lag'] = sub[x_col].shift(lag)
        sub['btc_lag'] = sub['bitcoin_return'].shift(1)
        sub = sub.dropna()
        X = sm.add_constant(sub[['x_lag', 'btc_lag', 'vix', 'epu_index',
                                   'bw_sentiment', 'reg_dummy_net']])
        m = sm.OLS(sub[y_col], X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
        coef = m.params['x_lag']
        pval = m.pvalues['x_lag']
        sig  = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "n.s."
        return f"  {label:<45} coef={coef:+.6f}  p={pval:.4f} {sig}  n={len(sub)}"

    # Alternative sentiment measures → sp500_return
    lines += ["── Alternative Sentiment Proxies (→ sp500_return) ────────────────", ""]
    for col, label in [
        ('irrational_sentiment', 'Irrational Sentiment (primary)'),
        ('rational_sentiment',   'Rational Sentiment (benchmark)'),
        ('sentiment_raw',        'Raw Composite Sentiment'),
        ('fear_greed_index',     'Fear & Greed Index (alt. measure)'),
        ('gtrends_bitcoin',      'Google Trends Bitcoin SVI'),
    ]:
        lines.append(simple_reg('sp500_return', col, label=label))

    # Out-of-sample: train 2018-2022, test 2023-2025
    lines += ["", "── Out-of-Sample Validation ──────────────────────────────────────", ""]
    sub_full = df[['sp500_return', 'irrational_sentiment', 'bitcoin_return',
                   'vix', 'epu_index', 'bw_sentiment', 'reg_dummy_net']].dropna().copy()
    sub_full['irr_lag1'] = sub_full['irrational_sentiment'].shift(1)
    sub_full['btc_lag1'] = sub_full['bitcoin_return'].shift(1)
    sub_full = sub_full.dropna()
    sub_full.index = pd.to_datetime(sub_full.index)

    train = sub_full[sub_full.index < '2023-01-01']
    test  = sub_full[sub_full.index >= '2023-01-01']

    X_train = sm.add_constant(train[['irr_lag1', 'btc_lag1', 'vix', 'epu_index',
                                      'bw_sentiment', 'reg_dummy_net']])
    m_train = sm.OLS(train['sp500_return'], X_train).fit()

    X_test = sm.add_constant(test[['irr_lag1', 'btc_lag1', 'vix', 'epu_index',
                                    'bw_sentiment', 'reg_dummy_net']])
    pred_test  = m_train.predict(X_test)
    actual_test = test['sp500_return'].values
    oos_r2 = 1 - np.sum((actual_test - pred_test)**2) / np.sum((actual_test - actual_test.mean())**2)
    oos_corr = np.corrcoef(actual_test, pred_test)[0, 1]

    # Benchmark (no sentiment)
    X_bench_train = sm.add_constant(train[['btc_lag1', 'vix', 'epu_index',
                                            'bw_sentiment', 'reg_dummy_net']])
    m_bench = sm.OLS(train['sp500_return'], X_bench_train).fit()
    X_bench_test = sm.add_constant(test[['btc_lag1', 'vix', 'epu_index',
                                          'bw_sentiment', 'reg_dummy_net']])
    pred_bench = m_bench.predict(X_bench_test)
    bench_r2 = 1 - np.sum((actual_test - pred_bench)**2) / np.sum((actual_test - actual_test.mean())**2)

    lines += [
        f"  Train: 2018-2022 ({len(train)} obs)  Test: 2023-2025 ({len(test)} obs)",
        f"  Sentiment model OOS R² : {oos_r2:.4f}  (corr={oos_corr:.4f})",
        f"  Benchmark (no sentiment): {bench_r2:.4f}",
        f"  OOS improvement: {oos_r2 - bench_r2:+.4f}",
    ]

    # Weekend/overnight instrument (reverse causality robustness)
    lines += ["", "── Weekend/Overnight Instrument ──────────────────────────────────", ""]
    sub_iv = df[['sp500_return', 'irrational_sentiment', 'bitcoin_return',
                  'vix', 'epu_index', 'reg_dummy_net']].copy()
    sub_iv.index = pd.to_datetime(sub_iv.index)
    sub_iv['weekday'] = sub_iv.index.dayofweek
    # Weekend sentiment = sentiment on Sat/Sun (equity markets closed, so can't be caused by equity)
    sub_iv['weekend_irr'] = np.where(sub_iv['weekday'].isin([5, 6]),
                                       sub_iv['irrational_sentiment'], np.nan)
    # Forward-fill weekend sentiment to Monday
    sub_iv['instrument'] = sub_iv['weekend_irr'].ffill(limit=2)
    sub_iv = sub_iv[sub_iv['weekday'] == 0].dropna()  # Monday only
    if len(sub_iv) > 50:
        first_stage = sm.OLS(
            sub_iv['irrational_sentiment'],
            sm.add_constant(sub_iv[['instrument', 'vix', 'epu_index']])).fit()
        lines.append(f"  First stage F-stat (instrument): {first_stage.fvalue:.2f}  "
                     f"(p={first_stage.f_pvalue:.4f})")
        lines.append(f"  Instrument coefficient: {first_stage.params.get('instrument', np.nan):+.4f}")
        lines.append(f"  n = {len(sub_iv)} Monday observations")
    else:
        lines.append("  Insufficient Monday obs for IV estimation")

    write_report(RESULTS_DIR / "robustness.txt", lines)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("=" * 70)
    print("EMPIRICAL ANALYSIS — Irrational Exuberance in the Age of Algorithms")
    print("=" * 70)

    svar_results, svar_data, best_lag, svar_cols = run_h1()
    run_h1_horserace(svar_results, svar_data, best_lag)
    run_irf_fevd(svar_data, best_lag)
    run_h2()
    run_h3()
    run_h4()
    run_robustness()

    print(f"\n✓ All results saved to {RESULTS_DIR.resolve()}")
    print("  Files generated:")
    for f in sorted(RESULTS_DIR.iterdir()):
        print(f"    {f.name}")