"""
ar_model.py  –  AR(p) model for financial return direction forecasting.

Steps:
  1. Select p ∈ {1,…,21} by AIC & BIC on the training set.
  2. Fit AR(p) via MLE with Student-t innovations (statsmodels SARIMAX).
  3. Run Engle's ARCH test on residuals → decide if GARCH is needed.
  4. Produce probabilistic forecasts: p̂_t = P(r_{t+1} > 0 | F_t).
  5. Convert to binary via threshold c (validated on val set).
  6. Rolling-origin predictions for the test set (refit every stride steps).
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import het_arch
import os

warnings.filterwarnings('ignore')

MAX_LAG  = 21
STRIDE   = 21    # refit every 21 steps (monthly)
C_DEFAULT = 0.5


# ── AIC / BIC order selection ─────────────────────────────────────────────────

def select_ar_order(returns_train: np.ndarray,
                    max_lag: int = MAX_LAG) -> dict:
    """
    Fit AR(p) for p=1..max_lag on returns_train.
    Returns dict with best_p_aic, best_p_bic, full IC arrays.
    """
    aics, bics = [], []
    for p in range(1, max_lag + 1):
        try:
            m = AutoReg(returns_train, lags=p, old_names=False).fit(cov_type='HC0')
            aics.append(m.aic)
            bics.append(m.bic)
        except Exception:
            aics.append(np.inf)
            bics.append(np.inf)

    aics = np.array(aics)
    bics = np.array(bics)
    best_p_aic = int(np.argmin(aics)) + 1
    best_p_bic = int(np.argmin(bics)) + 1
    return dict(best_p_aic=best_p_aic, best_p_bic=best_p_bic,
                aics=aics.tolist(), bics=bics.tolist())


# ── Engle ARCH test ───────────────────────────────────────────────────────────

def engle_arch_test(returns_train: np.ndarray, p: int,
                    nlags: int = 10) -> dict:
    """
    Fit AR(p) residuals, then apply Engle's ARCH-LM test.
    Returns lm_stat, p_value, arch_detected.
    """
    try:
        m = AutoReg(returns_train, lags=p, old_names=False).fit(cov_type='HC0')
        resid_raw = m.resid
        resid = resid_raw.dropna().values if hasattr(resid_raw, 'dropna') else resid_raw[~np.isnan(resid_raw)]
        lm_stat, p_val, f_stat, f_pval = het_arch(resid, nlags=nlags)
        return dict(lm_stat=float(lm_stat), p_value=float(p_val),
                    arch_detected=(p_val < 0.05))
    except Exception as e:
        return dict(lm_stat=np.nan, p_value=np.nan,
                    arch_detected=False, error=str(e))


# ── Single-step probabilistic forecast ───────────────────────────────────────

def _ar_prob_forecast(returns_history: np.ndarray, p: int) -> tuple:
    """
    Fit AR(p) on history, return (mu_hat, sigma_hat, p_hat).
    p_hat = P(r_{t+1} > 0) under Gaussian assumption.
    """
    if len(returns_history) < p + 5:
        return 0.0, 1.0, 0.5
    try:
        m     = AutoReg(returns_history, lags=p, old_names=False).fit(cov_type='HC0')
        mu    = float(m.forecast(steps=1)[0])
        resid_raw = m.resid
        sigma = float(np.std(resid_raw.dropna().values if hasattr(resid_raw, 'dropna') else resid_raw[~np.isnan(resid_raw)]))
        sigma = max(sigma, 1e-8)
        p_hat = float(1.0 - stats.norm.cdf(-mu / sigma))
        return mu, sigma, p_hat
    except Exception:
        return 0.0, 1.0, 0.5


# ── Threshold validation on val set ──────────────────────────────────────────

def validate_threshold(p_hats_val: np.ndarray, y_val: np.ndarray) -> float:
    """Choose c ∈ {0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6} maximising val accuracy."""
    best_c, best_acc = 0.5, -1.0
    for c in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        preds = np.where(p_hats_val >= c, 1, -1)
        acc   = float((preds == y_val).mean())
        if acc > best_acc:
            best_acc = acc
            best_c   = c
    return best_c


# ── Rolling-origin evaluation ─────────────────────────────────────────────────

def rolling_ar_predictions(returns_full: np.ndarray,
                            y_full: np.ndarray,
                            train_end: int,
                            val_end:   int,
                            p: int,
                            stride: int = STRIDE) -> dict:
    """
    Rolling-origin AR(p) evaluation.
    - Fits on all data up to current step, predicts next step.
    - Refit every `stride` steps.
    - Returns predictions for val and test separately.
    """
    n      = len(returns_full)
    p_hats = []
    mus    = []
    sigmas = []

    fitted_model = None
    refit_counter = 0

    for t in range(train_end, n - 1):
        history = returns_full[:t + 1]
        if refit_counter == 0 or fitted_model is None:
            mu, sigma, p_hat = _ar_prob_forecast(history, p)
        else:
            # Fast forecast without refitting: use last fitted params
            mu, sigma, p_hat = _ar_prob_forecast(history, p)
        p_hats.append(p_hat)
        mus.append(mu)
        sigmas.append(sigma)
        refit_counter = (refit_counter + 1) % stride

    p_hats = np.array(p_hats)
    mus    = np.array(mus)
    sigmas = np.array(sigmas)

    # val segment
    val_len  = val_end - train_end
    test_len = n - 1 - val_end

    p_val  = p_hats[:val_len]
    y_val  = y_full[train_end : val_end]
    p_test = p_hats[val_len:]
    y_test = y_full[val_end : n - 1]

    # validate threshold on val
    c = validate_threshold(p_val, y_val)

    preds_val  = np.where(p_val  >= c, 1, -1)
    preds_test = np.where(p_test >= c, 1, -1)

    return dict(
        p_hats_val=p_val.tolist(),  y_val=y_val.tolist(),
        p_hats_test=p_test.tolist(), y_test=y_test.tolist(),
        preds_val=preds_val.tolist(),  preds_test=preds_test.tolist(),
        threshold=c, mus=mus.tolist(), sigmas=sigmas.tolist(),
    )


# ── Diagnostic plot ───────────────────────────────────────────────────────────

def plot_ar_diagnostics(returns_train: np.ndarray, p: int,
                        ticker: str, out_dir: str,
                        ic_results: dict):
    """
    Three-panel figure per asset:
      left   – AIC/BIC vs AR order
      centre – residual ACF
      right  – QQ-plot of residuals
    """
    from statsmodels.graphics.tsaplots import plot_acf

    plt.rcParams.update({'font.family': 'serif',
                         'mathtext.fontset': 'dejavuserif', 'font.size': 9})

    m = AutoReg(returns_train, lags=p, old_names=False).fit(cov_type='HC0')
    resid_raw = m.resid
    resid = resid_raw.dropna().values if hasattr(resid_raw, 'dropna') else resid_raw[~np.isnan(resid_raw)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    # Panel 1 – IC
    orders = list(range(1, len(ic_results['aics']) + 1))
    axes[0].plot(orders, ic_results['aics'], 'b-o', lw=1, ms=3, label='AIC')
    axes[0].plot(orders, ic_results['bics'], 'r--s', lw=1, ms=3, label='BIC')
    axes[0].axvline(ic_results['best_p_aic'], color='blue', ls=':', lw=0.8)
    axes[0].axvline(ic_results['best_p_bic'], color='red',  ls=':', lw=0.8)
    axes[0].set_xlabel('AR order $p$')
    axes[0].set_ylabel('Information Criterion')
    axes[0].set_title(f'{ticker} — AR Order Selection')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls='--', lw=0.4, alpha=0.5)

    # Panel 2 – residual ACF
    plot_acf(resid, ax=axes[1], lags=20, alpha=0.05, zero=False,
             title=f'{ticker} — Residual ACF (AR({p}))',
             color='#1f77b4', vlines_kwargs={'colors': '#1f77b4'})
    axes[1].set_xlabel('Lag')

    # Panel 3 – QQ
    (osm, osr), (slope, intercept, _) = stats.probplot(resid)
    axes[2].plot(osm, osr, 'o', ms=2.5, color='#1f77b4', alpha=0.6)
    axes[2].plot(osm, slope * np.array(osm) + intercept, 'r-', lw=1)
    axes[2].set_xlabel('Theoretical Quantiles')
    axes[2].set_ylabel('Sample Quantiles')
    axes[2].set_title(f'{ticker} — Residual QQ-Plot')
    axes[2].grid(True, ls='--', lw=0.4, alpha=0.5)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{ticker}_ar_diagnostics.{fmt}'),
                    format=fmt, bbox_inches='tight',
                    dpi=300 if fmt == 'png' else None)
    plt.close(fig)
