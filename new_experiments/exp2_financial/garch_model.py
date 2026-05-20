"""
garch_model.py  –  AR(p)+GARCH(1,1) with Student-t innovations.

Uses the `arch` library for fitting.
Rolling-origin: refit every stride=21 steps (monthly).
"""

import warnings
import numpy as np
from scipy import stats

warnings.filterwarnings('ignore')

STRIDE = 21


# ── single-step conditional forecast ───────────────────────────────────────

def _garch_forecast(returns_history: np.ndarray, p: int) -> tuple:
    """
    Fit ARX(p)+GARCH(1,1) with t-innovations on history.
    Return (mu_hat, sigma_hat, df_hat, aic, bic).
    """
    from arch import arch_model
    if len(returns_history) < max(p + 10, 50):
        return 0.0, 1.0, 5.0, np.nan, np.nan
    try:
        # Scale returns to % (helps GARCH convergence)
        r   = returns_history * 100.0
        am  = arch_model(r, vol='Garch', p=1, q=1,
                         mean='ARX', lags=p,
                         dist='t')
        res = am.fit(disp='off', show_warning=False)
        fc  = res.forecast(horizon=1, reindex=False)
        mu    = float(fc.mean.values[-1, 0]) / 100.0
        var   = float(fc.variance.values[-1, 0]) / 10000.0
        sigma = float(np.sqrt(max(var, 1e-10)))
        df    = float(res.params.get('nu', 5.0))
        return mu, sigma, df, float(res.aic), float(res.bic)
    except Exception:
        return 0.0, 1.0, 5.0, np.nan, np.nan


# ── rolling-origin evaluation ─────────────────────────────────────────────────

def rolling_garch_predictions(returns_full: np.ndarray,
                               y_full: np.ndarray,
                               train_end: int,
                               p: int,
                               stride: int = STRIDE,
                               n_jobs: int = -1) -> dict:
    """
    Walk-forward GARCH predictions.
    Refits every `stride` steps in parallel chunks.
    """
    from joblib import Parallel, delayed

    n = len(returns_full)
    steps = list(range(train_end - 1, n - 1))

    # Group steps into chunks of `stride`; refit at start of each chunk
    def _run_chunk(chunk_steps):
        results = []
        for t in chunk_steps:
            history = returns_full[:t + 1]
            mu, sigma, df, aic, bic = _garch_forecast(history, p)
            results.append((mu, sigma, df, aic, bic))
        return results

    # Build chunks (refit = one fit per chunk)
    chunks = [steps[i:i+stride] for i in range(0, len(steps), stride)]

    chunk_results = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(_run_chunk)(chunk) for chunk in chunks
    )

    # Flatten
    flat = [item for chunk in chunk_results for item in chunk]
    mus    = np.array([x[0] for x in flat])
    sigmas = np.array([x[1] for x in flat])
    aics   = np.array([x[3] for x in flat])
    bics   = np.array([x[4] for x in flat])

    y_test   = y_full[train_end : n]
    preds_test = mus

    return dict(
        y_test=y_test.tolist(),
        preds_test=preds_test.tolist(),
        mean_aic=float(np.nanmean(aics)),
        mean_bic=float(np.nanmean(bics)),
        mus=mus.tolist(), sigmas=sigmas.tolist(),
    )
