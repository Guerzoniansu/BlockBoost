"""
models.py  –  BlockBoost VaR estimation via Strategy 1.

Strategy 1: Similarity-Weighted Historical Simulation
------------------------------------------------------
1. Fit a BlockBoostR2 model on training returns to predict volatility y = |r|.
2. For each test day, run the ensemble forward to obtain F_current
   (predicted volatility for that day).
3. Assign Gaussian kernel weights to each training observation based on how
   close its in-sample prediction F_train_i is to F_current.
4. Compute the weighted quantile of training returns at level alpha → VaR.

Rolling window: step_size is controlled by STEP_SIZE from data.py.
Model is re-estimated every STEP_SIZE days and produces a block of forecasts.
"""

import os
import sys
import numpy as np

# ── import BlockBoost from model/ ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)

from model.blockboost import BlockBoostR2Regressor
from joblib import Parallel, delayed

from data import STEP_SIZE

# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 Core
# ─────────────────────────────────────────────────────────────────────────────

def _strategy1_var_opt(F_train: np.ndarray, r_sorted: np.ndarray,
                       F_current: float, alpha: float, h: float, idx_sort: np.ndarray) -> float:
    """
    Gaussian-kernel weighted historical simulation VaR (Optimized inner loop).
    """
    n = len(r_sorted)

    # Kernel weights using pre-computed bandwidth
    w = np.exp(-0.5 * ((F_train - F_current) / h) ** 2)
    w_sum = w.sum()
    if w_sum < 1e-30:
        w = np.ones(n) / n
    else:
        w /= w_sum

    # Weighted quantile: apply pre-computed sort index
    w_sorted  = w[idx_sort]
    cum_w     = np.cumsum(w_sorted)
    idx_q     = np.searchsorted(cum_w, alpha)
    idx_q     = min(idx_q, n - 1)

    var = -r_sorted[idx_q]   # negate: VaR is a positive loss
    return float(var)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling window worker
# ─────────────────────────────────────────────────────────────────────────────

def _process_rolling_step(t, X, r, window_size, block_size,
                          n_estimators, alpha_levels):
    """
    Worker function for a rolling-window step block.

    Trains BlockBoostR2 on observations [t-window_size : t],
    forecasts VaR for observation block [t : t + STEP_SIZE].
    """
    X_train = X[t - window_size : t]
    r_train = r[t - window_size : t]
    X_test_block = X[t : t + STEP_SIZE]

    y_train = np.abs(r_train)

    model = BlockBoostR2Regressor(
        block_size   = block_size,
        n_estimators = n_estimators,
        max_depth    = 4,
    )
    model.fit(X_train, y_train, returns=r_train)
    F_train = model.train_predictions_
    
    # Predict volatility for all days in the block
    F_current_block = model.predict(X_test_block)

    # Pre-compute fixed components for the entire block
    n_train = len(r_train)
    h = np.std(F_train) * 1.06 * (n_train ** -0.2)
    if h < 1e-8: h = 1e-8
    
    idx_sort = np.argsort(r_train)
    r_sorted = r_train[idx_sort]

    block_forecasts = {alpha: [] for alpha in alpha_levels}
    for i in range(len(X_test_block)):
        F_curr = float(F_current_block[i])
        for alpha in alpha_levels:
            var_t = _strategy1_var_opt(F_train, r_sorted, F_curr, alpha, h, idx_sort)
            block_forecasts[alpha].append(var_t)

    return block_forecasts


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def rolling_forecast(X: np.ndarray, r: np.ndarray,
                     block_size: int,
                     n_estimators: int = 100,
                     alpha_levels: tuple = (0.05, 0.01),
                     window_size: int = 1000,
                     n_jobs: int = -1) -> dict:
    """
    Parallel rolling-window VaR forecast with jump-step logic.

    Parameters
    ----------
    X             : full feature matrix  (shape: N × n_lags)
    r             : full return series   (shape: N,)
    block_size    : BlockBoost block parameter a_T
    n_estimators  : boosting rounds
    alpha_levels  : tuple of tail probabilities
    window_size   : initial training window length
    n_jobs        : joblib parallel jobs

    Returns
    -------
    dict {alpha: np.ndarray of VaR forecasts}  length = N - window_size
    """
    n      = len(X)
    # Starts jump by STEP_SIZE (e.g. 63 days)
    starts = list(range(window_size, n, STEP_SIZE))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_rolling_step)(
            t, X, r, window_size, block_size, n_estimators, alpha_levels
        )
        for t in starts
    )

    # Flatten the blocks into a single continuous forecast series per alpha
    forecasts = {alpha: [] for alpha in alpha_levels}
    for block in results:
        for alpha in alpha_levels:
            forecasts[alpha].extend(block[alpha])

    # Truncate to ensure length matches total test observations (N - window_size)
    n_out = n - window_size
    return {alpha: np.array(forecasts[alpha])[:n_out] for alpha in alpha_levels}


