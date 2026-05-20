"""
data.py  –  Simulated AR(1) process with Regime-Switching GARCH(1,1) volatility.

Generates a single synthetic time series (Target: y_{t+1}) evaluated strictly 
out-of-sample to benchmark AdaBoost, BlockBoost, AR, and AR+GARCH.

Assets : SIM_AR1_RS
Period : 3500 observations
Target : y_{t+1} = c + \phi y_t + \sigma_{regime,t} \cdot \sigma_{garch,t} \cdot z_{t+1}
Features: [y_t, y_{t-1}, …, y_{t-L+1}]  (L = N_LAGS = 21)
Split  : 70 % train | 30 % test  (chronological)
"""

import os
import numpy as np
import pandas as pd

ASSETS   = ['SIM_AR1_RS']
N_LAGS   = 21        # lag window for feature matrix
TRAIN_F  = 0.70

HERE = os.path.dirname(__file__)


# ── Simulator ─────────────────────────────────────────────────────────────────

def generate_regime_switching_ar(n=3500, seed=42):
    """
    Generate realistic financial returns with regime switching.
    - Stochastic regime switching via Markov transition matrix
    - Separate sigma_regime from base GARCH sigma
    - Weak AR(1) mean component
    """
    np.random.seed(seed)
    
    # ── AR(1) parameters
    c = 0.0005
    phi = 0.08
    
    # ── Regime parameters (0=normal, 1=volatile, 2=crisis)
    # P[i, j] = Process transition probability from regime i to regime j
    P = np.array([
        [0.985, 0.010, 0.005],  # from normal
        [0.050, 0.930, 0.020],  # from volatile
        [0.080, 0.120, 0.800]   # from crisis
    ])
    sigma_regimes = [0.01, 0.02, 0.04]
    
    returns = np.zeros(n)
    sigma_garch = np.zeros(n)
    regimes = np.zeros(n, dtype=int)
    
    # Initial state
    regimes[0] = 0
    sigma_garch[0] = 1.0  # Base unit GARCH volatility
    returns[0] = c + sigma_regimes[0] * sigma_garch[0] * np.random.standard_t(df=6)
    
    for t in range(1, n):
        # Markov regime transition
        curr_regime = regimes[t-1]
        next_regime = np.random.choice([0, 1, 2], p=P[curr_regime])
        regimes[t] = next_regime
        
        sigma_regime = sigma_regimes[next_regime]
        
        # Base unit GARCH(1,1) dynamics (uses strictly the standardized residuals)
        # Note: We track the base volatility independent of the regime multiplier
        # e_t is the innovation without the regime scaling
        e_prev = returns[t-1] - (c + phi * returns[t-2] if t > 1 else c)
        z_prev = e_prev / (sigma_regimes[curr_regime] * sigma_garch[t-1])
        
        # Unit variance GARCH scaling: omega + alpha * z_{t-1}^2 + beta * sigma_{t-1}^2
        # where omega + alpha + beta = 1 to target unit base variance
        omega, alpha, beta = 0.05, 0.10, 0.85
        sigma_garch[t] = np.sqrt(omega + alpha * z_prev**2 + beta * sigma_garch[t-1]**2)
        
        # Student-t innovations
        z_t = np.random.standard_t(df=6)
        
        # AR(1) observation
        returns[t] = c + phi * returns[t-1] + sigma_regime * sigma_garch[t] * z_t
        
    return pd.Series(returns, name='SIM_AR1_RS')


# ── Feature Matrix Builder ────────────────────────────────────────────────────

def _make_features_labels(returns: pd.Series, n_lags: int = N_LAGS):
    """
    Build lag-feature matrix and direction labels.
    Row t of X  : [y_t, y_{t-1}, …, y_{t-n_lags+1}]
    Row t of y  : y_{t+1}
    Valid rows  : t ∈ [n_lags-1, T-2]  (need one future step for label)
    """
    r  = np.abs(returns.values.astype(float))
    T  = len(r)
    rows = []
    for t in range(n_lags - 1, T - 1):
        feat = r[t - n_lags + 1 : t + 1][::-1]   # [y_t, y_{t-1}, …]
        rows.append(feat)
    X     = np.array(rows)
    
    # Fake dates for API compatibility
    dates = pd.date_range(start='2010-01-01', periods=len(rows), freq='D')
    
    y_raw = returns.values.astype(float)[n_lags : T] # Keep raw for reference if needed
    y = r[n_lags : T]
    return X, y, dates, y_raw


def _split(X, y, dates, y_raw):
    n = len(y)
    i1 = int(n * TRAIN_F)
    return dict(
        X_train=X[:i1],  y_train=y[:i1],  dates_train=dates[:i1],  r_train=y_raw[:i1],
        X_test =X[i1:],  y_test =y[i1:],  dates_test =dates[i1:],  r_test =y_raw[i1:],
        returns_full=y_raw, dates_full=dates,
        X_full=X, y_full=y,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_all(seed=42, n_lags=N_LAGS):
    """
    Generates simulated standard assets for a specific random seed.
    Returns dict containing X_train/test, y_*, dates_*, etc.
    """
    ticker = f'SIM_Seed_{seed}'
    print(f'  [{ticker}] Generating synthetic AR(1) sequence …')
    returns = generate_regime_switching_ar(n=3500, seed=seed)

    X, y, dates, y_raw = _make_features_labels(returns, n_lags)
    split = _split(X, y, dates, y_raw)
    split['returns_series'] = returns
    split['ticker'] = ticker

    # summary
    n = len(y)
    mean_ret = y.mean()
    std_ret  = y.std()
    print(f'    n={n}, mean_ret={mean_ret:.5f}, std={std_ret:.5f}, '
          f'train={split["X_train"].shape[0]}, '
          f'test={split["X_test"].shape[0]}')

    return {ticker: split}


if __name__ == '__main__':
    data = load_all()
    for t, d in data.items():
        print(t, d['X_train'].shape, d['y_train'].shape)
