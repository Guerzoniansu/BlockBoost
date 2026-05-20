"""
data.py  –  Data loading for the multi-dataset BlockBoost VaR study.

Dataset registry
----------------
Each entry maps a short dataset key to:
  - t_csv     : path to data_t_*.csv  (training+test returns, column 'returns')
  - garch_01  : path to data_*_01.csv (GARCH VaR at α=0.01 + realised returns)
  - garch_05  : path to data_*_05.csv (GARCH VaR at α=0.05 + realised returns)
  - ret_col   : name of the return column in the garch CSV (for alignment check)
  - label     : human-readable name for plots/tables

Structure of data_t_*.csv  →  date | returns
Structure of data_*_01/05.csv  →  date | ret_col | VaRGARCH_N … VaRaGARCH_S
"""

import os
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
_DATA = os.path.join(_ROOT, 'data')

# ── Experiment constants ───────────────────────────────────────────────────────
N_LAGS      = 21      # lagged feature count
STEP_SIZE   = 1       # rolling step = 63 days

# GARCH column names present in every _01 / _05 CSV
GARCH_COLS = [
    'VaRGARCH_N', 'VaRGARCH_S',
    'VaReGARCH_N', 'VaReGARCH_S',
    'VaRgGARCH_N', 'VaRgGARCH_S',
    'VaRaGARCH_N', 'VaRaGARCH_S',
]

# ── Dataset registry ───────────────────────────────────────────────────────────
DATASETS = {
    'c': dict(
        t_csv    = os.path.join(_DATA, 'data_t_c.csv'),
        garch_01 = os.path.join(_DATA, 'data_c_01.csv'),
        garch_05 = os.path.join(_DATA, 'data_c_05.csv'),
        ret_col  = 'r_log_BRVM',
        label    = 'BRVM Composite',
    ),
    'f': dict(
        t_csv    = os.path.join(_DATA, 'data_t_f.csv'),
        garch_01 = os.path.join(_DATA, 'data_f_01.csv'),
        garch_05 = os.path.join(_DATA, 'data_f_05.csv'),
        ret_col  = 'r_log_F',
        label    = 'BRVM Finance',
    ),
    'ca': dict(
        t_csv    = os.path.join(_DATA, 'data_t_ca.csv'),
        garch_01 = os.path.join(_DATA, 'data_ca_01.csv'),
        garch_05 = os.path.join(_DATA, 'data_ca_05.csv'),
        ret_col  = 'r_log_ca',
        label    = 'CAC 40',
    ),
    's': dict(
        t_csv    = os.path.join(_DATA, 'data_t_s.csv'),
        garch_01 = os.path.join(_DATA, 'data_s_01.csv'),
        garch_05 = os.path.join(_DATA, 'data_s_05.csv'),
        ret_col  = 'r_log_s',
        label    = 'S&P 500',
    ),
}


# ── Feature engineering ────────────────────────────────────────────────────────

def _build_features(returns: np.ndarray, n_lags: int = N_LAGS) -> np.ndarray:
    """
    Feature matrix: absolute value of lags 1..n_lags of r_t.
    Rows 0..(n_lags-1) are dropped (affected by np.roll boundary).
    Returns X of shape (len(returns) - n_lags, n_lags).
    """
    cols = []
    for lag in range(1, n_lags + 1):
        cols.append(np.roll(np.abs(returns), lag))
    X = np.column_stack(cols)
    return X[n_lags:]


# ── Main loader ────────────────────────────────────────────────────────────────

def load_dataset(key: str, n_lags: int = N_LAGS):
    """
    Load a dataset by its registry key (e.g. 'c', 'f', 'ca', 's').

    Returns
    -------
    X           : np.ndarray  shape (N_total - n_lags, n_lags)  – feature matrix
    r           : np.ndarray  shape (N_total - n_lags,)          – log returns
    dates       : pd.DatetimeIndex  aligned with X and r
    garch_01    : dict {col_name: np.ndarray}  – GARCH VaR at α=0.01
    garch_05    : dict {col_name: np.ndarray}  – GARCH VaR at α=0.05
    r_test_garch: np.ndarray  – realised returns aligned with GARCH test rows
    dates_test  : pd.DatetimeIndex  – dates aligned with GARCH test rows
    window_size : dynamically calculated split index (start of test period)
    meta        : dict  – {'label': ..., 'ret_col': ...}
    """
    info = DATASETS[key]

    # ── 1. Load full return series (training + test) ──
    t_df = pd.read_csv(info['t_csv'], parse_dates=['date'])
    t_df = t_df.dropna(subset=['returns']).reset_index(drop=True)
    returns_all = t_df['returns'].values.astype(np.float64)

    # Build features (drops first n_lags rows)
    X       = _build_features(returns_all, n_lags)
    r       = returns_all[n_lags:]
    dates   = pd.DatetimeIndex(t_df['date'].values[n_lags:])
    n_total = len(r)

    # ── 2. Load GARCH benchmarks (test period already, with ret_col) ──
    def _load_garch_csv(path, ret_col):
        df = pd.read_csv(path, parse_dates=['date'])
        df = df.dropna(subset=[ret_col]).reset_index(drop=True)
        garch_vars = {}
        for col in GARCH_COLS:
            if col in df.columns:
                garch_vars[col] = df[col].values.astype(np.float64)
        r_test = df[ret_col].values.astype(np.float64)
        dates_test = pd.DatetimeIndex(df['date'].values)
        return garch_vars, r_test, dates_test

    garch_01, r_test_01, dates_01 = _load_garch_csv(info['garch_01'], info['ret_col'])
    garch_05, r_test_05, dates_05 = _load_garch_csv(info['garch_05'], info['ret_col'])

    r_test_garch = r_test_01
    dates_test   = dates_01

    # ── 3. Find exact split idx based on dates ──
    # Locate where dates_01 starts inside dates
    first_test_date = dates_01[0]
    try:
        split_idx = int(np.where(dates == first_test_date)[0][0])
    except IndexError:
        raise ValueError(f"First test date {first_test_date} not found in t_csv dates!")

    window_size = split_idx
    n_test      = len(r) - window_size

    print(f'  [data:{key}] N={n_total}  train={window_size}  test={n_test}')
    print(f'  [data:{key}] GARCH test rows: 01→{len(r_test_01)}  05→{len(r_test_05)}')

    return (X, r, dates,
            garch_01, garch_05,
            r_test_garch, dates_test, window_size,
            {'label': info['label'], 'ret_col': info['ret_col']})
