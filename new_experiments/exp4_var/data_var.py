"""
data_var.py  –  Data acquisition and feature building for VaR exceedance prediction.

Assets : SPY, AAPL, MSFT, JPM, XOM, GS
Target : y_t = 1[r_t < -VaR_{a,t}]
Split  : 70 % train | 15 % val | 15 % test  (chronological)
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

ASSETS   = ['SPY', 'AAPL', 'MSFT', 'JPM', 'XOM', 'GS']
START    = '2010-01-01'
END      = '2024-12-31'
TRAIN_F  = 0.70
VAL_F    = 0.15

HERE = os.path.dirname(__file__)


def _download(ticker: str) -> pd.Series:
    raw = yf.download(ticker, start=START, end=END,
                      auto_adjust=True, progress=False)
    close = raw['Close'].squeeze().dropna()
    return close


def load_returns(cache_dir=None):
    """
    Download (or load from cache) all assets.
    Returns dict: ticker → pd.Series of log returns.
    """
    cache_dir = cache_dir or os.path.join(HERE, 'results', 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    returns_dict = {}
    for ticker in ASSETS:
        cache_path = os.path.join(cache_dir, f'{ticker}_returns.csv')
        if os.path.exists(cache_path):
            returns = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze()
            print(f'  [{ticker}] loaded from cache')
        else:
            print(f'  [{ticker}] downloading …')
            close   = _download(ticker)
            returns = np.log(close / close.shift(1)).dropna()
            returns.name = ticker
            returns.to_csv(cache_path)
        returns_dict[ticker] = returns
    return returns_dict


def build_var_dataset(returns_series: pd.Series, sigmas: np.ndarray, 
                      var_limits: np.ndarray, hits: np.ndarray, mus: np.ndarray,
                      p_lags: int = 21, q_lags: int = 21, d_lags: int = 21):
    """
    Build feature matrix (X) and target (y) for VaR exceedance prediction.
    Features lag structure:
      r_{t-1} ... r_{t-p}
      sigma_{t-1} ... sigma_{t-q}
      VaR_{t-1}
      hit_{t-1} ... hit_{t-d}
      
    Target:
      y_t = hit_t mapped to {-1, +1} (or {0, 1})
      We use {-1, 1} for boosting compatibility. Exceedance = +1, no exceedance = -1.
    """
    r     = returns_series.values.astype(float)
    dates = returns_series.index
    n     = len(r)
    
    # We must skip the warmup period where VaR is NaN
    first_valid = np.where(~np.isnan(var_limits))[0][0]
    
    # Forward-fill any intermediate NaNs in sigmas or var_limits just in case GARCH failed for a block
    s_series = pd.Series(sigmas)
    sigmas = s_series.ffill().values
    v_series = pd.Series(var_limits)
    var_limits = v_series.ffill().values
    
    # Skip the initial lags needed AFTER the first valid GARCH prediction
    max_lag = max(p_lags, q_lags, d_lags)
    start_idx = first_valid + max_lag
    
    X, y, t_dates, raw_ret = [], [], [], []
    for t in range(start_idx, n):
        row = []
        # Lagged returns
        row.extend(r[t - p_lags : t][::-1])
        # Lagged sigmas
        row.extend(sigmas[t - q_lags : t][::-1])
        # Lagged VaR (1 day ago)
        row.append(var_limits[t - 1])
        # Lagged hits
        row.extend(hits[t - d_lags : t][::-1])
        
        X.append(row)
        # Class 1 if hit, -1 if no hit
        y.append(1 if hits[t] == 1 else -1)
        t_dates.append(dates[t])
        raw_ret.append(r[t])
        
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    
    # Split
    m = len(y)
    i1 = int(m * TRAIN_F)
    i2 = int(m * (TRAIN_F + VAL_F))
    
    return dict(
        X_train=X[:i1],  y_train=y[:i1],  dates_train=t_dates[:i1],  r_train=raw_ret[:i1],
        X_val  =X[i1:i2],y_val  =y[i1:i2],dates_val  =t_dates[i1:i2],r_val  =raw_ret[i1:i2],
        X_test =X[i2:],  y_test =y[i2:],  dates_test =t_dates[i2:],  r_test =raw_ret[i2:],
        X_full=X, y_full=y, dates_full=t_dates, returns_full=raw_ret,
        ticker=returns_series.name,
        # Save exact indexes mapping array -> full history (useful for DM test etc.)
        idx_train=np.arange(start_idx, start_idx + i1),
        idx_val=np.arange(start_idx + i1, start_idx + i2),
        idx_test=np.arange(start_idx + i2, start_idx + m)
    )
