"""
garch_model.py  –  AR(p)+GARCH(1,1) with Student-t innovations.

Uses the `arch` library for fitting.
Rolling-origin: refit every stride=21 steps (monthly).
"""

import warnings
import numpy as np
from scipy import stats
from ar_model import validate_threshold

warnings.filterwarnings('ignore')

STRIDE = 21


# ── out-of-sample GARCH features ──────────────────────────────────────────────

def generate_garch_features(returns_full: np.ndarray, p: int, alpha: float = 0.05,
                            warmup: int = 252, stride: int = 126):
    """
    Generate out-of-sample 1-step-ahead GARCH(1,1) conditional volatilities and VaR.
    Refits the model every `stride` days on an expanding window.
    
    Returns:
        sigmas: array of length N (first `warmup` are NaNs or in-sample)
        var_limits: array of length N (VaR thresholds, positive numbers typically)
        hits: array of length N (1 if returns_full < -var_limits else 0)
    """
    from arch import arch_model
    n = len(returns_full)
    r = returns_full * 100.0  # scale to %
    
    sigmas     = np.full(n, np.nan)
    var_limits = np.full(n, np.nan)
    mus        = np.full(n, np.nan)
    
    # We will step through the data
    # At each step t, we fit on r[:t], then use those params to get sigmas for r[:t+stride]
    
    t_start = warmup
    while t_start < n:
        t_end = min(t_start + stride, n)
        
        # Fit on history
        history = r[:t_start]
        try:
            am = arch_model(history, vol='Garch', p=1, q=1, mean='ARX', lags=p, dist='t')
            res = am.fit(disp='off', show_warning=False)
            params = res.params
            df = params.get('nu', 5.0)
        except Exception:
            # Fallback to normal if t fails
            try:
                am = arch_model(history, vol='Garch', p=1, q=1, mean='ARX', lags=p, dist='normal')
                res = am.fit(disp='off', show_warning=False)
                params = res.params
                df = None
            except Exception:
                params = None
                df = None

        if params is not None:
            # Filter the extended window
            window = r[:t_end]
            am_fwd = arch_model(window, vol='Garch', p=1, q=1, mean='ARX', lags=p, 
                                dist='t' if df is not None else 'normal')
            res_fwd = am_fwd.fix(params)
            
            # The conditional volatility at t is the forecast for t given data up to t-1
            # `res_fwd.conditional_volatility` has the same length as `window`
            chunk_sigmas = res_fwd.conditional_volatility[t_start:t_end] / 100.0
            chunk_mus    = (window - res_fwd.resid)[t_start:t_end] / 100.0
            
            # VaR calculation
            if df is not None:
                q = stats.t.ppf(alpha, df=df)
            else:
                q = stats.norm.ppf(alpha)
                
            chunk_var = -(chunk_mus + chunk_sigmas * q)  # VaR is typically expressed as a positive number
            
            sigmas[t_start:t_end] = chunk_sigmas
            mus[t_start:t_end]    = chunk_mus
            var_limits[t_start:t_end] = chunk_var
        
        t_start = t_end

    # Hit sequence: 1 if return < -VaR
    hits = np.zeros(n, dtype=int)
    valid = ~np.isnan(var_limits)
    hits[valid] = (returns_full[valid] < -var_limits[valid]).astype(int)
    
    return sigmas, var_limits, hits, mus
