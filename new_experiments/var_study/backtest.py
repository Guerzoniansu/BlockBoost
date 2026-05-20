"""
backtest.py  –  VaR backtesting: Kupiec test + summary statistics.

Convention: VaR is expressed as a *positive* loss threshold.
A violation occurs when   actual_return < −VaR.
"""

import numpy as np
from scipy import stats


def kupiec_test(actual: np.ndarray, var_forecasts: np.ndarray, alpha: float = 0.05):
    """
    Kupiec (1995) Proportion of Failures (POF) test.

    H0: violation probability equals alpha (unconditional coverage).

    Parameters
    ----------
    actual        : array of realised returns (signed, negative = loss)
    var_forecasts : array of VaR estimates as *positive* numbers
    alpha         : nominal coverage tail probability (e.g. 0.05)

    Returns
    -------
    lr_stat   : likelihood-ratio statistic
    p_value   : chi2(1) p-value (large = accept H0 = model well-calibrated)
    """
    violations = actual < -var_forecasts
    n   = len(actual)
    x   = int(np.sum(violations))
    p_hat = x / n if n > 0 else 0.0

    if x == 0 or x == n:
        # Boundary case — return trivial result
        return 0.0, 1.0

    lr_stat = -2.0 * (
        x * np.log(alpha / p_hat)
        + (n - x) * np.log((1.0 - alpha) / (1.0 - p_hat))
    )
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))
    return float(lr_stat), p_value


def christoffersen_test(actual: np.ndarray, var_forecasts: np.ndarray):
    """
    Christoffersen (1998) Independence test.
    H0: violations are independent.
    """
    violations = (actual < -var_forecasts).astype(int)
    n = len(violations)
    
    if n <= 1:
        return 0.0, 1.0
        
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        if violations[i-1] == 0 and violations[i] == 0: n00 += 1
        elif violations[i-1] == 0 and violations[i] == 1: n01 += 1
        elif violations[i-1] == 1 and violations[i] == 0: n10 += 1
        elif violations[i-1] == 1 and violations[i] == 1: n11 += 1
        
    pi_01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11) if (n00 + n01 + n10 + n11) > 0 else 0
    
    lr_ind = 0.0
    p_ind = 1.0
    
    # avoid log(0)
    eps = 1e-12
    # Ensure probabilities are strictly bounded for log computation
    pi_01_b = max(eps, min(1-eps, pi_01))
    pi_11_b = max(eps, min(1-eps, pi_11))
    pi_b    = max(eps, min(1-eps, pi))
    
    if (n01 + n11) > 0:
        term_null = (n00 + n10) * np.log(1 - pi_b) + (n01 + n11) * np.log(pi_b)
        
        term_alt = 0
        if (n00 + n01) > 0:
            term_alt += n00 * np.log(1 - pi_01_b) + n01 * np.log(pi_01_b)
        if (n10 + n11) > 0:
            term_alt += n10 * np.log(1 - pi_11_b) + n11 * np.log(pi_11_b)
            
        lr_ind = -2.0 * (term_null - term_alt)
        lr_ind = max(0.0, float(lr_ind))  # numeric precision 
        p_ind = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
        
    return lr_ind, p_ind


def evaluate_var(actual: np.ndarray, var_forecasts: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Comprehensive VaR evaluation.

    Returns a dict with:
      viol_rate  : observed violation frequency
      n_viol     : number of violations
      kupiec_lr  : Kupiec LR statistic
      kupiec_p   : Kupiec p-value
      christ_lr  : Christoffersen LR statistic
      christ_p   : Christoffersen p-value
      mae        : mean absolute error  |r_t + VaR_t|  (over all days)
      rmse       : root mean squared error
      es         : expected shortfall (mean loss on violation days)
    """
    if len(actual) == 0:
        return dict(viol_rate=0.0, n_viol=0, kupiec_lr=0.0, kupiec_p=1.0, 
                    christ_lr=0.0, christ_p=1.0, mae=0.0, rmse=0.0, es=0.0)

    violations = actual < -var_forecasts
    n_viol     = int(np.sum(violations))
    viol_rate  = n_viol / len(actual)

    lr_stat, kupiec_p = kupiec_test(actual, var_forecasts, alpha)
    christ_lr, christ_p = christoffersen_test(actual, var_forecasts)

    mae  = float(np.mean(np.abs(actual + var_forecasts)))
    rmse = float(np.sqrt(np.mean((actual + var_forecasts) ** 2)))

    if n_viol > 0:
        es = float(-np.mean(actual[violations]))
    else:
        es = 0.0

    return dict(
        viol_rate = viol_rate,
        n_viol    = n_viol,
        kupiec_lr = lr_stat,
        kupiec_p  = kupiec_p,
        christ_lr = christ_lr,
        christ_p  = christ_p,
        mae       = mae,
        rmse      = rmse,
        es        = es,
    )
