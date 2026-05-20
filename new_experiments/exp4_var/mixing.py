"""
mixing.py  –  Nonparametric β-mixing coefficient estimation.

Reference : McDonald, D.J., Shalizi, C.R., & Schervish, M. (2011).
            "Estimating beta-mixing coefficients."
            AISTATS 2011.

Method
------
For a block size b, split the sequence into non-overlapping blocks of length b.
Estimate β(b) as the L1 distance between the joint distribution of two
consecutive blocks and the product of their marginals, approximated via
k-NN density estimation on the 2b-dimensional joint space vs the b-dimensional
marginal spaces.

We use a simplified, computationally robust version:
    β̂(b) = (1/2) E[ |p(X_{1:b}, X_{b+1:2b}) / (p(X_{1:b}) p(X_{b+1:2b})) - 1| ]
which is estimated via the Kozachenko-Leonenko k-NN log density estimator
applied to the joint and marginal samples.

Block sizes : b ∈ {1,2,3,4,5,6,7,8,9,10,15,20,30,40,50}
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

HERE = os.path.dirname(__file__)
BLOCK_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50]
K_NN = 5   # number of nearest neighbours


# ── k-NN log-density estimator (Kozachenko-Leonenko) ─────────────────────────

def _kl_log_density(X: np.ndarray, k: int = K_NN) -> np.ndarray:
    """
    Per-sample log density estimate  log p̂(x_i)  using KL estimator.
    X : (n, d) array
    Returns (n,) array of log density values.
    """
    n, d = X.shape
    tree = cKDTree(X)
    # k+1 because the query point itself is included
    dists, _ = tree.query(X, k=k + 1)
    rho = dists[:, -1]   # distance to k-th neighbour (exclude self)
    rho = np.maximum(rho, 1e-10)
    # KL estimator: log p̂(x) = log(k) - log(n-1) - d*log(rho) - log(V_d)
    # V_d = pi^{d/2} / Gamma(d/2+1) but constant cancels in ratios, we keep it:
    import math
    log_Vd = (d / 2) * np.log(np.pi) - math.lgamma(d / 2 + 1)
    log_p  = np.log(k) - np.log(n - 1) - d * np.log(rho) - log_Vd
    return log_p


def _beta_one_block(returns: np.ndarray, b: int, n_pairs: int = 200,
                    k: int = K_NN, rng_seed: int = 0) -> float:
    """
    Estimate β(b) for a single block size using random consecutive block pairs.
    returns : 1-D array of log-returns
    """
    rng  = np.random.default_rng(rng_seed)
    n    = len(returns)
    # Build non-overlapping blocks of size b
    n_blocks = n // b
    if n_blocks < 4:
        return np.nan
    blocks = [returns[i*b : (i+1)*b] for i in range(n_blocks)]

    # Sample random consecutive pairs of blocks
    max_pairs = min(n_pairs, n_blocks - 1)
    pair_idx  = rng.choice(n_blocks - 1, size=max_pairs, replace=False)

    # Joint samples: (X_{k}, X_{k+1}) concatenated → 2b-dim
    joint     = np.stack([np.concatenate([blocks[i], blocks[i+1]])
                          for i in pair_idx])    # (n_pairs, 2b)
    marginal1 = np.stack([blocks[i  ] for i in pair_idx])  # (n_pairs, b)
    marginal2 = np.stack([blocks[i+1] for i in pair_idx])  # (n_pairs, b)

    if joint.shape[0] < K_NN + 2:
        return np.nan

    # Log densities
    log_p_joint = _kl_log_density(joint, k)
    log_p_m1    = _kl_log_density(marginal1, k)
    log_p_m2    = _kl_log_density(marginal2, k)

    # log-ratio  r = log p(X,Y) – log p(X) – log p(Y)
    log_ratio = log_p_joint - log_p_m1 - log_p_m2

    # β̂(b) = (1/2) E[|exp(log_ratio) – 1|]  capped at 1
    ratio     = np.exp(np.clip(log_ratio, -10, 10))
    beta_hat  = 0.5 * np.mean(np.abs(ratio - 1.0))
    return float(np.clip(beta_hat, 0.0, 1.0))


# ── parallel over block sizes ─────────────────────────────────────────────────

def estimate_beta_mixing(returns: np.ndarray,
                         block_sizes=None, n_jobs=-1) -> dict:
    """
    Estimate β(b) for all block sizes in parallel.
    Returns dict: b → β̂(b)
    """
    if block_sizes is None:
        block_sizes = BLOCK_SIZES
    betas = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_beta_one_block)(returns, b, n_pairs=300, k=K_NN)
        for b in block_sizes
    )
    return {b: float(v) if v is not None else np.nan
            for b, v in zip(block_sizes, betas)}


# ── exponential fit ───────────────────────────────────────────────────────────

def _fit_exponential(bs, betas):
    """Fit β(b) ≈ C * exp(-λ*b); return (C, λ, mixing_horizon)."""
    from scipy.optimize import curve_fit
    valid = [(b, v) for b, v in zip(bs, betas) if not np.isnan(v) and v > 1e-6]
    if len(valid) < 3:
        return None, None, None
    bv  = np.array([x[0] for x in valid], dtype=float)
    betv = np.array([x[1] for x in valid], dtype=float)
    try:
        popt, _ = curve_fit(lambda b, C, lam: C * np.exp(-lam * b),
                            bv, betv, p0=[1.0, 0.1], maxfev=5000)
        C, lam = popt
        horizon = int(np.ceil(1.0 / lam)) if lam > 1e-6 else None
        return float(C), float(lam), horizon
    except Exception:
        return None, None, None


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_mixing(all_betas: dict, out_dir: str):
    """
    all_betas : dict  ticker → {b: β̂(b)}
    """
    plt.rcParams.update({'font.family': 'serif',
                         'mathtext.fontset': 'dejavuserif', 'font.size': 9})
    fig, ax = plt.subplots(figsize=(7, 4))
    tickers = list(all_betas.keys())
    cmap    = plt.cm.get_cmap('tab10', len(tickers))

    fitted = {}
    for i, ticker in enumerate(tickers):
        bd  = all_betas[ticker]
        bs  = [b for b in BLOCK_SIZES if b in bd]
        bv  = [bd[b] for b in bs if not np.isnan(bd.get(b, np.nan))]
        bs2 = [b for b in bs if not np.isnan(bd.get(b, np.nan))]
        if not bs2:
            continue
        ax.plot(bs2, bv, 'o-', color=cmap(i), lw=1.2,
                markersize=4, label=ticker)
        C, lam, horizon = _fit_exponential(bs2, bv)
        fitted[ticker] = dict(C=C, lam=lam, mixing_horizon=horizon)
        if C is not None and lam is not None:
            b_fine = np.linspace(bs2[0], bs2[-1], 200)
            ax.plot(b_fine, C * np.exp(-lam * b_fine),
                    '--', color=cmap(i), lw=0.8, alpha=0.6)

    ax.set_xlabel('Block size $b$', fontsize=10)
    ax.set_ylabel('$\\hat{\\beta}(b)$', fontsize=10)
    ax.set_title('Estimated $\\beta$-Mixing Coefficients by Asset', fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, ls='--', lw=0.4, alpha=0.5, color='#aaaaaa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'beta_mixing.{fmt}'),
                    format=fmt, bbox_inches='tight',
                    dpi=300 if fmt == 'png' else None)
    plt.close(fig)
    return fitted


# ── run standalone ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))
    from new_experiments.exp2_financial.data import load_all
    datasets = load_all()
    res = {}
    for ticker, d in datasets.items():
        print(f'[{ticker}] estimating beta-mixing …')
        ret = d['returns_series'].values.astype(float)
        res[ticker] = estimate_beta_mixing(ret)
    os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    plot_mixing(res, os.path.join(HERE, 'figures'))
    with open(os.path.join(HERE, 'results', 'beta_mixing.json'), 'w') as f:
        json.dump({t: {str(b): v for b, v in d.items()} for t, d in res.items()}, f, indent=2)
    print('Done.')
