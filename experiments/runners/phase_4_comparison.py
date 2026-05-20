"""
phase_4_comparison.py

Compare BlockBoostWeighted (γ*, λ* from phase_4_optimize_gamma_lambda),
standard BlockBoost, and AdaBoost across 12 block sizes on AR(5) with Markov noise.

Protocol
--------
DGP     : AR(5) with ar_coeffs = [0.4, 0.2, 0.1, 0.05, 0.05], n_features = 10
          n_train = 1000 (noisy), n_test = 1000 (clean)
Noise   : Markov noise, eta = 0.15, alpha_markov = 0.8
M       : 200 boosting rounds
Block sizes: [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

For each block size a_T, we run:
  - BlockBoostWeighted(gamma=γ*, lam=λ*)
  - BlockBoost (standard, gamma=0 ≡ uniform)
  - AdaBoost   (a_T = 1 by definition, plotted as a horizontal reference)

Plot: 4×3 grid of subplots, one per block size, each showing the test
      accuracy learning curve of the three algorithms over boosting rounds.
      Professional, ready for a scientific paper (serif font, muted colours).
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.utils.reproducibility import set_seed
from experiments.dgp.ar_process import generate_ar_process, generate_labels
from experiments.dgp.noise import generate_markov_noise
from model.blockboost import BlockBoostClassifier
from model.blockboost_weighted import BlockBoostWeightedClassifier
from model.adaboost import AdaBoostM1

# ─── Configuration ────────────────────────────────────────────────────────────
AR_COEFFS    = [0.4, 0.2, 0.1, 0.05, 0.05]
N_FEATURES   = 10
N_TRAIN      = 1000
N_TEST       = 1000
ETA          = 0.15
ALPHA_MARKOV = 0.8
M_ROUNDS     = 200
BLOCK_SIZES  = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
SEED         = 42

OPT_JSON = os.path.join(os.path.dirname(__file__),
                        '../results/phase_4_gamma_lambda/optimal_gamma_lambda.json')
OUT_DIR  = os.path.join(os.path.dirname(__file__),
                        '../results/phase_4_comparison')


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pad(lst, length, fill_val=None):
    if fill_val is None: fill_val = lst[-1] if lst else 0.0
    lst = list(lst)
    if len(lst) < length:
        lst += [fill_val] * (length - len(lst))
    return np.array(lst[:length])


def run_one_block_size(a_T, X_tr, y_tr, X_te, y_te, gamma, lam):
    """Return (bb_weighted_acc, bb_std_acc, ada_acc) all of length M_ROUNDS."""
    stump = lambda: DecisionTreeClassifier(max_depth=1)

    # BlockBoostWeighted
    bw = BlockBoostWeightedClassifier(estimator=stump(), n_estimators=M_ROUNDS,
                                      block_size=a_T, gamma=gamma, lam=lam)
    bw.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    bw_acc = _pad(bw.test_accuracy_, M_ROUNDS)

    # Standard BlockBoost
    bs = BlockBoostClassifier(estimator=stump(), n_estimators=M_ROUNDS,
                              block_size=a_T)
    bs.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    bs_acc = _pad(bs.test_accuracy_, M_ROUNDS)

    # AdaBoost (block_size=1 is standard AdaBoost)
    ada = AdaBoostM1(estimator=stump(), n_estimators=M_ROUNDS)
    ada.fit(X_tr, y_tr, X_val=X_te, y_val=y_te)
    ada_acc = _pad(ada.test_accuracy_, M_ROUNDS)

    return bw_acc, bs_acc, ada_acc


# ─── main ─────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    # Load optimal hypers
    if not os.path.exists(OPT_JSON):
        raise FileNotFoundError(
            f"Run phase_4_optimize_gamma_lambda.py first to generate {OPT_JSON}")
    with open(OPT_JSON) as f:
        opt = json.load(f)
    gamma_star = opt['best_gamma']
    lam_star   = opt['best_lambda']
    print(f"[*] Loaded optimal hypers: gamma={gamma_star}, lambda={lam_star}")

    # Generate data
    print("[*] Generating AR(5) data …")
    X_tr = generate_ar_process(N_TRAIN, N_FEATURES, AR_COEFFS,
                                noise_std=1.0, random_state=SEED)
    y_tr_clean = generate_labels(X_tr, random_state=SEED)
    y_tr = generate_markov_noise(y_tr_clean, ETA, ALPHA_MARKOV, SEED)

    X_te = generate_ar_process(N_TEST, N_FEATURES, AR_COEFFS,
                                noise_std=1.0, random_state=SEED + 1)
    y_te = generate_labels(X_te, random_state=SEED + 1)

    print(f"[*] Running experiments for {len(BLOCK_SIZES)} block sizes "
          f"(M={M_ROUNDS}) …")
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_one_block_size)(a_T, X_tr, y_tr, X_te, y_te,
                                   gamma_star, lam_star)
        for a_T in BLOCK_SIZES
    )

    # ── Plot ──────────────────────────────────────────────────────────────
    print("[*] Generating comparison grid plot …")
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams["font.family"]      = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    rounds = np.arange(1, M_ROUNDS + 1)
    COLOR_BW  = '#1f77b4'   # muted blue  – BlockBoostWeighted
    COLOR_BB  = '#d62728'   # muted red   – Standard BlockBoost
    COLOR_ADA = '#7f7f7f'   # gray        – AdaBoost

    fig, axes = plt.subplots(4, 3, figsize=(13, 11),
                             sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.28, wspace=0.12)
    axes_flat = axes.flatten()

    # Global y-limits (leave a small margin below minimum observed accuracy)
    all_accs = [v for tup in results for arr in tup for v in arr]
    y_min = max(0.0, min(all_accs) - 0.03)
    y_max = min(1.02, max(all_accs) + 0.02)

    for i, (a_T, (bw_acc, bs_acc, ada_acc)) in enumerate(
            zip(BLOCK_SIZES, results)):
        ax = axes_flat[i]

        ax.plot(rounds, ada_acc, color=COLOR_ADA, linewidth=0.9, alpha=0.75,
                linestyle=':', label='AdaBoost')
        ax.plot(rounds, bs_acc,  color=COLOR_BB,  linewidth=0.9, alpha=0.80,
                linestyle='--', label='BlockBoost')
        ax.plot(rounds, bw_acc,  color=COLOR_BW,  linewidth=1.2, alpha=0.90,
                label=f'BB-Weighted ($\\gamma^*={gamma_star}$, $\\lambda^*={lam_star}$)')

        ax.text(0.97, 0.07, f'$a_T = {a_T}$',
                transform=ax.transAxes, fontsize=10,
                ha='right', va='bottom',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, M_ROUNDS)
        ax.grid(True, linestyle='--', linewidth=0.45, alpha=0.55, color='gray')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    # Shared labels
    fig.text(0.50, 0.04, 'Boosting Round ($m$)',
             ha='center', va='center', fontsize=13)
    fig.text(0.04, 0.50, 'Test Accuracy (Clean)',
             ha='center', va='center', rotation='vertical', fontsize=13)

    # Single legend at the top-right of the full figure
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.50, 0.97))

    plt.suptitle("BlockBoostWeighted vs BlockBoost vs AdaBoost\n"
                 "AR(5) with Markov noise — Test Accuracy by Block Size",
                 fontsize=14, y=1.01)

    fig.savefig(os.path.join(OUT_DIR, 'comparison_grid.pdf'),
                format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'comparison_grid.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[+] Comparison grid saved to: {os.path.abspath(OUT_DIR)}")

    # Save raw data
    raw = {}
    for a_T, (bw_acc, bs_acc, ada_acc) in zip(BLOCK_SIZES, results):
        raw[str(a_T)] = dict(
            blockboost_weighted=bw_acc.tolist(),
            blockboost_standard=bs_acc.tolist(),
            adaboost=ada_acc.tolist(),
        )
    with open(os.path.join(OUT_DIR, 'comparison_raw.json'), 'w') as f:
        json.dump(raw, f, indent=2)
    print("[+] All done.")


if __name__ == '__main__':
    run()
