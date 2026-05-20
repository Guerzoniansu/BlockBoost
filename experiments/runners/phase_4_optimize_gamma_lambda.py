"""
phase_4_optimize_gamma_lambda.py

Cross-validation optimizer for the (gamma, lambda) hyperparameters of
BlockBoostWeighted on an AR(5) process with Markov label noise.

Protocol
--------
DGP     : AR(5) with ar_coeffs = [0.4, 0.2, 0.1, 0.05, 0.05] (rho ~ 0.89)
          n_train = 1000 (noisy), n_test = 1000 (clean)
Noise   : Markov noise with eta = 0.15, alpha_markov = 0.8
Booster : BlockBoostWeightedClassifier, n_estimators = 200, block_size = 10

Search  : Grid search over GAMMA_GRID × LAM_GRID using temporal
          walk-forward cross-validation (5 folds) on the training set.

Output  : JSON with best (gamma, lambda), grid of CV scores + heat-map PNG.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.utils.reproducibility import set_seed
from experiments.dgp.ar_process import generate_ar_process, generate_labels
from experiments.dgp.noise import generate_markov_noise, generate_iid_noise
from model.blockboost_weighted import BlockBoostWeightedClassifier

# ─── Configuration ────────────────────────────────────────────────────────────
AR_COEFFS     = [0.4, 0.2, 0.1, 0.05, 0.05]
N_FEATURES    = 10
N_TRAIN       = 1000
N_TEST        = 1000
ETA           = 0.15
ALPHA_MARKOV  = 0.8
M_ROUNDS      = 200
BLOCK_SIZE    = 1          # fixed during optimisation
N_CV_FOLDS    = 5
SEED          = 42

GAMMA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
LAM_GRID   = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0]

OUT_DIR = os.path.join(os.path.dirname(__file__),
                       '../results/phase_4_gamma_lambda')


# ─── helpers ──────────────────────────────────────────────────────────────────

def temporal_cv_folds(n: int, n_folds: int):
    """Walk-forward splits: train on 1..k*(n//n_folds), val on k*(n//n_folds)+1 .. (k+1)*(n//n_folds)"""
    fold_size = n // (n_folds + 1)
    for f in range(1, n_folds + 1):
        train_end = f * fold_size
        val_end   = (f + 1) * fold_size
        yield slice(0, train_end), slice(train_end, val_end)


def cv_score(X: np.ndarray, y: np.ndarray, gamma: float, lam: float) -> float:
    """Mean validation accuracy over temporal CV folds."""
    scores = []
    stump  = DecisionTreeClassifier(max_depth=1)
    for tr, va in temporal_cv_folds(len(X), N_CV_FOLDS):
        clf = BlockBoostWeightedClassifier(
            estimator=stump,
            n_estimators=M_ROUNDS,
            block_size=BLOCK_SIZE,
            gamma=gamma,
            lam=lam,
        )
        clf.fit(X[tr], y[tr])
        preds  = clf.predict(X[va])
        scores.append(float(np.mean(preds == y[va])))
    return float(np.mean(scores))


# ─── main ─────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    # 1. Generate data
    print("[*] Generating AR(5) training set …")
    X_train = generate_ar_process(N_TRAIN, N_FEATURES, AR_COEFFS,
                                  noise_std=1.0, random_state=SEED)
    y_train = generate_labels(X_train, random_state=SEED)
    y_train_noisy = generate_iid_noise(y_train, ETA, SEED)

    print("[*] Generating clean AR(5) test set …")
    X_test  = generate_ar_process(N_TEST, N_FEATURES, AR_COEFFS,
                                  noise_std=1.0, random_state=SEED + 1)
    y_test  = generate_labels(X_test, random_state=SEED + 1)

    # 2. Grid search
    print(f"[*] Grid search: {len(GAMMA_GRID)} x {len(LAM_GRID)} = "
          f"{len(GAMMA_GRID)*len(LAM_GRID)} combinations "
          f"(each with {N_CV_FOLDS} CV folds) …")

    grid = np.zeros((len(GAMMA_GRID), len(LAM_GRID)))
    for i, gam in enumerate(GAMMA_GRID):
        for j, lam in enumerate(LAM_GRID):
            score = cv_score(X_train, y_train_noisy, gam, lam)
            grid[i, j] = score
            print(f"  gamma={gam:.2f}  lambda={lam:.2f}  cv_acc={score:.4f}")

    best_idx = np.unravel_index(np.argmax(grid), grid.shape)
    best_gamma = GAMMA_GRID[best_idx[0]]
    best_lam   = LAM_GRID[best_idx[1]]
    best_score = float(grid[best_idx])
    print(f"\n[+] Best: gamma={best_gamma}, lambda={best_lam}, "
          f"cv_acc={best_score:.4f}")

    # 3. Evaluate on held-out test set with best params
    stump = DecisionTreeClassifier(max_depth=1)
    final_model = BlockBoostWeightedClassifier(
        estimator=stump,
        n_estimators=M_ROUNDS,
        block_size=BLOCK_SIZE,
        gamma=best_gamma,
        lam=best_lam,
    )
    final_model.fit(X_train, y_train_noisy)
    test_acc = float(np.mean(final_model.predict(X_test) == y_test))
    print(f"    Test accuracy with best hypers: {test_acc:.4f}")

    # 4. Save results
    results = dict(
        best_gamma=best_gamma,
        best_lambda=best_lam,
        best_cv_score=best_score,
        test_accuracy=test_acc,
        gamma_grid=GAMMA_GRID,
        lambda_grid=LAM_GRID,
        cv_score_matrix=grid.tolist(),
    )
    out_json = os.path.join(OUT_DIR, 'optimal_gamma_lambda.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"[+] Results saved to {out_json}")

    # 5. Heat-map
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(grid, aspect='auto', origin='lower',
                   cmap='viridis', vmin=grid.min(), vmax=grid.max())
    ax.set_xticks(range(len(LAM_GRID)));   ax.set_xticklabels([f'{v}' for v in LAM_GRID])
    ax.set_yticks(range(len(GAMMA_GRID))); ax.set_yticklabels([f'{v}' for v in GAMMA_GRID])
    ax.set_xlabel('$\\lambda$');  ax.set_ylabel('$\\gamma$')
    ax.set_title('Cross-Validation Accuracy Heatmap — BlockBoostWeighted\n'
                 'AR(5), Markov noise η=0.15, M=200, $a_T$=10')
    plt.colorbar(im, ax=ax, label='Mean CV validation accuracy')
    # Mark best
    ax.add_patch(plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5),
                                1, 1, fill=False, edgecolor='red', lw=2))
    ax.text(best_idx[1], best_idx[0], f'{best_score:.3f}',
            ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'cv_heatmap.pdf'), format='pdf', bbox_inches='tight')
    fig.savefig(os.path.join(OUT_DIR, 'cv_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("[+] Heat-map saved.")


if __name__ == '__main__':
    run()
