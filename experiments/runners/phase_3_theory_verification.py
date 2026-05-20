"""
phase_3_theory_verification.py

Phase 3: Direct Theory Verification (Proposition 1.12)

We will construct a 2-dimensional feature space X in R^{N x 2} that perfectly maps
to the theoretical margin densities. The underlying target label for the clean space
is strictly positive: y_i = +1 for all i.

DGP (iid):
  - gamma = 0.15, sigma = 0.5
  - Cluster A (Neutral Anchor):          N/4  samples from N([1, 0],       sigma^2 I)
  - Cluster B (Heavy Mass/Small Margin): N/2  samples from N([gamma, -gamma], sigma^2 I)
  - Cluster C (Sparse Outliers):         N/4  samples from N([gamma, 20*gamma], sigma^2 I)
  - Total N = 10,000, split evenly: T_train = 1000, T_test = 1000 (subsetted)

Noise Injection (IID, training set ONLY):
  - eta = 0.1 injected via inject_label_noise()

Boosting:
  - BlockBoost for block sizes a_T in [1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
  - AdaBoost (a_T = 1)
  - M = 5000 rounds
  - 10 independent seeds

Output plots:
  1. 4x3 grid of step sizes alpha_m per block size (phase_3_theory_verification folder)
  2. Adversarial DGP comparison at a_T = 10 (phase_3_adversarial folder) with
     accuracy, alpha_m, and sum of weights squared for BlockBoost vs AdaBoost
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier
import scipy.stats as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.utils.reproducibility import set_seed
from model.blockboost import BlockBoostClassifier
from model.adaboost import AdaBoostM1

# ─── Experiment Parameters ───────────────────────────────────────────────────
GAMMA     = 0.15
SIGMA     = 0.2
ETA       = 0.1
N_TOTAL   = 10000
N_TRAIN   = 1000
N_TEST    = 9000
M_ROUNDS  = 5000
A_T_COMPARE = 10   # Kept for reference, but we loop over BLOCK_SIZES now
BLOCK_SIZES = [10, 15, 20, 25, 30]
N_SEEDS   = 20


# ─── Helpers ─────────────────────────────────────────────────────────────────

def generate_adversarial_dgp(n_samples: int, gamma: float, sigma: float,
                              random_state: int):
    """3-cluster Gaussian DGP, all labels y = +1."""
    rng = np.random.default_rng(random_state)
    n_a = n_samples // 4
    n_b = n_samples // 2
    n_c = n_samples - n_a - n_b

    X_a = rng.normal(loc=[1.0, 0.0],          scale=sigma, size=(n_a, 2))
    X_b = rng.normal(loc=[gamma, -gamma],      scale=sigma, size=(n_b, 2))
    X_c = rng.normal(loc=[gamma, 20 * gamma],   scale=sigma, size=(n_c, 2))

    X = np.vstack([X_a, X_b, X_c])
    y = np.ones(n_samples)          # u_i = +1 universally

    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def inject_label_noise(y, noise_level, random_state=None):
    """Flip labels with given probability (iid)."""
    if random_state is not None:
        rng = np.random.RandomState(random_state)
    else:
        rng = np.random.RandomState()

    y_noisy = y.copy()
    n_noisy = int(noise_level * len(y))
    if n_noisy > 0:
        noisy_idx = rng.choice(len(y), size=n_noisy, replace=False)
        y_noisy[noisy_idx] *= -1
    return y_noisy


def run_blockboost(a_T: int, X_train, y_train, X_test=None, y_test=None,
                   m_rounds: int = M_ROUNDS):
    """Train BlockBoost and return alphas, sum_squared_weights, test_acc, train_acc."""
    stump = DecisionTreeClassifier(max_depth=2)
    bb = BlockBoostClassifier(estimator=stump, n_estimators=m_rounds,
                              block_size=a_T)
    bb.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    alphas    = list(bb.alphas_)
    ssws      = list(bb.sum_squared_weights_)
    test_acc  = list(bb.test_accuracy_)
    train_acc = list(bb.train_accuracy_)

    if len(alphas) < m_rounds:
        pad = m_rounds - len(alphas)
        alphas    += [0.0] * pad
        ssws      += [ssws[-1] if ssws else 0.0] * pad
        test_acc  += [test_acc[-1]  if test_acc  else 0.0] * pad
        train_acc += [train_acc[-1] if train_acc else 0.0] * pad

    return (np.array(alphas), np.array(ssws),
            np.array(test_acc), np.array(train_acc))


def run_adaboost(X_train, y_train, X_test, y_test, m_rounds: int = M_ROUNDS):
    """Train AdaBoost and return alphas, sum_squared_weights, test_acc, train_acc."""
    stump = DecisionTreeClassifier(max_depth=2)
    ada = AdaBoostM1(estimator=stump, n_estimators=m_rounds)
    ada.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    alphas    = list(ada.alphas_)
    ssws      = list(ada.sum_squared_weights_)
    test_acc  = list(ada.test_accuracy_)
    train_acc = list(ada.train_accuracy_)
    
    if len(alphas) < m_rounds:
        pad = m_rounds - len(alphas)
        alphas    += [0.0] * pad
        ssws      += [ssws[-1] if ssws else 0.0] * pad
        test_acc  += [test_acc[-1]  if test_acc  else 0.0] * pad
        train_acc += [train_acc[-1] if train_acc else 0.0] * pad

    return (np.array(alphas), np.array(ssws), 
            np.array(test_acc), np.array(train_acc))


def run_single_seed(seed: int):
    # 1. Generate data
    X_all, y_all = generate_adversarial_dgp(N_TOTAL, GAMMA, SIGMA, seed)
    
    X_train = X_all[:N_TRAIN];  y_train_clean = y_all[:N_TRAIN]
    X_test  = X_all[N_TRAIN:N_TRAIN+N_TEST];  y_test        = y_all[N_TRAIN:N_TRAIN+N_TEST]

    # 2. Inject IID noise into training set
    y_train = inject_label_noise(y_train_clean, noise_level=ETA, random_state=seed)

    # 3. AdaBoost
    ada_res = run_adaboost(X_train, y_train, X_test, y_test, m_rounds=M_ROUNDS)

    # 4. BlockBoost
    bb_res = {}
    for a_T in BLOCK_SIZES:
        bb_res[a_T] = run_blockboost(a_T, X_train, y_train, X_test, y_test, m_rounds=M_ROUNDS)

    return seed, ada_res, bb_res


# ─── Main ────────────────────────────────────────────────────────────────────

def run_phase_3():
    print("=" * 70)
    print("PHASE 3: Direct Theory Verification (Adversarial Comparison)")
    print("=" * 70)

    # Parallel over seeds
    print(f"[*] Running {N_SEEDS} independent seeds in parallel "
          f"(N_TRAIN={N_TRAIN}, M={M_ROUNDS})…")
    
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_single_seed)(seed) for seed in range(N_SEEDS)
    )

    # Structure data: arrays of shape (N_SEEDS, M_ROUNDS)
    ada_alphas = np.zeros((N_SEEDS, M_ROUNDS))
    ada_ssws   = np.zeros((N_SEEDS, M_ROUNDS))
    ada_test   = np.zeros((N_SEEDS, M_ROUNDS))
    ada_train  = np.zeros((N_SEEDS, M_ROUNDS))

    bb_data = {a_T: {
        'alphas': np.zeros((N_SEEDS, M_ROUNDS)),
        'ssws':   np.zeros((N_SEEDS, M_ROUNDS)),
        'test':   np.zeros((N_SEEDS, M_ROUNDS)),
        'train':  np.zeros((N_SEEDS, M_ROUNDS))
    } for a_T in BLOCK_SIZES}

    for i, (seed, ada_res, bb_res) in enumerate(results):
        ada_alphas[i], ada_ssws[i], ada_test[i], ada_train[i] = ada_res
        for a_T in BLOCK_SIZES:
            bb_data[a_T]['alphas'][i], bb_data[a_T]['ssws'][i], \
            bb_data[a_T]['test'][i], bb_data[a_T]['train'][i] = bb_res[a_T]

    # ── Plot : Adversarial DGP comparison for multiple a_T (Mean + CI) ─────────
    out_dir2 = os.path.join(os.path.dirname(__file__), '../results/phase_3_adversarial')
    os.makedirs(out_dir2, exist_ok=True)

    def plot_with_ci(ax, x, matrix, color, label, linestyle='-'):
        mean = np.mean(matrix, axis=0)
        ci   = 1.96 * np.std(matrix, axis=0) / np.sqrt(matrix.shape[0])
        ax.plot(x, mean, color=color, linewidth=1.5, linestyle=linestyle, label=label)
        ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.2)

    rounds = np.arange(1, M_ROUNDS + 1)
    
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 16,     # Larger Title
        "axes.labelsize": 16,     # Larger Axis labels
        "xtick.labelsize": 14,    # Larger X-axis ticks
        "ytick.labelsize": 14,    # Larger Y-axis ticks
        "legend.fontsize": 14     # Larger Legend
    })

    for a_T in BLOCK_SIZES:
        print(f"\n[*] Generating comparison plot and stats for a_T={a_T}...")
        
        # ── Statistical Testing ──
        final_ada = ada_test[:, -1]
        final_bb  = bb_data[a_T]['test'][:, -1]
        
        stat, p_val = st.wilcoxon(final_bb, final_ada, alternative='greater')
        print(f"Wilcoxon Signed-Rank Test (BlockBoost_aT{a_T} > AdaBoost):")
        print(f"Mean Ada Test Acc:   {np.mean(final_ada):.4f} ± {np.std(final_ada):.4f}")
        print(f"Mean BB({a_T}) Test Acc: {np.mean(final_bb):.4f} ± {np.std(final_bb):.4f}")
        print(f"Statistic: {stat:.2f}, p-value: {p_val:.4e}")

        # ── Plotting ──
        fig2, axes2 = plt.subplots(1, 4, figsize=(24, 6))
        
        # Panel A – Test Accuracy (clean holdout)
        plot_with_ci(axes2[0], rounds, ada_test, 'red', 'AdaBoost', linestyle='--')
        plot_with_ci(axes2[0], rounds, bb_data[a_T]['test'], 'blue', f'BlockBoost ($a_T={a_T}$)')
        axes2[0].set_xlabel('Boosting Round ($m$)')
        axes2[0].set_ylabel('Accuracy')
        axes2[0].set_title('Test Accuracy (Clean Holdout)')
        axes2[0].grid(True, linestyle='--', alpha=0.6)
        axes2[0].legend()

        # Panel B – Noisy Train Accuracy
        plot_with_ci(axes2[1], rounds, ada_train, 'red', 'AdaBoost', linestyle='--')
        plot_with_ci(axes2[1], rounds, bb_data[a_T]['train'], 'blue', f'BlockBoost ($a_T={a_T}$)')
        axes2[1].set_xlabel('Boosting Round ($m$)')
        axes2[1].set_ylabel('Accuracy')
        axes2[1].set_title('Noisy Train Accuracy')
        axes2[1].grid(True, linestyle='--', alpha=0.6)
        axes2[1].legend()

        # Panel C – Step Size
        plot_with_ci(axes2[2], rounds, ada_alphas, 'red', 'AdaBoost', linestyle='--')
        plot_with_ci(axes2[2], rounds, bb_data[a_T]['alphas'], 'blue', f'BlockBoost ($a_T={a_T}$)')
        axes2[2].set_xlabel('Boosting Round ($m$)')
        axes2[2].set_ylabel('Step Size ($\\alpha_m$)')
        axes2[2].set_title('Algorithm Momentum (Step Size)')
        axes2[2].grid(True, linestyle='--', alpha=0.6)
        axes2[2].legend()

        # Panel D – Sum of Weights Squared
        plot_with_ci(axes2[3], rounds, ada_ssws, 'red', 'AdaBoost', linestyle='--')
        plot_with_ci(axes2[3], rounds, bb_data[a_T]['ssws'], 'blue', f'BlockBoost ($a_T={a_T}$)')
        axes2[3].set_xlabel('Boosting Round ($m$)')
        axes2[3].set_ylabel('Sum of Weights Squared ($\\sum (w_t^{(m)})^2$)')
        axes2[3].set_title('Weight Concentration')
        axes2[3].grid(True, linestyle='--', alpha=0.6)
        axes2[3].legend()

        for ax in axes2:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir2, f'adversarial_dgp_comparison_aT_{a_T}.pdf'), format='pdf', bbox_inches='tight')
        fig2.savefig(os.path.join(out_dir2, f'adversarial_dgp_comparison_aT_{a_T}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"  -> Saved adversarial_dgp_comparison_aT_{a_T}.pdf/png")

    print("\n[+] All done.")


if __name__ == "__main__":
    run_phase_3()
