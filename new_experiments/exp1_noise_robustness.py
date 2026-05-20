"""
exp1_noise_robustness.py
========================
Experiment 1 — Noise Robustness: BlockBoost (a_T=10) vs AdaBoost.M1

AR(1) data, IID label noise at [0.0, 0.1, 0.2, 0.3],
averaged over 10 seeds, 1500 boosting rounds.

Output
------
results/exp1_results.json     – raw per-seed curves for every (eta, model)
figures/exp1_noise_grid.pdf   – paper-ready 4×2 figure
figures/exp1_noise_grid.png   – high-res PNG
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

# ── import models from parent Experiments/ ────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.blockboost import BlockBoostClassifier
from model.adaboost   import AdaBoostM1

# ── Output dirs ───────────────────────────────────────────────────────────────
HERE    = os.path.dirname(__file__)
os.makedirs(os.path.join(HERE, 'results'),  exist_ok=True)
os.makedirs(os.path.join(HERE, 'figures'),  exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3]
N_TRAIN    = 2000
N_TEST     = 1000
M_ROUNDS   = 1500
BLOCK_SIZE = 10
N_SEEDS    = 10
RHO        = 0.95
NOISE_STD  = 0.3


# ── DGP (user-provided) ───────────────────────────────────────────────────────

def generate_ar_data(n, rho=0.8, noise_std=0.3, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    X = np.zeros((n, 5))
    X[0] = np.random.randn(5) * noise_std
    innovation_scale = np.sqrt(1 - rho**2) * noise_std
    for t in range(1, n):
        X[t] = rho * X[t-1] + np.random.randn(5) * innovation_scale
    signal = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
    signal_with_lag = np.zeros(n)
    signal_with_lag[0] = signal[0]
    for t in range(1, n):
        signal_with_lag[t] = signal[t] + 0.2 * signal_with_lag[t-1]
    return X, np.where(signal_with_lag > 0, 1, -1)


def inject_label_noise(y, noise_level, random_state=None):
    rng = np.random.RandomState(random_state)
    y_noisy = y.copy()
    n_noisy = int(noise_level * len(y))
    if n_noisy > 0:
        idx = rng.choice(len(y), size=n_noisy, replace=False)
        y_noisy[idx] *= -1
    return y_noisy


# ── Per-round accuracy (incremental, O(M·N)) ──────────────────────────────────

def per_round_acc(estimators_, alphas_, X, y):
    f = np.zeros(len(y))
    accs = []
    for a, clf in zip(alphas_, estimators_):
        f += a * clf.predict(X)
        accs.append(float(np.mean(np.sign(f) == y)))
    return np.array(accs)


def _pad(arr, length):
    if len(arr) < length:
        arr = np.pad(arr, (0, length - len(arr)), mode='edge')
    return arr[:length]


# ── One seed ──────────────────────────────────────────────────────────────────

def run_seed(seed, eta):
    X_tr, y_tr_c = generate_ar_data(N_TRAIN, RHO, NOISE_STD, seed)
    X_te, y_te_c = generate_ar_data(N_TEST,  RHO, NOISE_STD, seed + 100_000)
    y_tr_n = inject_label_noise(y_tr_c, eta, seed)
    y_te_n = inject_label_noise(y_te_c, eta, seed + 200_000)

    stump = lambda: DecisionTreeClassifier(max_depth=1)

    # BlockBoost
    bb = BlockBoostClassifier(estimator=stump(), n_estimators=M_ROUNDS,
                              block_size=BLOCK_SIZE)
    bb.fit(X_tr, y_tr_n)
    bb_res = dict(
        train_noisy = _pad(np.array(bb.train_accuracy_), M_ROUNDS),
        test_noisy  = _pad(per_round_acc(bb.estimators_, bb.alphas_, X_te, y_te_n), M_ROUNDS),
        test_clean  = _pad(per_round_acc(bb.estimators_, bb.alphas_, X_te, y_te_c), M_ROUNDS),
    )

    # AdaBoost
    ada = AdaBoostM1(estimator=stump(), n_estimators=M_ROUNDS)
    ada.fit(X_tr, y_tr_n)
    ada_res = dict(
        train_noisy = _pad(np.array(ada.train_accuracy_), M_ROUNDS),
        test_noisy  = _pad(per_round_acc(ada.estimators_, ada.alphas_, X_te, y_te_n), M_ROUNDS),
        test_clean  = _pad(per_round_acc(ada.estimators_, ada.alphas_, X_te, y_te_c), M_ROUNDS),
    )
    return bb_res, ada_res


# ── Experiment loop ───────────────────────────────────────────────────────────

def run_all():
    raw = {}
    for eta in NOISE_LEVELS:
        print(f"\n[*] eta = {eta}  ({N_SEEDS} seeds, M = {M_ROUNDS}) …")
        seed_results = Parallel(n_jobs=-1, verbose=3)(
            delayed(run_seed)(s, eta) for s in range(N_SEEDS)
        )
        bb_stack  = {k: np.stack([r[0][k] for r in seed_results]) for k in seed_results[0][0]}
        ada_stack = {k: np.stack([r[1][k] for r in seed_results]) for k in seed_results[0][1]}

        raw[str(eta)] = dict(
            blockboost = {k: v.tolist() for k, v in bb_stack.items()},
            adaboost   = {k: v.tolist() for k, v in ada_stack.items()},
        )
        print(f"    BB  final clean-test: {bb_stack['test_clean'][:, -1].mean():.4f}"
              f"  ±{bb_stack['test_clean'][:, -1].std():.4f}")
        print(f"    Ada final clean-test: {ada_stack['test_clean'][:, -1].mean():.4f}"
              f"  ±{ada_stack['test_clean'][:, -1].std():.4f}")

    out = os.path.join(HERE, 'results', 'exp1_results.json')
    with open(out, 'w') as f:
        json.dump(raw, f, indent=2)
    print(f"\n[+] Results → {out}")
    return raw


# ── Figure ────────────────────────────────────────────────────────────────────

# Minimal, paper-ready colour map
C_TRAIN = '#0000FF'   # blue  – noisy train
C_NOISY = '#FF0000'   # red   – noisy test
C_CLEAN = '#32CD32'   # green – clean test


def _panel(ax, tr_m, tr_s, tn_m, tn_s, tc_m, tc_s, title, show_ylabel, show_noisy=True):
    x = np.arange(1, M_ROUNDS + 1)
    kw = dict(alpha=0.10)

    ax.fill_between(x, tr_m - tr_s, tr_m + tr_s, color=C_TRAIN, **kw)
    ax.plot(x, tr_m, color=C_TRAIN, lw=1.0, label='Train (noisy)')

    if show_noisy:
        ax.fill_between(x, tn_m - tn_s, tn_m + tn_s, color=C_NOISY, **kw)
        ax.plot(x, tn_m, color=C_NOISY, lw=1.0, label='Test (noisy)')

    ax.fill_between(x, tc_m - tc_s, tc_m + tc_s, color=C_CLEAN, **kw)
    ax.plot(x, tc_m, color=C_CLEAN, lw=1.2, ls='--',
            label='Test (clean)' if show_noisy else 'Test accuracy')

    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xlim(0, M_ROUNDS)
    if show_ylabel:
        ax.set_ylabel('Accuracy', fontsize=9)
    ax.grid(True, ls='--', lw=0.4, alpha=0.55, color='#bbbbbb')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.tick_params(labelsize=8)


def make_figure(raw):
    plt.rcParams.update({
        'font.family':      'serif',
        'mathtext.fontset': 'dejavuserif',
        'font.size':        9,
    })

    nrows = len(NOISE_LEVELS)
    fig, axes = plt.subplots(nrows, 2, figsize=(11, 3.0 * nrows),
                             sharex=True, sharey='row')
    fig.subplots_adjust(hspace=0.38, wspace=0.08)  # narrower gap since y-axes match

    for row, eta in enumerate(NOISE_LEVELS):
        r   = raw[str(eta)]
        bb  = {k: np.array(v) for k, v in r['blockboost'].items()}
        ada = {k: np.array(v) for k, v in r['adaboost'].items()}

        _panel(axes[row, 0],
               bb['train_noisy'].mean(0),  bb['train_noisy'].std(0),
               bb['test_noisy'].mean(0),   bb['test_noisy'].std(0),
               bb['test_clean'].mean(0),   bb['test_clean'].std(0),
               title=f'BlockBoost  ($a_T = 10$),  $\\eta = {eta}$',
               show_ylabel=True,
               show_noisy=(eta > 0.0))

        _panel(axes[row, 1],
               ada['train_noisy'].mean(0), ada['train_noisy'].std(0),
               ada['test_noisy'].mean(0),  ada['test_noisy'].std(0),
               ada['test_clean'].mean(0),  ada['test_clean'].std(0),
               title=f'AdaBoost.M1,  $\\eta = {eta}$',
               show_ylabel=False,
               show_noisy=(eta > 0.0))

    for ax in axes[-1, :]:
        ax.set_xlabel('Number of Boosting Rounds', fontsize=9)

    # Collect the union of legend entries from first noisy row (eta>0) for full legend,
    # and from first row (eta=0) for the no-noisy-test row
    handles_full, labels_full = axes[1, 0].get_legend_handles_labels()  # row with noise
    handles_zero, labels_zero = axes[0, 0].get_legend_handles_labels()  # row without
    # Use the full set (3 entries) for the legend
    fig.legend(handles_full, labels_full, loc='upper center', ncol=3, fontsize=9,
               frameon=True, edgecolor='#cccccc',
               bbox_to_anchor=(0.50, 0.995))  # very close to top of figure

    for fmt in ('pdf', 'png'):
        path = os.path.join(HERE, 'figures', f'exp1_noise_grid.{fmt}')
        fig.savefig(path, format=fmt, bbox_inches='tight',
                    dpi=300 if fmt == 'png' else None)
        print(f"[+] Figure → {path}")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Experiment 1 — Noise Robustness: BlockBoost vs AdaBoost")
    print("=" * 60)
    raw = run_all()
    print("\n[*] Generating figure …")
    make_figure(raw)
    print("[+] All done.")
