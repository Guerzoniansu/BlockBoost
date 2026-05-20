"""
phase_1_1_A_plateau.py

Phase 1.1.A — Protocol A: Plateau Verification

PURPOSE: To show that the BlockBoost and AdaBoost learning curves stabilize
at the theoretical plateau value as M grows large. This is NOT a statistical
comparison experiment. We use very few seeds (2) and many rounds (M=3000) so
the full learning curve trajectory is clearly visible.

Limitation note: With only 2 seeds we DO NOT have statistical power to compare
final accuracies between algorithms. The purpose of this script is purely to
verify that the plateau exist and overlaying the theoretical reference line
  theoretical_plateau = eta + (1 - 2*eta) * (1 - bayes_risk_clean)
for the noisy test accuracy curve.

For the clean test accuracy curve, the reference line is simply:
  theoretical_clean_plateau = 1 - bayes_risk_clean  (approximate, under clean Bayes)

Plots saved to: experiments/results/phase_1_1_A_plateau/plots/

Protocol:
  - n_seeds: 2
  - M: 3000
  - eta: [0.10, 0.20]
  - Metrics tracked per round: clean_train_acc, noisy_train_acc, clean_test_acc
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.configs.schema import Phase1Config
from experiments.utils.reproducibility import set_seed, generate_seeds
from experiments.dgp.ar_process import (
    generate_ar_process,
    generate_labels,
    compute_theta,
    compute_theoretical_optimal_block_size,
)
from experiments.dgp.noise import generate_markov_noise, compute_bayes_risk

from model.adaboost import AdaBoostM1
from model.blockboost import BlockBoostClassifier

# Protocol A fixed parameters
PROTOCOL_A_N_SEEDS = 2
PROTOCOL_A_M = 1500
PROTOCOL_A_ETAS = [0.10, 0.20]


def run_single_seed(seed: int, eta: float, config: Phase1Config, a_T: int) -> dict:
    """
    Runs one full AdaBoost and BlockBoost trajectory for M=3000 rounds.
    Returns per-round accuracy arrays for all three tracked metrics.
    """
    set_seed(seed)

    X = generate_ar_process(
        n_samples=config.n_samples,
        n_features=config.n_features,
        ar_coeffs=config.ar_coeffs,
        noise_std=config.noise_std,
        random_state=seed
    )
    y_clean = generate_labels(X, random_state=seed)
    y_noisy = generate_markov_noise(y_clean, eta=eta, alpha_markov=config.alpha_markov, random_state=seed)

    # Temporal split
    X_train, X_test, y_train_clean, y_test_clean = train_test_split(X, y_clean, test_size=0.3, shuffle=False)
    _, _, y_train_noisy, _ = train_test_split(X, y_noisy, test_size=0.3, shuffle=False)

    stump = DecisionTreeClassifier(max_depth=1)

    # AdaBoost: track noisy train + clean test per round
    ada = AdaBoostM1(estimator=stump, n_estimators=PROTOCOL_A_M)
    ada_noisy_train_curve, ada_clean_test_curve = ada.fit(
        X_train, y_train_noisy, X_val=X_test, y_val=y_test_clean
    )
    # Track clean train accuracy: evaluate after each estimator using staged_predict
    ada_clean_train_curve = []
    y_pred_acc = np.zeros(len(X_train))
    for alpha, clf in zip(ada.alphas_, ada.estimators_):
        y_pred_acc += alpha * clf.predict(X_train)
        preds = np.sign(y_pred_acc)
        mapped = np.where(preds == -1, ada.classes_[0], ada.classes_[1])
        ada_clean_train_curve.append(np.mean(mapped == y_train_clean))

    # BlockBoost
    block = BlockBoostClassifier(estimator=stump, n_estimators=PROTOCOL_A_M, block_size=a_T)
    block_noisy_train_curve, block_clean_test_curve = block.fit(
        X_train, y_train_noisy, X_val=X_test, y_val=y_test_clean
    )
    # BlockBoost clean train curve
    block_clean_train_curve = []
    y_pred_acc = np.zeros(len(X_train))
    for alpha, clf in zip(block.alphas_, block.estimators_):
        y_pred_acc += alpha * clf.predict(X_train)
        preds = np.sign(y_pred_acc)
        mapped = np.where(preds == -1, block.classes_[0], block.classes_[1])
        block_clean_train_curve.append(np.mean(mapped == y_train_clean))

    return {
        'seed': seed,
        'eta': eta,
        'a_T': a_T,
        'ada_noisy_train_curve': ada_noisy_train_curve,
        'ada_clean_train_curve': ada_clean_train_curve,
        'ada_clean_test_curve': ada_clean_test_curve,
        'block_noisy_train_curve': block_noisy_train_curve,
        'block_clean_train_curve': block_clean_train_curve,
        'block_clean_test_curve': block_clean_test_curve,
    }


def plot_learning_curves(all_runs: list, eta: float, theoretical_plateau: float,
                         out_dir: str, a_T: int):
    """
    Plots learning curves for all seeds for a given eta value.
    Shows three subplots stacked vertically:
      1. Noisy train accuracy
      2. Clean train accuracy
      3. Clean test accuracy (most important — should plateau at theoretical value)
    The theoretical plateau is overlaid as a dashed horizontal reference line on panel 3.
    """
    runs = [r for r in all_runs if r['eta'] == eta]
    if not runs:
        return

    rounds = range(1, len(runs[0]['ada_clean_test_curve']) + 1)

    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        f'Phase 1.1.A — Learning Curves | AR(5) | $\\eta$ = {eta} | $a_T$ = {a_T} | M = {PROTOCOL_A_M}',
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 1, hspace=0.38)
    ax_noisy_train = fig.add_subplot(gs[0])
    ax_clean_train = fig.add_subplot(gs[1])
    ax_clean_test = fig.add_subplot(gs[2])

    colors_ada = ['#1565C0', '#1E88E5']
    colors_bb = ['#B71C1C', '#E53935']

    for i, run in enumerate(runs):
        seed = run['seed']
        label_suffix = f'seed={seed}'

        ada_noisy = np.array(run['ada_noisy_train_curve'])
        ada_clean_tr = np.array(run['ada_clean_train_curve'])
        ada_clean_te = np.array(run['ada_clean_test_curve'])
        bb_noisy = np.array(run['block_noisy_train_curve'])
        bb_clean_tr = np.array(run['block_clean_train_curve'])
        bb_clean_te = np.array(run['block_clean_test_curve'])

        m_ada = len(ada_noisy)
        m_bb = len(bb_noisy)

        # Panel 1: Noisy train accuracy
        ax_noisy_train.plot(range(1, m_ada + 1), ada_noisy,
                            color=colors_ada[i], lw=1.2, alpha=0.85, label=f'AdaBoost {label_suffix}')
        ax_noisy_train.plot(range(1, m_bb + 1), bb_noisy,
                            color=colors_bb[i], lw=1.2, alpha=0.85, linestyle='--', label=f'BlockBoost {label_suffix}')

        # Panel 2: Clean train accuracy
        ax_clean_train.plot(range(1, len(ada_clean_tr) + 1), ada_clean_tr,
                            color=colors_ada[i], lw=1.2, alpha=0.85, label=f'AdaBoost {label_suffix}')
        ax_clean_train.plot(range(1, len(bb_clean_tr) + 1), bb_clean_tr,
                            color=colors_bb[i], lw=1.2, alpha=0.85, linestyle='--', label=f'BlockBoost {label_suffix}')

        # Panel 3: Clean test accuracy
        ax_clean_test.plot(range(1, len(ada_clean_te) + 1), ada_clean_te,
                           color=colors_ada[i], lw=1.5, alpha=0.9, label=f'AdaBoost {label_suffix}')
        ax_clean_test.plot(range(1, len(bb_clean_te) + 1), bb_clean_te,
                           color=colors_bb[i], lw=1.5, alpha=0.9, linestyle='--', label=f'BlockBoost {label_suffix}')

    # Theoretical plateau reference line on clean test accuracy panel
    ax_clean_test.axhline(
        y=theoretical_plateau,
        color='#2E7D32', lw=2.0, linestyle=':', alpha=0.9,
        label=f'Theoretical plateau $\\eta + (1-2\\eta)R^*$ = {theoretical_plateau:.4f}'
    )

    # Formatting
    for ax, title, ylabel in [
        (ax_noisy_train, 'Noisy Train Accuracy', 'Accuracy'),
        (ax_clean_train, 'Clean Train Accuracy', 'Accuracy'),
        (ax_clean_test, 'Clean Test Accuracy (Plateau Verification)', 'Accuracy'),
    ]:
        ax.set_title(title, fontsize=11, pad=5)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0.45, 1.02)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=8, loc='lower right')

    ax_clean_test.set_xlabel('Boosting Round (M)', fontsize=10)

    out_path = os.path.join(out_dir, f'learning_curves_eta_{str(eta).replace(".", "_")}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {out_path}")


def run_phase_1_1_A(config: Phase1Config):
    print("=" * 60)
    print("PHASE 1.1.A — PROTOCOL A: PLATEAU VERIFICATION")
    print("=" * 60)
    print()
    print("PURPOSE: Verify that learning curves stabilize at the theoretical")
    print("         plateau value. NOT a statistical power experiment.")
    print(f"         Limitation: With {PROTOCOL_A_N_SEEDS} seeds we cannot reliably")
    print("         compare algorithm means — see Protocol B (Phase 1.1) for that.")
    print()

    # DGP setup
    theta = compute_theta(config.ar_coeffs)
    print(f"AR({len(config.ar_coeffs)}) theta = {theta:.4f}")

    # Load lambda_optimal from Phase 0
    phase0_path = os.path.join(os.path.dirname(__file__), '../results/phase_0/summary.json')
    if os.path.exists(phase0_path):
        with open(phase0_path) as f:
            phase0 = json.load(f)
        lambda_optimal = phase0.get('lambda_optimal', 1.0)
        print(f"lambda_optimal = {lambda_optimal:.4f} (from Phase 0)")
    else:
        lambda_optimal = 1.0
        print("[!] Phase 0 summary not found — using lambda=1.0")

    seeds = generate_seeds(PROTOCOL_A_N_SEEDS, config.base_seed)
    etas = PROTOCOL_A_ETAS

    out_dir = os.path.join(os.path.dirname(__file__), '../results/phase_1_1_A_plateau')
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    all_runs = []

    for eta in etas:
        a_T_theory = compute_theoretical_optimal_block_size(config.n_samples, theta, eta)
        a_T = max(1, int(np.ceil(lambda_optimal * a_T_theory)))
        bayes_risk_noisy = compute_bayes_risk(eta)

        # Theoretical clean test accuracy plateau: eta + (1-2*eta) * (1 - bayes_risk_clean)
        # We approximate bayes_risk_clean ~ 1 - clean_stump_baseline.
        # But since we don't run stump here, use the known eta-derived bound:
        # For noisy labels, the test accuracy under ideal conditions is ~ 1 - bayes_risk_noisy
        # = 1 - eta. For clean test acc, the limit is closer to 1 - 0 in a perfectly separable
        # problem. We use: theoretical_plateau = eta + (1 - 2*eta) * (1 - bayes_risk_noisy)
        # = eta + (1-2*eta) * (1-eta).
        theoretical_plateau = eta + (1.0 - 2.0 * eta) * (1.0 - bayes_risk_noisy)

        print(f"\n{'=' * 60}")
        print(f"ETA = {eta}  |  a_T_theory = {a_T_theory}  |  a_T_eff = {a_T}")
        print(f"Theoretical plateau (clean test reference): {theoretical_plateau:.4f}")
        print(f"{'=' * 60}")

        for seed in seeds:
            print(f"  Running seed = {seed}  (M = {PROTOCOL_A_M} rounds) ...")
            run_data = run_single_seed(seed, eta, config, a_T)
            all_runs.append(run_data)

            # Print per-seed summary
            final_ada = run_data['ada_clean_test_curve'][-1] if run_data['ada_clean_test_curve'] else float('nan')
            final_bb = run_data['block_clean_test_curve'][-1] if run_data['block_clean_test_curve'] else float('nan')
            print(f"    AdaBoost  final clean test acc: {final_ada:.4f}")
            print(f"    BlockBoost final clean test acc: {final_bb:.4f}")

        # Generate and save learning curve plot for this eta
        print(f"\nGenerating learning curve plot for eta={eta} ...")
        plot_learning_curves(all_runs, eta, theoretical_plateau, plots_dir, a_T)

    # Serialize raw results (curves as lists for JSON compatibility)
    serializable = []
    for run in all_runs:
        serializable.append({
            'seed': run['seed'],
            'eta': run['eta'],
            'a_T': run['a_T'],
            'ada_noisy_train_curve': list(run['ada_noisy_train_curve']),
            'ada_clean_train_curve': list(run['ada_clean_train_curve']),
            'ada_clean_test_curve': list(run['ada_clean_test_curve']),
            'block_noisy_train_curve': list(run['block_noisy_train_curve']),
            'block_clean_train_curve': list(run['block_clean_train_curve']),
            'block_clean_test_curve': list(run['block_clean_test_curve']),
        })

    metadata = {
        'protocol': 'A — Plateau Verification',
        'n_seeds': PROTOCOL_A_N_SEEDS,
        'M': PROTOCOL_A_M,
        'etas': PROTOCOL_A_ETAS,
        'ar_coeffs': config.ar_coeffs,
        'lambda_optimal': lambda_optimal,
        'n_samples': config.n_samples,
        'note': (
            'Protocol A is designed solely to visualize curve stabilization. '
            'With 2 seeds there is insufficient statistical power for algorithm comparison. '
            'See Phase 1.1 (Protocol B) for statistical comparison results.'
        )
    }

    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    with open(os.path.join(out_dir, 'raw_curves.json'), 'w') as f:
        json.dump(serializable, f, indent=4)

    print(f"\nAll results and plots saved to: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    # Protocol A overrides: 2 seeds, M=3000, eta=[0.10, 0.20].
    # All other config fields (T, ar_coeffs, etc.) from defaults.
    config = Phase1Config(n_seeds=PROTOCOL_A_N_SEEDS, M=PROTOCOL_A_M)
    run_phase_1_1_A(config)
