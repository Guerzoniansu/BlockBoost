"""
phase_1_1_1_unit_root.py

Phase 1.1.1: Noise Rate Experiment under a Unit-Root AR(5) Process.

This mirrors Phase 1.1 exactly, but uses AR(5) coefficients that sum to 1.0,
placing the process at the boundary of stationarity (unit root boundary).
The spectral radius (rho) equals 1.0, making the theoretical block size formula
undefined. This experiment tests how BlockBoost and AdaBoost behave under
near-non-stationary temporal dependence.

AR(5) coefficients: [0.4, 0.25, 0.15, 0.1, 0.1]  (sum = 1.0)

All other experimental parameters are inherited from Phase1Config defaults.
"""

import json
import os
import sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.configs.schema import Phase1Config
from experiments.utils.reproducibility import generate_seeds
from experiments.dgp.ar_process import (
    generate_ar_process,
    generate_labels,
    compute_companion_matrix_spectral_radius,
    compute_theta,
    compute_theoretical_optimal_block_size,
)
from experiments.dgp.noise import generate_markov_noise, compute_bayes_risk
from experiments.metrics.tracking import compute_plateau_average, identify_collapse

from model.adaboost import AdaBoostM1
from model.blockboost import BlockBoostClassifier

# Unit-root AR(5) coefficients: sum = 1.0, boundary of stationarity.
UNIT_ROOT_AR_COEFFS = [0.4, 0.25, 0.15, 0.1, 0.1]

# Fixed block size fallback for unit-root case (rho = 1 -> theta = 0 -> formula undefined).
# We use the lambda_optimal-scaled heuristic from Phase 0 if available,
# otherwise fall back to a sensible fixed value.
FALLBACK_BLOCK_SIZE = 6


def evaluate_single_run(eta: float, seed: int, config: Phase1Config, a_T: int) -> dict:
    """
    Evaluates AdaBoost and BlockBoost for a specific noise rate and seed under
    the unit-root AR(5) DGP.
    """
    X = generate_ar_process(
        n_samples=config.n_samples,
        n_features=config.n_features,
        ar_coeffs=UNIT_ROOT_AR_COEFFS,
        noise_std=config.noise_std,
        random_state=seed
    )
    y_clean = generate_labels(X, random_state=seed)
    y_noisy = generate_markov_noise(y_clean, eta=eta, alpha_markov=config.alpha_markov, random_state=seed)

    # Temporal split: train on first 70%, test on last 30%.
    X_train, X_test, y_train_clean, y_test_clean = train_test_split(X, y_clean, test_size=0.3, shuffle=False)
    _, _, y_train_noisy, y_test_noisy = train_test_split(X, y_noisy, test_size=0.3, shuffle=False)

    # AdaBoost
    stump = DecisionTreeClassifier(max_depth=1)
    ada = AdaBoostM1(estimator=stump, n_estimators=config.M)
    ada_train_acc, ada_test_acc = ada.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_noisy)
    ada_final_clean_acc = ada.score(X_test, y_test_clean)
    ada_collapse_round = identify_collapse(ada_test_acc)
    ada_plateau = compute_plateau_average(ada_test_acc, 400, 500)

    # BlockBoost
    block = BlockBoostClassifier(estimator=stump, n_estimators=config.M, block_size=a_T)
    block_train_acc, block_test_acc = block.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_noisy)
    block_final_clean_acc = block.score(X_test, y_test_clean)
    block_collapse_round = identify_collapse(block_test_acc)
    block_plateau = compute_plateau_average(block_test_acc, 400, 500)

    return {
        'eta': eta,
        'seed': seed,
        'a_T': a_T,
        'ada_final_train_noisy': ada_train_acc[-1] if ada_train_acc else 0.0,
        'ada_final_test_noisy': ada_test_acc[-1] if ada_test_acc else 0.0,
        'ada_final_test_clean': float(ada_final_clean_acc),
        'ada_collapse_round': ada_collapse_round,
        'ada_plateau': ada_plateau,
        'block_final_train_noisy': block_train_acc[-1] if block_train_acc else 0.0,
        'block_final_test_noisy': block_test_acc[-1] if block_test_acc else 0.0,
        'block_final_test_clean': float(block_final_clean_acc),
        'block_collapse_round': block_collapse_round,
        'block_plateau': block_plateau,
    }


def run_phase_1_1_1(config: Phase1Config):
    print("=" * 60)
    print("PHASE 1.1.1: UNIT-ROOT AR(5) NOISE RATE EXPERIMENTS")
    print("=" * 60)

    # Report DGP properties for the unit-root process.
    rho = compute_companion_matrix_spectral_radius(UNIT_ROOT_AR_COEFFS)
    print(f"\n--- Unit-Root AR(5) DGP Properties ---")
    print(f"[*] AR Coefficients: {UNIT_ROOT_AR_COEFFS}  (sum = {sum(UNIT_ROOT_AR_COEFFS):.2f})")
    print(f"    Spectral Radius (rho): {rho:.6f}")
    if rho >= 0.999:
        print("    [!] Process is AT or BEYOND the stationarity boundary (rho >= 1).")
        print("         Theoretical theta and a_T* formula are undefined.")
    else:
        theta = compute_theta(UNIT_ROOT_AR_COEFFS)
        print(f"    theta = {theta:.6f}  (very small -> very slow mixing)")

    # Load lambda_optimal from Phase 0 to set the block size.
    # Since the formula is undefined, we apply lambda_optimal * fallback.
    phase0_path = os.path.join(os.path.dirname(__file__), '../results/phase_0/summary.json')
    if os.path.exists(phase0_path):
        with open(phase0_path) as f:
            phase0 = json.load(f)
        lambda_optimal = phase0.get('lambda_optimal', 1.0)
        a_T_theory_stationary = phase0.get('a_T_theory', FALLBACK_BLOCK_SIZE)
        a_T_eff = max(1, int(np.ceil(lambda_optimal * a_T_theory_stationary)))
        print(f"\nLoaded from Phase 0: lambda_optimal={lambda_optimal:.4f}, a_T_theory={a_T_theory_stationary}")
        print(f"Effective block size for unit-root phase: a_T_eff = {a_T_eff}")
    else:
        a_T_eff = FALLBACK_BLOCK_SIZE
        lambda_optimal = None
        print(f"\n[!] Phase 0 summary not found. Using fallback block size = {a_T_eff}")

    seeds = generate_seeds(config.n_seeds, config.base_seed)
    etas = config.etas

    print(f"\nTotal noise rates: {len(etas)} | Seeds per rate: {config.n_seeds}")
    print(f"Boosting rounds (M): {config.M}")

    all_results = []
    start_time = datetime.now()

    for eta in etas:
        print(f"\n{'=' * 60}")
        print(f"STARTING ETA = {eta} | a_T_eff = {a_T_eff}")
        print(f"{'=' * 60}")

        eta_tasks = [(eta, seed, config, a_T_eff) for seed in seeds]

        eta_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(evaluate_single_run)(*task) for task in eta_tasks
        )
        all_results.extend(eta_results)

        # Immediate summary for this eta level.
        ada_collapses = sum(1 for r in eta_results if r['ada_collapse_round'] != -1) / len(eta_results)
        bb_collapses = sum(1 for r in eta_results if r['block_collapse_round'] != -1) / len(eta_results)
        ada_clean_accs = [r['ada_final_test_clean'] for r in eta_results]
        bb_clean_accs = [r['block_final_test_clean'] for r in eta_results]

        print(f"\n>>>>> INTERMEDIATE SUMMARY FOR ETA={eta} <<<<<")
        print(f"AdaBoost Collapse Fraction:   {ada_collapses:.2f}")
        print(f"BlockBoost Collapse Fraction: {bb_collapses:.2f}")
        print(f"AdaBoost Clean Test Acc:   {np.mean(ada_clean_accs):.4f} \u00b1 {np.std(ada_clean_accs):.4f}")
        print(f"BlockBoost Clean Test Acc: {np.mean(bb_clean_accs):.4f} \u00b1 {np.std(bb_clean_accs):.4f}")
        print("<" * 20 + ">" * 20 + "\n")

    print(f"Finished in {datetime.now() - start_time}.")

    # Aggregate results.
    aggregated = {}
    for eta in set(etas):
        eta_results = [r for r in all_results if r['eta'] == eta]
        agg = {}
        for metric in eta_results[0].keys():
            if metric in ('eta', 'seed', 'a_T'):
                continue
            values = [r[metric] for r in eta_results]
            if metric.endswith('collapse_round'):
                agg[metric.replace('_round', '_fraction')] = float(
                    sum(1 for v in values if v != -1) / len(values)
                )
            else:
                valid = [v for v in values if not np.isnan(v)]
                agg[f'{metric}_mean'] = float(np.mean(valid)) if valid else float('nan')
                agg[f'{metric}_std'] = float(np.std(valid)) if valid else 0.0
        aggregated[eta] = agg

    print("\n--- Final Summary: Phase 1.1.1 (Unit-Root AR(5)) ---\n")
    for eta in sorted(aggregated.keys()):
        print(f"ETA = {eta}:")
        print(f"  AdaBoost Collapse Fraction:   {aggregated[eta]['ada_collapse_fraction']:.2f}")
        print(f"  BlockBoost Collapse Fraction: {aggregated[eta]['block_collapse_fraction']:.2f}")
        print(f"  AdaBoost Clean Test Acc:   {aggregated[eta]['ada_final_test_clean_mean']:.4f} \u00b1 {aggregated[eta]['ada_final_test_clean_std']:.4f}")
        print(f"  BlockBoost Clean Test Acc: {aggregated[eta]['block_final_test_clean_mean']:.4f} \u00b1 {aggregated[eta]['block_final_test_clean_std']:.4f}")
        print("")

    # Save results.
    out_dir = os.path.join(os.path.dirname(__file__), '../results/phase_1_1_1_unit_root')
    os.makedirs(out_dir, exist_ok=True)

    metadata = {
        'ar_coeffs': UNIT_ROOT_AR_COEFFS,
        'rho': float(rho),
        'lambda_optimal': lambda_optimal,
        'a_T_eff': a_T_eff,
        'M': config.M,
        'n_seeds': config.n_seeds,
        'n_samples': config.n_samples,
        'etas': list(etas),
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    with open(os.path.join(out_dir, 'aggregated.json'), 'w') as f:
        json.dump(aggregated, f, indent=4)
    with open(os.path.join(out_dir, 'raw_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"Results saved to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    config = Phase1Config()  # Inherits defaults: T=3000, M=500, 10 seeds, 5 etas
    run_phase_1_1_1(config)
