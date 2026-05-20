import json
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Add the parent directory to the path so we can import 'experiments' and 'model'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.configs.schema import Phase0Config
from experiments.utils.reproducibility import set_seed
from experiments.dgp.ar_process import (
    generate_ar_process,
    generate_labels,
    compute_companion_matrix_spectral_radius,
    compute_theta,
    compute_theoretical_optimal_block_size
)
from experiments.dgp.noise import generate_markov_noise, compute_bayes_risk
from experiments.dgp.lambda_optimizer import optimize_lambda


def run_phase_0(config: Phase0Config):
    print("=" * 50)
    print("PHASE 0: PRELIMINARIES AND BASELINES")
    print("=" * 50)

    # 0.1 Fix the data generating process completely.
    set_seed(config.random_state)
    print(f"[0.1] Generating data with seed = {config.random_state}")

    X = generate_ar_process(
        n_samples=config.n_samples,
        n_features=config.n_features,
        ar_coeffs=config.ar_coeffs,
        noise_std=config.noise_std,
        random_state=config.random_state
    )
    y_clean = generate_labels(X, random_state=config.random_state)

    # Referential noise level for baseline reporting (eta=0.10)
    eta = 0.10
    alpha_markov = 0.5
    y_noisy = generate_markov_noise(y_clean, eta=eta, alpha_markov=alpha_markov, random_state=config.random_state)

    # 0.2 Compute Analytical DGP Properties
    rho = compute_companion_matrix_spectral_radius(config.ar_coeffs)
    theta = compute_theta(config.ar_coeffs)
    a_T_theory = compute_theoretical_optimal_block_size(config.n_samples, theta, eta=0.0)
    bayes_risk = compute_bayes_risk(eta)

    print("\n--- Analytical DGP Properties ---")
    print(f"[*] AR({len(config.ar_coeffs)}) Coefficients: {config.ar_coeffs}")
    print(f"      - Spectral Radius (rho):             {rho:.4f}")
    if rho >= 1.0:
        print("      - [!] WARNING: Process is NON-STATIONARY (rho >= 1).")
    else:
        print(f"      - Mixing Decay Rate (theta):         {theta:.4f}")
        print(f"      - Theoretical Opt. Block Size (a_T*): {a_T_theory}")
    print(f"      - Bayes Risk (eta={eta}):              {bayes_risk:.4f}")

    # 0.3 Establish weak learner baseline using temporal train/test split.
    X_train, X_test, y_train_clean, y_test_clean = train_test_split(X, y_clean, test_size=0.3, shuffle=False)
    _, _, y_train_noisy, y_test_noisy = train_test_split(X, y_noisy, test_size=0.3, shuffle=False)

    stump = DecisionTreeClassifier(max_depth=1, random_state=config.random_state)
    stump.fit(X_train, y_train_clean)
    clean_acc_train = stump.score(X_train, y_train_clean)
    clean_acc_test = stump.score(X_test, y_test_clean)

    stump_noisy = DecisionTreeClassifier(max_depth=1, random_state=config.random_state)
    stump_noisy.fit(X_train, y_train_noisy)
    noisy_acc_train = stump_noisy.score(X_train, y_train_noisy)
    noisy_acc_test = stump_noisy.score(X_test, y_test_noisy)

    print(f"\n[0.3] Weak Learner Baseline (Stump):")
    print(f"      - Train Accuracy (Clean): {clean_acc_train:.4f}")
    print(f"      - Test Accuracy (Clean):  {clean_acc_test:.4f}")
    print(f"      - Train Accuracy (Noisy): {noisy_acc_train:.4f}")
    print(f"      - Test Accuracy (Noisy):  {noisy_acc_test:.4f}")

    if not (0.55 <= clean_acc_test <= 1.00):
        print("\n[!] FATAL: Clean test accuracy outside [0.55, 1.0]. Validation FAILED.")
        sys.exit(1)
    else:
        print("\n[+] VALIDATION PASSED. Ready for boosting experiments.")

    # 0.4 Optimize the block-size scaling factor lambda via temporal cross-validation.
    # We optimize on the training split only (no leakage from test split).
    # The effective block size used by all downstream experiments is:
    #   a_T = ceil(lambda * a_T_theory)
    print("\n--- Lambda Optimization for Block Size ---")
    print(f"Running grid search over lambda candidates using {3} temporal CV folds...")

    best_lambda, best_a_T, cv_scores = optimize_lambda(
        X_train=X_train,
        y_train=y_train_clean,
        a_T_theory=a_T_theory,
        eta=eta,
        random_state=config.random_state,
    )

    print(f"\nCV Scores per lambda:")
    for lam, score in sorted(cv_scores.items()):
        marker = " <<< OPTIMAL" if lam == best_lambda else ""
        a_T_candidate = max(1, int(np.ceil(lam * a_T_theory)))
        print(f"    lambda={lam:.2f} (a_T={a_T_candidate:4d}):  val_acc={score:.4f}{marker}")

    print(f"\n[+] Optimal lambda = {best_lambda:.4f}")
    print(f"[+] Optimal a_T    = ceil({best_lambda:.4f} * {a_T_theory}) = {best_a_T}")

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), '../results/phase_0')
    os.makedirs(out_dir, exist_ok=True)
    results = {
        'rho': rho,
        'theta': theta,
        'a_T_theory': a_T_theory,
        'lambda_optimal': best_lambda,
        'a_T_optimal': best_a_T,
        'cv_scores': cv_scores,
        'bayes_risk_eta_0_1': bayes_risk,
        'clean_test_acc': clean_acc_test,
        'noisy_test_acc': noisy_acc_test
    }
    out_path = os.path.join(out_dir, 'summary.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n      - Results saved to {os.path.abspath(out_path)}")


def reproducibility_test():
    print("\n--- Running Reproducibility Test ---")
    config = Phase0Config()

    set_seed(config.random_state)
    X1 = generate_ar_process(config.n_samples, config.n_features, config.ar_coeffs, config.noise_std, config.random_state)
    y1 = generate_labels(X1, config.random_state)

    set_seed(config.random_state)
    X2 = generate_ar_process(config.n_samples, config.n_features, config.ar_coeffs, config.noise_std, config.random_state)
    y2 = generate_labels(X2, config.random_state)

    assert np.all(X1 == X2), "Reproducibility failed on X"
    assert np.all(y1 == y2), "Reproducibility failed on y"
    print("[+] Reproducibility test passed. Fixed seeds guarantee deterministic DGP.")


if __name__ == "__main__":
    config = Phase0Config(n_samples=3000, n_features=10)
    run_phase_0(config)
    reproducibility_test()
