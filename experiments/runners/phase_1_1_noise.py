import json
import os
import sys
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

# Add parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.configs.schema import Phase1Config
from experiments.utils.reproducibility import generate_seeds
from experiments.dgp.ar_process import generate_ar_process, generate_labels, compute_theta, compute_theoretical_optimal_block_size
from experiments.dgp.noise import generate_markov_noise, compute_bayes_risk
from experiments.metrics.tracking import compute_plateau_average, identify_collapse

from model.adaboost import AdaBoostM1
from model.blockboost import BlockBoostClassifier

def evaluate_single_run(eta: float, seed: int, config: Phase1Config, a_T_star: int):
    """
    Evaluates AdaBoost and BlockBoost for a specific noise rate and seed.
    Returns a dict of metrics.
    """
    X = generate_ar_process(
        n_samples=config.n_samples,
        n_features=config.n_features,
        ar_coeffs=config.ar_coeffs,
        noise_std=config.noise_std,
        random_state=seed
    )
    y_clean = generate_labels(X, random_state=seed)
    y_noisy = generate_markov_noise(y_clean, eta=eta, alpha_markov=config.alpha_markov, random_state=seed)
    
    X_train, X_test, y_train_clean, y_test_clean = train_test_split(X, y_clean, test_size=0.3, shuffle=False)
    _, _, y_train_noisy, y_test_noisy = train_test_split(X, y_noisy, test_size=0.3, shuffle=False)
    
    # 1. Train AdaBoost
    stump = DecisionTreeClassifier(max_depth=1)
    ada = AdaBoostM1(estimator=stump, n_estimators=config.M)
    # Fit and collect metrics round by round
    ada_train_acc, ada_test_acc = ada.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_noisy)
    # Compute clean test accuracy at the end
    ada_final_clean_acc = ada.score(X_test, y_test_clean)
    
    # Identify collapse
    ada_collapse_round = identify_collapse(ada_test_acc)
    ada_plateau = compute_plateau_average(ada_test_acc, 1000, 1500)
    
    # 2. Train BlockBoost
    block = BlockBoostClassifier(estimator=stump, n_estimators=config.M, block_size=a_T_star)
    block_train_acc, block_test_acc = block.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_noisy)
    block_final_clean_acc = block.score(X_test, y_test_clean)
    
    block_collapse_round = identify_collapse(block_test_acc)
    block_plateau = compute_plateau_average(block_test_acc, 1000, 1500)
    
    return {
        'eta': eta,
        'seed': seed,
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

def run_phase_1_1(config: Phase1Config):
    print("=" * 50)
    print("PHASE 1.1: NOISE RATE ETA EXPERIMENTS")
    print("=" * 50)
    
    theta = compute_theta(config.ar_coeffs)
    
    # Load optimal lambda calibrated in Phase 0.
    phase0_summary_path = os.path.join(os.path.dirname(__file__), '../results/phase_0/summary.json')
    if os.path.exists(phase0_summary_path):
        with open(phase0_summary_path) as f:
            phase0_results = json.load(f)
        lambda_optimal = phase0_results.get('lambda_optimal', 1.0)
        print(f"Loaded lambda_optimal = {lambda_optimal:.4f} from Phase 0 results.")
    else:
        lambda_optimal = 1.0
        print("[!] Phase 0 summary not found. Defaulting lambda_optimal = 1.0 (raw theoretical block size).")
    
    seeds = generate_seeds(config.n_seeds, config.base_seed)
    etas = config.etas
    
    print(f"Total noise rates to test: {len(etas)} with {config.n_seeds} seeds each")
    print(f"Max boosting rounds (M): {config.M}")
    
    all_results = []
    
    start_time = datetime.now()
    
    for eta in etas:
        a_T_theory = compute_theoretical_optimal_block_size(config.n_samples, theta, eta)
        a_T_star = max(1, int(np.ceil(lambda_optimal * a_T_theory)))
        print(f"\n==================================================")
        print(f"STARTING ETA = {eta} | a_T_theory = {a_T_theory} | lambda = {lambda_optimal:.3f} | a_T_eff = {a_T_star}")
        print(f"==================================================")
        
        eta_tasks = [(eta, seed, config, a_T_star) for seed in seeds]
        
        # Run tasks for current eta in parallel
        eta_results = Parallel(n_jobs=-1, verbose=10)(
            delayed(evaluate_single_run)(*task) for task in eta_tasks
        )
        all_results.extend(eta_results)
        
        # Immediate aggregation for this eta
        ada_collapses = sum(1 for r in eta_results if r['ada_collapse_round'] != -1) / len(eta_results)
        bb_collapses = sum(1 for r in eta_results if r['block_collapse_round'] != -1) / len(eta_results)
        ada_clean_accs = [r['ada_final_test_clean'] for r in eta_results]
        bb_clean_accs = [r['block_final_test_clean'] for r in eta_results]
        
        print(f"\n>>>>> INTERMEDIATE SUMMARY FOR ETA={eta} <<<<<")
        print(f"AdaBoost Collapse Fraction:   {ada_collapses:.2f}")
        print(f"BlockBoost Collapse Fraction: {bb_collapses:.2f}")
        print(f"AdaBoost Clean Test Acc:   {np.mean(ada_clean_accs):.4f} \u00b1 {np.std(ada_clean_accs):.4f}")
        print(f"BlockBoost Clean Test Acc: {np.mean(bb_clean_accs):.4f} \u00b1 {np.std(bb_clean_accs):.4f}")
        print("<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>\n")
    
    results = all_results
    print(f"Finished in {datetime.now() - start_time}.")
    
    # Aggregate results by eta
    aggregated = {}
    for eta in set(etas):
        eta_results = [r for r in results if r['eta'] == eta]
        
        # Calculate means and stds for each metric across 20 seeds
        # Also fraction of collapse for AdaBoost
        agg_eta = {}
        for metric in eta_results[0].keys():
            if metric in ('eta', 'seed'):
                continue
            
            values = [r[metric] for r in eta_results]
            if metric.endswith('collapse_round'):
                collapse_fraction = sum(1 for v in values if v != -1) / len(values)
                agg_eta[metric.replace('_round', '_fraction')] = float(collapse_fraction)
            else:
                # filter out nans for plateau average if algorithm stopped early
                valid_values = [v for v in values if not np.isnan(v)]
                if len(valid_values) == 0:
                    agg_eta[f"{metric}_mean"] = float('nan')
                    agg_eta[f"{metric}_std"] = 0.0
                else:
                    agg_eta[f"{metric}_mean"] = float(np.mean(valid_values))
                    agg_eta[f"{metric}_std"] = float(np.std(valid_values))
                
        aggregated[eta] = agg_eta
        
    print("\n--- Summary of Phase 1.1 ---\n")
    for eta in sorted(aggregated.keys()):
        print(f"ETA = {eta}:")
        print(f"  AdaBoost Collapse Fraction:   {aggregated[eta]['ada_collapse_fraction']:.2f}")
        print(f"  BlockBoost Collapse Fraction: {aggregated[eta]['block_collapse_fraction']:.2f}")
        print(f"  AdaBoost Clean Test Acc:   {aggregated[eta]['ada_final_test_clean_mean']:.4f} \u00b1 {aggregated[eta]['ada_final_test_clean_std']:.4f}")
        print(f"  BlockBoost Clean Test Acc: {aggregated[eta]['block_final_test_clean_mean']:.4f} \u00b1 {aggregated[eta]['block_final_test_clean_std']:.4f}")
        print("")
        
    # Save raw and aggregated results
    out_dir = os.path.join(os.path.dirname(__file__), '../results/phase_1_1_noise')
    os.makedirs(out_dir, exist_ok=True)
    
    with open(os.path.join(out_dir, 'aggregated.json'), 'w') as f:
        json.dump(aggregated, f, indent=4)
        
    with open(os.path.join(out_dir, 'raw_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    # Test quickly
    # config = Phase1Config(n_seeds=1, n_samples=1200)
    config = Phase1Config() # Uses the new defaults: 3000, 10 seeds, M=500
    run_phase_1_1(config)
