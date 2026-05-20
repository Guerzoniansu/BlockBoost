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
from experiments.dgp.ar_process import generate_ar_process, generate_labels, compute_companion_matrix_spectral_radius, compute_theoretical_optimal_block_size
from experiments.dgp.noise import generate_markov_noise
from experiments.metrics.tracking import identify_collapse

from model.adaboost import AdaBoostM1
from model.blockboost import BlockBoostClassifier

def evaluate_single_run_sample_size(T: int, seed: int, config: Phase1Config):
    eta = 0.10
    
    # Generate data with new sample size T
    X = generate_ar_process(
        n_samples=T,
        n_features=config.n_features,
        ar_coeffs=config.ar_coeffs,
        noise_std=config.noise_std,
        random_state=seed
    )
    y_clean = generate_labels(X, random_state=seed)
    y_noisy = generate_markov_noise(y_clean, eta=eta, alpha_markov=config.alpha_markov, random_state=seed)
    
    X_train, X_test, y_train_clean, y_test_clean = train_test_split(X, y_clean, test_size=0.3, shuffle=False)
    _, _, y_train_noisy, y_test_noisy = train_test_split(X, y_noisy, test_size=0.3, shuffle=False)
    
    rho = compute_companion_matrix_spectral_radius(config.ar_coeffs)
    a_T_star = compute_theoretical_optimal_block_size(T, rho)

    stump = DecisionTreeClassifier(max_depth=1)
    
    # 1. AdaBoost
    ada = AdaBoostM1(estimator=stump, n_estimators=config.M)
    ada_train_acc, ada_test_acc = ada.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_noisy)
    
    ada_final_train = ada_train_acc[-1] if ada_train_acc else 0.0
    ada_final_test = ada_test_acc[-1] if ada_test_acc else 0.0
    ada_gap = abs(ada_final_train - ada_final_test)
    ada_collapse_round = identify_collapse(ada_test_acc)
    
    # 2. BlockBoost
    block = BlockBoostClassifier(estimator=stump, n_estimators=config.M, block_size=a_T_star)
    block_train_acc, block_test_acc = block.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_noisy)
    
    block_final_train = block_train_acc[-1] if block_train_acc else 0.0
    block_final_test = block_test_acc[-1] if block_test_acc else 0.0
    block_gap = abs(block_final_train - block_final_test)
    
    return {
        'T': T,
        'seed': seed,
        'a_T_star': a_T_star,
        'ada_gap': ada_gap,
        'ada_collapse_round': ada_collapse_round,
        'block_gap': block_gap
    }

def run_phase_1_3(config: Phase1Config):
    print("=" * 50)
    print("PHASE 1.3: SAMPLE SIZE (T) VALIDATION")
    print("=" * 50)
    
    seeds = generate_seeds(config.n_seeds, config.base_seed)
    sample_sizes = [500, 1000, 2000, 3000, 5000, 10000]
    
    tasks = [(T, seed, config) for T in sample_sizes for seed in seeds]
    print(f"Total runs: {len(tasks)} ({len(sample_sizes)} sample sizes x {config.n_seeds} seeds)")
    
    start_time = datetime.now()
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_single_run_sample_size)(*task) for task in tasks
    )
    print(f"Finished in {datetime.now() - start_time}.")
    
    aggregated = {}
    for T in sample_sizes:
        T_res = [r for r in results if r['T'] == T]
        agg = {}
        for metric in T_res[0].keys():
            if metric in ('T', 'seed'): continue
            vals = [r[metric] for r in T_res]
            
            if metric.endswith('collapse_round'):
                collapse_frac = sum(1 for v in vals if v != -1) / len(vals)
                agg[metric.replace('_round', '_fraction')] = float(collapse_frac)
            else:
                agg[f"{metric}_mean"] = float(np.mean(vals))
                agg[f"{metric}_std"] = float(np.std(vals))
        aggregated[T] = agg
        print(f"T = {T:5d}: a_T* = {agg.get('a_T_star_mean', 0):.1f}, "
              f"Ada Gap: {agg.get('ada_gap_mean', 0):.4f}, "
              f"Block Gap: {agg.get('block_gap_mean', 0):.4f}, "
              f"Ada Collapse Frac: {agg.get('ada_collapse_fraction', 0):.2f}")

    out_dir = os.path.join(os.path.dirname(__file__), '../results/phase_1_3_sample_size')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'aggregated.json'), 'w') as f:
        json.dump(aggregated, f, indent=4)
        
    print(f"Results saved to {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    config = Phase1Config(n_seeds=5)
    run_phase_1_3(config)
