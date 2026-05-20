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
from experiments.metrics.tracking import compute_plateau_average, identify_collapse

from model.adaboost import AdaBoostM1
from model.blockboost import BlockBoostClassifier


def evaluate_single_run_boosting_rounds(M: int, seed: int, config: Phase1Config):
    eta = 0.10
    
    # Generate data
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
    
    rho = compute_companion_matrix_spectral_radius(config.ar_coeffs)
    a_T_star = compute_theoretical_optimal_block_size(config.n_samples, rho)

    stump = DecisionTreeClassifier(max_depth=1)
    
    ada = AdaBoostM1(estimator=stump, n_estimators=M)
    ada_train_acc, ada_test_acc = ada.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_noisy)
    ada_collapse_round = identify_collapse(ada_test_acc)
    
    # We define plateau as the average of the last 1/3 of the rounds if M is small, or 1000-1500 if M is large.
    start_plat = int(M * 2/3)
    ada_plateau = compute_plateau_average(ada_test_acc, start_plat, M)
    ada_clean_acc = ada.score(X_test, y_test_clean)
    
    block = BlockBoostClassifier(estimator=stump, n_estimators=M, block_size=a_T_star)
    block_train_acc, block_test_acc = block.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_noisy)
    block_collapse_round = identify_collapse(block_test_acc)
    block_plateau = compute_plateau_average(block_test_acc, start_plat, M)
    block_clean_acc = block.score(X_test, y_test_clean)
    
    return {
        'M': M,
        'seed': seed,
        'ada_final_noisy': ada_test_acc[-1] if ada_test_acc else 0.0,
        'ada_final_clean': float(ada_clean_acc),
        'ada_collapse_round': ada_collapse_round,
        'ada_plateau': ada_plateau,
        'block_final_noisy': block_test_acc[-1] if block_test_acc else 0.0,
        'block_final_clean': float(block_clean_acc),
        'block_collapse_round': block_collapse_round,
        'block_plateau': block_plateau
    }

def run_phase_1_4(config: Phase1Config):
    print("=" * 50)
    print("PHASE 1.4: BOOSTING ROUNDS (M) VALIDATION")
    print("=" * 50)
    
    seeds = generate_seeds(config.n_seeds, config.base_seed)
    rounds_list = [100, 200, 500, 1000, 1500, 2000, 3000]
    
    tasks = [(M, seed, config) for M in rounds_list for seed in seeds]
    print(f"Total runs: {len(tasks)} ({len(rounds_list)} rounds amounts x {config.n_seeds} seeds)")
    
    # M=3000 takes longest, let's execute in descending order so 3000 gets scheduled first
    tasks = sorted(tasks, key=lambda x: x[0], reverse=True)
    
    start_time = datetime.now()
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_single_run_boosting_rounds)(*task) for task in tasks
    )
    print(f"Finished in {datetime.now() - start_time}.")
    
    aggregated = {}
    for M in rounds_list:
        M_res = [r for r in results if r['M'] == M]
        agg = {}
        for metric in M_res[0].keys():
            if metric in ('M', 'seed'): continue
            vals = [r[metric] for r in M_res if r[metric] is not None and not np.isnan(r[metric])]

            if metric.endswith('collapse_round'):
                if not vals:
                    agg[metric.replace('_round', '_fraction')] = 0.0
                else:
                    agg[metric.replace('_round', '_fraction')] = sum(1 for v in vals if v != -1) / len(vals)
            else:
                if vals:
                    agg[f"{metric}_mean"] = float(np.mean(vals))
                    agg[f"{metric}_std"] = float(np.std(vals))
                else:
                    agg[f"{metric}_mean"] = float('nan')
                    agg[f"{metric}_std"] = 0.0
        
        aggregated[M] = agg
        print(f"M = {M:4d}: "
              f"Ada Plateau: {agg.get('ada_plateau_mean', 0):.4f}, "
              f"Block Plateau: {agg.get('block_plateau_mean', 0):.4f}, "
              f"Ada Collapse Frac: {agg.get('ada_collapse_fraction', 0):.2f}")

    out_dir = os.path.join(os.path.dirname(__file__), '../results/phase_1_4_boosting_rounds')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'aggregated.json'), 'w') as f:
        json.dump(aggregated, f, indent=4)
        
    print(f"Results saved to {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    config = Phase1Config(n_seeds=5)
    run_phase_1_4(config)
