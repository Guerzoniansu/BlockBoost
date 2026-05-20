import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from experiments.configs.schema import Phase1Config
from experiments.utils.reproducibility import set_seed, generate_seeds
from experiments.dgp.ar_process import generate_ar_process, generate_labels
from experiments.dgp.noise import generate_markov_noise

from model.adaboost import AdaBoostM1
from model.blockboost import BlockBoostClassifier

def evaluate_single_run_block_size(a_T: int, seed: int, config: Phase1Config):
    # Using fixed eta=0.10 for block size validation
    eta = 0.10
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
    
    stump = DecisionTreeClassifier(max_depth=1)
    
    # Train BlockBoost
    block = BlockBoostClassifier(estimator=stump, n_estimators=config.M, block_size=a_T)
    # Track noisy train and clean test accuracies per round
    block_train_acc, block_test_acc_clean = block.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_clean)
    
    # Recover epsilons to see where it plateaus
    alphas = np.array(block.alphas_)
    # epsilon = 1 / (1 + exp(2*alpha))
    epsilons = 1.0 / (1.0 + np.exp(2 * alphas))
    min_epsilon = float(np.min(epsilons))
    
    # Stability: std over late rounds (e.g., 700-1000 if M=1000)
    stability_start = int(0.7 * config.M)
    stability_metric = float(np.std(block_test_acc_clean[stability_start:])) if len(block_test_acc_clean) > stability_start else float('nan')
    
    return {
        'a_T': a_T,
        'seed': seed,
        'clean_test_curve': block_test_acc_clean,
        'final_test_clean': float(block_test_acc_clean[-1]) if block_test_acc_clean else 0.0,
        'stability': stability_metric,
        'empirical_eps_star': min_epsilon
    }

def run_phase_1_2(config: Phase1Config):
    print("=" * 50)
    print("PHASE 1.2: BLOCK SIZE (a_T) VALIDATION (M=1000)")
    print("=" * 50)
    
    seeds = generate_seeds(config.n_seeds, config.base_seed)
    # Sample block sizes spanning from small to large
    block_sizes = [2, 6, 12, 25, 50, 100]
    
    tasks = [(a_T, seed, config) for a_T in block_sizes for seed in seeds]
    print(f"Total runs: {len(tasks)} ({len(block_sizes)} block sizes x {config.n_seeds} seeds)")
    
    start_time = datetime.now()
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_single_run_block_size)(*task) for task in tasks
    )
    print(f"Finished in {datetime.now() - start_time}.")
    
    # Aggregate curves and metrics
    aggregated = {}
    curves_to_plot = {}
    
    for a_T in block_sizes:
        a_T_res = [r for r in results if r['a_T'] == a_T]
        
        # Calculate mean learning curve for this block size
        all_curves = [r['clean_test_curve'] for r in a_T_res]
        # Pad with last value if any seed terminated early
        max_len = max(len(c) for c in all_curves)
        padded_curves = []
        for c in all_curves:
            if len(c) < max_len:
                padded_curves.append(c + [c[-1]] * (max_len - len(c)))
            else:
                padded_curves.append(c)
        
        mean_curve = np.mean(padded_curves, axis=0)
        curves_to_plot[a_T] = mean_curve
        
        # Aggregate scalar metrics
        agg = {}
        for metric in ['final_test_clean', 'stability', 'empirical_eps_star']:
            vals = [r[metric] for r in a_T_res if not np.isnan(r[metric])]
            if len(vals) == 0:
                agg[f"{metric}_mean"] = float('nan')
                agg[f"{metric}_std"] = 0.0
            else:
                agg[f"{metric}_mean"] = float(np.mean(vals))
                agg[f"{metric}_std"] = float(np.std(vals))
        
        aggregated[a_T] = agg
        print(f"a_T = {a_T:3d}: Clean Test: {agg['final_test_clean_mean']:.4f}, Stability: {agg['stability_mean']:.4f}, eps_star: {agg['empirical_eps_star_mean']:.4f}")

    # Plotting
    out_dir = os.path.join(os.path.dirname(__file__), '../results/phase_1_2_block_size')
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    for a_T, curve in curves_to_plot.items():
        plt.plot(range(1, len(curve) + 1), curve, label=f'a_T = {a_T}')
    
    plt.title(f'BlockBoost Clean Test Accuracy vs Round (M={config.M}) for Varying a_T', fontsize=14)
    plt.xlabel('Boosting Round', fontsize=12)
    plt.ylabel('Clean Test Accuracy', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Block Size')
    
    plot_path = os.path.join(out_dir, 'block_size_comparison.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")

    with open(os.path.join(out_dir, 'aggregated.json'), 'w') as f:
        json.dump(aggregated, f, indent=4)
        
    print(f"Results saved to {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    # n_seeds=5 for reasonably stable curves without excessive compute
    config = Phase1Config(n_seeds=5, M=1000)
    run_phase_1_2(config)
