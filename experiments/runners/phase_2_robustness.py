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

from model.adaboost import AdaBoostM1
from model.blockboost import BlockBoostClassifier

def generate_nonstationary_ar(n_samples: int, n_features: int, ar_coeffs: list[float], noise_std: float, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    p = len(ar_coeffs)
    X = np.zeros((n_samples + p, n_features))
    e = rng.normal(0, noise_std, size=(n_samples + p, n_features))
    
    half_t = n_samples // 2
    # Second half coeffs: reverse them or alter them to create a structural break
    ar_coeffs_2 = ar_coeffs[::-1] 
    
    for t in range(p, n_samples + p):
        phi = ar_coeffs if (t - p) < half_t else ar_coeffs_2
        X[t] = sum(phi[i] * X[t - 1 - i] for i in range(p)) + e[t]
        
    return X[p:]

def evaluate_single_run_robustness(experiment_type: str, param_val, seed: int, config: Phase1Config):
    eta = 0.10
    ar_coeffs = config.ar_coeffs
    alpha_markov = config.alpha_markov
    is_nonstationary = False
    
    if experiment_type == 'ar_order':
        ar_coeffs = param_val
    elif experiment_type == 'markov_intensity':
        alpha_markov = param_val
    elif experiment_type == 'structural_break':
        is_nonstationary = True
        
    if is_nonstationary:
        X = generate_nonstationary_ar(config.n_samples, config.n_features, ar_coeffs, config.noise_std, seed)
    else:
        X = generate_ar_process(config.n_samples, config.n_features, ar_coeffs, config.noise_std, seed)
        
    y_clean = generate_labels(X, random_state=seed)
    y_noisy = generate_markov_noise(y_clean, eta=eta, alpha_markov=alpha_markov, random_state=seed)
    
    X_train, X_test, y_train_clean, y_test_clean = train_test_split(X, y_clean, test_size=0.3, shuffle=False)
    _, _, y_train_noisy, y_test_noisy = train_test_split(X, y_noisy, test_size=0.3, shuffle=False)
    
    rho = compute_companion_matrix_spectral_radius(ar_coeffs)
    a_T_star = compute_theoretical_optimal_block_size(config.n_samples, rho)

    stump = DecisionTreeClassifier(max_depth=1)
    
    ada = AdaBoostM1(estimator=stump, n_estimators=config.M)
    ada_train_acc, ada_test_acc = ada.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_noisy)
    ada_clean_acc = ada.score(X_test, y_test_clean)
    
    block = BlockBoostClassifier(estimator=stump, n_estimators=config.M, block_size=a_T_star)
    block_train_acc, block_test_acc = block.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_noisy)
    block_clean_acc = block.score(X_test, y_test_clean)
    
    return {
        'experiment': experiment_type,
        'param': str(param_val),
        'seed': seed,
        'ada_final_noisy': ada_test_acc[-1] if ada_test_acc else 0.0,
        'ada_final_clean': float(ada_clean_acc),
        'block_final_noisy': block_test_acc[-1] if block_test_acc else 0.0,
        'block_final_clean': float(block_clean_acc)
    }

def run_phase_2(config: Phase1Config):
    print("=" * 50)
    print("PHASE 2: ROBUSTNESS")
    print("=" * 50)
    
    seeds = generate_seeds(config.n_seeds, config.base_seed)
    
    # 2.1 Varying AR order
    ar_orders = {
        'AR(1)': [0.8],
        'AR(2)': [0.5, 0.3],
        'AR(5)': [0.4, 0.2, 0.1, 0.05, 0.05],
        'AR(10)': [0.2, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.02, 0.01]
    }
    
    # 2.2 Structural Break
    break_configs = ['Break_at_T/2']
    
    # 2.3 Markov noise intensity
    alpha_markovs = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    tasks = []
    for order_name, coeffs in ar_orders.items():
        for s in seeds:
            tasks.append(('ar_order', coeffs, s, config))
            
    for s in seeds:
        tasks.append(('structural_break', 'Break_at_T/2', s, config))
        
    for alpha in alpha_markovs:
        for s in seeds:
            tasks.append(('markov_intensity', alpha, s, config))
            
    print(f"Total runs: {len(tasks)}")
    
    start_time = datetime.now()
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_single_run_robustness)(*task) for task in tasks
    )
    print(f"Finished in {datetime.now() - start_time}.")
    
    aggregated = {}
    for r in results:
        exp = r['experiment']
        if exp not in aggregated:
            aggregated[exp] = {}
        param = r['param']
        if param not in aggregated[exp]:
            aggregated[exp][param] = []
        aggregated[exp][param].append(r)
        
    summary = {}
    for exp, params in aggregated.items():
        summary[exp] = {}
        for param, runs in params.items():
            agg = {}
            for metric in ['ada_final_clean', 'block_final_clean']:
                vals = [run[metric] for run in runs]
                agg[f"{metric}_mean"] = float(np.mean(vals))
                agg[f"{metric}_std"] = float(np.std(vals))
            summary[exp][param] = agg
            print(f"[{exp}] {param}: Ada Clean: {agg['ada_final_clean_mean']:.4f}, Block Clean: {agg['block_final_clean_mean']:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), '../results/phase_2_robustness')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'aggregated.json'), 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"Results saved to {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    config = Phase1Config(n_seeds=5)
    run_phase_2(config)
