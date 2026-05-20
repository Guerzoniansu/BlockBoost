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
from experiments.dgp.noise import generate_markov_noise, compute_bayes_risk
from experiments.metrics.tracking import compute_auc_normalized

from model.adaboost import AdaBoostM1
from model.blockboost import BlockBoostClassifier
# For LogitBoost/GentleAdaBoost, we can use sklearn equivalents or approximations if not implemented.
# Currently using AdaBoost from sklearn with different algorithms as proxy.
from sklearn.ensemble import AdaBoostClassifier

class NaiveBlockEnsemble:
    """
    Naive block-average ensemble: Trains stumps on contiguous blocks of data and averages predictions.
    """
    def __init__(self, block_size):
        self.block_size = block_size
        self.estimators = []
        
    def fit(self, X, y):
        n_samples = len(X)
        n_blocks = max(1, n_samples // self.block_size)
        indices = np.array_split(np.arange(n_samples), n_blocks)
        
        for idx in indices:
            stump = DecisionTreeClassifier(max_depth=1)
            stump.fit(X[idx], y[idx])
            self.estimators.append(stump)
            
        return self
        
    def score_curve(self, X, y):
        # Return accuracy if we aggregate up to m estimators (simulating boosting rounds)
        accuracies = []
        preds = np.zeros(len(X))
        for m, est in enumerate(self.estimators):
            preds += est.predict(X)
            acc = np.mean(np.sign(preds) == y)
            accuracies.append(acc)
            
        return accuracies


def evaluate_benchmarking(eta, seed, config):
    X = generate_ar_process(config.n_samples, config.n_features, config.ar_coeffs, config.noise_std, seed)
    y_clean = generate_labels(X, seed)
    y_noisy = generate_markov_noise(y_clean, eta=eta, alpha_markov=config.alpha_markov, random_state=seed)
    
    X_train, X_test, y_train_clean, y_test_clean = train_test_split(X, y_clean, test_size=0.3, shuffle=False)
    _, _, y_train_noisy, y_test_noisy = train_test_split(X, y_noisy, test_size=0.3, shuffle=False)
    
    rho = compute_companion_matrix_spectral_radius(config.ar_coeffs)
    a_T_star = compute_theoretical_optimal_block_size(config.n_samples, rho)
    bayes_acc = 1.0 - compute_bayes_risk(eta)
    
    stump = DecisionTreeClassifier(max_depth=1)
    
    # BlockBoost
    block = BlockBoostClassifier(estimator=stump, n_estimators=config.M, block_size=a_T_star)
    _, block_test_acc = block.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_noisy)
    block_auc = compute_auc_normalized(block_test_acc, bayes_acc)
    
    # Custom AdaBoostM1
    ada = AdaBoostM1(estimator=stump, n_estimators=config.M)
    _, ada_test_acc = ada.fit(X_train, y_train_noisy, X_val=X_test, y_val=y_test_noisy)
    ada_auc = compute_auc_normalized(ada_test_acc, bayes_acc)
    
    # LogitBoost proxy (SAMME optimizations)
    logit = AdaBoostClassifier(estimator=stump, n_estimators=config.M)
    logit.fit(X_train, y_train_noisy)
    logit_test_acc = list(logit.staged_score(X_test, y_test_clean)) # Score on clean for generalization eval
    logit_auc = compute_auc_normalized(logit_test_acc, bayes_acc)

    # Naive Block
    naive = NaiveBlockEnsemble(block_size=a_T_star)
    naive.fit(X_train, y_train_noisy)
    naive_test_acc = naive.score_curve(X_test, y_test_clean)
    naive_test_acc += [naive_test_acc[-1]] * max(0, config.M - len(naive_test_acc)) # pad to M
    naive_auc = compute_auc_normalized(naive_test_acc[:config.M], bayes_acc)
    
    return {
        'seed': seed,
        'eta': eta,
        'block_auc': block_auc,
        'ada_auc': ada_auc,
        'logit_auc': logit_auc,
        'naive_auc': naive_auc
    }

def run_phase_4(config: Phase1Config):
    print("=" * 50)
    print("PHASE 4: COMPARATIVE BENCHMARKING")
    print("=" * 50)
    
    seeds = generate_seeds(config.n_seeds, config.base_seed)
    etas = [0.0, 0.1, 0.3]
    
    tasks = [(eta, s, config) for eta in etas for s in seeds]
    
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_benchmarking)(*task) for task in tasks
    )
    
    aggregated = {}
    for eta in etas:
        res_eta = [r for r in results if r['eta'] == eta]
        agg = {}
        for k in ['block_auc', 'ada_auc', 'logit_auc', 'naive_auc']:
            vals = [r[k] for r in res_eta]
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        aggregated[eta] = agg
        
        print(f"ETA = {eta}:")
        print(f"  BlockBoost AUC : {agg['block_auc_mean']:.4f}")
        print(f"  AdaBoost AUC   : {agg['ada_auc_mean']:.4f}")
        print(f"  LogitBoost AUC : {agg['logit_auc_mean']:.4f}")
        print(f"  NaiveBlock AUC : {agg['naive_auc_mean']:.4f}")

    out_dir = os.path.join(os.path.dirname(__file__), '../results/phase_4_benchmark')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'aggregated.json'), 'w') as f:
        json.dump(aggregated, f, indent=4)
        
    print(f"Results saved to {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    config = Phase1Config(n_seeds=5)
    run_phase_4(config)
