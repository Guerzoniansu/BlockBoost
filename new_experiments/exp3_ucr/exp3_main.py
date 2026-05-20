"""
exp3_main.py  –  UCR Time Series Classification Experiment.
Pivots to UCR 'Yoga' dataset to compare BlockBoost vs AdaBoost.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent and local dirs to path
HERE = os.path.dirname(__file__)
sys.path.append(HERE)
sys.path.append(os.path.dirname(HERE))
sys.path.append(os.path.join(os.path.dirname(HERE), 'exp2_financial'))

from data_ucr   import load_ucr_dataset, summarize_dataset
from evaluation import compute_metrics, run_boosting_curves, plot_learning_curves, save_table

# ── setup ─────────────────────────────────────────────────────────────────────
EXP_DIR = os.path.join(HERE, 'results')
FIG_DIR = os.path.join(HERE, 'figures')
TAB_DIR = os.path.join(HERE, 'tables')

for d in [EXP_DIR, FIG_DIR, TAB_DIR]:
    os.makedirs(d, exist_ok=True)

DATASET_NAME = "ECG200"
M_ROUNDS = 1500
BLOCK_SIZE = 5

def main():
    print('\n' + '='*65)
    print(f'EXP 3 – UCR TSC: {DATASET_NAME}')
    print('='*65)

    # 1. Load Data
    X_train, y_train, X_test, y_test = load_ucr_dataset(DATASET_NAME)
    summarize_dataset(DATASET_NAME, X_train, y_train, X_test, y_test)

    # 2. Run Boosting Models
    print(f'\nRunning BlockBoost & AdaBoost ({M_ROUNDS} rounds, block_size={BLOCK_SIZE}) ...')
    start = time.time()
    
    # We wrap it in a dict to match evaluation.py's expected structure for plotting
    # Evaluation.py expects { ticker: results_from_run_boosting_curves }
    results_all = {
        DATASET_NAME: run_boosting_curves(
            X_train, y_train, X_test, y_test,
            m_rounds=M_ROUNDS, block_size=BLOCK_SIZE
        )
    }
    
    elapsed = time.time() - start
    print(f'  Done in {elapsed:.2f}s.')

    # 3. Plot Learning Curves
    print('\nGenerating learning curves ...')
    plot_learning_curves(results_all, FIG_DIR)
    
    # 4. Generate Performance Table
    print('\nCompiling results table ...')
    res = results_all[DATASET_NAME]
    
    header = ['Dataset', 'Model', 'Accuracy', 'F1', 'MCC']
    rows = []
    
    # BlockBoost
    m_bb = compute_metrics(y_test, res['bb_preds'])
    rows.append([DATASET_NAME, 'BlockBoost', 
                 f"{m_bb['accuracy']:.4f}", f"{m_bb['f1']:.4f}", f"{m_bb['mcc']:.4f}"])
    
    # AdaBoost
    m_ada = compute_metrics(y_test, res['ada_preds'])
    rows.append([DATASET_NAME, 'AdaBoost', 
                 f"{m_ada['accuracy']:.4f}", f"{m_ada['f1']:.4f}", f"{m_ada['mcc']:.4f}"])
    
    print(f"\nResults for {DATASET_NAME}:")
    print(pd.DataFrame(rows, columns=header).to_string(index=False))
    
    save_table(rows, header, f'Model Comparison on UCR {DATASET_NAME}',
               os.path.join(TAB_DIR, 'ucr_comparison.pdf'))
    
    print(f'\nAll results saved to {HERE}')

if __name__ == "__main__":
    main()
