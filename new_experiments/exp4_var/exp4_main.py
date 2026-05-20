"""
exp4_main.py  –  Orchestrates Experiment 4: VaR Exceedance Prediction
"""

import os, sys, json, time
import numpy as np
from scipy import stats

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)   # so sub-modules can import each other

from data_var       import load_returns, build_var_dataset, ASSETS
from ar_model       import select_ar_order
from garch_model    import generate_garch_features
from evaluation_var import compute_metrics, run_boosting_curves, plot_learning_curves, save_table, dm_test

# ── output dirs ───────────────────────────────────────────────────────────────
RES_DIR = os.path.join(HERE, 'results')
FIG_DIR = os.path.join(HERE, 'figures')
TBL_DIR = os.path.join(HERE, 'tables')
for d in (RES_DIR, FIG_DIR, TBL_DIR):
    os.makedirs(d, exist_ok=True)


def step1_data_and_features():
    print('\n' + '='*65)
    print('STEP 1 – Data, AR Order, and GARCH features')
    print('='*65)
    returns_dict = load_returns(cache_dir=os.path.join(RES_DIR, 'cache'))
    
    datasets = {}
    ar_info = {}
    
    for ticker, rets in returns_dict.items():
        print(f'\n  [{ticker}] Processing ...')
        r_vals = rets.values.astype(float)
        
        # AR(p) selection on first 70%
        train_len = int(len(r_vals) * 0.70)
        ic = select_ar_order(r_vals[:train_len])
        p = ic['best_p_bic']
        print(f'    AR order (BIC): p={p}')
        ar_info[ticker] = p
        
        # Out-of-sample GARCH features
        print(f'    Generating rolling GARCH(1,1) features (warmup=252, stride=126) ...')
        t0 = time.time()
        sigmas, var_limits, hits, mus = generate_garch_features(r_vals, p, alpha=0.05)
        print(f'    GARCH features done in {time.time()-t0:.1f}s.')
        
        # Build dataset
        ds = build_var_dataset(rets, sigmas, var_limits, hits, mus, p_lags=21, q_lags=21, d_lags=21)
        datasets[ticker] = ds
        
        y_train = ds['y_train']
        pos_frac = (y_train == 1).mean()
        print(f'    Train: n={len(y_train)}, exceedance rate={pos_frac:.3f}')
        
    return datasets, ar_info


def step2_boosting(datasets):
    print('\n' + '='*65)
    print('STEP 2 – BlockBoost & AdaBoost learning curves')
    print('='*65)
    
    results = {}
    for ticker, d in datasets.items():
        print(f'  [{ticker}] running boosting …')
        results[ticker] = run_boosting_curves(
            d['X_train'], d['y_train'], d['X_test'], d['y_test'],
            m_rounds=1500, block_size=10
        )
    
    plot_learning_curves(results, FIG_DIR)
    return results


def step3_tables(datasets, boosting):
    print('\n' + '='*65)
    print('STEP 3 – Compiling metrics and generating PDF tables')
    print('='*65)

    header = ['Asset', 'Model', 'Christ. LR_cc', 'p-value', 'pi_11', 'Precision', 'DM Test Stat', 'DM p-val']
    rows   = []
    alpha = 0.05
    
    for ticker in ASSETS:
        d = datasets[ticker]
        y_test = d['y_test']
        y_test_01 = np.where(y_test == 1, 1, 0)
        
        # GARCH Baseline: constant 5% probability forecast
        garch_prob = np.full(len(y_test), alpha)
        garch_brier_loss = (y_test_01 - garch_prob)**2
        
        # BlockBoost Loss
        bb_preds = boosting[ticker]['bb_preds']
        bb_preds_01 = np.where(np.array(bb_preds) == 1, 1, 0)
        bb_brier_loss = (y_test_01 - bb_preds_01)**2
        
        # AdaBoost Loss
        ada_preds = boosting[ticker]['ada_preds']
        ada_preds_01 = np.where(np.array(ada_preds) == 1, 1, 0)
        ada_brier_loss = (y_test_01 - ada_preds_01)**2
        
        # Metrics 
        m_garch = compute_metrics(y_test, np.full(len(y_test), -1), y_prob=garch_prob, baseline_hits=y_test_01)
        m_bb    = compute_metrics(y_test, bb_preds, baseline_hits=y_test_01)
        m_ada   = compute_metrics(y_test, ada_preds, baseline_hits=y_test_01)
        
        # DM Test vs GARCH (using Brier score difference, negative stat means Boosting is better)
        dm_bb_stat, dm_bb_p = dm_test(bb_brier_loss, garch_brier_loss, h=1)
        dm_ada_stat, dm_ada_p = dm_test(ada_brier_loss, garch_brier_loss, h=1)
        
        # GARCH row
        rows.append([ticker, 'GARCH (Baseline)',
                     f'{m_garch["lr_cc"]:.2f}',
                     f'{m_garch["p_cc"]:.4f}',
                     f'{m_garch["pi_11"]:.4f}',
                     '—', '—', '—'])
                     
        # BlockBoost row
        rows.append(['', 'BlockBoost',
                     f'{m_bb["lr_cc"]:.2f}',
                     f'{m_bb["p_cc"]:.4f}',
                     f'{m_bb["pi_11"]:.4f}',
                     f'{m_bb["precision"]:.4f}',
                     f'{dm_bb_stat:.2f}',
                     f'{dm_bb_p:.4f}'])
                     
        # AdaBoost row
        rows.append(['', 'AdaBoost',
                     f'{m_ada["lr_cc"]:.2f}',
                     f'{m_ada["p_cc"]:.4f}',
                     f'{m_ada["pi_11"]:.4f}',
                     f'{m_ada["precision"]:.4f}',
                     f'{dm_ada_stat:.2f}',
                     f'{dm_ada_p:.4f}'])

    out_file = os.path.join(TBL_DIR, 'var_exceedance_results.pdf')
    save_table(rows, header, 'Table 1 — VaR Exceedance Prediction (Test Set)', out_file)
    print(f'  [+] Table saved to {out_file}')

if __name__ == '__main__':
    print('=' * 65)
    print('Experiment 4 — Financial Time Series VaR Exceedance Prediction')
    print(f'Assets : {ASSETS}')
    print('=' * 65)
    t_start = time.time()

    sets, ar_info = step1_data_and_features()
    boosting_results = step2_boosting(sets)
    step3_tables(sets, boosting_results)

    print(f'\n[+] All done in {(time.time()-t_start)/60:.1f} min.')
    print(f'    Figures → {FIG_DIR}')
    print(f'    Tables  → {TBL_DIR}')
    print(f'    Results → {RES_DIR}')
