import os, sys, time, json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from data     import DATASETS, GARCH_COLS
from backtest import evaluate_var
from plots    import (plot_var_fan, plot_violation_distance, save_result_table, plot_violation_clustering)

# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR = os.path.join(_HERE, 'figures');  os.makedirs(FIG_DIR, exist_ok=True)
TBL_DIR = os.path.join(_HERE, 'tables');   os.makedirs(TBL_DIR, exist_ok=True)
RES_DIR = os.path.join(_HERE, 'results');  os.makedirs(RES_DIR, exist_ok=True)

BLOCK_SIZES  = [1, 5, 10, 15, 20, 30, 35, 40]
ALPHA_LEVELS = (0.05, 0.01)
DATASET_KEYS = ['c', 'f', 'ca', 's']

def evaluate_var_periods(actual: np.ndarray, var_forecasts: np.ndarray, dates_test: pd.DatetimeIndex, alpha: float) -> dict:
    """Evaluate VaR on all test data and partitioned (before/after 2022)."""
    mask_pre = dates_test < pd.to_datetime('2022-01-01')
    mask_post = dates_test >= pd.to_datetime('2022-01-01')

    return {
        'global': evaluate_var(actual, var_forecasts, alpha),
        'pre2022': evaluate_var(actual[mask_pre], var_forecasts[mask_pre], alpha),
        'post2022': evaluate_var(actual[mask_post], var_forecasts[mask_post], alpha),
    }

def _save_comprehensive_metrics(results_bb, garch_metrics, key, label):
    """Save periodized (global, pre-2022, post-2022) metrics CSV — same schema as main.py."""
    rows = []
    periods = ['global', 'pre2022', 'post2022']
    loaded_blocks = [b for b in BLOCK_SIZES if b in results_bb.get(0.05, {})]

    for alpha in ALPHA_LEVELS:
        for a_T in loaded_blocks:
            if a_T not in results_bb[alpha]: continue
            row = {'dataset': label, 'model': 'BB-R2', 'block': a_T, 'alpha': alpha}
            for p in periods:
                m = results_bb[alpha][a_T][p]
                row[f'{p}_viol_pct']  = round(m['viol_rate'] * 100, 4)
                row[f'{p}_kupiec_lr'] = round(m['kupiec_lr'],        6)
                row[f'{p}_kupiec_p']  = round(m['kupiec_p'],         6)
                row[f'{p}_christ_p']  = round(m['christ_p'],         6)
                row[f'{p}_mae']       = round(m['mae'],               8)
                row[f'{p}_es']        = round(m['es'],                8)
            rows.append(row)
        for col, m_dict in garch_metrics[alpha].items():
            row = {'dataset': label, 'model': col, 'block': None, 'alpha': alpha}
            for p in periods:
                m = m_dict[p]
                row[f'{p}_viol_pct']  = round(m['viol_rate'] * 100, 4)
                row[f'{p}_kupiec_lr'] = round(m['kupiec_lr'],        6)
                row[f'{p}_kupiec_p']  = round(m['kupiec_p'],         6)
                row[f'{p}_christ_p']  = round(m['christ_p'],         6)
                row[f'{p}_mae']       = round(m['mae'],               8)
                row[f'{p}_es']        = round(m['es'],                8)
            rows.append(row)

    df = pd.DataFrame(rows)
    id_cols = ['dataset', 'model', 'block', 'alpha']
    metric_cols = [f'{p}_{m}'
                   for p in periods
                   for m in ['viol_pct', 'kupiec_lr', 'kupiec_p', 'christ_p', 'mae', 'es']]
    df = df[id_cols + metric_cols]
    path = os.path.join(RES_DIR, f'metrics_{key}.csv')
    df.to_csv(path, index=False)
    print(f"  [+] Saved Metrics CSV -> {path}")

def _save_metrics_json(results_bb, garch_metrics, key, label):
    """Save a structured JSON containing all period metrics for fast lookup."""
    periods = ['global', 'pre2022', 'post2022']
    out = {'dataset': label, 'key': key, 'models': {}}

    for alpha in ALPHA_LEVELS:
        alpha_str = str(alpha)
        out['models'].setdefault(alpha_str, {})

        # BlockBoost rows
        for a_T in BLOCK_SIZES:
            if a_T not in results_bb[alpha]: continue
            model_key = f'BB-R2_block{a_T}'
            out['models'][alpha_str][model_key] = {}
            for p in periods:
                m = results_bb[alpha][a_T][p]
                out['models'][alpha_str][model_key][p] = {
                    'viol_rate_pct': round(m['viol_rate'] * 100, 4),
                    'kupiec_lr':     round(m['kupiec_lr'], 6),
                    'kupiec_p':      round(m['kupiec_p'], 6),
                    'christ_p':      round(m['christ_p'], 6),
                    'mae':           round(m['mae'], 8),
                    'es':            round(m['es'], 8),
                }

        # GARCH rows
        for col, m_dict in garch_metrics[alpha].items():
            out['models'][alpha_str][col] = {}
            for p in periods:
                m = m_dict[p]
                out['models'][alpha_str][col][p] = {
                    'viol_rate_pct': round(m['viol_rate'] * 100, 4),
                    'kupiec_lr':     round(m['kupiec_lr'], 6),
                    'kupiec_p':      round(m['kupiec_p'], 6),
                    'christ_p':      round(m['christ_p'], 6),
                    'mae':           round(m['mae'], 8),
                    'es':            round(m['es'], 8),
                }

    path = os.path.join(RES_DIR, f'metrics_{key}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  [+] Saved Metrics JSON -> {path}")

def run_fast_eval(key):
    info  = DATASETS[key]
    label = info['label'].replace('BRVM ', '')
    print(f"\n[+] Fast Eval Dataset: {label} (key={key})")
    
    results_bb = {a: {} for a in ALPHA_LEVELS}
    garch_metrics = {a: {} for a in ALPHA_LEVELS}
    var_raw = {a: {} for a in ALPHA_LEVELS}
    garch_bench = {a: {} for a in ALPHA_LEVELS}
    
    dates_test = None
    r_test_true = None
        
    for alpha in ALPHA_LEVELS:
        csv_path = os.path.join(RES_DIR, f'forecasts_{key}_{alpha}.csv')
        if not os.path.exists(csv_path):
            print(f"  [!] Missing {csv_path}, make sure you ran main.py first.")
            return None, None
            
        df = pd.read_csv(csv_path, parse_dates=['date'])
        dates_test = pd.DatetimeIndex(df['date'])
        r_test_true = df['actual_return'].values
        
        # Parse BB variants
        for a_T in BLOCK_SIZES:
            col_name = f"BB_{a_T}"
            if col_name in df.columns:
                preds = df[col_name].values
                var_raw[alpha][a_T] = preds
                results_bb[alpha][a_T] = evaluate_var_periods(r_test_true, preds, dates_test, alpha)
                
        # Parse GARCH variants
        for col in GARCH_COLS:
            if col in df.columns:
                preds = df[col].values
                garch_bench[alpha][col] = preds
                garch_metrics[alpha][col] = evaluate_var_periods(r_test_true, preds, dates_test, alpha)
                
    loaded_blocks = [b for b in BLOCK_SIZES if b in results_bb.get(0.05, {})]
    if not loaded_blocks:
        print(f"  [!] No BB blocks loaded for {key}. Skipping.")
        return None, None

    best_block = min(loaded_blocks, key=lambda a: abs(results_bb[0.05][a]['post2022']['viol_rate'] - 0.05))
    print(f"  [best block post-2022] a_T={best_block}")
    
    # Generate Plots
    plot_var_fan(dates_test, r_test_true, var_raw, garch_bench, FIG_DIR, best_block, key, ALPHA_LEVELS)
    plot_violation_distance(results_bb, garch_metrics, loaded_blocks, ALPHA_LEVELS, FIG_DIR, key)
    plot_violation_clustering(results_bb, r_test_true, dates_test, var_raw, ALPHA_LEVELS, loaded_blocks, FIG_DIR, key)
    
    # Save standard global tables and our new structured comprehensive table
    for alpha in ALPHA_LEVELS:
        save_result_table(results_bb, loaded_blocks, alpha, garch_metrics, TBL_DIR, key)
        
    _save_comprehensive_metrics(results_bb, garch_metrics, key, label)
    _save_metrics_json(results_bb, garch_metrics, key, label)
        
    return results_bb, garch_metrics

if __name__ == '__main__':
    t0 = time.time()
    for key in DATASET_KEYS:
        run_fast_eval(key)
    print(f"\n[+] Fast Evaluation complete in {time.time()-t0:.2f} seconds.")
