"""
main.py  –  BlockBoost VaR Study — Full Multi-Dataset Pipeline
==============================================================

Run from the Experiments/ root:
    .\\.venv\\Scripts\\python new_experiments\\var_study\\main.py

Pipeline (per dataset)
----------------------
1. Load data_t_*.csv  → feature matrix X, returns r  (data.py)
2. For each block_size in BLOCK_SIZES:
     - Train BlockBoostR2 + step_size=1 rolling forecast for both alpha levels
     - Evaluate violation rate, Kupiec, MAE, ES  (backtest.py)
3. Load GARCH benchmarks from data_*_01.csv / data_*_05.csv
     - Evaluate the same metrics for every GARCH model
4. Generate:
     - Fig 1: VaR fan chart  (best BB block + blue GARCH curve)
     - Fig 2: Violation-distance chart  (replaces heatmap)
     - Tables: per alpha, per dataset

Datasets
--------
  c  → BRVM Composite
  f  → BRVM Finance
  ca → CAC 40
  s  → S&P 500
"""

import os, sys, time
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from data     import load_dataset, DATASETS, GARCH_COLS
from models   import rolling_forecast
from backtest import evaluate_var
from plots    import (plot_var_fan, plot_violation_distance, save_result_table)

# ── Output directories ────────────────────────────────────────────────────────
FIG_DIR = os.path.join(_HERE, 'figures');  os.makedirs(FIG_DIR, exist_ok=True)
TBL_DIR = os.path.join(_HERE, 'tables');   os.makedirs(TBL_DIR, exist_ok=True)
RES_DIR = os.path.join(_HERE, 'results');  os.makedirs(RES_DIR, exist_ok=True)

def evaluate_var_periods(actual: np.ndarray, var_forecasts: np.ndarray, dates_test: pd.DatetimeIndex, alpha: float) -> dict:
    """Evaluate VaR on all test data and partitioned (before/after 2022)."""
    mask_pre = dates_test < pd.to_datetime('2022-01-01')
    mask_post = dates_test >= pd.to_datetime('2022-01-01')

    return {
        'global': evaluate_var(actual, var_forecasts, alpha),
        'pre2022': evaluate_var(actual[mask_pre], var_forecasts[mask_pre], alpha),
        'post2022': evaluate_var(actual[mask_post], var_forecasts[mask_post], alpha),
    }

# ── Experiment parameters ─────────────────────────────────────────────────────
BLOCK_SIZES  = [1, 5, 10, 15, 20, 30, 35, 40] #BLOCK_SIZES  = [1, 5, 10, 15, 20]
ALPHA_LEVELS = (0.05, 0.01)
N_ESTIMATORS = 10
DATASET_KEYS = ['c', 'f', 'ca', 's']   # process in this order


# =============================================================================
# Utility: save per-dataset violation-rate summary as CSV
# =============================================================================

def _save_comprehensive_metrics(results_bb: dict, garch_metrics: dict,
                                block_sizes: list, alpha_levels: tuple,
                                dataset_key: str, label: str):
    """
    Write a tidy CSV with violation rates, Kupiec p, Christoffersen p,
    MAE and ES for Global, Pre-2022 and Post-2022 periods,
    for every model / block / alpha combination.
    """
    rows = []
    periods = ['global', 'pre2022', 'post2022']

    for alpha in alpha_levels:
        # ── BB-R2 rows
        for a_T in block_sizes:
            if a_T not in results_bb[alpha]:
                continue
            m_periods = results_bb[alpha][a_T]
            row = {'dataset': label, 'model': 'BB-R2', 'block': a_T, 'alpha': alpha}
            for p in periods:
                m = m_periods[p]
                row[f'{p}_viol_pct']  = round(m['viol_rate'] * 100, 4)
                row[f'{p}_kupiec_lr'] = round(m['kupiec_lr'],        6)
                row[f'{p}_kupiec_p']  = round(m['kupiec_p'],         6)
                row[f'{p}_christ_p']  = round(m['christ_p'],         6)
                row[f'{p}_mae']       = round(m['mae'],               8)
                row[f'{p}_es']        = round(m['es'],                8)
            rows.append(row)

        # ── GARCH rows
        for col, m_periods in garch_metrics[alpha].items():
            row = {'dataset': label, 'model': col, 'block': None, 'alpha': alpha}
            for p in periods:
                m = m_periods[p]
                row[f'{p}_viol_pct']  = round(m['viol_rate'] * 100, 4)
                row[f'{p}_kupiec_lr'] = round(m['kupiec_lr'],        6)
                row[f'{p}_kupiec_p']  = round(m['kupiec_p'],         6)
                row[f'{p}_christ_p']  = round(m['christ_p'],         6)
                row[f'{p}_mae']       = round(m['mae'],               8)
                row[f'{p}_es']        = round(m['es'],                8)
            rows.append(row)

    df = pd.DataFrame(rows)
    # Column order: identifiers first, then grouped by period
    id_cols = ['dataset', 'model', 'block', 'alpha']
    metric_cols = [f'{p}_{m}'
                   for p in periods
                   for m in ['viol_pct', 'kupiec_lr', 'kupiec_p', 'christ_p', 'mae', 'es']]
    df = df[id_cols + metric_cols]
    path = os.path.join(RES_DIR, f'metrics_{dataset_key}.csv')
    df.to_csv(path, index=False)
    print(f'  [+] Metrics CSV -> {path}')


# =============================================================================
# Per-dataset processing
# =============================================================================

def run_dataset(key: str):
    info  = DATASETS[key]
    label = info['label']

    print('\n' + '='*65)
    print(f'  Dataset: {label}  (key={key})')
    print('='*65)
    t_ds = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    (X, r, dates,
     garch_01, garch_05,
     r_test_garch, dates_test, window_size, meta) = load_dataset(key)

    r_test_true = r[window_size:]      # test returns aligned with BB forecasts
    n_test      = len(r_test_true)

    # Quick sanity check
    print(f'  [ok] n_test (BB window) = {n_test}')
    print(f'  [ok] n_test (GARCH CSV) = {len(r_test_garch)}')

    # ── 2. BlockBoost rolling forecasts ──────────────────────────────────────
    print(f'\n[+] Rolling forecasts — {len(BLOCK_SIZES)} block sizes, '
          f'n_estimators={N_ESTIMATORS} …')

    results_bb = {a: {} for a in ALPHA_LEVELS}
    var_raw    = {a: {} for a in ALPHA_LEVELS}


    for a_T in BLOCK_SIZES:
        t0 = time.time()
        print(f'  [BB-R2  a_T={a_T:2d}] fitting …', end=' ', flush=True)

        forecasts = rolling_forecast(
            X            = X,
            r            = r,
            block_size   = a_T,
            n_estimators = N_ESTIMATORS,
            alpha_levels = ALPHA_LEVELS,
            window_size  = window_size,
        )

        for alpha in ALPHA_LEVELS:
            var_raw[alpha][a_T]    = forecasts[alpha]
            results_bb[alpha][a_T] = evaluate_var_periods(r_test_true, forecasts[alpha], dates[window_size:], alpha)

        dt  = time.time() - t0
        vr5 = results_bb[0.05][a_T]['global']['viol_rate']
        vr1 = results_bb[0.01][a_T]['global']['viol_rate']
        kp5 = results_bb[0.05][a_T]['global']['kupiec_p']
        kp1 = results_bb[0.01][a_T]['global']['kupiec_p']
        print(f'done in {dt:.1f}s  |  '
              f'viol5%={vr5*100:.2f}% (p={kp5:.3f})  '
              f'viol1%={vr1*100:.2f}% (p={kp1:.3f})')


    # ── 3. GARCH benchmark evaluation ─────────────────────────────────────────
    print('\n[+] Evaluating GARCH benchmarks …')

    garch_metrics = {alpha: {} for alpha in ALPHA_LEVELS}

    # α = 0.01
    for col in GARCH_COLS:
        if col not in garch_01:
            continue
        m = evaluate_var_periods(r_test_garch, garch_01[col], dates_test, 0.01)
        garch_metrics[0.01][col] = m
        print(f'  [{col}  α=0.01]  viol={m["global"]["viol_rate"]*100:.2f}%  '
              f'Kup p={m["global"]["kupiec_p"]:.3f}')

    # α = 0.05
    for col in GARCH_COLS:
        if col not in garch_05:
            continue
        m = evaluate_var_periods(r_test_garch, garch_05[col], dates_test, 0.05)
        garch_metrics[0.05][col] = m
        print(f'  [{col}  α=0.05]  viol={m["global"]["viol_rate"]*100:.2f}%  '
              f'Kup p={m["global"]["kupiec_p"]:.3f}')

    # ── 4. Best block (closest violation rate to 5% nominal) ─────────────────
    best_block = min(BLOCK_SIZES,
                     key=lambda a: abs(results_bb[0.05][a]['global']['viol_rate'] - 0.05))
    print(f'\n  [best block at α=5%] a_T={best_block}  '
          f'viol={results_bb[0.05][best_block]["global"]["viol_rate"]*100:.2f}%')

    # ── 5. Build aligned GARCH series for fan chart ───────────────────────────
    # The fan chart needs GARCH VaR series aligned to test dates (positive values).
    # We use garch_05 arrays for the 5% panel and garch_01 for 1% — but
    # for simplicity in the fan chart we pass garch_05 as the benchmark dict
    # (the _best_garch_for_alpha helper picks the right one per alpha internally).
    # We combine both dicts: if same col exists in both we need to separate them.
    # Solution: build a unified dict keyed by (col, alpha) but the helper accesses
    # the right one. Pass a merged approach handled inside plot_var_fan.

    # We pass two separate dicts, one per alpha, as a nested structure:
    garch_bench_per_alpha = {
        0.05: {col: garch_05[col] for col in garch_05},
        0.01: {col: garch_01[col] for col in garch_01},
    }

    # ── 5.5 Save Raw VaR Forecasts ────────────────────────────────────────────
    print('\n[+] Saving raw VaR forecasts to CSV …')
    for alpha in ALPHA_LEVELS:
        df_out = pd.DataFrame({
            'date': dates[window_size:],
            'actual_return': r_test_true
        })
        
        # Add GARCH series
        for col, preds in garch_bench_per_alpha[alpha].items():
            df_out[col] = preds
            
        # Add BB series
        for a_T, preds in var_raw[alpha].items():
            df_out[f'BB_{a_T}'] = preds
            
        csv_path = os.path.join(RES_DIR, f'forecasts_{key}_{alpha}.csv')
        df_out.to_csv(csv_path, index=False)
        print(f"  [+] Saved forecasts -> {csv_path}")

    # ── 6. Figures ────────────────────────────────────────────────────────────
    print('\n[+] Generating figures …')

    plot_var_fan(
        dates       = dates[window_size:],
        r_test      = r_test_true,
        var_bb_r2   = var_raw,
        garch_bench = garch_bench_per_alpha,
        out_dir     = FIG_DIR,
        best_block  = best_block,
        dataset_key = key,
        alpha_levels= ALPHA_LEVELS,
    )

    plot_violation_distance(
        results_bb   = results_bb,
        garch_metrics= garch_metrics,
        block_sizes  = BLOCK_SIZES,
        alpha_levels = ALPHA_LEVELS,
        out_dir      = FIG_DIR,
        dataset_key  = key,
    )

    # ── 7. Tables ─────────────────────────────────────────────────────────────
    print('\n[+] Generating tables …')
    for alpha in ALPHA_LEVELS:
        save_result_table(
            results_bb   = results_bb,
            block_sizes  = BLOCK_SIZES,
            alpha        = alpha,
            garch_metrics= garch_metrics,
            out_dir      = TBL_DIR,
            dataset_key  = key,
        )

    # ── 8. Save metrics CSV ───────────────────────────────────────────────────
    _save_comprehensive_metrics(results_bb, garch_metrics, BLOCK_SIZES, ALPHA_LEVELS, key, label)

    elapsed = (time.time() - t_ds) / 60
    print(f'\n  [done] {label} finished in {elapsed:.1f} min.')

    return results_bb, garch_metrics


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    t_total = time.time()

    print('\n' + '='*65)
    print('  BlockBoost VaR Study — Full Multi-Dataset Run')
    print('='*65)
    print(f'  Datasets : {DATASET_KEYS}')
    print(f'  Blocks   : {BLOCK_SIZES}')
    print(f'  Alpha    : {ALPHA_LEVELS}')
    print(f'  Window   : Dynamic (aligned to GARCH test dates)  |  Step : 1')
    print(f'  Est.     : {N_ESTIMATORS} estimators')

    all_results = {}
    for key in DATASET_KEYS:
        results_bb, garch_metrics = run_dataset(key)
        all_results[key] = (results_bb, garch_metrics)

    total_min = (time.time() - t_total) / 60
    print('\n' + '='*65)
    print(f'  ALL DATASETS DONE in {total_min:.1f} min.')
    print(f'  Figures → {FIG_DIR}')
    print(f'  Tables  → {TBL_DIR}')
    print(f'  Results → {RES_DIR}')
    print('='*65)
