"""
exp2_main.py  –  Orchestrates Experiment 2: Financial Time Series Comparison.

Runs in this order:
  0. Download / load data (SPY, AAPL, MSFT, JPM, XOM, GS)
  1. Estimate beta-mixing coefficients → plot + JSON
  2. AR(p): order selection, ARCH test, residual diagnostics → plots + tables
  3. GARCH rolling predictions (in parallel, stride=21)
  4. AR rolling predictions
  5. BlockBoost + AdaBoost learning curves (10 seeds each)
  6. Compile all metrics → comparison table (PDF)
  7. GARCH & AR fit-summary table (PDF)
  8. Beta-mixing summary table (PDF)

Run from the Experiments/ root:
    .\.venv\Scripts\python new_experiments\exp2_financial\exp2_main.py
"""

import os, sys, json, time
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── paths ─────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)   # so sub-modules can import each other

from data        import load_all, ASSETS
from mixing      import estimate_beta_mixing, plot_mixing, BLOCK_SIZES
from ar_model    import (select_ar_order, engle_arch_test,
                         rolling_ar_predictions, plot_ar_diagnostics)
from garch_model import rolling_garch_predictions
from evaluation  import (compute_metrics,
                         save_table)

# ── output dirs ───────────────────────────────────────────────────────────────
RES_DIR = os.path.join(HERE, 'results')
FIG_DIR = os.path.join(HERE, 'figures')
TBL_DIR = os.path.join(HERE, 'tables')
for d in (RES_DIR, FIG_DIR, TBL_DIR):
    os.makedirs(d, exist_ok=True)



# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Beta-mixing
# ─────────────────────────────────────────────────────────────────────────────
def step1_mixing(datasets):
    print('\n' + '='*65)
    print('STEP 1 – Beta-mixing estimation')
    print('='*65)
    all_betas = {}
    for ticker, d in datasets.items():
        print(f'  [{ticker}] …')
        ret = d['returns_series'].values.astype(float)
        all_betas[ticker] = estimate_beta_mixing(ret, block_sizes=BLOCK_SIZES)

    # Save raw estimates
    out_json = os.path.join(RES_DIR, 'beta_mixing.json')
    with open(out_json, 'w') as f:
        json.dump({t: {str(b): v for b, v in d.items()}
                   for t, d in all_betas.items()}, f, indent=2)

    # Plot
    fitted = plot_mixing(all_betas, FIG_DIR)
    print(f'  [+] Mixing estimates saved.')
    return all_betas, fitted


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – AR(p) validation
# ─────────────────────────────────────────────────────────────────────────────
def step2_ar_validation(datasets):
    print('\n' + '='*65)
    print('STEP 2 – AR(p) order selection & ARCH test')
    print('='*65)
    ar_info = {}
    for ticker, d in datasets.items():
        returns_train = d['r_train']
        print(f'  [{ticker}] AIC/BIC order selection …')
        ic  = select_ar_order(returns_train)
        p   = ic['best_p_bic']
        print(f'    best_p_bic={p}, best_p_aic={ic["best_p_aic"]}')
        arch_res = engle_arch_test(returns_train, p)
        print(f'    ARCH test: lm={arch_res["lm_stat"]:.3f}, '
              f'p={arch_res["p_value"]:.4f}, '
              f'arch_detected={arch_res["arch_detected"]}')
        # AR diagnostic plot disabled for bulk seed runs (generates per-seed files)
        # plot_ar_diagnostics(returns_train, p, ticker, FIG_DIR, ic)
        ar_info[ticker] = dict(best_p_bic=p, best_p_aic=ic['best_p_aic'],
                               **arch_res, ic=ic)
    return ar_info


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – Rolling AR & GARCH predictions
# ─────────────────────────────────────────────────────────────────────────────
def step3_rolling_predictions(datasets, ar_info):
    print('\n' + '='*65)
    print('STEP 3 – Rolling-origin AR + GARCH predictions')
    print('='*65)
    all_ar_preds    = {}
    all_garch_preds = {}

    for ticker, d in datasets.items():
        returns_full = d['returns_full']
        y_full       = d['y_full']
        train_end    = d['X_train'].shape[0]
        p            = ar_info[ticker]['best_p_bic']

        # AR(p)
        print(f'  [{ticker}] Rolling AR({p}) …')
        t0 = time.time()
        ar_preds = rolling_ar_predictions(
            returns_full, y_full, train_end, p, stride=63)
        ar_mse = mean_squared_error(ar_preds["y_test"], ar_preds["preds_test"])
        ar_mae = mean_absolute_error(ar_preds["y_test"], ar_preds["preds_test"])
        print(f'    Done in {time.time()-t0:.1f}s  test_mae={ar_mae:.6f}')
        print(f'    Sample Actual y : {[round(y, 6) for y in ar_preds["y_test"][:5]]}')
        print(f'    Sample AR preds : {[round(p, 6) for p in ar_preds["preds_test"][:5]]}')
        all_ar_preds[ticker] = ar_preds

        # AR+GARCH (parallel chunks)
        if ar_info[ticker]['arch_detected']:
            print(f'  [{ticker}] ARCH detected → Rolling AR({p})+GARCH(1,1) …')
        else:
            print(f'  [{ticker}] No ARCH → Running GARCH anyway for comparison …')
        t0 = time.time()
        garch_preds = rolling_garch_predictions(
            returns_full, y_full, train_end, p, stride=63, n_jobs=-1)
        garch_mse = mean_squared_error(garch_preds["y_test"], garch_preds["preds_test"])
        garch_mae = mean_absolute_error(garch_preds["y_test"], garch_preds["preds_test"])
        print(f'    Done in {time.time()-t0:.1f}s  test_mae={garch_mae:.6f}')
        print(f'    Sample GARCH preds : {[round(p, 6) for p in garch_preds["preds_test"][:5]]}')
        all_garch_preds[ticker] = garch_preds

    return all_ar_preds, all_garch_preds


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – BlockBoost + AdaBoost Regression (paced expanding window)
# ─────────────────────────────────────────────────────────────────────────────
def step4_boosting(datasets):
    print('\n' + '='*65)
    print('STEP 4 – BlockBoost & AdaBoost Regression Evaluation')
    print('='*65)
    
    from evaluation import BLOCK_SIZES, rolling_boosting_regression
    results = {}
    for ticker, d in datasets.items():
        t0 = time.time()
        print(f'  [{ticker}] training boosting models …')
        train_end = d['X_train'].shape[0]
        results[ticker] = rolling_boosting_regression(
            d['X_full'], d['y_full'], train_end,
            m_rounds=20, block_sizes=BLOCK_SIZES, stride=63, n_jobs=-1
        )
        print(f'    Done in {time.time()-t0:.1f}s')
    
    return results



# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Aggregation & Reporting (20 Seeds)
# ─────────────────────────────────────────────────────────────────────────────
def compile_seed_metrics(ticker, datasets, ar_info, all_ar_preds, all_garch_preds, all_boosting):
    d = datasets[ticker]
    y_test = d['y_test']
    
    ar_preds = all_ar_preds[ticker]['preds_test']
    garch_preds = all_garch_preds[ticker]['preds_test']
    boost = all_boosting[ticker]
    
    # Align y_test for AR which might be shorter depending on lag
    n_ar = len(ar_preds)
    y_aligned = y_test[-n_ar:]
    
    return {
        'y_test': y_aligned.tolist(),
        'preds': {
            'AR(p)': ar_preds,
            'AR+GARCH': garch_preds,
            'AdaBoost.R2': boost['ada_preds_test'],
            'BlockBoost.R2': boost['bb_r2_preds_test']  # dict a_T -> list
        },
        'learning_curves': {
            'AdaBoost.R2': boost['ada_test_mae'],
            'BlockBoost.R2': boost['bb_r2_test_mae'] # dict a_T -> list
        }
    }

def plot_collapse_diagnostics(all_seed_results, best_a_T, out_dir):
    """
    Two scatter plots aggregated over all seeds to check whether the best
    BlockBoost.R2 model has collapsed to a near-constant prediction.

    Panel A: actual  y  vs  pred_BB  (should track y=x if BB predicts well).
    Panel B: pred_AR vs  pred_BB     (spread shows BB differs from AR mean).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
        'font.size': 10, 'axes.linewidth': 0.8,
    })

    y_all      = []
    bb_all     = []
    ar_all     = []

    for res in all_seed_results:
        y_all .extend(res['y_test'])
        bb_all.extend(res['preds']['BlockBoost.R2'][best_a_T])
        ar_all.extend(res['preds']['AR(p)'])

    y_all  = np.array(y_all)
    bb_all = np.array(bb_all)
    ar_all = np.array(ar_all)

    # Statistics printed to console
    print(f'\n  [Collapse Check] best a_T={best_a_T}  (aggregated across all seeds)')
    print(f'    std(y)        = {np.std(y_all):.5f}')
    print(f'    std(pred_BB)  = {np.std(bb_all):.5f}')
    print(f'    std(pred_AR)  = {np.std(ar_all):.5f}')
    print(f'    bias_BB       = {np.mean(bb_all - y_all):.5f}')
    print(f'    bias_AR       = {np.mean(ar_all - y_all):.5f}')
    corr_bb = np.corrcoef(y_all, bb_all)[0, 1]
    corr_ar = np.corrcoef(y_all, ar_all)[0, 1]
    print(f'    corr(y, BB)   = {corr_bb:.4f}')
    print(f'    corr(y, AR)   = {corr_ar:.4f}')
    from sklearn.linear_model import LinearRegression
    lr_bb = LinearRegression().fit(bb_all.reshape(-1, 1), y_all)
    print(f'    slope y~BB    = {lr_bb.coef_[0]:.4f}  intercept={lr_bb.intercept_:.5f}')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # ── Panel A: actual vs predicted BB ──────────────────────────────────────
    ax = axes[0]
    ax.scatter(y_all, bb_all, s=2, color='black', alpha=0.25, rasterized=True)
    # y = x reference
    lo, hi = min(y_all.min(), bb_all.min()), max(y_all.max(), bb_all.max())
    ax.plot([lo, hi], [lo, hi], color='red', lw=1, ls='--', label='$y=x$')
    ax.set_xlabel('Actual $y_t$', fontsize=10)
    ax.set_ylabel(r'$\hat{y}_t$ (BlockBoost.R2)', fontsize=10)
    ax.set_title(fr'Actual vs. Predicted  ($a_T={best_a_T}$)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # ── Panel B: AR pred vs BB pred ───────────────────────────────────────────
    ax = axes[1]
    ax.scatter(ar_all, bb_all, s=2, color='black', alpha=0.25, rasterized=True)
    lo2, hi2 = min(ar_all.min(), bb_all.min()), max(ar_all.max(), bb_all.max())
    ax.plot([lo2, hi2], [lo2, hi2], color='red', lw=1, ls='--', label='$y=x$')
    ax.set_xlabel(r'$\hat{y}_t$ (AR)', fontsize=10)
    ax.set_ylabel(r'$\hat{y}_t$ (BlockBoost.R2)', fontsize=10)
    ax.set_title(r'AR vs.\ BlockBoost.R2 Predictions', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', lw=0.5, alpha=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    fig.suptitle('Prediction Diagnostics', fontsize=11)
    fig.tight_layout()
    for fmt in ('pdf', 'png'):
        fig.savefig(
            os.path.join(out_dir, f'exp2_collapse_diagnostics_aT{best_a_T}.{fmt}'),
            format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None
        )
    plt.close(fig)
    print(f'  [+] Collapse diagnostic figure saved.')


def generate_aggregate_report(all_seed_results):
    from evaluation import BLOCK_SIZES, plot_learning_curves, diebold_mariano_test, fmt_sci
    import scipy.stats as stats

    # ── 1. Aggregate Learning Curves (with per-seed confidence bands) ──────────
    per_seed_curves = {
        'AdaBoost.R2': [res['learning_curves']['AdaBoost.R2'] for res in all_seed_results],
        'BlockBoost.R2': {
            a_T: [res['learning_curves']['BlockBoost.R2'][a_T] for res in all_seed_results]
            for a_T in BLOCK_SIZES
        }
    }
    agg_learning_curves = {
        'ada_test_mae': np.mean(per_seed_curves['AdaBoost.R2'], axis=0).tolist(),
        'bb_r2_test_mae': {
            a_T: np.mean(per_seed_curves['BlockBoost.R2'][a_T], axis=0).tolist()
            for a_T in BLOCK_SIZES
        }
    }
    plot_learning_curves('SIM_Aggregate_20_Seeds', agg_learning_curves, FIG_DIR,
                         block_sizes=BLOCK_SIZES, per_seed_curves=per_seed_curves)

    # ── 2. Compile per-seed metrics and concatenate predictions ───────────────
    concat_y_test = []
    concat_preds = {
        'AR(p)': [], 'AR+GARCH': [], 'AdaBoost.R2': [],
        'BlockBoost.R2': {a_T: [] for a_T in BLOCK_SIZES}
    }
    model_metrics = {
        'AR(p)'      : {'mae': [], 'mse': []},
        'AR+GARCH'   : {'mae': [], 'mse': []},
        'AdaBoost.R2': {'mae': [], 'mse': []},
        'BlockBoost.R2': {a_T: {'mae': [], 'mse': []} for a_T in BLOCK_SIZES}
    }

    for res in all_seed_results:
        y = np.array(res['y_test'])
        concat_y_test.extend(y)
        for m in ['AR(p)', 'AR+GARCH', 'AdaBoost.R2']:
            p = np.array(res['preds'][m])
            concat_preds[m].extend(p)
            model_metrics[m]['mae'].append(mean_absolute_error(y, p))
            model_metrics[m]['mse'].append(mean_squared_error(y, p))
        for a_T in BLOCK_SIZES:
            p = np.array(res['preds']['BlockBoost.R2'][a_T])
            concat_preds['BlockBoost.R2'][a_T].extend(p)
            model_metrics['BlockBoost.R2'][a_T]['mae'].append(mean_absolute_error(y, p))
            model_metrics['BlockBoost.R2'][a_T]['mse'].append(mean_squared_error(y, p))

    y_full     = np.array(concat_y_test)
    ar_full    = np.array(concat_preds['AR(p)'])
    garch_full = np.array(concat_preds['AR+GARCH'])

    # ── 3. Table 1: Mean ± Std, DM vs AR and vs AR+GARCH, scientific notation ──
    header = ['Model',
              'Test MAE  (mean ± std)',
              'Test MSE  (mean ± std)',
              'DM p-val vs AR',
              'DM p-val vs AR+GARCH']
    rows = []

    def _row(label, mae_arr, mse_arr, pred_full, is_ar=False, is_garch=False):
        mae_m, mae_s = np.mean(mae_arr), np.std(mae_arr)
        mse_m, mse_s = np.mean(mse_arr), np.std(mse_arr)
        mae_str = f'{fmt_sci(mae_m)} ± {fmt_sci(mae_s)}'
        mse_str = f'{fmt_sci(mse_m)} ± {fmt_sci(mse_s)}'
        if is_ar:
            return [label, mae_str, mse_str, '—', '—']
        _, pval_ar    = diebold_mariano_test(y_full, ar_full,    np.array(pred_full), h=1)
        if is_garch:
            return [label, mae_str, mse_str, f'{pval_ar:.3g}', '—']
        _, pval_garch = diebold_mariano_test(y_full, garch_full, np.array(pred_full), h=1)
        return [label, mae_str, mse_str, f'{pval_ar:.3g}', f'{pval_garch:.3g}']

    rows.append(_row('AR(p)',      model_metrics['AR(p)']['mae'],       model_metrics['AR(p)']['mse'],       concat_preds['AR(p)'],       is_ar=True))
    rows.append(_row('AR+GARCH',  model_metrics['AR+GARCH']['mae'],    model_metrics['AR+GARCH']['mse'],    concat_preds['AR+GARCH'],    is_garch=True))
    rows.append(_row('AdaBoost.R2', model_metrics['AdaBoost.R2']['mae'], model_metrics['AdaBoost.R2']['mse'], concat_preds['AdaBoost.R2']))

    # All BlockBoost.R2 block sizes (all, not just best)
    best_a_T = min(BLOCK_SIZES, key=lambda a: np.mean(model_metrics['BlockBoost.R2'][a]['mae']))
    for a_T in BLOCK_SIZES:
        label = f'BB.R2 (a_T={a_T})'
        rows.append(_row(label,
                         model_metrics['BlockBoost.R2'][a_T]['mae'],
                         model_metrics['BlockBoost.R2'][a_T]['mse'],
                         concat_preds['BlockBoost.R2'][a_T]))

    save_table(rows, header,
               'Table 1 — Aggregate Model Performance across 20 Seeds',
               os.path.join(TBL_DIR, 'table1_model_comparison_20seeds.pdf'),
               col_widths=None)

    # ── 4. Prediction collapse / diagnostic scatters ──────────────────────────
    plot_collapse_diagnostics(all_seed_results, best_a_T, FIG_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 65)
    print('Experiment 2 — Simulated Regime-Switching AR(1) (20 Seeds)')
    print('=' * 65)
    t_start = time.time()

    SEEDS = list(range(42, 62))
    all_seed_results = []
    
    for seed in SEEDS:
        print(f"\n" + "▼"*65)
        print(f"▼ STARTING INDEPENDENT RUN FOR SEED {seed}")
        print("▼"*65 + "\n")
        
        datasets       = load_all(seed=seed)
        ticker         = f'SIM_Seed_{seed}'
        
        all_betas, _   = step1_mixing(datasets)
        ar_info        = step2_ar_validation(datasets)
        ar_preds, garch_preds = step3_rolling_predictions(datasets, ar_info)
        boosting       = step4_boosting(datasets)
        
        seed_metrics = compile_seed_metrics(ticker, datasets, ar_info, ar_preds, garch_preds, boosting)
        all_seed_results.append(seed_metrics)

    print('\n' + '='*65)
    print('STEP 5 – Aggregating metrics across 20 seeds')
    print('='*65)
    generate_aggregate_report(all_seed_results)

    print(f'\n[+] All done in {(time.time()-t_start)/60:.1f} min.')
    print(f'    Figures → {FIG_DIR}')
    print(f'    Tables  → {TBL_DIR}')
    print(f'    Results → {RES_DIR}')

