"""
evaluation.py  –  Metrics, boosting evaluation, and PDF table generation.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor

# ── import boosting models ────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from model.blockboost import BlockBoostR2Regressor
from model.adaboost   import AdaBoostR2Regressor

M_ROUNDS   = 100
BLOCK_SIZES = [5, 10, 15, 20]

# ── Scientific notation formatter ─────────────────────────────────────────────

def fmt_sci(value, decimals=2):
    """Format a float in standard scientific notation, e.g. 1.34e-02."""
    if value == 0:
        return '0'
    return f'{value:.{decimals}e}'


# ── regression metrics ────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2  = float(r2_score(y_true, y_pred))
    return dict(mse=mse, mae=mae, r2=r2)

def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1):
    """
    Returns the DM test statistic and p-value.
    H0: Two models (pred1 and pred2) have the same forecast accuracy.
    Uses MAE loss differential by default.
    """
    import scipy.stats
    e1 = np.abs(y_true - y_pred1)
    e2 = np.abs(y_true - y_pred2)
    d = e1 - e2
    mean_d = np.mean(d)
    
    def autocovariance(xi, N, k):
        autoCov = 0.0
        x_mean = np.mean(xi)
        for i in np.arange(0, N-k):
            autoCov += ((xi[i+k])-x_mean)*(xi[i]-x_mean)
        return (1/(N))*autoCov
        
    N = len(d)
    gamma = [autocovariance(d, N, lag) for lag in range(0, h)]
    V_d = gamma[0] + 2 * sum(gamma[1:]) if h > 1 else gamma[0]
    
    if V_d <= 0:
        return 0.0, 1.0
        
    DM_stat = mean_d / np.sqrt((1/N)*V_d)
    p_value = 2 * scipy.stats.norm.cdf(-np.abs(DM_stat))
    return float(DM_stat), float(p_value)


# ── Paced Rolling Boosting Evaluation ──────────────────────

def rolling_boosting_regression(X_full, y_full, train_end,
                                m_rounds=50, block_sizes=BLOCK_SIZES,
                                stride=63, n_jobs=-1):
    """
    Paced rolling-origin validation (expanding window).
    Moves the training window forward by `stride` steps.
    """
    from joblib import Parallel, delayed
    n = X_full.shape[0]
    steps = list(range(train_end, n))
    chunks = [steps[i:i+stride] for i in range(0, len(steps), stride)]
    
    def _run_chunk(chunk_steps):
        t_refit = chunk_steps[0]
        # expanding window (from 0 to t_refit)
        X_history = X_full[:t_refit]
        y_history = y_full[:t_refit]
        
        X_chunk = X_full[chunk_steps]
        y_chunk = y_full[chunk_steps]
        
        stump = lambda: DecisionTreeRegressor(max_depth=4)
        
        # Train AdaBoost.R2
        ada = AdaBoostR2Regressor(estimator=stump(), n_estimators=m_rounds)
        ada.fit(X_history, y_history, X_val=X_chunk, y_val=y_chunk)
        ada_preds_chunk = ada.predict(X_chunk)
        
        # Train BlockBoost
        bb_r2_preds_chunk = {a_T: [] for a_T in block_sizes}
        bb_r2_maes = {}
        
        for a_T in block_sizes:
            bb_r2 = BlockBoostR2Regressor(max_depth=4, n_estimators=m_rounds, block_size=a_T)
            bb_r2.fit(X_history, y_history, X_val=X_chunk, y_val=y_chunk)
            bb_r2_preds_chunk[a_T] = bb_r2.predict(X_chunk)
            bb_r2_maes[a_T] = bb_r2.test_mae_
            
        chunk_results = []
        for i in range(len(chunk_steps)):
            chunk_results.append({
                'ada': float(ada_preds_chunk[i]),
                'bb_r2': {a_T: float(bb_r2_preds_chunk[a_T][i]) for a_T in block_sizes}
            })
            
        return chunk_results, ada.test_mae_, bb_r2_maes

    chunk_outputs = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(_run_chunk)(chunk) for chunk in chunks
    )
    
    # Unpack predictions vs learning curves
    flat_preds = [item for chunk, _, _ in chunk_outputs for item in chunk]
    
    ada_preds_test = np.array([x['ada'] for x in flat_preds])
    bb_r2_preds_test = {a_T: np.array([x['bb_r2'][a_T] for x in flat_preds]) for a_T in block_sizes}
    
    def _pad_curves(curves):
        # curves is a list of lists. Pad shorter curves with their last value
        max_len = max((len(c) for c in curves if c), default=0)
        if max_len == 0:
            return []
        padded = []
        for c in curves:
            if not c:
                padded.append([np.nan] * max_len)
            elif len(c) < max_len:
                padded.append(c + [c[-1]] * (max_len - len(c)))
            else:
                padded.append(c)
        return np.mean(padded, axis=0).tolist()
    
    # Average learning curves across all rolling chunks
    ada_test_mae = _pad_curves([outputs[1] for outputs in chunk_outputs])
    
    bb_r2_test_mae = {}
    for a_T in block_sizes:
        bb_r2_test_mae[a_T] = _pad_curves([outputs[2][a_T] for outputs in chunk_outputs])
        
    return dict(
        ada_preds_test=ada_preds_test.tolist(),
        bb_r2_preds_test={a_T: bb_r2_preds_test[a_T].tolist() for a_T in block_sizes},
        ada_test_mae=ada_test_mae,
        bb_r2_test_mae=bb_r2_test_mae
    )


# ── Single-panel learning-curve figure ────────────────────────────────────────

def plot_learning_curves(ticker, boosting_results, out_dir, block_sizes=BLOCK_SIZES,
                         per_seed_curves=None):
    """
    Single-panel figure overlaying all boosting learning curves.

    AdaBoost.R2  – solid red line.
    BlockBoost.R2 (each block size) – solid/dashed black lines.
    Curve labels are drawn in-line near the end of each curve (no legend box).

    Optional ``per_seed_curves`` is a dict:
        {
          'AdaBoost.R2': list[list[float]],   # one list per seed
          'BlockBoost.R2': {a_T: list[list[float]]}
        }
    When provided, a 1-std confidence band is drawn for each BB.R2 curve.
    """
    plt.rcParams.update({
        'font.family'      : 'serif',
        'mathtext.fontset' : 'dejavuserif',
        'font.size'        : 10,
        'axes.linewidth'   : 0.8,
    })

    # Black linestyles for the different block sizes
    BB_LINESTYLES = ['-', '--', ':', '-.']

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ada_test_mae = np.array(boosting_results['ada_test_mae'])
    rounds_ada   = np.arange(1, len(ada_test_mae) + 1)
    line_ada, = ax.plot(rounds_ada, ada_test_mae,
                        color='red', lw=1.6, ls='-', zorder=4)

    # Annotate AdaBoost at the end of the curve
    ax.annotate(r'AdaBoost.R2', xy=(rounds_ada[-1], ada_test_mae[-1]),
                xytext=(4, 0), textcoords='offset points',
                color='red', fontsize=8, va='center')

    for idx, a_T in enumerate(block_sizes):
        bb_mae = np.array(boosting_results['bb_r2_test_mae'].get(a_T, []))
        if len(bb_mae) == 0:
            continue
        rounds_bb = np.arange(1, len(bb_mae) + 1)
        ls = BB_LINESTYLES[idx % len(BB_LINESTYLES)]

        ax.plot(rounds_bb, bb_mae, color='black', lw=1.3, ls=ls, zorder=3)

        # Confidence bands removed — aggregate mean is shown without CI

        # Annotate per curve near the right end
        y_end = float(bb_mae[-1])
        ax.annotate(fr'$a_T={a_T}$', xy=(rounds_bb[-1], y_end),
                    xytext=(4, 0), textcoords='offset points',
                    color='black', fontsize=8, va='center')

    ax.set_xlabel('Boosting Round', fontsize=10)
    ax.set_ylabel('Test MAE', fontsize=10)
    ax.set_title('Learning Curves', fontsize=11, fontweight='normal')
    ax.grid(True, ls=':', lw=0.5, alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'exp2_learning_curves_{ticker}.{fmt}'),
                    format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close(fig)
    print(f'  [+] Learning-curve figure saved for {ticker}.')





# ── PDF table generation ──────────────────────────────────────────────────────

def _make_pdf_table_reportlab(rows, header, title, out_path, col_widths=None):
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib           import colors
    from reportlab.platypus      import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles    import getSampleStyleSheet
    from reportlab.lib.units     import cm

    doc    = SimpleDocTemplate(out_path, pagesize=landscape(A4),
                               leftMargin=1.5*cm, rightMargin=1.5*cm,
                               topMargin=1.5*cm,  bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    elems  = [Paragraph(title, styles['Title']), Spacer(1, 0.3*cm)]

    data  = [header] + rows
    ncols = len(header)
    cw    = col_widths or [4.5*cm] * ncols

    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f3864')),
        ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
        ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0, 0), (-1, -1), 8),
        ('ALIGN',      (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#e8edf4')]),
        ('GRID',       (0, 0), (-1, -1), 0.5, colors.HexColor('#aaaaaa')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ])

    tbl = Table(data, colWidths=cw)
    tbl.setStyle(style)
    elems.append(tbl)
    doc.build(elems)


def _make_pdf_table_matplotlib(rows, header, title, out_path):
    """Fallback matplotlib table."""
    n_rows = len(rows) + 1
    n_cols = len(header)
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.8), 0.45 * n_rows + 1.2))
    ax.axis('off')
    ax.set_title(title, fontsize=10, pad=8, fontfamily='serif')
    all_data = [header] + rows
    tbl = ax.table(cellText=all_data[1:], colLabels=all_data[0],
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor('#1f3864')
            cell.set_text_props(color='white', fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#e8edf4')
    plt.tight_layout()
    for fmt in ('pdf', 'png'):
        p = out_path.replace('.pdf', f'.{fmt}')
        fig.savefig(p, format=fmt, bbox_inches='tight',
                    dpi=200 if fmt == 'png' else None)
    plt.close(fig)


def save_table(rows, header, title, out_path, col_widths=None):
    """Try reportlab first, fall back to matplotlib."""
    try:
        _make_pdf_table_reportlab(rows, header, title, out_path, col_widths)
        print(f'  [+] Table (reportlab) → {out_path}')
    except Exception as e:
        print(f'  [!] reportlab failed ({e}); using matplotlib fallback.')
        _make_pdf_table_matplotlib(rows, header, title, out_path)
