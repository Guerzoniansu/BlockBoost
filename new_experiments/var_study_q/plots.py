"""
plots.py  –  Paper-ready figures and tables for the BlockBoost VaR study.

Conventions
-----------
- Serif font, academic paper aesthetic.
- VaR fan chart: actual returns (black) + Q-BB best-block line (red dashed)
  + best GARCH line (blue solid).
- Violation-distance chart: horizontal bar chart, bars coloured by Kupiec outcome.
  Replaces the old heatmap.
- Tables via reportlab (PDF), fallback matplotlib.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

_SERIF = {
    'font.family'       : 'serif',
    'mathtext.fontset'  : 'dejavuserif',
    'font.size'         : 10,
    'axes.linewidth'    : 0.8,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_garch_for_alpha(garch_bench: dict, r_test: np.ndarray, alpha: float):
    """
    Return name and array of the GARCH model whose violation rate is
    closest to alpha on the test set.
    """
    from backtest import evaluate_var
    best_name, best_arr, best_dist = None, None, np.inf
    for col, arr in garch_bench.items():
        m = evaluate_var(r_test, arr, alpha)
        dist = abs(m['viol_rate'] - alpha)
        if dist < best_dist:
            best_dist, best_name, best_arr = dist, col, arr
    return best_name, best_arr


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – VaR fan chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_var_fan(dates, r_test: np.ndarray,
                 var_bb_r2: dict,
                 garch_bench: dict,
                 out_dir: str,
                 best_block: int,
                 dataset_key: str,
                 alpha_levels=(0.05, 0.01)):
    """
    Fan chart: actual returns overlaid with −VaR lines for:
      * Q-BB best block size  (red dashed)
      * Best GARCH model for that alpha  (blue solid)

    Parameters
    ----------
    var_bb_r2   : {alpha: {block_size: np.ndarray}}
    garch_bench : {alpha: {col_name: np.ndarray}}  (positive VaR, test-period aligned,
                  keyed by confidence level so 0.05 uses garch_05 arrays etc.)
    best_block  : block size chosen by best 5%-violation proximity
    dataset_key : short key (e.g. 'c', 'f', 'ca', 's') for file naming
    """
    plt.rcParams.update(_SERIF)

    n_alpha = len(alpha_levels)
    fig, axes = plt.subplots(n_alpha, 1,
                             figsize=(10, 3.8 * n_alpha),
                             sharex=True)
    if n_alpha == 1:
        axes = [axes]

    x = np.arange(len(r_test))

    for ax, alpha in zip(axes, alpha_levels):
        # ── Realised returns ──
        ax.fill_between(x, r_test, 0, where=(r_test < 0),
                        color='black', alpha=0.12, lw=0)
        ax.plot(x, r_test, color='black', lw=0.55, label='Returns')

        # ── Q-BB best block (red dashed) ──
        if alpha in var_bb_r2 and best_block in var_bb_r2[alpha]:
            v = var_bb_r2[alpha][best_block]
            ax.plot(x, -v, color='#c0392b', lw=1.4, ls='--',
                    label=fr'Q-BB $a_T$={best_block}')

        # ── Best GARCH (blue solid) — use the alpha-specific VaR arrays ──
        garch_alpha_dict = garch_bench.get(alpha, {})
        garch_name, garch_arr = _best_garch_for_alpha(garch_alpha_dict, r_test, alpha)
        if garch_arr is not None:
            ax.plot(x, -garch_arr, color='#2980b9', lw=1.2, ls='-',
                    label=f'Best GARCH ({garch_name})')

        pct = int(100 * (1 - alpha))
        ax.set_ylabel(f'Log return  ({pct}% VaR)', fontsize=10)
        ax.set_title(
            fr'$\alpha = {alpha}$ — VaR Forecast vs. Realised Returns',
            fontsize=10, fontweight='normal',
        )
        ax.grid(True, ls=':', lw=0.4, alpha=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(8, integer=True))
        ax.legend(loc='lower left', fontsize=8, frameon=False)

    axes[-1].set_xlabel('Test observation', fontsize=10)
    fig.tight_layout()

    fname = f'fig1_var_fan_{dataset_key}'
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{fname}.{fmt}'),
                    format=fmt, bbox_inches='tight',
                    dpi=300 if fmt == 'png' else None)
    plt.close(fig)
    print(f'  [+] Figure 1 fan chart ({dataset_key}) saved.')


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Violation-distance chart  (replaces heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def plot_violation_distance(results_bb: dict,
                            garch_metrics: dict,
                            block_sizes: list,
                            alpha_levels: tuple,
                            out_dir: str,
                            dataset_key: str):
    """
    Horizontal bar chart: deviation of observed violation rate from nominal α,
    coloured by Kupiec test outcome.

    Convention (matching existing figures in figures/):
      - Green  : Q-BB passes Kupiec (p ≥ 0.05)
      - Blue   : GARCH passes Kupiec (p ≥ 0.05)
      - Red    : any model fails Kupiec (p < 0.05)

    Parameters
    ----------
    results_bb   : {alpha: {block_size: metrics_dict}}   — Q-BB results
    garch_metrics: {alpha: {col_name: metrics_dict}}      — GARCH results
    block_sizes  : list of block sizes
    alpha_levels : tuple of alpha values
    dataset_key  : short key for file naming
    """
    plt.rcParams.update(_SERIF)
    KUPIEC_THRESH = 0.05   # significance level for Kupiec colour coding

    n_alpha = len(alpha_levels)
    fig, axes = plt.subplots(1, n_alpha, figsize=(6.5 * n_alpha, 8))
    if n_alpha == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alpha_levels):
        labels, deviations, colors = [], [], []

        # GARCH models first (top of chart)
        for col, m in garch_metrics[alpha].items():
            dev  = (m['global']['viol_rate'] - alpha) * 100   # percentage points
            kupiec_pass = m['global']['kupiec_p'] >= KUPIEC_THRESH
            labels.append(col)
            deviations.append(dev)
            colors.append('#2980b9' if kupiec_pass else '#e74c3c')

        # Q-BB rows
        for a_T in block_sizes:
            m   = results_bb[alpha][a_T]
            dev = (m['global']['viol_rate'] - alpha) * 100
            kupiec_pass = m['global']['kupiec_p'] >= KUPIEC_THRESH
            labels.append(f'Q-BB (a_T={a_T})')
            deviations.append(dev)
            colors.append('#27ae60' if kupiec_pass else '#e74c3c')


        y_pos  = np.arange(len(labels))
        deviations = np.array(deviations)

        ax.barh(y_pos, deviations, color=colors, height=0.6,
                edgecolor='none', alpha=0.85)

        ax.axvline(0, color='black', lw=0.9, ls='-')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_xlabel(f'Deviation from Nominal α ({int(alpha*100)}%)\n'
                      f'[violation rate − α,  in percentage points]',
                      fontsize=9)
        ax.set_title(
            fr'$\alpha = {int(alpha*100)}\%$ Violation Rate Distance',
            fontsize=11, fontweight='bold',
        )
        ax.grid(axis='x', ls=':', lw=0.4, alpha=0.7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend
        from matplotlib.patches import Patch
        legend_elems = [
            Patch(facecolor='#27ae60', label='BB Passes Kupiec'),
            Patch(facecolor='#2980b9', label='GARCH Passes Kupiec'),
            Patch(facecolor='#e74c3c', label='Fails Kupiec'),
        ]
        ax.legend(handles=legend_elems, fontsize=8, frameon=True,
                  loc='upper right' if deviations.max() > abs(deviations.min()) else 'upper left')

    fig.tight_layout(pad=2.0)
    fname = f'fig2_violation_dist_{dataset_key}'
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{fname}.{fmt}'),
                    format=fmt, bbox_inches='tight',
                    dpi=300 if fmt == 'png' else None)
    plt.close(fig)
    print(f'  [+] Figure 2 violation-distance ({dataset_key}) saved.')


# ─────────────────────────────────────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_sci(v, d=2):
    if v == 0:
        return '0'
    return f'{v:.{d}e}'


def _fmt_pval(v):
    """Display p-values with significance stars."""
    if v < 0.001:
        return f'{v:.2e}***'
    elif v < 0.01:
        return f'{v:.3f}**'
    elif v < 0.05:
        return f'{v:.3f}*'
    return f'{v:.3f}'


def _make_table_rows(results_bb: dict, block_sizes: list, alpha: float,
                     garch_metrics: dict) -> tuple:
    """Build header + rows for the summary comparison table."""
    header = [
        'Model', 'Block',
        'Viol% (All)', 'Kup-p (All)', 'Chr-p (All)',
        'Viol% (Pre)', 'Kup-p (Pre)', 'Chr-p (Pre)',
        'Viol% (Post)', 'Kup-p (Post)', 'Chr-p (Post)',
        'MAE',
    ]
    rows = []

    def _row(model_name, block_str, m):
        g, pre, post = m['global'], m['pre2022'], m['post2022']
        return [
            model_name, block_str,
            f'{g["viol_rate"]*100:.2f}',   _fmt_pval(g['kupiec_p']),   _fmt_pval(g['christ_p']),
            f'{pre["viol_rate"]*100:.2f}', _fmt_pval(pre['kupiec_p']), _fmt_pval(pre['christ_p']),
            f'{post["viol_rate"]*100:.2f}',_fmt_pval(post['kupiec_p']),_fmt_pval(post['christ_p']),
            _fmt_sci(g['mae']),
        ]

    for a_T in block_sizes:
        if a_T in results_bb[alpha]:
            rows.append(_row('Q-BB', str(a_T), results_bb[alpha][a_T]))

    # GARCH rows (no block column)
    for col, m_dict in garch_metrics[alpha].items():
        rows.append(_row(col, '-', m_dict))

    return header, rows


def save_result_table(results_bb: dict, block_sizes: list,
                      alpha: float, garch_metrics: dict,
                      out_dir: str, dataset_key: str = ''):
    """Build and save a PDF/PNG table for a given alpha level and dataset."""
    header, rows  = _make_table_rows(results_bb, block_sizes, alpha, garch_metrics)
    n_test_str    = rows[0][2] if rows else '?'
    title  = (f'BlockBoost VaR Backtest — '
              f'α = {alpha}  dataset = {dataset_key}')
    tag    = f'_{dataset_key}' if dataset_key else ''
    out_path = os.path.join(out_dir,
                            f'table_var_alpha{int(alpha*100):02d}{tag}.pdf')

    try:
        _save_reportlab(header, rows, title, out_path)
        print(f'  [+] Table (reportlab) -> {out_path}')
    except Exception as e:
        print(f'  [!] reportlab failed ({e}); using matplotlib fallback.')
        _save_mpl(header, rows, title, out_path)


def _save_reportlab(header, rows, title, out_path):
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib           import colors
    from reportlab.platypus      import (SimpleDocTemplate, Table,
                                         TableStyle, Paragraph, Spacer)
    from reportlab.lib.styles    import getSampleStyleSheet
    from reportlab.lib.units     import cm

    doc    = SimpleDocTemplate(out_path, pagesize=landscape(A4),
                               leftMargin=1.5*cm, rightMargin=1.5*cm,
                               topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    elems  = [Paragraph(title, styles['Title']), Spacer(1, 0.3*cm)]

    data   = [header] + rows
    ncols  = len(header)
    cw     = [3.5*cm] * ncols
    cw[0]  = 4.5*cm

    style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1f3864')),
        ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,-1), 8),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),
         [colors.white, colors.HexColor('#e8edf4')]),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#aaaaaa')),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ])
    tbl = Table(data, colWidths=cw)
    tbl.setStyle(style)
    elems.append(tbl)
    doc.build(elems)


def _save_mpl(header, rows, title, out_path):
    n_rows = len(rows) + 1
    n_cols = len(header)
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.8), 0.45 * n_rows + 1.2))
    ax.axis('off')
    ax.set_title(title, fontsize=9, pad=8, fontfamily='serif')
    tbl = ax.table(cellText=rows, colLabels=header,
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

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – Violation Clustering Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_violation_clustering(results_bb: dict,
                              r_test: np.ndarray,
                              dates: pd.DatetimeIndex,
                              var_bb_r2: dict,
                              alpha_levels: tuple,
                              block_sizes: list,
                              out_dir: str,
                              dataset_key: str):
    """
    Plots the timeline of violations to visually identify clustering.
    Only top 4 block sizes (by Post-2022 violation rate proximity to alpha) are shown per alpha.
    """
    import matplotlib.dates as mdates
    plt.rcParams.update(_SERIF)

    n_alpha = len(alpha_levels)
    fig, axes = plt.subplots(n_alpha, 1, figsize=(10, 2.5 * n_alpha), sharex=True)
    if n_alpha == 1:
        axes = [axes]

    for ax, alpha in zip(axes, alpha_levels):
        # Identify top 4 blocks by absolute deviation from alpha in the post2022 period
        def post2022_dev(b):
            try:
                # the metrics list evaluates 'post2022' viol_rate
                return abs(results_bb[alpha][b]['post2022']['viol_rate'] - alpha)
            except KeyError:
                return 999
        
        sorted_blocks = sorted(block_sizes, key=post2022_dev)
        top_blocks = sorted_blocks[:4]
        
        y_ticks = []
        y_labels = []
        
        for i, b in enumerate(top_blocks):
            y_val = i + 1
            y_ticks.append(y_val)
            y_labels.append(f'QBB {b}')
            
            # Find violations (returns < negative VaR), var_bb_r2 are positive VaR magnitudes
            var = var_bb_r2[alpha][b]
            violations = r_test < -var
            viol_dates = dates[violations]
            
            # Plot
            ax.scatter(viol_dates, [y_val]*len(viol_dates), color='#e74c3c', marker='|', s=200, lw=1.5)
            ax.axhline(y_val, color='gray', linestyle=':', lw=0.5, alpha=0.5)
            
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.set_title(fr"Violation Clustering ($\alpha={alpha}$)", fontsize=10, fontweight='bold')
        ax.grid(True, axis='x', ls=':', alpha=0.5)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
    plt.tight_layout()
    fname = f'fig3_violation_cluster_{dataset_key}'
    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{fname}.{fmt}'), format=fmt, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
    plt.close(fig)
    print(f'  [+] Figure 3 Clustering chart ({dataset_key}) saved.')
