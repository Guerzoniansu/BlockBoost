"""
structural_break.py  –  Structural break tests for variance (squared returns).

Performs Zivot-Andrews test and CUSUM test on squared returns 
across the four datasets to identify structural breaks.
Plots all CUSUM results in a 2x2 grid and generates a PDF table of results.
"""

import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import zivot_andrews
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Paths & Setup ─────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SERIF = {
    'font.family': 'serif',
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10,
    'axes.linewidth': 0.8
}

sys.path.insert(0, _HERE)
from data import DATASETS
# Import table helpers from plots.py if possible, otherwise we define a simplified version
try:
    from plots import _save_reportlab, _save_mpl
except ImportError:
    _save_reportlab = None

# ─────────────────────────────────────────────────────────────────────────────
# Statistical Helpers
# ─────────────────────────────────────────────────────────────────────────────

def cusum_var_break(returns: np.ndarray):
    """Inclan and Tiao (1994) CUSUM test for variance breaks."""
    T = len(returns)
    r_sq = returns ** 2
    sum_r_sq = np.sum(r_sq)
    cum_sq = np.cumsum(r_sq)
    k_arr = np.arange(1, T + 1)
    
    # D_k is the centered cumulative sum of squares
    D = (cum_sq / sum_r_sq) - (k_arr / T)
    
    # Break index is where D is maximized in absolute value
    break_idx = np.argmax(np.abs(D))
    
    # IT Statistic: max |D_k| * sqrt(T / 2)
    max_D = np.abs(D[break_idx])
    it_stat = max_D * np.sqrt(T / 2)
    
    # Asymptotic p-value approximation via Brownian bridge limit distribution
    # p ≈ 2 * sum_{j=1}^{...} (-1)^(j-1) * exp(-2 * j^2 * IT^2)
    p_val = 0.0
    for j in range(1, 100):
        term = ((-1) ** (j - 1)) * np.exp(-2 * (j ** 2) * (it_stat ** 2))
        p_val += term
        if abs(term) < 1e-10:
            break
    p_val *= 2.0
    p_val = min(max(p_val, 0.0), 1.0)
    
    return break_idx, D, it_stat, p_val

# ─────────────────────────────────────────────────────────────────────────────
# Table Rendering
# ─────────────────────────────────────────────────────────────────────────────

def save_za_table(za_results: list, out_dir: str):
    """Save Zivot-Andrews results to a formal PDF table."""
    header = ['Dataset', 'ZA Stat', 'ZA p-val', 'ZA Break', 'IT CUSUM', 'IT p-val', 'CUSUM Break']
    rows = []
    for res in za_results:
        za_pval_str = f"{res['pval']:.3e}" if res['pval'] < 0.001 else f"{res['pval']:.3f}"
        it_pval_str = f"{res['it_pval']:.3e}" if res['it_pval'] < 0.001 else f"{res['it_pval']:.3f}"
        
        rows.append([
            res['label'],
            f"{res['stat']:.3f}",
            za_pval_str,
            res['za_date'],
            f"{res['it_stat']:.3f}",
            it_pval_str,
            res['cusum_date']
        ])
    
    title = "Structural Break Analysis: Volatility (Squared Returns)"
    out_path = os.path.join(out_dir, 'table_structural_breaks.pdf')
    
    if _save_reportlab:
        try:
            from reportlab.lib.units import cm
            # Slight modification to plots.py's _save_reportlab if needed, 
            # but usually it works with standard list of lists
            _save_reportlab(header, rows, title, out_path)
            print(f"  [+] ZA Table (PDF) saved to {out_path}")
            return
        except Exception as e:
            print(f"  [!] Reportlab failed for ZA table: {e}")
    
    # Fallback to matplotlib table
    print("  [!] Falling back to Matplotlib for ZA table...")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=header, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.5)
    plt.savefig(out_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Main Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    plt.rcParams.update(_SERIF)
    fig_dir = os.path.join(_HERE, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # 1x4 Grid setup
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5), sharex=False, sharey=False)
    coords = {
        'c': 0, 'f': 1,
        'ca': 2, 's': 3
    }
    
    za_results = []
    
    print("\n" + "="*95)
    print(f"{'Dataset':<20} | {'ZA Stat':<10} | {'p-val':<10} | {'ZA Break':<12} | {'IT Stat':<10} | {'IT p-val':<10} | {'CUSUM Break':<12}")
    print("-" * 95)
    
    for key in ['c', 'f', 'ca', 's']:
        info = DATASETS[key]
        label = info['label'].replace('BRVM ', '')
        csv_path = info['t_csv']
        
        df = pd.read_csv(csv_path, parse_dates=['date'])
        # Ensure data is sorted by date
        df = df.sort_values('date').dropna(subset=['returns']).reset_index(drop=True)
        returns = df['returns'].values
        dates   = df['date'].values
        r_sq    = returns ** 2
        
        # 1. Zivot-Andrews
        try:
            # regression='c' tests for a break in intercept (volatility level shift)
            za_stat, pval, cvs, baselag, bpidx = zivot_andrews(r_sq, regression='c', autolag='AIC')
            za_date = pd.to_datetime(dates[bpidx]).strftime('%Y-%m-%d')
            cv1 = cvs['1%']
        except Exception as e:
            print(f"Error in ZA for {key}: {e}")
            za_stat, pval, cv1, bpidx, za_date = 0, 1, 0, -1, "Error"
            
        # 2. CUSUM
        c_idx, D, it_stat, it_pval = cusum_var_break(returns)
        c_date = pd.to_datetime(dates[c_idx]).strftime('%Y-%m-%d')
        
        za_results.append({
            'key': key, 'label': label, 'stat': za_stat, 'pval': pval, 
            'cv1%': cv1, 'za_date': za_date, 'cusum_date': c_date,
            'it_stat': it_stat, 'it_pval': it_pval
        })
        
        print(f"{label:<20} | {za_stat:>10.3f} | {pval:>10.4f} | {za_date:<12} | {it_stat:>10.3f} | {it_pval:>10.4f} | {c_date:<12}")
        
        # ── Plotting ──
        c = coords[key]
        ax = axes[c]
        
        d_objs = pd.to_datetime(dates)
        ax.plot(d_objs, D, color='black', lw=0.6, label=r'$D_k$')
        
        # Plot IT CUSUM Break only if significant (p < 0.05)
        if it_pval < 0.05:
            ax.axvline(d_objs[c_idx], color=(1, 0, 0), ls='-', lw=0.8, 
                       label='IT break')
        
        # Plot ZA Break only if significant (p < 0.05)
        if bpidx != -1 and pval < 0.05:
            ax.axvline(d_objs[bpidx], color=(0, 0, 1), ls='-', lw=0.8, 
                       label='ZA break')
            
        # Scientific Style
        ax.text(0.02, 0.98, fr'{label}', 
                transform=ax.transAxes, verticalalignment='top', 
                fontsize=10, fontweight='bold', usetex=False)
        
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        # Rotate x labels for 1x4 horizontal layout
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha('right')
            
        ax.grid(True, ls=':', lw=0.5, alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        # Remove spines for clean look
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        
        # Clean legend (only show if something was plotted)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)

    print("-" * 95 + "\n")
    
    fig.tight_layout()
    # Save in both formats
    for fmt in ['png', 'pdf']:
        out_path = os.path.join(fig_dir, f'fig3_structural_breaks.{fmt}')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    print(f"  [+] Combined Structural Break Plot (PNG/PDF) saved.")
    
    # Save formal table
    save_za_table(za_results, os.path.join(_HERE, 'tables'))

if __name__ == '__main__':
    run_tests()
