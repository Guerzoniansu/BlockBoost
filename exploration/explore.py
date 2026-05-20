"""
explore.py  —  Data Exploration: Returns Plot + Descriptive Statistics Table
=============================================================================
Outputs
-------
- exploration_returns.pdf / .png   : 2x2 panel of log-returns (black, thin)
- desc_stats.tex                   : LaTeX table with descriptive stats & tests
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.stats as sps
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import jarque_bera

warnings.filterwarnings('ignore', category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, '..', 'data', 'data.csv')

# ── Series metadata ────────────────────────────────────────────────────────────
SERIES_COLS   = ['BRVM-COMPOSITE', 'BRVM-FINANCES', 'CAC_40', 'S&P_500']
SERIES_LABELS = ['BRVM Composite', 'BRVM Finance', 'CAC 40', 'S\&P 500']  # LaTeX-safe

# ── Helper: scientific notation for p-values ───────────────────────────────────
def fmt_sci(p):
    """Format a p-value in LaTeX-friendly scientific notation a \times 10^{b}."""
    if p == 0.0 or p < 5e-324:
        return r'$< 10^{-300}$'
    exp = int(np.floor(np.log10(p)))
    base = p / 10**exp
    # Avoid "1.00 × 10^0" for values >= 1 just in case
    if exp == 0:
        return f'{p:.4f}'
    return fr'${base:.2f} \times 10^{{{exp}}}$'

def stars(p):
    if p < 0.01:  return r'$^{***}$'
    if p < 0.05:  return r'$^{**}$'
    if p < 0.10:  return r'$^{*}$'
    return ''

# ── 1. Load Data & Compute Log Returns independently ──────────────────────────
df = pd.read_csv(DATA_FILE, parse_dates=['date'])

rets_data = {}  # {col: (dates, returns)}
for col in SERIES_COLS:
    # Drop NaNs only for the current series to maximize available data
    subset = df[['date', col]].dropna().sort_values('date')
    if len(subset) < 2:
        continue
    
    P     = subset[col].values
    dates_subset = subset['date'].values[1:]
    r     = 100.0 * np.log(P[1:] / P[:-1])
    
    rets_data[col] = (pd.DatetimeIndex(dates_subset), r)

# ── 1.1 Save Individual Clean Data Sets ───────────────────────────────────────
csv_mapping = {
    'BRVM-COMPOSITE': '../data/data_t_c.csv',
    'BRVM-FINANCES':  '../data/data_t_f.csv',
    'CAC_40':         '../data/data_t_ca.csv',
    'S&P_500':        '../data/data_t_s.csv'
}

for col, filename in csv_mapping.items():
    if col in rets_data:
        d_col, r_col = rets_data[col]
        out_df = pd.DataFrame({'date': d_col, 'returns': r_col})
        out_path = os.path.join(HERE, filename)
        out_df.to_csv(out_path, index=False)
        print(f'[+] Saved {col} returns to {filename}')

# ── 2. Returns Plot ────────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':       'serif',
    'font.size':         11,
    'axes.labelsize':    11,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'figure.dpi':        300,
})

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
axes = axes.flatten()

for ax, col, label in zip(axes, SERIES_COLS, SERIES_LABELS):
    if col not in rets_data:
        continue
    
    d_col, r = rets_data[col]

    # Series line (thin, black)
    ax.plot(d_col, r, color='black', linewidth=0.5, rasterized=True)

    # Zero line (red, dashed)
    ax.axhline(0, color='red', linewidth=0.8, linestyle='--', zorder=2)

    # Subplot title (series name)
    raw_label = label.replace(r'\&', '&')  # plain text version for matplotlib
    ax.set_title(raw_label, fontsize=12, fontweight='bold', pad=6)

    # Year ticks — one per year, plain 4-digit format, rotated 45°
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)

    # Faded grid
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.35, color='grey')
    ax.set_axisbelow(True)

    # Remove box (all four borders)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # No tick marks — just labels
    ax.tick_params(axis='both', length=0)

fig.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.5)

for fmt in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'exploration_returns.{fmt}'),
                format=fmt, bbox_inches='tight')
print('[+] Plot saved.')
plt.close()

# ── 3. Descriptive Statistics + Tests ─────────────────────────────────────────
rows = []
# ── 3. Descriptive Statistics + Tests ─────────────────────────────────────────
rows = []
for col in SERIES_COLS:
    if col not in rets_data:
        continue
    _, r = rets_data[col]
    n = len(r)

    mean   = np.mean(r)
    std    = np.std(r, ddof=1)
    vmin   = np.min(r)
    vmax   = np.max(r)
    vskew  = sps.skew(r)
    vkurt  = sps.kurtosis(r, fisher=True)   # excess=False → normal = 3

    # ADF (H0: unit root)
    adf_p  = adfuller(r, autolag='AIC')[1]

    # KPSS (H0: stationary)
    kpss_p = kpss(r, regression='c', nlags='auto')[1]

    # Jarque-Bera (H0: normality)
    jb_stat, jb_p, _, _ = jarque_bera(r)

    rows.append({
        'col':    col,
        'mean':   mean,
        'std':    std,
        'min':    vmin,
        'max':    vmax,
        'skew':   vskew,
        'kurt':   vkurt,
        'adf_p':  adf_p,
        'kpss_p': kpss_p,
        'jb_p':   jb_p,
    })

# ── 4. Build LaTeX Table ───────────────────────────────────────────────────────
def num_cell(v):
    return f'{v:.2f}'

def p_cell(p):
    return fmt_sci(p) + stars(p)

ncols = len(SERIES_COLS)
col_spec = 'l' + 'c' * ncols

header_str = ' & '.join(SERIES_LABELS)

# Numeric rows
num_row_defs = [
    ('Mean',      'mean'),
    ('Std. Dev.', 'std'),
    ('Min',       'min'),
    ('Max',       'max'),
    ('Skewness',  'skew'),
    ('Kurtosis',  'kurt'),
]

# p-value rows
p_row_defs = [
    ('ADF $p$-value',         'adf_p'),
    ('KPSS $p$-value',        'kpss_p'),
    ('Jarque-Bera $p$-value', 'jb_p'),
]

def make_row(label, key, formatter):
    cells = ' & '.join(formatter(row[key]) for row in rows)
    return f'{label} & {cells} \\\\'

lines = [
    r'\begin{table}[htbp]',
    r'\centering',
    r'\caption{Descriptive Statistics and Stationarity Tests for Daily Stock Index Returns}',
    r'\label{tab:desc_stats}',
    r'\begin{tabular}{' + col_spec + r'}',
    r'\hline\hline',
    r'Statistic & ' + header_str + r' \\',
    r'\hline',
]

for label, key in num_row_defs:
    lines.append(make_row(label, key, num_cell))

lines.append(r'\hline')

for label, key in p_row_defs:
    lines.append(make_row(label, key, p_cell))

lines += [
    r'\hline\hline',
    r'\multicolumn{' + str(ncols + 1) + r'}{l}{\footnotesize \textit{Notes:} '
    r'$^{***}$, $^{**}$, and $^{*}$ denote significance at 1\%, 5\%, and 10\% levels. '
    r'ADF: Augmented Dickey--Fuller test; $H_0$: unit root. '
    r'KPSS: Kwiatkowski--Phillips--Schmidt--Shin test; $H_0$: stationarity. '
    r'Jarque--Bera: $H_0$: normality.} \\',
    r'\end{tabular}',
    r'\end{table}',
]

tex_path = os.path.join(HERE, 'desc_stats.tex')
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('[+] LaTeX table saved.')
