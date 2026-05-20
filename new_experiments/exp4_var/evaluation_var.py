"""
evaluation.py  –  Metrics, boosting evaluation, and PDF table generation.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef, brier_score_loss
from sklearn.tree import DecisionTreeClassifier

# ── import boosting models ────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
from model.blockboost import BlockBoostClassifier
from model.adaboost   import AdaBoostM1

M_ROUNDS   = 1500
BLOCK_SIZE = 10
N_SEEDS    = 10
C_TRAIN    = '#0000FF'
C_NOISY    = '#FF0000'
C_CLEAN    = '#32CD32'


# ── classification metrics ────────────────────────────────────────────────────

def christoffersen_test(hits: np.ndarray, alpha: float = 0.05):
    """
    Christoffersen's framework for Conditional Coverage.
    hits: 1D array of {0, 1} indicating VaR exceedances.
    Returns: dict with LR_uc, p_uc, LR_ind, p_ind, LR_cc, p_cc, pi_11
    """
    from scipy.stats import chi2
    hits = np.array(hits, dtype=int)
    n = len(hits)
    if n == 0: return {}
    n1 = np.sum(hits)
    n0 = n - n1
    pi_hat = n1 / n
    
    # 1. Unconditional Coverage (LR_uc)
    # Log-likelihood under null (pi = alpha) vs alternative (pi = pi_hat)
    # Be careful with 0*log(0)
    ll_null = n0 * np.log(1 - alpha) + n1 * np.log(alpha)
    ll_alt_uc = (n0 * np.log(1 - pi_hat) + n1 * np.log(pi_hat)) if 0 < pi_hat < 1 else ll_null
    lr_uc = -2 * (ll_null - ll_alt_uc)
    p_uc = 1 - chi2.cdf(lr_uc, 1)
    
    # 2. Independence (LR_ind)
    # Transition counts: T_ij is transition from i to j
    T00 = T01 = T10 = T11 = 0
    for t in range(1, n):
        if hits[t-1] == 0 and hits[t] == 0: T00 += 1
        elif hits[t-1] == 0 and hits[t] == 1: T01 += 1
        elif hits[t-1] == 1 and hits[t] == 0: T10 += 1
        elif hits[t-1] == 1 and hits[t] == 1: T11 += 1
        
    pi_01 = T01 / (T00 + T01) if (T00 + T01) > 0 else 0.0
    pi_11 = T11 / (T10 + T11) if (T10 + T11) > 0 else 0.0
    
    ll_alt_ind = (T00 * np.log(1 - pi_01) if 1 - pi_01 > 0 else 0) + \
                 (T01 * np.log(pi_01) if pi_01 > 0 else 0) + \
                 (T10 * np.log(1 - pi_11) if 1 - pi_11 > 0 else 0) + \
                 (T11 * np.log(pi_11) if pi_11 > 0 else 0)
                 
    # In independence test, the null is that the probability of hit next day does not depend on today (pi_01 = pi_11 = pi_hat)
    # We use the empirical pi_hat from transitions: (T01 + T11) / (T00 + T01 + T10 + T11)
    n_trans = T00 + T01 + T10 + T11
    pi_hat_trans = (T01 + T11) / n_trans if n_trans > 0 else 0.0
    ll_null_ind = ((T00 + T10) * np.log(1 - pi_hat_trans) if 1 - pi_hat_trans > 0 else 0) + \
                  ((T01 + T11) * np.log(pi_hat_trans) if pi_hat_trans > 0 else 0)
                  
    lr_ind = max(0.0, -2 * (ll_null_ind - ll_alt_ind))
    p_ind = 1 - chi2.cdf(lr_ind, 1)
    
    # 3. Conditional Coverage (LR_cc)
    lr_cc = lr_uc + lr_ind
    p_cc = 1 - chi2.cdf(lr_cc, 2)
    
    return dict(
        lr_uc=float(lr_uc), p_uc=float(p_uc),
        lr_ind=float(lr_ind), p_ind=float(p_ind),
        lr_cc=float(lr_cc), p_cc=float(p_cc),
        pi_11=float(pi_11), pi_hat=float(pi_hat)
    )

def dm_test(loss1: np.ndarray, loss2: np.ndarray, h: int = 1) -> tuple:
    """
    Diebold-Mariano test comparing two loss arrays.
    Null hypothesis: E[loss1] = E[loss2].
    Returns (dm_stat, p_value). Negative dm_stat implies model 1 is better (lower loss).
    """
    from scipy.stats import norm
    d = np.array(loss1) - np.array(loss2)
    mean_d = np.mean(d)
    
    def autocovariance(xi, k):
        N = len(xi)
        if k >= N: return 0.0
        return np.mean((xi[:N-k] - np.mean(xi)) * (xi[k:] - np.mean(xi)))
        
    variance = autocovariance(d, 0)
    for k in range(1, h):
        variance += 2 * autocovariance(d, k)
        
    if variance <= 0:
        return 0.0, 1.0
        
    dm_stat = mean_d / np.sqrt(variance / len(d))
    p_value = 2 * (1 - norm.cdf(np.abs(dm_stat)))
    return float(dm_stat), float(p_value)

def compute_metrics(y_true, y_pred, y_prob=None, baseline_hits=None) -> dict:
    """
    y_true: true hit sequence mapped to {-1, +1} (or {0, 1}). We convert to {0, 1} for evaluation.
    y_pred: predicted hit sequence {-1, +1}
    baseline_hits: the actual raw {0, 1} hits from GARCH VaR, used to calculate residual hits.
    """
    from sklearn.metrics import precision_score
    
    # Convert labels to 0 and 1 for metric clarity
    y_true_01 = np.where(np.array(y_true) == 1, 1, 0)
    y_pred_01 = np.where(np.array(y_pred) == 1, 1, 0)
    
    acc   = float((y_true_01 == y_pred_01).mean())
    f1    = float(f1_score(y_true_01, y_pred_01, average='macro', zero_division=0))
    mcc   = float(matthews_corrcoef(y_true_01, y_pred_01))
    prec  = float(precision_score(y_true_01, y_pred_01, pos_label=1, zero_division=0))
    
    if y_prob is not None:
        brier = float(brier_score_loss(y_true_01, y_prob))
    else:
        # For ensemble boosting margin, we can estimate prob or just use 0/1 Brier
        brier = float(brier_score_loss(y_true_01, y_pred_01))
        
    # Christoffersen test on *unpredicted* (residual) exceedances
    # If the model predicts an exceedance (y_pred=1) and it occurs (y_true=1), it is "predicted".
    # Residual hits are hits that occurred but the model missed -> y_true_01 * (1 - y_pred_01)
    if baseline_hits is None:
        resid_hits = y_true_01
    else:
        # Base hits are those that occurred (rt < -VaRt). Model is attempting to predict these.
        baseline_hits = np.array(baseline_hits)
        resid_hits = np.maximum(0, baseline_hits - y_pred_01)
        
    ctest = christoffersen_test(resid_hits, alpha=0.05)
    
    return dict(accuracy=acc, f1=f1, mcc=mcc, brier=brier, precision=prec, **ctest)


# ── per-round accuracy helper (incremental) ───────────────────────────────────

def _per_round_acc(estimators_, alphas_, X, y):
    f    = np.zeros(len(y))
    accs = []
    for a, clf in zip(alphas_, estimators_):
        f += a * clf.predict(X)
        accs.append(float(np.mean(np.sign(f) == y)))
    return np.array(accs)


def _pad(arr, length):
    arr = np.array(arr)
    if len(arr) < length:
        arr = np.pad(arr, (0, length - len(arr)), mode='edge')
    return arr[:length]


# ── BlockBoost + AdaBoost learning curves (multi-seed) ───────────────────────

def run_boosting_curves(X_train, y_train, X_test, y_test,
                        m_rounds=M_ROUNDS,
                        block_size=BLOCK_SIZE):
    """
    Run BlockBoost and AdaBoost once on chronological data (no permutation).
    """
    stump = lambda: DecisionTreeClassifier(max_depth=1)
    
    # BlockBoost
    bb = BlockBoostClassifier(estimator=stump(), n_estimators=m_rounds,
                              block_size=block_size)
    bb.fit(X_train, y_train)
    bb_train = _pad(bb.train_accuracy_, m_rounds)
    bb_test  = _pad(_per_round_acc(bb.estimators_, bb.alphas_, X_test, y_test), m_rounds)

    # AdaBoost
    ada = AdaBoostM1(estimator=stump(), n_estimators=m_rounds)
    ada.fit(X_train, y_train)
    ada_train = _pad(ada.train_accuracy_, m_rounds)
    ada_test  = _pad(_per_round_acc(ada.estimators_, ada.alphas_, X_test, y_test), m_rounds)

    # Final predictions
    bb_preds  = np.sign(sum(a * clf.predict(X_test) for a, clf in zip(bb.alphas_, bb.estimators_)))
    ada_preds = np.sign(sum(a * clf.predict(X_test) for a, clf in zip(ada.alphas_, ada.estimators_)))

    # Wrap in array to maintain (1, M) shape for the plotting logic's mean/std
    return dict(
        bb_train  = bb_train[np.newaxis, :],
        bb_test   = bb_test[np.newaxis, :],
        ada_train = ada_train[np.newaxis, :],
        ada_test  = ada_test[np.newaxis, :],
        bb_preds  = bb_preds.tolist(),
        ada_preds = ada_preds.tolist(),
    )


# ── learning-curve figure (BlockBoost vs AdaBoost, one row per asset) ─────────

def plot_learning_curves(curves_by_ticker: dict, out_dir: str,
                         m_rounds: int = M_ROUNDS):
    plt.rcParams.update({'font.family': 'serif',
                         'mathtext.fontset': 'dejavuserif', 'font.size': 9})

    tickers = list(curves_by_ticker.keys())
    n_rows  = len(tickers)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.2 * n_rows),
                             sharex=True, sharey='row')
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.subplots_adjust(hspace=0.35, wspace=0.08)

    x = np.arange(1, m_rounds + 1)
    kw_fill = dict(alpha=0.12)

    def _panel(ax, tr_m, tr_s, te_m, te_s, title, show_ylabel):
        ax.fill_between(x, tr_m - tr_s, tr_m + tr_s, color=C_TRAIN, **kw_fill)
        ax.plot(x, tr_m, color=C_TRAIN, lw=1.0, label='Train')
        ax.fill_between(x, te_m - te_s, te_m + te_s, color=C_CLEAN, **kw_fill)
        ax.plot(x, te_m, color=C_CLEAN, lw=1.2, ls='--', label='Test')
        ax.set_title(title, fontsize=10, pad=4)
        ax.grid(True, ls='--', lw=0.4, alpha=0.55, color='#bbbbbb')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if show_ylabel:
            ax.set_ylabel('Accuracy', fontsize=9)
        ax.set_xlim(0, m_rounds)

    for row, ticker in enumerate(tickers):
        c = curves_by_ticker[ticker]
        bb_tr_m   = c['bb_train'].mean(0);  bb_tr_s  = c['bb_train'].std(0)
        bb_te_m   = c['bb_test'].mean(0);   bb_te_s  = c['bb_test'].std(0)
        ada_tr_m  = c['ada_train'].mean(0); ada_tr_s = c['ada_train'].std(0)
        ada_te_m  = c['ada_test'].mean(0);  ada_te_s = c['ada_test'].std(0)

        _panel(axes[row, 0], bb_tr_m,  bb_tr_s,  bb_te_m,  bb_te_s,
               f'BlockBoost ($a_T=10$) — {ticker}', show_ylabel=True)
        _panel(axes[row, 1], ada_tr_m, ada_tr_s, ada_te_m, ada_te_s,
               f'AdaBoost.M1 — {ticker}', show_ylabel=False)

    for ax in axes[-1, :]:
        ax.set_xlabel('Number of Boosting Rounds', fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=9,
               frameon=True, edgecolor='#cccccc', bbox_to_anchor=(0.5, 0.995))

    for fmt in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'exp2_learning_curves.{fmt}'),
                    format=fmt, bbox_inches='tight',
                    dpi=300 if fmt == 'png' else None)
    plt.close(fig)
    print(f'[+] Learning curve figure saved.')


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
