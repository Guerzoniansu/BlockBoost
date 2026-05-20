"""
lambda_optimizer.py

Optimizes the scaling factor lambda in the effective block size formula:
    a_T = ceil(lambda * a_T_theory)

Strategy: fast grid search over a candidate set of lambda values.
For each lambda, we fit BlockBoost with a small number of rounds (M_probe)
on the training fold and evaluate on a held-out validation fold.
We pick the lambda that maximizes mean validation accuracy over
n_cv independent random temporal splits of the training data.

This runs inside Phase 0 and records the optimal lambda alongside
the theoretical block size for all downstream Phase 1-4 runners to use.
"""

import math
import sys
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from model.blockboost import BlockBoostClassifier


# Grid of candidate lambda values to search over.
# 0.1 = very small blocks (may under-smooth),
# 2.0 = 2x the theoretical block size (may over-smooth).
LAMBDA_GRID = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00]

# Number of boosting rounds for each probe evaluation (small for speed).
M_PROBE = 50

# Number of cross-validation temporal folds. One split per CV run.
# We use a single forward-expanding train/val split for temporal integrity.
N_CV_FOLDS = 3


def optimize_lambda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    a_T_theory: int,
    eta: float,
    random_state: int = 42,
    lambda_grid: list = None,
    m_probe: int = M_PROBE,
    n_cv_folds: int = N_CV_FOLDS,
) -> tuple:
    """
    Finds the optimal lambda for the effective block size formula:
        a_T = ceil(lambda * a_T_theory)

    Uses temporal cross-validation: we carve out n_cv_folds non-overlapping
    held-out segments at the tail of the training set, each of size
    val_fraction of the total, and average the blockboost validation accuracy.

    Parameters
    ----------
    X_train : np.ndarray, shape (T_train, n_features)
    y_train : np.ndarray, shape (T_train,)
    a_T_theory : int
        The theoretical optimal block size from the analytical formula.
    eta : float
        The noise rate (used for informational purposes only here).
    random_state : int
        Global random seed.
    lambda_grid : list of float, optional
        Candidate lambda values to search over.
    m_probe : int
        Number of boosting rounds for each probe evaluation.
    n_cv_folds : int
        Number of temporal CV folds to average over.

    Returns
    -------
    best_lambda : float
    best_a_T : int
    cv_scores : dict mapping lambda -> mean_val_accuracy
    """
    if lambda_grid is None:
        lambda_grid = LAMBDA_GRID

    T = len(X_train)

    # Use 20% of training data as each validation fold
    val_size = max(1, int(0.20 * T))

    cv_scores = {}

    for lam in lambda_grid:
        a_T = max(1, math.ceil(lam * a_T_theory))
        fold_scores = []

        for fold_idx in range(n_cv_folds):
            # Temporal val fold: use the last val_size samples from a shrinking prefix
            # Prefix length: T - val_size * (n_cv_folds - fold_idx)
            prefix_end = T - val_size * (n_cv_folds - fold_idx)
            if prefix_end < val_size:
                # Not enough data for this fold; skip
                continue

            X_cv_train = X_train[:prefix_end]
            y_cv_train = y_train[:prefix_end]
            X_cv_val = X_train[prefix_end: prefix_end + val_size]
            y_cv_val = y_train[prefix_end: prefix_end + val_size]

            stump = DecisionTreeClassifier(max_depth=1, random_state=random_state)
            bb = BlockBoostClassifier(
                estimator=stump,
                n_estimators=m_probe,
                block_size=a_T,
                learning_rate=1.0,
            )

            try:
                bb.fit(X_cv_train, y_cv_train)
                acc = bb.score(X_cv_val, y_cv_val)
                fold_scores.append(acc)
            except Exception:
                # Degenerate block configuration; penalize
                fold_scores.append(0.0)

        if fold_scores:
            cv_scores[lam] = float(np.mean(fold_scores))
        else:
            cv_scores[lam] = 0.0

    best_lambda = max(cv_scores, key=cv_scores.get)
    best_a_T = max(1, math.ceil(best_lambda * a_T_theory))

    return best_lambda, best_a_T, cv_scores
