# adaboost.py

"""
Self implementation of original adaboost
References
----------
.. [1] Y. Freund, R. Schapire, "A Decision-Theoretic Generalization of
        on-Line Learning and an Application to Boosting", 1995.
"""

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def _safe_normalize(w):
    """Normalize weight vector; reset to uniform if NaN/inf detected."""
    if not np.all(np.isfinite(w)) or np.sum(w) < 1e-300:
        return np.full_like(w, 1.0 / len(w))
    return w / np.sum(w)


class AdaBoostM1(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, n_estimators=50):
        self.estimator = estimator if estimator is not None else DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators

        self.estimators_ = []
        self.alphas_ = []
        self.sum_squared_weights_ = []

        self.train_accuracy_ = []
        self.test_accuracy_ = []

    def fit(self, X, y, X_val=None, y_val=None):
        X, y = check_X_y(X, y)

        # map labels to {-1, +1}
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("AdaBoostM1 supports binary classification only.")
        y = np.where(y == self.classes_[0], -1, 1)
        if y_val is not None:
            y_val= np.where(y_val == self.classes_[0], -1, 1)

        n_samples = X.shape[0]
        D = np.full(n_samples, 1.0 / n_samples)

        # Optimization: Track cumulative scores to avoid O(M^2) redundant predictions
        f_train = np.zeros(n_samples)
        f_val = np.zeros(X_val.shape[0]) if X_val is not None else None

        for _ in range(self.n_estimators):
            self.sum_squared_weights_.append(np.sum(D**2))
            clf = sklearn.base.clone(self.estimator)
            clf.fit(X, y, sample_weight=D)

            pred = clf.predict(X)
            incorrect = (pred != y)

            epsilon = np.sum(D[incorrect])

            # if epsilon <= 0 or epsilon >= 0.5:
            #     # Still record accuracy if we stop
            #     self.train_accuracy_.append(np.mean(np.sign(f_train) == y) if len(self.alphas_) > 0 else 0.0)
            #     break

            alpha = 0.5 * np.log((1 - epsilon) / (epsilon + 1e-10))

            # weight update (exponential loss) — log-space for stability
            log_D = np.log(np.maximum(D, 1e-300))
            log_D -= alpha * y * pred
            log_D -= np.max(log_D)          # shift for numerical safety
            D = np.exp(log_D)
            D = _safe_normalize(D)

            self.estimators_.append(clf)
            self.alphas_.append(alpha)

            # Incremental score updates
            f_train += alpha * pred
            # map {-1, 1} to original classes via sign and mapping
            # self.score uses self.predict, which replicates this logic.
            # We can just compute accuracy directly from f_train.
            self.train_accuracy_.append(np.mean(np.sign(f_train) == y))

            if X_val is not None and y_val is not None:
                pred_val = clf.predict(X_val)
                f_val += alpha * pred_val
                self.test_accuracy_.append(np.mean(np.sign(f_val) == y_val))

        return self.train_accuracy_, self.test_accuracy_

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        F = np.zeros(X.shape[0])
        for alpha, clf in zip(self.alphas_, self.estimators_):
            F += alpha * clf.predict(X)

        preds = np.sign(F)
        return np.where(preds == -1, self.classes_[0], self.classes_[1])


class AdaBoostR2Regressor(BaseEstimator):
    """
    AdaBoost.R2 for regression (Drucker 1997).
    """
    def __init__(self, estimator=None, n_estimators=50):
        self.estimator = estimator if estimator is not None else DecisionTreeRegressor(max_depth=3)
        self.n_estimators = n_estimators

        self.estimators_ = []
        self.betas_ = []
        self.train_mse_ = []
        self.test_mse_ = []
        self.train_mae_ = []
        self.test_mae_ = []

    @staticmethod
    def _weighted_median_predict(predictions, weights):
        """
        Compute weighted median over columns of predictions using weights.
        predictions: (N, M) array, weights: (M,) array.
        """
        n_samples, n_est = predictions.shape
        y_pred = np.zeros(n_samples)
        sum_w = np.sum(weights)
        for i in range(n_samples):
            sorted_idx = np.argsort(predictions[i])
            cum_w = np.cumsum(weights[sorted_idx])
            median_idx = np.searchsorted(cum_w, 0.5 * sum_w)
            y_pred[i] = predictions[i, sorted_idx[median_idx]]
        return y_pred

    def fit(self, X, y, X_val=None, y_val=None):
        X, y = check_X_y(X, y)

        n_samples = X.shape[0]
        D = np.full(n_samples, 1.0 / n_samples)

        # Pre-allocate cache for predictions to avoid redundant tree calls
        pred_cache = np.empty((n_samples, self.n_estimators))
        pred_cache_val = None
        if X_val is not None:
            pred_cache_val = np.empty((X_val.shape[0], self.n_estimators))

        m = 0   # actual number of estimators added
        for m_round in range(self.n_estimators):
            clf = sklearn.base.clone(self.estimator)
            clf.fit(X, y, sample_weight=D)
            pred = clf.predict(X)

            # Linear loss
            abs_errors = np.abs(y - pred)
            D_max = np.max(abs_errors)
            
            if D_max < 1e-10:
                # Perfect fit, zero loss
                self.estimators_.append(clf)
                self.betas_.append(1e-10)
                pred_cache[:, m] = pred
                m += 1
                break

            L_t = abs_errors / D_max
            L_bar = np.sum(D * L_t)

            # No stopping condition if L_bar > 0.5, per user instruction.
            beta = L_bar / (1.0 - L_bar + 1e-10)
            
            # Avoid beta = 0 which causes log(1/beta) = inf
            beta = max(beta, 1e-10)
            
            # Update distribution — log-space for stability
            log_D = np.log(np.maximum(D, 1e-300))
            log_D += (1.0 - L_t) * np.log(max(beta, 1e-300))
            log_D -= np.max(log_D)
            D = np.exp(log_D)
            D = _safe_normalize(D)

            self.estimators_.append(clf)
            self.betas_.append(beta)
            pred_cache[:, m] = pred
            m += 1

            # Compute train metrics from cache (no redundant tree calls)
            weights = np.log(1.0 / np.array(self.betas_))
            y_hat = self._weighted_median_predict(pred_cache[:, :m], weights)
            self.train_mse_.append(float(np.mean((y_hat - y)**2)))
            self.train_mae_.append(float(np.mean(np.abs(y_hat - y))))

            if X_val is not None and y_val is not None:
                pred_cache_val[:, m - 1] = clf.predict(X_val)
                y_hat_val = self._weighted_median_predict(pred_cache_val[:, :m], weights)
                self.test_mse_.append(float(np.mean((y_hat_val - y_val)**2)))
                self.test_mae_.append(float(np.mean(np.abs(y_hat_val - y_val))))

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        n_samples = X.shape[0]

        predictions = np.zeros((n_samples, len(self.estimators_)))
        for i, clf in enumerate(self.estimators_):
            predictions[:, i] = clf.predict(X)

        weights = np.log(1.0 / np.array(self.betas_))
        return self._weighted_median_predict(predictions, weights)














