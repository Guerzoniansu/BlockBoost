# blockboost.py

"""
Boosting with blocking scheme, provides two variations: classifier and regressor
- Classifier: takes in sample observation of time series (x_t,y_t), t=1,...,T and \
outputs y_t+1, requires covariates x_t
- Regressor: takes in  sample observation of time series (x_t,y_t), t=1,...,T and \
outputs y_t+1

Because BlockBoost is originally designed for time series data and dependence \
is really important in time series, we use Shallow trees instead decision stumps \
Therefore, depth should be 2 or 3, not too expressive either.
"""

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


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

# BlockBoost classifier: Empirical Risk Minimization approach

class BlockBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator=None, n_estimators=50, block_size=10, learning_rate=1.0):
        self.estimator = estimator if estimator is not None else DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.block_size = block_size
        self.learning_rate = learning_rate

        self.alphas_ = []
        self.epsilons_ = [] # Track the weighted block error at each round
        self.sum_squared_weights_ = []
        self.estimators_ = []
        # self.invert_ = []
        self.train_accuracy_ = []
        self.test_accuracy_ = []
    
    def fit(self, X, y, X_val=None, y_val=None):
        X, y = check_X_y(X, y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("BlockBoost supports binary classification only.")
        y = np.where(y == self.classes_[0], -1, 1)

        if y_val is not None:
            y_val= np.where(y_val == self.classes_[0], -1, 1)

        n_samples = X.shape[0]
        K = max(1, round(n_samples/self.block_size))

        indices = np.arange(n_samples)
        blocks = np.array_split(indices, K)

        W = np.full(K, 1.0 / K)
        D = np.full(n_samples, 1.0 / n_samples)

        # Optimization: Track cumulative scores to avoid O(M^2) redundant predictions
        f_train = np.zeros(n_samples)
        f_val = np.zeros(X_val.shape[0]) if X_val is not None else None

        for _ in range(self.n_estimators):
            self.sum_squared_weights_.append(np.sum(D**2))
            clf = sklearn.base.clone(self.estimator)
            clf.fit(X, y, sample_weight=D)
            
            predictions = clf.predict(X)
            incorrect = (predictions != y).astype(int)

            r = np.zeros(K)
            for k in range(K):
                block_indices = blocks[k]
                r[k] = np.mean(incorrect[block_indices])

            epsilon = np.sum(W * r)
            # is_inverted = False

            # stop if weighted error is close to 0, add estimator with large weight
            if epsilon < 1e-6:
                alpha = 0.5 * np.log((1.0 - epsilon) / (epsilon + 1e-6))
                self.alphas_.append(alpha)
                self.epsilons_.append(epsilon)
                self.estimators_.append(clf)
                # self.invert_.append(is_inverted)
                # Final accuracy update
                f_train += alpha * predictions
                self.train_accuracy_.append(np.mean(np.sign(f_train) == y))
                break

            alpha = 0.5 * np.log((1.0 - epsilon) / (epsilon))
            
            self.alphas_.append(alpha)
            self.epsilons_.append(epsilon)
            self.estimators_.append(clf)
            # self.invert_.append(is_inverted)

            # Incremental score updates
            f_train += alpha * predictions
            self.train_accuracy_.append(np.mean(np.sign(f_train) == y))

            if X_val is not None and y_val is not None:
                pred_val = clf.predict(X_val)
                f_val += alpha * pred_val
                self.test_accuracy_.append(np.mean(np.sign(f_val) == y_val))

            # added feature: learning rate to control weight update
            log_W = np.log(np.maximum(W, 1e-300))
            log_W += self.learning_rate * alpha * (2 * r - 1)
            log_W -= np.max(log_W)
            W = np.exp(log_W)
            W = W / np.sum(W)

            D = np.empty(n_samples)
            for k, blk in enumerate(blocks):
                D[blk] = W[k] / len(blk)
            D = D / np.sum(D)
            
        return self.train_accuracy_, self.test_accuracy_
    
    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        
        # collect predictions from all weak learners
        # convert {0, 1} or other labels to {-1, 1} for weighted voting
        y_pred = np.zeros(X.shape[0])
        
        for alpha, clf in zip(self.alphas_, self.estimators_):
            # if is_inverted:
            #     preds = 1 - clf.predict(X)
            # else:
            #     preds = clf.predict(X)

            # preds = clf.predict(X)
            # # Map predictions to -1 and 1
            # preds_mapped = np.where(preds == 0, -1, 1)
            y_pred += alpha * clf.predict(X)
        
        preds = np.sign(y_pred)
        return np.where(preds == -1, self.classes_[0], self.classes_[1])


class BlockBoostR2Regressor(BaseEstimator):
    """
    BlockBoost.R2 for regression (simplified).
    Extension of AdaBoost.R2 with blockwise reweighting.
    """
    def __init__(self, block_size=20, n_estimators=50, max_depth=3):
        self.block_size = block_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(self, X, y, returns=None, X_val=None, y_val=None):
        X, y = check_X_y(X, y)
        n_samples = len(X)
        K = max(1, round(n_samples / self.block_size))
        indices = np.arange(n_samples)
        blocks = np.array_split(indices, K)

        W_k = np.ones(K) / K
        D = np.ones(n_samples) / n_samples

        self.estimators_ = []
        self.betas_ = []
        self.blocks_ = blocks
        self.train_returns_ = returns if returns is not None else y
        self.test_mae_ = []

        pred_cache = np.zeros((n_samples, self.n_estimators))
        pred_cache_val = np.zeros((X_val.shape[0], self.n_estimators)) if X_val is not None else None

        for m in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=10, random_state=m)
            tree.fit(X, y, sample_weight=D)
            pred = tree.predict(X)

            abs_errors = np.abs(y - pred)
            D_max = np.max(abs_errors)

            if D_max < 1e-12:
                self.estimators_.append(tree)
                self.betas_.append(1e-12)
                pred_cache[:, m] = pred
                if X_val is not None:
                    pred_cache_val[:, m] = tree.predict(X_val)
                    weights = np.log(1.0 / np.maximum(self.betas_, 1e-12))
                    y_hat_val = _weighted_median_predict(pred_cache_val[:, :m+1], weights)
                    self.test_mae_.append(float(np.mean(np.abs(y_hat_val - y_val))))
                break

            L_t = abs_errors / D_max
            r = np.array([np.mean(L_t[blk]) for blk in blocks])
            epsilon = np.sum(W_k * r)

            beta = epsilon / (1.0 - epsilon + 1e-12)
            if beta >= 1.0: beta = 1.0

            W_k_new = W_k * (max(beta, 1e-12) ** (1.0 - r))
            W_k = W_k_new / np.sum(W_k_new)

            for k, blk in enumerate(blocks):
                D[blk] = W_k[k] / len(blk)

            self.estimators_.append(tree)
            self.betas_.append(beta)
            pred_cache[:, m] = pred

            if X_val is not None and y_val is not None:
                pred_cache_val[:, m] = tree.predict(X_val)
                weights = np.log(1.0 / np.maximum(self.betas_, 1e-12))
                y_hat_val = _weighted_median_predict(pred_cache_val[:, :m+1], weights)
                self.test_mae_.append(float(np.mean(np.abs(y_hat_val - y_val))))

        weights = np.log(1.0 / np.maximum(self.betas_, 1e-12))
        self.train_predictions_ = _weighted_median_predict(pred_cache[:, :len(self.estimators_)], weights)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        predictions = np.zeros((X.shape[0], len(self.estimators_)))
        for i, tree in enumerate(self.estimators_):
            predictions[:, i] = tree.predict(X)
        weights = np.log(1.0 / np.maximum(self.betas_, 1e-12))
        return _weighted_median_predict(predictions, weights)


class BlockBoostRegressor(BaseEstimator):
    """
    BlockBoost Regressor using residual fitting and exact line search (Gradient-based).
    """
    def __init__(self, block_size=20, n_estimators=50, learning_rate=0.5, max_depth=3):
        self.block_size = block_size
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(self, X, y, returns=None):
        X, y = check_X_y(X, y)
        n_samples = len(X)
        K = max(1, round(n_samples / self.block_size))
        indices = np.arange(n_samples)
        blocks = np.array_split(indices, K)

        W_k = np.ones(K) / K
        D = np.ones(n_samples) / n_samples
        self.f0_ = np.mean(y)
        F = np.full(n_samples, self.f0_)

        self.estimators_ = []
        self.alphas_ = []
        self.block_influences_ = []
        self.blocks_ = blocks
        self.train_returns_ = returns if returns is not None else y

        for m in range(self.n_estimators):
            residuals = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=10, random_state=m)
            tree.fit(X, residuals, sample_weight=D)
            pred = tree.predict(X)

            eps_k = np.array([np.mean((pred[blk] - residuals[blk])**2) for blk in blocks])
            eps_m = np.sum(W_k * eps_k)

            if eps_m < 1e-12:
                break

            num = np.sum(D * residuals * pred)
            den = np.sum(D * pred**2)
            alpha = num / (den + 1e-12)

            # Clip exponent to prevent float64 overflow for high-variance blocks
            exponent = self.learning_rate * eps_k / (eps_m + 1e-12)
            W_k_new = W_k * np.exp(np.clip(exponent, -700, 700))
            W_k = W_k_new / np.sum(W_k_new)

            infl = np.array([np.sum(D[blk]) for blk in blocks])

            for k, blk in enumerate(blocks):
                D[blk] = W_k[k] / len(blk)

            F = F + alpha * pred

            self.estimators_.append(tree)
            self.alphas_.append(alpha)
            self.block_influences_.append(infl)

        self.train_predictions_ = F
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        F = np.full(X.shape[0], self.f0_)
        for alpha, tree in zip(self.alphas_, self.estimators_):
            F += alpha * tree.predict(X)
        return F


class QBlockBoostR2Regressor(BaseEstimator):
    """
    Q-BlockBoost.R2 for regression.
    Uses Pinball Loss and a single quantile regression tree as the weak learner.
    Target quantile tau = 1 - alpha.
    """
    def __init__(self, block_size=20, n_estimators=50, max_depth=3, alpha=0.05):
        self.block_size = block_size
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.alpha = alpha
        self.tau = 1.0 - alpha

    def _pinball_loss(self, y, f):
        error = y - f
        return np.where(error >= 0, self.tau * error, (self.tau - 1.0) * error)

    def fit(self, X, y, returns=None, X_val=None, y_val=None):
        X, y = check_X_y(X, y)
        n_samples = len(X)
        K = max(1, round(n_samples / self.block_size))
        indices = np.arange(n_samples)
        blocks = np.array_split(indices, K)

        W_k = np.ones(K) / K
        D = np.ones(n_samples) / n_samples

        self.estimators_ = []
        self.betas_ = []
        self.blocks_ = blocks
        self.train_returns_ = returns if returns is not None else y
        self.test_loss_ = []

        pred_cache = np.zeros((n_samples, self.n_estimators))
        pred_cache_val = np.zeros((X_val.shape[0], self.n_estimators)) if X_val is not None else None

        for m in range(self.n_estimators):
            tree = GradientBoostingRegressor(
                loss='quantile',
                alpha=self.tau,
                n_estimators=1,
                learning_rate=1.0,
                max_depth=self.max_depth,
                min_samples_leaf=10,
                random_state=42
            )
            tree.fit(X, y, sample_weight=D)
            pred = tree.predict(X)

            errors = self._pinball_loss(y, pred)
            D_max = np.max(errors)

            if D_max < 1e-12:
                self.estimators_.append(tree)
                self.betas_.append(1e-12)
                pred_cache[:, m] = pred
                if X_val is not None:
                    pred_cache_val[:, m] = tree.predict(X_val)
                    weights = np.log(1.0 / np.maximum(self.betas_, 1e-12))
                    y_hat_val = _weighted_median_predict(pred_cache_val[:, :m+1], weights)
                    self.test_loss_.append(float(np.mean(self._pinball_loss(y_val, y_hat_val))))
                break

            L_t = errors / D_max
            r = np.array([np.mean(L_t[blk]) for blk in blocks])
            epsilon = np.sum(W_k * r)

            beta = epsilon / (1.0 - epsilon + 1e-12)
            if beta >= 1.0: beta = 1.0

            W_k_new = W_k * (max(beta, 1e-12) ** (1.0 - r))
            W_k = W_k_new / np.sum(W_k_new)

            for k, blk in enumerate(blocks):
                D[blk] = W_k[k] / len(blk)

            self.estimators_.append(tree)
            self.betas_.append(beta)
            pred_cache[:, m] = pred

            if X_val is not None and y_val is not None:
                pred_cache_val[:, m] = tree.predict(X_val)
                weights = np.log(1.0 / np.maximum(self.betas_, 1e-12))
                y_hat_val = _weighted_median_predict(pred_cache_val[:, :m+1], weights)
                self.test_loss_.append(float(np.mean(self._pinball_loss(y_val, y_hat_val))))

        weights = np.log(1.0 / np.maximum(self.betas_, 1e-12))
        self.train_predictions_ = _weighted_median_predict(pred_cache[:, :len(self.estimators_)], weights)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        predictions = np.zeros((X.shape[0], len(self.estimators_)))
        for i, tree in enumerate(self.estimators_):
            predictions[:, i] = tree.predict(X)
        weights = np.log(1.0 / np.maximum(self.betas_, 1e-12))
        return _weighted_median_predict(predictions, weights)
