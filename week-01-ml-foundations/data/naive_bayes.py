"""Gaussian Naive Bayes classifier, implemented from scratch."""

from __future__ import annotations

import numpy as np

from .logger import get_logger


logger = get_logger(__name__)


class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier.

    Assumes each feature within each class is normally distributed.
    Uses the naive Bayes assumption: features are conditionally
    independent given the class.

    fit(X, y):
        X: array of shape (n_samples, n_features) — numeric features
        y: array of shape (n_samples,) — class labels (strings or ints)

    Attributes set by fit:
        classes_        : list of unique classes seen in y
        class_priors_   : dict {class: P(class)} — fractions in training
        class_means_    : dict {class: np.array of shape (n_features,)}
        class_vars_     : dict {class: np.array of shape (n_features,)}
    """

    def __init__(self, var_smoothing: float = 1e-9):
        """
        var_smoothing: small value added to every variance to prevent
                       division by zero on constant features.
        """
        if var_smoothing < 0:
            raise ValueError("var_smoothing must be non-negative")
        self.var_smoothing = var_smoothing

        # Fitted state (declare in __init__, set in fit — sklearn convention)
        self.is_fitted = False
        self.classes_ = None
        self.class_priors_ = None
        self.class_means_ = None
        self.class_vars_ = None

    def fit(self, X, y) -> "GaussianNaiveBayes":
        """Learn per-class priors, means, and variances."""
        # ── 1. Convert X, y to numpy arrays (in case caller passes lists) ──
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        # ── 2. Basic validation ─────────────────────────────────
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have different lengths: {X.shape[0]} vs {y.shape[0]}")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on empty data")

        self.is_fitted = False

        # ── 3. Learn classes and priors ─────────────────────────
        # TODO: self.classes_ = np.unique(y).tolist()  (sorted unique)
        # TODO: self.class_priors_ = {c: float((y == c).mean()) for c in self.classes_}
        self.classes_ = np.unique(y).tolist()
        self.class_priors_ = {c: float((y == c).mean()) for c in self.classes_}
        # ── 4. Learn means and variances per class ──────────────
        self.class_means_ = {}
        self.class_vars_ = {}

        for c in self.classes_:
            # TODO: X_c = X[y == c]                # rows belonging to class c
            # TODO: self.class_means_[c] = X_c.mean(axis=0)
            # TODO: raw_var = X_c.var(axis=0)
            # TODO: self.class_vars_[c] = raw_var + self.var_smoothing
            X_c = X[y == c]
            self.class_means_[c] = X_c.mean(axis=0)
            raw_var = X_c.var(axis=0)
            self.class_vars_[c] = raw_var + self.var_smoothing
            
        self.is_fitted = True
        logger.info(
            "GaussianNB fitted: n_classes=%d n_features=%d n_samples=%d",
            len(self.classes_),
            X.shape[1],
            X.shape[0],
        )
        return self

    def _log_likelihoods(self, X: np.ndarray) -> np.ndarray:
        """Compute log P(X | class) for every class.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Entry [i, j] = sum over features of log Gaussian density
            of X[i] under class classes_[j].
        """
        if not self.is_fitted:
            raise RuntimeError("Must fit before predict")

        # ── For each class, compute the per-feature log Gaussian ──
        # Use the vectorized formula:
        #
        #   log N(x; mu, var) = -0.5 * [ log(2*pi*var) + (x - mu)^2 / var ]
        #
        # For a batch X of shape (n_samples, n_features):
        #   log_density[c] should be shape (n_samples, n_features)
        #   sum over features (axis=1) → shape (n_samples,)
        #
        # TODO: initialize scores = np.empty((X.shape[0], len(self.classes_)))
        # TODO: for j, c in enumerate(self.classes_):
        # TODO:     mean = self.class_means_[c]              # (n_features,)
        # TODO:     var  = self.class_vars_[c]                # (n_features,)
        # TODO:     log_density = -0.5 * (np.log(2*np.pi*var) + (X - mean)**2 / var)
        # TODO:     scores[:, j] = log_density.sum(axis=1)
        # TODO: return scores
        scores = np.empty((X.shape[0], len(self.classes_)))
        for j,c in enumerate(self.classes_):
            mean = self.class_means_[c]
            var = self.class_vars_[c]
            log_den = -0.5 * (np.log(2*np.pi*var) + (X - mean)**2 / var)
            scores[:, j] = log_den.sum(axis=1)
        return scores
        
        
        

    def predict(self, X) -> np.ndarray:
        """Return predicted class per row."""
        X = np.asarray(X, dtype=float)
        if not self.is_fitted:
            raise RuntimeError("Must fit before predict")

        # TODO: log_likelihoods = self._log_likelihoods(X)          # (n_samples, n_classes)
        # TODO: log_priors = np.array([np.log(self.class_priors_[c]) for c in self.classes_])
        # TODO: scores = log_likelihoods + log_priors               # broadcasting
        # TODO: idx = np.argmax(scores, axis=1)                     # (n_samples,)
        # TODO: return np.array([self.classes_[i] for i in idx])    # map indices to labels
        log_likelihoods = self._log_likelihoods(X)
        log_priors = np.array([np.log(self.class_priors_[c]) for c in self.classes_])
        scores = log_likelihoods + log_priors
        idx = np.argmax(scores, axis=1)
        return np.array([self.classes_[i] for i in idx]) 
        
        
    def predict_proba(self, X) -> np.ndarray:
        """Return probability of each class per row. Shape (n_samples, n_classes)."""
        X = np.asarray(X, dtype=float)
        if not self.is_fitted:
            raise RuntimeError("Must fit before predict")

        # TODO: compute scores exactly like in predict
        # TODO: apply softmax (numerically stable — subtract max per row)
        # TODO: return probabilities  (rows sum to 1)
        log_likelihoods = self._log_likelihoods(X)
        log_priors = np.array([np.log(self.class_priors_[c]) for c in self.classes_])
        scores = log_likelihoods + log_priors
        
        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
 
        return probabilities

    def score(self, X, y) -> float:
        """Return accuracy: fraction of rows where predict(X) == y."""
        # TODO: predictions = self.predict(X)
        # TODO: return float((predictions == np.asarray(y)).mean())
        predictions = self.predict(X)
        return float((predictions == np.asarray(y)).mean())