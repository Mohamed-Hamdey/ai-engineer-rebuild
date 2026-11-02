
"""Principal Component Analysis, implemented from scratch."""

from __future__ import annotations

import numpy as np

from .logger import get_logger


logger = get_logger(__name__)


class PCA:
    """Principal Component Analysis via covariance eigendecomposition."""

    def __init__(self, n_components: int | None = None):
        if n_components is not None:
            if not isinstance(n_components, (int, np.integer)):
                raise TypeError("n_components must be an integer or None")
            if n_components < 1:
                raise ValueError(
                    "n_components must be a positive integer or None"
                )

        self.n_components = n_components

        # Fitted state
        self.is_fitted = False
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.n_components_ = None

    def fit(self, X) -> "PCA":
        """Learn the principal components from X."""

        X = np.asarray(X, dtype=float)

        # 1. Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("Cannot fit on empty data")

        if X.shape[0] < 2:
            raise ValueError(
                "Need at least 2 samples to compute covariance"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        self.is_fitted = False

        n, d = X.shape

        # 2. Center the data
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_

        # 3. Calculate covariance matrix
        C = (X_c.T @ X_c) / (n - 1)

        # 4. Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # 5. Sort eigenvalues and corresponding eigenvectors
        idx = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 6. Select number of components
        k = (
            self.n_components
            if self.n_components is not None
            else d
        )

        if k > min(n, d):
            raise ValueError(
                f"n_components={k} is too large. "
                f"Maximum is min(n_samples, n_features) = {min(n, d)}"
            )

        # 7. Select top k eigenvalues and eigenvectors
        top_vals = eigenvalues[:k]
        top_vecs = eigenvectors[:, :k]

        # 8. Store learned PCA parameters
        self.components_ = top_vecs.T
        self.explained_variance_ = top_vals

        total_variance = eigenvalues.sum()

        if total_variance > 0:
            self.explained_variance_ratio_ = (
                top_vals / total_variance
            )
        else:
            self.explained_variance_ratio_ = np.zeros(k)

        # Singular values corresponding to selected components
        self.singular_values_ = np.sqrt(
            np.maximum(top_vals, 0) * (n - 1)
        )

        self.n_components_ = k

        self.is_fitted = True

        logger.info(
            "PCA fitted: n_samples=%d n_features=%d "
            "n_components=%d explained_variance_total=%.4f",
            n,
            d,
            self.n_components_,
            self.explained_variance_ratio_.sum(),
        )

        return self

    def transform(self, X) -> np.ndarray:
        """Project X onto the principal components."""

        if not self.is_fitted:
            raise RuntimeError("Must fit before transform")

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")

        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, "
                f"expected {self.mean_.shape[0]}"
            )

        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")

        # Center using the training mean
        X_c = X - self.mean_

        # Project onto the selected principal directions
        return X_c @ self.components_.T

    def fit_transform(self, X) -> np.ndarray:
        """Fit PCA and transform X."""

        return self.fit(X).transform(X)

    def inverse_transform(self, X_pca) -> np.ndarray:
        """Reconstruct approximate original-space data."""

        if not self.is_fitted:
            raise RuntimeError(
                "Must fit before inverse_transform"
            )

        X_pca = np.asarray(X_pca, dtype=float)

        if X_pca.ndim != 2:
            raise ValueError(
                f"X_pca must be 2-D, got shape {X_pca.shape}"
            )

        if X_pca.shape[1] != self.n_components_:
            raise ValueError(
                f"X_pca has {X_pca.shape[1]} components, "
                f"expected {self.n_components_}"
            )

        if not np.all(np.isfinite(X_pca)):
            raise ValueError("X_pca must contain only finite values")

        # Reconstruct centered data
        X_c_approx = X_pca @ self.components_

        # Restore the original feature means
        return X_c_approx + self.mean_
