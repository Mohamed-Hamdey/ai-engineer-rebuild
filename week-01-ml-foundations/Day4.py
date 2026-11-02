"""Day 4 checkpoints — verify PCA on toy data (corrected)."""

import numpy as np
from data.pca import PCA


print("=" * 60)
print("Checkpoint A — fit() matches independent numpy reference")
print("=" * 60)

rng = np.random.default_rng(42)
X = rng.standard_normal((50, 2))
X[:, 1] = 0.7 * X[:, 0] + 0.7 * rng.standard_normal(50)

pca = PCA(n_components=2)
pca.fit(X)

# Independent reference — compute covariance + eigh ourselves
X_c = X - X.mean(axis=0)
C_expected = (X_c.T @ X_c) / (len(X) - 1)
expected_evals, _ = np.linalg.eigh(C_expected)
expected_evals = expected_evals[::-1]
expected_ratio = expected_evals / expected_evals.sum()

print("my eigenvalues:   ", pca.explained_variance_)
print("ref eigenvalues:  ", expected_evals)
print("my evr:           ", pca.explained_variance_ratio_)
print("ref evr:          ", expected_ratio)

np.testing.assert_allclose(pca.mean_, X.mean(axis=0), atol=1e-9)
np.testing.assert_allclose(pca.explained_variance_, expected_evals, atol=1e-6)
np.testing.assert_allclose(pca.explained_variance_ratio_, expected_ratio, atol=1e-9)
print("✓ Checkpoint A passed\n")


print("=" * 60)
print("Checkpoint B — transform() projects correctly")
print("=" * 60)

pca1 = PCA(n_components=1)
pca1.fit(X)
X_pca1 = pca1.transform(X)
print("X_pca1 shape:", X_pca1.shape)     # should be (50, 1)
assert X_pca1.shape == (50, 1), f"Expected (50, 1), got {X_pca1.shape}"
print("✓ Checkpoint B passed\n")


print("=" * 60)
print("Checkpoint C — inverse_transform exact when k=d")
print("=" * 60)

pca_full = PCA()
X_reconstructed = pca_full.inverse_transform(pca_full.fit_transform(X))
np.testing.assert_allclose(X_reconstructed, X, atol=1e-9)
print("✓ Checkpoint C passed (reconstruction exact)\n")


print("=" * 60)
print("Checkpoint D — matches sklearn")
print("=" * 60)

from sklearn.decomposition import PCA as SkPCA

sk = SkPCA(n_components=2)
sk.fit(X)

my_pca = PCA(n_components=2)
my_pca.fit(X)

np.testing.assert_allclose(
    my_pca.explained_variance_, sk.explained_variance_, rtol=1e-6
)
np.testing.assert_allclose(
    my_pca.explained_variance_ratio_, sk.explained_variance_ratio_, rtol=1e-6
)

my_t = my_pca.transform(X)
sk_t = sk.transform(X)
np.testing.assert_allclose(np.abs(my_t), np.abs(sk_t), atol=1e-6)

print("✓ Checkpoint D passed (matches sklearn up to sign)\n")


print("=" * 60)
print("🎉 ALL CHECKPOINTS PASSED")
print("=" * 60)