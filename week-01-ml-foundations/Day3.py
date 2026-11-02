"""Day 3 checkpoints — verify GaussianNaiveBayes on toy data."""

import numpy as np
from data.naive_bayes import GaussianNaiveBayes


print("=" * 60)
print("Checkpoint A — fit() learns correct statistics")
print("=" * 60)

# Toy dataset from Block 3 §3.6 — hand-verified means/vars
X = np.array([
    [1, 100], [2, 110], [3, 90], [4, 105], [5, 95],     # class A
    [10, 50], [12, 60], [11, 55], [13, 45], [14, 40],   # class B
], dtype=float)
y = np.array(["A"] * 5 + ["B"] * 5)

model = GaussianNaiveBayes(var_smoothing=1e-9)
model.fit(X, y)

print("classes_:", model.classes_)
print("class_priors_:", model.class_priors_)
print("class_means_['A']:", model.class_means_["A"])
print("class_vars_['A']:", model.class_vars_["A"])
print("class_means_['B']:", model.class_means_["B"])
print("class_vars_['B']:", model.class_vars_["B"])

# Assertions — these values come from Block 3 hand-calculation
assert model.classes_ == ["A", "B"], f"classes_ = {model.classes_}"
assert model.class_priors_ == {"A": 0.5, "B": 0.5}, f"priors = {model.class_priors_}"
np.testing.assert_allclose(model.class_means_["A"], [3, 100], atol=1e-6)
np.testing.assert_allclose(model.class_means_["B"], [12, 50], atol=1e-6)
np.testing.assert_allclose(model.class_vars_["A"], [2, 50], atol=1e-6)
np.testing.assert_allclose(model.class_vars_["B"], [2, 50], atol=1e-6)
print("✓ Checkpoint A passed\n")


print("=" * 60)
print("Checkpoint B — predict() on toy data")
print("=" * 60)

# Predictions from Block 3 hand-trace
pred_A = model.predict(np.array([[3, 100]]))[0]
pred_B = model.predict(np.array([[12, 55]]))[0]
pred_mid = model.predict(np.array([[7, 75]]))[0]

print(f"predict([3, 100])  = {pred_A!r}   (expected 'A')")
print(f"predict([12, 55])  = {pred_B!r}   (expected 'B')")
print(f"predict([7, 75])   = {pred_mid!r} (boundary — either class OK)")

assert pred_A == "A", f"Expected A, got {pred_A}"
assert pred_B == "B", f"Expected B, got {pred_B}"
print("✓ Checkpoint B passed\n")


print("=" * 60)
print("Checkpoint C — predict_proba() and score()")
print("=" * 60)

probs_A = model.predict_proba(np.array([[3, 100]]))[0]
probs_B = model.predict_proba(np.array([[12, 55]]))[0]
print(f"predict_proba([3, 100])  = {probs_A}")
print(f"predict_proba([12, 55])  = {probs_B}")

# Rows must sum to 1
all_probs = model.predict_proba(X)
row_sums = all_probs.sum(axis=1)
print(f"row sums (should all be 1.0): {row_sums}")
np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-9)

# Probability of the winning class should be > 0.5
assert probs_A.max() > 0.9, f"Confident A expected, got {probs_A}"
assert probs_B.max() > 0.9, f"Confident B expected, got {probs_B}"

# Training accuracy — toy data is separable, should be 1.0
train_acc = model.score(X, y)
print(f"train accuracy: {train_acc:.4f}  (should be 1.0)")
assert train_acc == 1.0, f"Expected 1.0, got {train_acc}"
print("✓ Checkpoint C passed\n")


print("=" * 60)
print("Checkpoint D — matches sklearn on toy data")
print("=" * 60)

from sklearn.naive_bayes import GaussianNB

sk = GaussianNB(var_smoothing=1e-9)
sk.fit(X, y)

my_preds = model.predict(X)
sk_preds = sk.predict(X)
print("mine:   ", my_preds)
print("sklearn:", sk_preds)
print("matches:", np.array_equal(my_preds, sk_preds))

assert np.array_equal(my_preds, sk_preds), "Predictions differ from sklearn"

# Compare fitted means (sklearn stores theta_ = means per class, sorted)
# sklearn's theta_[0] is for classes_[0], theta_[1] for classes_[1]
my_means = np.array([model.class_means_[c] for c in model.classes_])
print("my means:     ", my_means)
print("sklearn theta_:", sk.theta_)
np.testing.assert_allclose(my_means, sk.theta_, atol=1e-6)

# Compare fitted variances (sklearn stores sigma_ = variances)
my_vars = np.array([model.class_vars_[c] for c in model.classes_])
print("my vars:        ", my_vars)
print("sklearn var_: ", sk.var_)
np.testing.assert_allclose(my_vars, sk.var_, rtol=1e-3)
print("✓ Checkpoint D passed\n")


print("=" * 60)
print("🎉 ALL CHECKPOINTS PASSED")
print("=" * 60)