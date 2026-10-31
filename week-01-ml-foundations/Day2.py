from data import load, train_val_test_split
from data.preprocessing import Preprocessor
from pathlib import Path

KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

DATA_PATH = Path(__file__).parent / "data" / "samples" / "KDDTrain+.txt"
df = load(DATA_PATH, header=None, names=KDD_COLUMNS)
train, val, test = train_val_test_split(df, val_size=0.15, test_size=0.15)

# ── Fit with label excluded ─────────────────────────────────
p = Preprocessor(
    standardize=True,
    fill_missing=True,
    exclude_cols=["label"],          # ← THE FIX
)
p.fit(train)

# ── A: Fit attributes ───────────────────────────────────────
print("\n=== Checkpoint A ===")
print("numeric cols:", len(p.numeric_cols_))
print("categorical cols:", p.categorical_cols_)
print("output columns:", len(p.output_columns_))
assert "label" not in p.categorical_cols_, \
    f"LEAKAGE: label in {p.categorical_cols_}"
assert not any(c.startswith("label__") for c in p.output_columns_), \
    f"LEAKAGE: {[c for c in p.output_columns_ if 'label' in c]}"

# ── B: Transform train ──────────────────────────────────────
train_t = p.transform(train)
print("\n=== Checkpoint B ===")
print("shape:", train_t.shape)
print("NaN count:", train_t.isna().sum().sum())
assert train_t.isna().sum().sum() == 0
assert train_t.columns.tolist() == p.output_columns_

# ── C: Belt + suspenders leak test ──────────────────────────
leaked = [c for c in p.output_columns_ if "label" in c]
print("\n=== Checkpoint C ===")
print("label-related columns in output:", leaked)
assert not leaked

# ── D: Train/test consistency ───────────────────────────────
test_t = p.transform(test)
print("\n=== Checkpoint D ===")
print("train:", train_t.shape, "test:", test_t.shape)
assert train_t.columns.tolist() == test_t.columns.tolist()

# ── E: Unseen category ──────────────────────────────────────
test_bad = test.copy()
test_bad.loc[0, "service"] = "totally_new_service"
result = p.transform(test_bad)
print("\n=== Checkpoint E ===")
print("row 0 service cols sum:", result.filter(like="service__").iloc[0].sum())

print("\n✅ ALL CHECKPOINTS PASS")