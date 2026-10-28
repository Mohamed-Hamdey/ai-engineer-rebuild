from data import (
    load, load_csv, stream_jsonl, LoadError,
    get_logger, log_execution_time,
    Preprocessor,
    train_val_test_split,
    validate_schema, validate_no_nulls, ValidationReport, ValidationError,
)

# 1. Logger works, no duplicates
log = get_logger("smoke")
log.info("hello")
log2 = get_logger("smoke")
assert log is log2, "logger should be cached"

# 2. Load works
df = load_csv(__import__("pathlib").Path("E:/Desktop/ai-engineer-rebuild/week-01-ml-foundations/Dataset.csv"))
print("shape:", df.shape)

# 3. Validation works
rep = validate_schema(df, {"col1": int})
print("validation:", rep.passed)

# 4. Preprocessor works
p = Preprocessor(fill_missing=True)
out = p.fit_transform(df)
print("fit_transform ok")

# 5. Split works
tr, va, te = train_val_test_split(df, val_size=0.2, test_size=0.2)
print("split:", tr.shape, va.shape, te.shape)

print("✅ Day 1 package imports and runs end-to-end")