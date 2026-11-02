# 📘 Day 2 Revision Sheet --- NumPy, Pandas & the Preprocessor

This sheet covers the concepts from today's work: NumPy arrays and

broadcasting, pandas operations, missing values, feature engineering,

data leakage, and the `fit`/`transform` preprocessing contract.

  

------------------------------------------------------------------------

  

## Part 1 --- NumPy: Arrays, Vectorization, and Data Types

  

### 1.1 What NumPy is

  

NumPy provides efficient multidimensional arrays and vectorized

numerical operations.

  

A Python list can contain values of different types. A NumPy array

usually stores values using one common dtype.

  

``` python

import numpy as np

  

a = np.array([1, 2, 3])          # shape (3,), usually int64

m = np.array([[1, 2, 3],

              [4, 5, 6]])        # shape (2, 3)

```

  

**Mental model:** An array is a structured grid of values, not a

general-purpose container like a Python list.

  

### 1.2 Vectorization --- why NumPy is useful

  

Vectorization means expressing an operation over an entire array instead

of writing an explicit Python loop.

  

``` python

a = np.array([1, 2, 3])

  

a * 2       # array([2, 4, 6])

a + 10      # array([11, 12, 13])

a[a > 1]    # array([2, 3])

```

  

NumPy performs many array operations in optimized compiled code, so

vectorized operations are often much faster than equivalent Python

loops. The exact speedup depends on the operation, array size, and

hardware.

  

For example:

  

``` python

df["col"] - df["col"].mean()

```

  

This computes the mean and subtracts it from each value using vectorized

pandas/NumPy operations.

  

### 1.3 `shape`, `ndim`, `size`, and `dtype`

  

  Attribute   Meaning                                       Example

  ----------- --------------------------------------------- ------------------------------

  `.shape`    Tuple containing the size of each dimension   `(2, 3)`

  `.ndim`     Number of dimensions                          `2`

  `.size`     Total number of elements                      `6`

  `.dtype`    Element data type                             `int64`, `float64`, `object`

  

NumPy arrays generally use a common dtype. Values may be converted to a

compatible type:

  

``` python

np.array([1, "hello"])  # promotes values to a string dtype

np.array([1, None])     # commonly produces dtype=object

```

  

**Pandas consequence:** A column containing mixed types may have

`object` dtype. This can use more memory and make numerical operations

harder or slower. Check dtypes before preprocessing; do not assume that

a column containing numbers is actually numeric.

  

### 1.4 Boolean masks --- a pattern you will use constantly

  

``` python

a = np.array([10, 20, 30, 40, 50])

  

a[a > 25]  # array([30, 40, 50])

```

  

Trace it:

  

1.  `a > 25` produces `[False, False, True, True, True]`.

2.  `a[mask]` selects the positions where the mask is `True`.

  

In pandas, the same idea appears here:

  

``` python

df[df["sales"] > 0]

```

  

**Gotcha:** Use `&`, `|`, and `~` for element-wise boolean logic with

NumPy arrays and pandas Series. Put each comparison in parentheses.

  

``` python

(a > 10) & (a < 50)   # correct

# (a > 10) and (a < 50)  # raises an ambiguity-related error

```

  

### 1.5 NumPy functions to recognize

  

Common functions and methods include:

  

`np.array`, `np.arange`, `np.zeros`, `np.ones`, `np.isnan`, `.mean()`,

`.std()`, `.sum()`, `.min()`, `.max()`, `.reshape()`, `.astype()`,

`np.where`, `np.concatenate`, and `np.unique`.

  

You do not need to memorize every function immediately. Learn to

recognize the problem each one solves.

  

------------------------------------------------------------------------

  

## Part 2 --- Broadcasting: The Concept That Makes Array Math Readable

  

### 2.1 The rule

  

NumPy compares shapes from **right to left**. Two dimensions are

compatible when they are equal or one of them is `1`. Missing leading

dimensions are treated as `1`.

  

``` python

a.shape = (2, 3)

b.shape = (3,)

  

# b is treated as shape (1, 3), then broadcast across 2 rows.

a + b  # works; result shape is (2, 3)

  

a.shape = (2, 3)

c.shape = (2,)

  

# Compare from the right: 3 vs 2. Neither is 1, so this fails.

a + c  # ValueError

```

  

"Stretch" is a mental model. Broadcasting usually does not require

physically copying the smaller array.

  

### 2.2 Common cases

  

  -----------------------------------------------------------------------

  Case                    Example shapes          Result

  ----------------------- ----------------------- -----------------------

  Scalar + array          scalar and `(2, 3)`     Scalar applies to every

                                                  element

  

  Vector + matrix         `(3,)` and `(2, 3)`     Vector applies to each

                                                  row

  

  Column + matrix         `(2, 1)` and `(2, 3)`   Each row's column value

                                                  applies across that row

  

  Incompatible            `(2,)` and `(2, 3)`     `ValueError`

  -----------------------------------------------------------------------

  

When you see an error such as:

  

``` text

operands could not be broadcast together with shapes (2,3) (2,)

```

  

Compare dimensions from the right. The final dimensions are `3` and `2`;

they differ, and neither is `1`.

  

### 2.3 Why broadcasting matters in preprocessing

  

``` python

# x.mean() and x.std() are scalars.

z = (x - x.mean()) / x.std()

```

  

The scalar mean and standard deviation are applied across the elements

of `x`.

  

For grouped pandas calculations:

  

``` python

df["dept_mean"] = (

    df.groupby("dept")["salary"].transform("mean")

)

```

  

`transform()` returns a result aligned to the original rows. This is a

pandas group operation rather than a NumPy broadcasting example, but it

solves a related problem: putting group-level information back onto each

row without manually looping over groups.

  

------------------------------------------------------------------------

  

## Part 3 --- Pandas: Series, DataFrames, and `[]` vs `[[]]`

  

### 3.1 Series

  

A pandas Series is a one-dimensional labeled array. It has values, an

index, a dtype, and optionally a name.

  

``` python

import pandas as pd

  

s = pd.Series([10, 20, 30], index=["a", "b", "c"])

  

s.to_numpy()  # NumPy array([10, 20, 30])

s.index       # Index(['a', 'b', 'c'])

s.dtype       # int64

```

  

Series operations are generally vectorized:

  

``` python

s * 2

s[s > 15]

```

  

### 3.2 DataFrame

  

A DataFrame is a two-dimensional labeled table. A useful mental model is

a collection of Series sharing the same row index.

  

``` python

df = pd.DataFrame({

    "name": ["Alice", "Bob", "Carol"],

    "age": [25, 30, 35],

})

```

  

-   `df["name"]` returns a Series.

-   `df[["name"]]` returns a DataFrame.

-   `df.dtypes` returns a Series containing each column's dtype.

-   `df.shape` returns `(number_of_rows, number_of_columns)`, here

    `(3, 2)`.

  

### 3.3 The `[]` vs `[[]]` distinction

  

``` python

df["age"]       # Series: shape (n,)

df[["age"]]     # DataFrame: shape (n, 1)

```

  

A single column name selects a Series. A list of column names selects a

DataFrame, even if the list contains only one name.

  

Why it matters:

  

-   Many scikit-learn estimators expect `X` to be two-dimensional. Use

    `df[["age"]]` for a one-feature input matrix; `df["age"]` is

    one-dimensional.

-   `df["new"] = values` creates or replaces one column.

-   `df.drop(["a", "b"], axis=1)` drops columns. `df.drop("a")` defaults

    to dropping a row label; use `axis=1` or `columns="a"` to drop a

    column.

  

### 3.4 The index is a label, not necessarily a row number

  

When pandas combines Series, it usually aligns values by index label

rather than by position.

  

``` python

s1 = pd.Series([1, 2, 3], index=["a", "b", "c"])

s2 = pd.Series([10, 20, 30], index=["c", "a", "b"])

  

s1 + s2

# a    21  (1 + 20)

# b    32  (2 + 30)

# c    13  (3 + 10)

```

  

This is useful because calculations can remain correct when rows are

reordered, provided the labels represent the same observations. If you

intend positional operations, make sure the indexes are aligned as

intended; reset or explicitly align indexes when appropriate.

  

### 3.5 Loading CSV-like datasets

  

``` python

df = pd.read_csv(

    path,

    header=None,

    names=COLUMNS,

    na_values=["?"],

)

```

  

-   `header=None`: the file does not contain a header row.

-   `names=COLUMNS`: use the supplied column names.

-   `na_values=["?"]`: interpret `?` as a missing value.

  

For the NSL-KDD `KDDTrain+` file, the standard schema contains 41

features, a `label` column, and a `difficulty` column: 43 columns total.

Verify the exact file variant and schema you downloaded before assigning

names.

  

### 3.6 `.loc` vs `.iloc`

  

``` python

df.loc[0:2]          # label-based; includes label 2

df.iloc[0:2]         # position-based; positions 0 and 1

df.loc[df["x"] > 0]  # boolean filter

df.loc[:, "col"]     # all rows, column named "col"

df.iloc[:, 1]        # all rows, second column by position

```

  

**Rule:** - `.loc` selects by labels. - `.iloc` selects by integer

positions. - Label slices with `.loc` include the endpoint; positional

slices with `.iloc` exclude it, like normal Python slicing.

  

------------------------------------------------------------------------

  

## Part 4 --- GroupBy and Merge

  

### 4.1 GroupBy: split → apply → combine

  

``` python

df.groupby("dept")["salary"].mean()

```

  

The operation follows three conceptual steps:

  

1.  **Split:** divide rows into groups based on `dept`.

2.  **Apply:** calculate the mean salary within each group.

3.  **Combine:** return the group results, indexed by department.

  

Common forms:

  

``` python

# One column, one aggregation

df.groupby("dept")["salary"].mean()

  

# One column, multiple aggregations

df.groupby("dept")["salary"].agg(["mean", "min", "max"])

  

# Multiple columns with different aggregations

df.groupby("dept").agg({

    "salary": ["mean", "max"],

    "bonus": "sum",

})

```

  

### 4.2 `.agg()` vs `.transform()`

  

  -----------------------------------------------------------------------

  Method                  Typical result shape    Use it when

  ----------------------- ----------------------- -----------------------

  `.agg(fn)`              One result per group    You want a summary

                                                  table

  

  `.transform(fn)`        Same number of rows as  You want a group

                          the input group         statistic attached to

                          selection               each original row

  -----------------------------------------------------------------------

  

``` python

# One result per department

df.groupby("dept")["salary"].agg("mean")

  

# One value per original row

df["dept_mean"] = (

    df.groupby("dept")["salary"].transform("mean")

)

```

  

Feature engineering often uses `transform()` when each original row

needs its group's statistic.

  

### 4.3 Merge: the pandas equivalent of a SQL JOIN

  

``` python

merged = pd.merge(left, right, on="id", how="inner")

```

  

  -----------------------------------------------------------------------

  `how`                               Meaning

  ----------------------------------- -----------------------------------

  `"inner"`                           Keep matching keys from both tables

  

  `"left"`                            Keep all left rows; unmatched

                                      right-side values become missing

  

  `"right"`                           Keep all right rows; unmatched

                                      left-side values become missing

  

  `"outer"`                           Keep keys from both tables

  -----------------------------------------------------------------------

  

**Potential silent row loss:** An inner merge removes rows whose keys do

not match. Check row counts and unmatched keys when that matters.

  

**Dtype trap:** `"1"` (string) and `1` (integer) are different values.

Check join-key dtypes before merging:

  

``` python

assert left["id"].dtype == right["id"].dtype, (

    "Join-key dtypes differ"

)

```

  

Matching dtypes are not sufficient by themselves: also check key

formatting, whitespace, missing keys, and duplicate keys.

  

### 4.4 GroupBy and merge together

  

To attach group statistics to every row:

  

``` python

# Option A: transform — concise, preserves row alignment

df["group_mean"] = (

    df.groupby("group")["value"].transform("mean")

)

  

# Option B: aggregate separately, then merge

stats = (

    df.groupby("group", as_index=False)["value"]

      .mean()

      .rename(columns={"value": "group_mean"})

)

  

df = df.merge(stats, on="group", how="left")

```

  

Use `transform()` when you want a same-length result aligned to the

original rows. Use a separate aggregation and merge when you also need

the summary table.

  

------------------------------------------------------------------------

  

## Part 5 --- Missing Values

  

### 5.1 What counts as missing?

  

Pandas recognizes values such as:

  

-   `None` in many pandas contexts.

-   `np.nan`.

-   `pd.NA`.

-   `pd.NaT` for missing datetime-like values.

  

Useful checks:

  

``` python

df.isna()        # Boolean DataFrame

df.isna().sum()  # Missing-value count per column

df.isna().mean() # Missing-value rate per column

```

  

These are useful first checks when inspecting a dataset.

  

Values such as `"?"`, `"NA"`, `"N/A"`, and `"-1"` are not automatically

missing in every context. They may be ordinary strings or valid domain

values. Decide what they mean for the dataset, then configure the loader

if appropriate:

  

``` python

pd.read_csv(path, na_values=["?", "NA", "N/A"])

```

  

A NumPy `int64` column cannot represent `np.nan`, so missing values may

cause it to be converted to `float64`. Pandas nullable integer dtypes

such as `"Int64"` can represent missing values without that conversion.

  

### 5.2 Drop vs fill --- make a reasoned decision

  

There is no universal rule that says to drop every row with a missing

target or every row with a missing ID. Decide based on the task and data

semantics.

  

A practical checklist:

  

1.  **Target/label missing:** You generally cannot use that row for

    supervised training because the target is unknown. You may retain it

    for inference or other analyses.

2.  **Identifier missing:** Determine whether the identifier is

    essential. An ID is often not a model feature; a missing ID does not

    automatically mean the entire row must be deleted.

3.  **Feature mostly missing:** Consider dropping the feature, imputing

    it, or adding a missingness indicator. The right choice depends on

    why it is missing and whether it carries useful signal.

4.  **Partially missing numeric feature:** Median imputation is a common

    robust baseline.

5.  **Partially missing categorical feature:** Mode imputation or an

    explicit `"unknown"` category may be reasonable.

6.  **Time series:** Forward fill or backward fill may be appropriate,

    but only when consistent with time ordering and the prediction

    setting.

  

Never drop rows silently. Log what changed:

  

``` python

before = len(df)

df_clean = df.dropna()

dropped = before - len(df_clean)

  

drop_rate = dropped / before * 100 if before else 0.0

  

logger.info(

    "Dropped %d rows (%.1f%%)",

    dropped,

    drop_rate,

)

```

  

### 5.3 The data-leakage trap

  

**Wrong:** Calculate an imputation statistic from the test set.

  

``` python

test_df["age"] = test_df["age"].fillna(

    test_df["age"].median()

)

```

  

This uses information from the test data to determine preprocessing

behavior. It makes evaluation less representative of how the model will

behave on genuinely unseen data.

  

**Right:** Learn the statistic from training data, then reuse it.

  

``` python

train_median = train_df["age"].median()

  

train_df["age"] = train_df["age"].fillna(train_median)

test_df["age"] = test_df["age"].fillna(train_median)

```

  

This is why a reusable preprocessor separates `fit()` from

`transform()`:

  

-   `fit(train_df)` learns parameters such as medians.

-   `transform(train_df)` and `transform(test_df)` reuse those

    parameters.

-   `transform()` must not recalculate statistics from the new input.

  

### 5.4 Missingness can itself be a feature

  

Sometimes the fact that a value is missing contains useful information.

  

For example, the absence of a measurement may be associated with a

process or outcome. A missingness indicator lets a model use that

signal.

  

``` python

# Create the indicator before imputation.

train["age_was_missing"] = (

    train["age"].isna().astype(int)

)

  

# Then fill the missing values.

train["age"] = train["age"].fillna(train_median)

```

  

**Order matters:** If you fill first, `.isna()` will no longer identify

the original missing values. Create the indicator before filling.

  

------------------------------------------------------------------------

  

## Part 6 --- Feature Engineering

  

### 6.1 Ratios: create informative signals

  

``` python

df["bytes_per_sec"] = (

    df["src_bytes"] / (df["duration"] + 1)

)

```

  

Things to consider:

  

-   Adding `1` prevents a zero denominator in this particular

    expression, but it changes the meaning of the ratio. Choose a

    denominator rule that makes sense for the domain.

-   Keep the original columns unless you have a reason to remove them.

-   Inspect extreme values and non-finite results. A small denominator

    can produce very large ratios.

-   Check the result:

  

``` python

df["bytes_per_sec"].describe()

```

  

For robust handling, explicitly inspect:

  

``` python

np.isfinite(df["bytes_per_sec"]).all()

```

  

### 6.2 Binning

  

``` python

pd.cut(

    x,

    bins=[0, 18, 35, 55, 100],

    labels=["minor", "young", "mid", "senior"],

)

  

pd.qcut(x, q=5)  # approximately equal-frequency bins

```

  

Use binning when: - Domain knowledge gives meaningful boundaries. - You

have a reason to model a numeric feature as ranges. - You want to

represent a non-linear relationship using categories.

  

Avoid binning automatically. Tree-based models can often learn useful

split points directly.

  

**`qcut` gotcha:** Repeated values can make it impossible to form the

requested number of distinct quantile bins. You can investigate

`duplicates="drop"` or use domain-based fixed bins with `cut()`.

Dropping duplicate edges may produce fewer bins than requested.

  

If bin boundaries are learned from data, learn them using training data

only and reuse them for validation and test data.

  

### 6.3 One-hot encoding

  

``` python

pd.get_dummies(df, columns=["protocol_type"])

```

  

Why encode? Integer codes such as `tcp=0`, `udp=1`, and `icmp=2` can

imply an order that does not exist. One-hot encoding represents each

category with a separate binary feature.

  

For example:

  

  protocol     protocol_tcp   protocol_udp   protocol_icmp

  ---------- -------------- -------------- ---------------

  tcp                     1              0               0

  udp                     0              1               0

  icmp                    0              0               1

  

`drop_first=True` can help avoid perfect multicollinearity in some

linear-model designs when an intercept is included. It is not

universally required, and it is generally unnecessary for tree models.

  

**Train/test consistency problem:**

  

``` python

train_oh = pd.get_dummies(train["service"])

test_oh = pd.get_dummies(test["service"])

  

# These may have different columns.

```

  

A simple alignment step is:

  

``` python

test_oh = test_oh.reindex(

    columns=train_oh.columns,

    fill_value=0,

)

```

  

A fitted encoder is usually a better reusable solution: learn the

categories from training data, then use the same output columns for

validation, test, and inference data. Unseen categories need an explicit

policy, such as all-zero encoding or an unknown category.

  

### 6.4 Date extraction

  

``` python

df["dt"] = pd.to_datetime(df["order_date"])

  

df["year"] = df["dt"].dt.year

df["month"] = df["dt"].dt.month

df["dayofweek"] = df["dt"].dt.dayofweek

df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

```

  

For periodic features such as month, hour, or day of week, cyclical

encoding can represent the wrap-around relationship:

  

``` python

df["month_sin"] = np.sin(

    2 * np.pi * df["month"] / 12

)

df["month_cos"] = np.cos(

    2 * np.pi * df["month"] / 12

)

```

  

December and January are adjacent in the calendar even though their

integer values are far apart. Sine and cosine represent the cycle on a

circle.

  

### 6.5 The fit-on-train rule

  

Any preprocessing step that **learns something from the data** must

learn it from the training split only.

  

  -----------------------------------------------------------------------

  Operation                           What is learned?

  ----------------------------------- -----------------------------------

  `src_bytes / (duration + 1)`        Nothing, if the formula is fixed in

                                      advance

  

  Quantile binning                    Bin boundaries, if calculated from

                                      data

  

  One-hot encoding                    Categories observed during fitting

  

  Median imputation                   Training median

  

  Standardization                     Training mean and standard

                                      deviation

  -----------------------------------------------------------------------

  

The workflow is:

  

``` text

Split data into train / validation / test

  

fit(train_df)

    learns preprocessing state

  

transform(train_df)

transform(validation_df)

transform(test_df)

    applies the same learned state

```

  

**Target leakage:** The target/label must not be included among the

input features. If `label` is one-hot encoded as an input feature, the

model receives the answer it is supposed to predict. This can produce

misleadingly high evaluation scores.

  

Keep the target separate:

  

``` python

y_train = train_df["label"]

X_train = train_df.drop(columns=["label"])

  

y_test = test_df["label"]

X_test = test_df.drop(columns=["label"])

```

  

If your `Preprocessor` supports an `exclude_cols` argument, use it to

exclude the target and any other columns that must not become features.

Otherwise, separate the target before calling `fit()`.

  

------------------------------------------------------------------------

  

## Part 7 --- The Preprocessor Contract

  

### 7.1 Design

  

The intended contract for a reusable preprocessor is:

  

``` text

fit(train_df):

    -> validates the input schema

    -> selects feature columns (excluding target/ignored columns)

    -> learns medians_, means_, stds_, and modes_

    -> learns missing_indicators_ using the configured threshold

    -> learns category_levels_ from training data

    -> learns output_columns_

    -> sets is_fitted = True

    -> returns self

  

transform(df):

    -> requires is_fitted

    -> checks that required feature columns are present

    -> creates missing indicators before imputation

    -> fills numeric values using learned medians

    -> fills categorical values using learned modes

    -> standardizes numeric columns using learned means/stds

    -> one-hot encodes using learned category levels

    -> aligns output to output_columns_

    -> returns a new DataFrame

```

  

The exact steps depend on which options are enabled. For example, if

categorical encoding is disabled, categorical columns may remain in the

output rather than being one-hot encoded.

  

### 7.2 Crucial properties

  

1.  **Stable output schema:** Every transformed batch should have the

    same output columns in the same order, regardless of which

    categories appear in that batch.

2.  **No input mutation:** `transform()` should start from a copy, such

    as `out = df.copy()`.

3.  **No statistic recomputation:** Imputation and standardization must

    use learned `self.*_` attributes, not statistics calculated from the

    data passed to `transform()`.

4.  **Fail clearly on misuse:** Calling `transform()` before `fit()`

    should raise `RuntimeError`. Missing required columns and invalid

    all-null features should produce clear errors.

5.  **Target separation:** The target must not accidentally become an

    input feature.

  

### 7.3 Important implementation checks

  

Before treating your preprocessor as complete, test these cases:

  

-   `transform()` before `fit()`.

-   Empty input and duplicate column names.

-   Missing required columns at transform time.

-   All-null numeric and categorical columns.

-   Constant numeric columns when standardizing.

-   Missing indicators are created before imputation.

-   Unseen categories at transform time.

-   Training and test output columns are identical.

-   The original input DataFrame is unchanged.

-   The target column is excluded from model features.

  

**Implementation note:** Keep this revision sheet aligned with the

actual code. If your current `Preprocessor` does not implement

`exclude_cols`, do not assume that passing `exclude_cols=["label"]`

works. Either add and test that parameter or remove it from the call and

separate `X` and `y` before preprocessing.

  

------------------------------------------------------------------------

  

## Part 8 --- Blank-Page Test

  

Close this sheet and try to answer these without looking.

  

### NumPy

  

1.  What is the difference between a Python list and a NumPy array?

2.  What does vectorization mean?

3.  What do `shape`, `ndim`, `size`, and `dtype` tell you?

4.  How do boolean masks work?

5.  What is the broadcasting compatibility rule?

  

### Pandas

  

6.  What is the difference between a Series and a DataFrame?

7.  What is the difference between `df["age"]` and `df[["age"]]`?

8.  Why does pandas align Series by index?

9.  When should you use `.loc` versus `.iloc`?

10. What is the difference between `groupby().agg()` and

    `groupby().transform()`?

11. What do the four merge types do?

  

### Data preparation

  

12. Which values does pandas recognize as missing, and which may need

    `na_values`?

13. When might you drop a row, drop a column, or impute a value?

14. What is data leakage?

15. Why must missing indicators be created before imputation?

16. Why must learned preprocessing statistics come from training data

    only?

17. Why must train and test one-hot encoded data have the same columns?

  

### Preprocessor

  

18. What belongs in `fit()`?

19. What belongs in `transform()`?

20. Why should `transform()` never recalculate medians, means, or

    category levels?

21. How should the preprocessor handle an unseen category?

22. Why must the target column be excluded from input features?

  

**Self-check:** If you can explain each answer and write a small example

without looking, you understand the core of Day 2. If not, revisit only

the sections you struggled to explain.