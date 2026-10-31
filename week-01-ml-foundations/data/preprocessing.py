"""Preprocessing pipeline with fit/transform separation."""

from __future__ import annotations

import pandas as pd

from .logger import get_logger


logger = get_logger(__name__)


class Preprocessor:
    """Reusable preprocessing pipeline.

    fit():
        Learns statistics and output schema from training data.

    transform():
        Applies learned parameters without modifying the input.

    fit_transform():
        Fits and transforms the same dataset.
    """

    def __init__(
        self,
        fill_missing: bool = True,
        standardize: bool = False,
        encode_categoricals: bool = True,
        add_missing_indicators: bool = True,
        missing_threshold: float = 0.05,
        exclude_cols: list[str] | None = None,
    ):
        if not 0 <= missing_threshold <= 1:
            raise ValueError("missing_threshold must be between 0 and 1")

        self.fill_missing = fill_missing
        self.standardize = standardize
        self.encode_categoricals = encode_categoricals
        self.add_missing_indicators = add_missing_indicators
        self.missing_threshold = missing_threshold
        self.exclude_cols = exclude_cols or []

        # Fitted state
        self.is_fitted = False

        self.columns_ = None            # FEATURE columns only (after exclude_cols)
        self.numeric_cols_ = None
        self.categorical_cols_ = None

        self.medians_ = None
        self.means_ = None
        self.stds_ = None
        self.modes_ = None

        self.missing_rates_ = None
        self.missing_indicators_ = None

        self.category_levels_ = None
        self.onehot_columns_ = None
        self.output_columns_ = None

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Learn preprocessing parameters from training data."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.empty:
            raise ValueError("Cannot fit on an empty DataFrame")

        if df.columns.duplicated().any():
            raise ValueError("Duplicate column names are not supported")

        # Validate exclusion list against the input BEFORE dropping
        missing_excludes = [c for c in self.exclude_cols if c not in df.columns]
        if missing_excludes:
            raise ValueError(f"exclude_cols not in df: {missing_excludes}")

        # All feature-level operations use `features`, NOT `df`.
        # This is what prevents target leakage when `label` is excluded.
        features = df.drop(columns=self.exclude_cols)

        # Invalidate old fitted state before learning new parameters.
        self.is_fitted = False

        # 1. Identify supported feature types
        numeric = features.select_dtypes(include="number")
        categorical = features.select_dtypes(
            include=["object", "category", "string", "bool"]
        )

        self.columns_ = features.columns.tolist()
        # `columns_` holds the FEATURE columns only (after exclude_cols removed).
        # transform() accepts the full df but selects only these columns.

        self.numeric_cols_ = numeric.columns.tolist()
        self.categorical_cols_ = categorical.columns.tolist()

        unsupported = [
            col for col in features.columns
            if col not in self.numeric_cols_ + self.categorical_cols_
        ]

        if unsupported:
            raise TypeError(
                f"Unsupported column types: {unsupported}"
            )

        if not self.numeric_cols_ and not self.categorical_cols_:
            raise ValueError("No supported feature columns found")

        # 2. Initialize learned statistics
        self.medians_ = {}
        self.means_ = {}
        self.stds_ = {}
        self.modes_ = {}

        self.category_levels_ = {}
        self.onehot_columns_ = []

        # 3. Learn missing rates — FEATURES only
        self.missing_rates_ = features.isna().mean().to_dict()

        # 4. Select missing indicators
        if self.add_missing_indicators:
            self.missing_indicators_ = [
                col
                for col, rate in self.missing_rates_.items()
                if rate > self.missing_threshold
            ]
        else:
            self.missing_indicators_ = []

        # 5. Learn numerical imputation statistics
        if self.fill_missing:
            for col in self.numeric_cols_:
                median = numeric[col].median()

                if pd.isna(median):
                    raise ValueError(
                        f"All-null numeric column: {col!r}"
                    )

                self.medians_[col] = float(median)

            # 6. Learn categorical imputation statistics
            for col in self.categorical_cols_:
                mode = categorical[col].mode(dropna=True)

                if mode.empty:
                    raise ValueError(
                        f"All-null categorical column: {col!r}"
                    )

                self.modes_[col] = mode.iloc[0]

        # 7. Learn standardization statistics
        if self.standardize:
            for col in self.numeric_cols_:
                series = numeric[col].copy()

                if self.fill_missing:
                    series = series.fillna(self.medians_[col])

                if series.isna().any():
                    raise ValueError(
                        f"Column {col!r} contains missing values. "
                        "Enable fill_missing or handle them first."
                    )

                mean = series.mean()
                std = series.std(ddof=0)

                if not pd.notna(mean) or not pd.notna(std):
                    raise ValueError(
                        f"Invalid statistics in column {col!r}"
                    )

                self.means_[col] = float(mean)
                self.stds_[col] = float(std) if std != 0 else 1.0

        # 8. Learn categorical levels for one-hot encoding
        if self.encode_categoricals:
            for col in self.categorical_cols_:
                # sorted() for reproducibility across fit() calls
                levels = sorted(categorical[col].dropna().unique().tolist())

                self.category_levels_[col] = levels

                for level in levels:
                    self.onehot_columns_.append(
                        f"{col}__{level}"
                    )

        # 9. Learn the exact output schema
        output_columns = list(self.numeric_cols_)

        if not self.encode_categoricals:
            output_columns += self.categorical_cols_

        output_columns += [
            f"{col}_was_missing"
            for col in self.missing_indicators_
        ]

        if self.encode_categoricals:
            output_columns += self.onehot_columns_

        if len(output_columns) != len(set(output_columns)):
            raise ValueError(
                "Generated feature names contain duplicates. "
                "Rename conflicting input columns."
            )

        self.output_columns_ = output_columns

        # 10. Mark fitted
        self.is_fitted = True

        logger.info(
            "Preprocessor fitted: rows=%d numeric=%d categorical=%d output=%d",
            len(features),
            len(self.numeric_cols_),
            len(self.categorical_cols_),
            len(self.output_columns_),
        )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned preprocessing parameters to new data."""

        if not self.is_fitted:
            raise RuntimeError(
                "Preprocessor must be fitted before transform."
            )

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        if df.columns.duplicated().any():
            raise ValueError("Duplicate column names are not supported")

        # 1. Validate input schema — require feature columns,
        # but tolerate extra columns (e.g. exclude_cols, target) in the input.
        required = set(self.columns_)
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing expected columns: {sorted(missing)}"
            )

        # Preserve training column order and select only features
        out = df[self.columns_].copy()

        # 2. Add missing indicators BEFORE imputation
        for col in self.missing_indicators_:
            out[f"{col}_was_missing"] = (
                out[col].isna().astype(int)
            )

        # 3. Fill missing numerical values
        if self.fill_missing:
            for col, median in self.medians_.items():
                out[col] = out[col].fillna(median)

        # 4. Fill missing categorical values
        if self.fill_missing:
            for col, mode in self.modes_.items():
                out[col] = out[col].fillna(mode)

        # 5. Standardize numerical values
        if self.standardize:
            for col in self.numeric_cols_:
                out[col] = (
                    out[col] - self.means_[col]
                ) / self.stds_[col]

        # 6. One-hot encode categorical features
        if self.encode_categoricals:
            for col, levels in self.category_levels_.items():

                # Detect categories not observed during fit
                unseen = (
                    out[col].notna()
                    & ~out[col].isin(levels)
                )

                if unseen.any():
                    logger.warning(
                        "Column %r contains %d unseen categories",
                        col,
                        int(unseen.sum()),
                    )

                for level in levels:
                    out[f"{col}__{level}"] = (
                        out[col] == level
                    ).astype(int)

            # Remove original categorical columns
            out = out.drop(
                columns=self.categorical_cols_
            )

        # 7. Enforce exact training output schema
        out = out.reindex(
            columns=self.output_columns_,
            fill_value=0,
        )

        logger.info(
            "Transformed rows=%d cols=%d",
            len(out),
            out.shape[1],
        )

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessor and transform the training data."""

        return self.fit(df).transform(df)