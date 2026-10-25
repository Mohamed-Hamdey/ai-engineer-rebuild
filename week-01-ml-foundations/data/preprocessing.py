"""Preprocessing pipeline with fit/transform separation."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .logger import get_logger


logger = get_logger(__name__)


class Preprocessor:
    """Reusable preprocessing pipeline.

    Contract:
      - fit(df) learns parameters from df, stores them as *_ attrs,
        returns self
      - transform(df) applies the LEARNED parameters, returns a new df
      - fit_transform(df) = fit(df).transform(df)
      - transform on an unfitted instance raises RuntimeError

    For today's minimum version, implement at least:
      - fill_missing: learn the median per column during fit;
        fill NaNs with it during transform
    Stretch (only if quick):
      - standardize: learn mean/std per column during fit;
        z-score during transform
    """
    def __init__(self, fill_missing: bool = True, standardize: bool = False):
        self.fill_missing = fill_missing
        self.standardize = standardize
        self.is_fitted = False
        # TODO: prepare state attributes (e.g. self.medians_: dict | None)
        self.columns_: list[str] | None = None
        self.medians_: dict[str, float] | None = None
        self.means_: dict[str, float] | None = None
        self.stds_: dict[str, float] | None = None
    def fit(self, df):
        numeric = df.select_dtypes(include="number")
        self.columns_ = list(numeric.columns)
        self.medians_, self.means_, self.stds_ = {}, {}, {}
        if self.fill_missing:
            for col in self.columns_:
                median = numeric[col].median()
                # fail fast on all-null columns
                if pd.isna(median) and numeric[col].isna().any():
                    raise ValueError(f"all-null column {col!r}")
                self.medians_[col] = float(median)
        if self.standardize:
            for col in self.columns_:
                series = numeric[col]
                if self.fill_missing:
                    series = series.fillna(self.medians_[col])
                if series.isna().any():
                    raise ValueError(f"column {col!r} has nulls")
                self.means_[col] = float(series.mean())
                std = float(series.std(ddof=0))
                self.stds_[col] = std if std != 0 else 1.0
        self.is_fitted = True                     # ← OUTSIDE both ifs
        logger.info("Preprocessor fitted rows=%d cols=%d", len(df), len(self.columns_))
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("preprocessor must be fit before transform")
        out = df.copy()
        missing = [c for c in self.columns_ if c not in out.columns]   # self.columns_, not out.columns
        if missing:
            raise ValueError(f"missing columns: {missing!r}")
        if self.fill_missing:
            for col, median in self.medians_.items():
                out[col] = out[col].fillna(median)
        if self.standardize:
            for col in self.columns_:
                out[col] = (out[col] - self.means_[col]) / self.stds_[col]
        logger.info("Transformed %d rows", len(out))
        return out
                
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit then transform."""
        # TODO: one-liner
        return self.fit(df).transform(df)