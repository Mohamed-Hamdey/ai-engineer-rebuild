
"""Train / validation / test splitting with stratification."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .logger import get_logger

logger = get_logger(__name__)


def train_val_test_split(
    df: pd.DataFrame,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    stratify_col: str | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train, validation, and test DataFrames."""

    # 1. Validate split fractions.
    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be less than 1.")

    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")

    # 2. Validate the stratification column, if provided.
    if stratify_col is not None:
        if stratify_col not in df.columns:
            raise ValueError(
                f"Stratification column {stratify_col!r} not found."
            )

        if df[stratify_col].isna().any():
            raise ValueError(
                f"Stratification column {stratify_col!r} contains nulls."
            )

        class_counts = df[stratify_col].value_counts()
        if (class_counts < 3).any():
            raise ValueError(
                "Each class needs at least 3 rows for a three-way "
                "stratified split. Some classes have fewer."
            )

    # y contains the class labels used to preserve class balance.
    y = df[stratify_col] if stratify_col is not None else None

    # 3. Split off the test set.
    remaining_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # 4. Calculate validation fraction relative to the remainder.
    val_fraction = val_size / (1 - test_size)

    # Keep labels aligned with the rows remaining after the first split.
    remaining_y = (
        remaining_df[stratify_col]
        if stratify_col is not None
        else None
    )

    # 5. Split the remainder into training and validation sets.
    train_df, val_df = train_test_split(
        remaining_df,
        test_size=val_fraction,
        random_state=seed,
        stratify=remaining_y,
    )

    logger.info(
        "Dataset split complete: train=%s, validation=%s, test=%s",
        train_df.shape,
        val_df.shape,
        test_df.shape,
    )

    return train_df, val_df, test_df
