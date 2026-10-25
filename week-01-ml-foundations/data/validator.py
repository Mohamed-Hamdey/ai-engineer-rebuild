
"""Validation utilities: verify DataFrames before use."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .loader import DataError
from .logger import get_logger

logger = get_logger(__name__)


class ValidationError(DataError):
    """Raised when validation fails in strict mode."""


@dataclass
class ValidationReport:
    """Result of a validation pass.

    Collects errors and warnings. A validator that finds
    problems returns a report with passed=False unless strict=True.
    """

    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Record an error and mark validation as failed."""
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        """Record a warning without failing validation."""
        self.warnings.append(msg)


def _finish_validation(
    report: ValidationReport,
    validation_name: str,
    strict: bool,
) -> ValidationReport:
    """Log the result and optionally raise on validation failure."""

    if report.passed:
        logger.info("%s validation passed", validation_name)
    else:
        logger.warning(
            "%s validation failed with %d error(s)",
            validation_name,
            len(report.errors),
        )

        if strict:
            details = "; ".join(report.errors)
            raise ValidationError(
                f"{validation_name} validation failed: {details}"
            )

    return report


def validate_schema(
    df: pd.DataFrame,
    schema: dict[str, type],
    *,
    strict: bool = False,
) -> ValidationReport:
    """Check required columns and their expected data types.

    Common Python types are checked using pandas dtype helpers.
    This allows compatible types such as int64 and int32.
    For strings, object dtype is also accepted for compatibility,
    although object columns are not guaranteed to contain only strings.
    """

    report = ValidationReport()

    for col, expected_type in schema.items():
        if col not in df.columns:
            report.add_error(f"Missing required column: {col!r}")
            continue

        series = df[col]

        if expected_type is int:
            valid = pd.api.types.is_integer_dtype(series.dtype)

        elif expected_type is float:
            valid = pd.api.types.is_float_dtype(series.dtype)

        elif expected_type is str:
            valid = (
                pd.api.types.is_string_dtype(series.dtype)
                or pd.api.types.is_object_dtype(series.dtype)
            )

        elif expected_type is bool:
            valid = pd.api.types.is_bool_dtype(series.dtype)

        else:
            # For other expected types, compare against the pandas dtype.
            valid = series.dtype == expected_type

        if not valid:
            expected_name = getattr(
                expected_type, "__name__", str(expected_type)
            )
            report.add_error(
                f"Column {col!r} has dtype {series.dtype}; "
                f"expected {expected_name}"
            )

    return _finish_validation(report, "Schema", strict)


def validate_no_nulls(
    df: pd.DataFrame,
    required_cols: list[str],
    *,
    strict: bool = False,
) -> ValidationReport:
    """Ensure required columns exist and contain no null values."""

    report = ValidationReport()

    for col in required_cols:
        if col not in df.columns:
            report.add_error(f"Missing required column: {col!r}")
            continue

        null_count = int(df[col].isna().sum())

        if null_count > 0:
            report.add_error(
                f"Column {col!r} contains {null_count} null value(s)"
            )

    return _finish_validation(report, "Null", strict)


def validate_ranges(
    df: pd.DataFrame,
    rules: dict[str, tuple[Any, Any]],
    *,
    strict: bool = False,
) -> ValidationReport:
    """Check that numeric values fall within inclusive [low, high] bounds.

    Example:
        {"age": (0, 120), "price": (0, 10_000)}

    Null values are ignored by this function. Use validate_no_nulls()
    separately if null values are not allowed.
    """

    report = ValidationReport()

    for col, (low, high) in rules.items():
        if col not in df.columns:
            report.add_error(f"Missing required column: {col!r}")
            continue

        try:
            if low > high:
                report.add_error(
                    f"Invalid range for {col!r}: lower bound "
                    f"{low!r} exceeds upper bound {high!r}"
                )
                continue
        except TypeError:
            report.add_error(
                f"Invalid bounds for {col!r}: "
                f"{low!r} and {high!r} cannot be compared"
            )
            continue

        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            report.add_error(
                f"Column {col!r} must be numeric for range validation"
            )
            continue

        non_null = df[col].dropna()

        try:
            outside = (non_null < low) | (non_null > high)
        except (TypeError, ValueError):
            report.add_error(
                f"Column {col!r} cannot be compared with "
                f"the supplied range [{low!r}, {high!r}]"
            )
            continue

        outside_count = int(outside.sum())

        if outside_count > 0:
            report.add_error(
                f"Column {col!r} has {outside_count} value(s) "
                f"outside the inclusive range [{low}, {high}]"
            )

    return _finish_validation(report, "Range", strict)
