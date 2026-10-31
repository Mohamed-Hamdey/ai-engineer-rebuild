
"""File loading utilities with strong error handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Any

import pandas as pd

from .logger import get_logger


logger = get_logger(__name__)


class DataError(Exception):
    """Base exception for data-related errors."""


class LoadError(DataError):
    """Raised when a file cannot be loaded."""


def _ensure_file(path: Path) -> None:
    """Validate that the given path points to an existing file."""

    if not path.is_file():
        raise LoadError(f"not an existing file: {path!r}")


def load_csv(
    path: Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a CSV or comma-separated text file.

    Additional keyword arguments are passed to pandas.read_csv().
    """

    _ensure_file(path)

    try:
        df = pd.read_csv(path, **kwargs)

    except (OSError, UnicodeError, pd.errors.ParserError) as e:
        logger.exception("failed to load CSV %r", path)
        raise LoadError(
            f"could not load CSV {path!r}: {e}"
        ) from e

    logger.info(
        "loaded CSV %r rows=%d cols=%d",
        path.name,
        len(df),
        df.shape[1],
    )

    return df


def stream_jsonl(path: Path) -> Iterator[dict]:
    """Stream JSONL records one line at a time."""

    _ensure_file(path)

    try:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):

                stripped = line.strip()

                if not stripped:
                    continue

                try:
                    record = json.loads(stripped)

                    if not isinstance(record, dict):
                        raise LoadError(
                            f"expected JSON object in {path!r} "
                            f"line {line_no}"
                        )

                    yield record

                except json.JSONDecodeError as e:
                    raise LoadError(
                        f"malformed JSON in {path!r} "
                        f"line {line_no}: {e.msg}"
                    ) from e

    except (OSError, UnicodeError) as e:
        raise LoadError(
            f"could not read JSONL {path!r}: {e}"
        ) from e


def load(
    path: Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a supported dataset based on its file extension.

    Supported formats:
        .csv  -> CSV DataFrame
        .txt  -> comma-separated text DataFrame
        .jsonl -> JSON Lines DataFrame

    Extra keyword arguments are forwarded to pandas.read_csv()
    for CSV and TXT files.
    """

    path = Path(path)

    suffix = path.suffix.lower()

    _ensure_file(path)

    if suffix in (".csv", ".txt"):
        return load_csv(path, **kwargs)

    if suffix == ".jsonl":
        if kwargs:
            raise TypeError(
                "Additional arguments are not supported for JSONL"
            )

        return pd.DataFrame(list(stream_jsonl(path)))

    raise LoadError(f"unsupported suffix: {suffix!r}")
