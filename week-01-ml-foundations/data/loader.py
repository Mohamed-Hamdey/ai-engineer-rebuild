"""File loading utilities with strong error handling."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator
import pandas as pd
from .logger import get_logger

logger = get_logger(__name__)

class DataError(Exception): ...      # or move to data/errors.py
class LoadError(DataError): ...

def _ensure_file(path: Path) -> None:
    if not path.is_file():
        raise LoadError(f"not an existing file: {path!r}")

def load_csv(path: Path) -> pd.DataFrame:
    _ensure_file(path)
    try:
        df = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as e:
        logger.exception("failed to load CSV %r", path)
        raise LoadError(f"could not load CSV {path!r}: {e}") from e
    logger.info("loaded CSV %r rows=%d cols=%d", path.name, len(df), df.shape[1])
    return df

def stream_jsonl(path: Path) -> Iterator[dict]:
    _ensure_file(path)
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue                       # skip blank lines
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as e:
                raise LoadError(
                    f"malformed JSON in {path!r} line {line_no}: {stripped!r}"
                ) from e

def load(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv(path)
    if suffix == ".jsonl":
        # load() returns a DataFrame, so materialize.
        # For huge JSONL files, callers use stream_jsonl directly.
        return pd.DataFrame(list(stream_jsonl(path)))
    raise LoadError(f"unsupported suffix: {suffix!r}")