"""Data engineering package for the AI rebuild roadmap."""

from .loader import load, load_csv, stream_jsonl, LoadError
from .logger import get_logger, log_execution_time
from .preprocessing import Preprocessor
from .split import train_val_test_split
from .validator import validate_schema, validate_no_nulls, ValidationReport, ValidationError

__all__ = [
    "load", "load_csv", "stream_jsonl", "LoadError",
    "get_logger", "log_execution_time",
    "Preprocessor",
    "train_val_test_split",
    "validate_schema", "validate_no_nulls", "ValidationReport", "ValidationError",
]