"""Logging utilities for the data package."""
import functools
import logging
import time

logger = logging.getLogger(__name__)     # decided: module-level logger

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

def get_logger(name, level=logging.INFO):
    log = logging.getLogger(name)
    log.setLevel(level)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        log.addHandler(handler)
    return log

def log_execution_time(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info("%s executed in %.4f seconds", fn.__name__, elapsed)
        return result
    return wrapper     # ← don't forget this