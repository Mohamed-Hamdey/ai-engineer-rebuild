# 📘 Day 1 Revision Sheet — Python Engineering (Low-Level, Logic-First)

**How to use this:** Read it once slowly. Then close it and try to explain each section out loud. Anything you can't explain, re-read that section. Then do the **Blank Page Test** at the end.

This is not a cheat sheet — it's a thinking sheet. Every section answers: *"why does this exist, and what breaks without it?"*

---

## PART 1 — OOP: Why Classes Exist

### 1.1 The core idea

A class bundles **state** (data that persists) + **behavior** (functions that use that data) into one object.

A function is just behavior. No persistent memory between calls.

### 1.2 When is a class the right tool?

Ask: **"Does this thing need to REMEMBER something between calls?"**

- `validate_schema(df, schema)` — takes input, returns report. No memory needed. → **Function** ✅
- `Preprocessor` — must remember the median it computed from training data, so `transform()` on test data can reuse it. Memory needed. → **Class** ✅
- `load_csv(path)` — takes path, returns df. No memory. → **Function** ✅
- `DataLoader` (later, Day 7) — remembers source, config, retries, stats. Memory. → **Class**

**The rule:** If the function has nothing to remember between calls, it's a function. If it must remember learned state, it's a class.

### 1.3 The anatomy of a class

```python
class Preprocessor:
    def __init__(self, fill_missing=True):      # ← runs ONCE when created
        self.fill_missing = fill_missing        # ← attribute (state)
        self.medians_ = None                    # ← placeholder for learned state
        self.is_fitted = False                  # ← flag: has fit() run yet?

    def fit(self, df):                          # ← LEARNS, stores on self
        self.medians_ = {...}                   # ← now the object REMEMBERS
        self.is_fitted = True
        return self                             # ← return self so you can chain

    def transform(self, df):                    # ← APPLIES what fit learned
        if not self.is_fitted:                  # ← guard against misuse
            raise RuntimeError("fit first")
        return df.copy()                        # ← never mutate input
```

**Line-by-line logic:**

| Line | What it does | Why |
| --- | --- | --- |
| `__init__` | Sets up the object | Runs once at `Preprocessor()` |
| `self.x = ...` | Stores on the instance | Persists across method calls |
| `fit` | Learns from data, stores as `self.xxx_` | This is what makes it a class, not a function |
| `return self` | Returns the object itself | Enables `p.fit(df).transform(df)` chaining |
| `transform` guard | Checks `is_fitted` | Prevents silent wrong output if user forgets `fit` |
| `df.copy()` | Never mutate caller's data | Side-effect-free transform = predictable |

### 1.4 The trailing `_` convention

`self.medians_` (with underscore) means: *"learned during `fit`, not passed in by the caller."*

- `self.fill_missing` — set by user, config
- `self.medians_` — learned from data, parameter
- `self.is_fitted` — status flag

sklearn uses this convention everywhere. Recognize it on sight.

### 1.5 Why `fit_transform` exists

```python
def fit_transform(self, df):
    return self.fit(df).transform(df)
```

- `fit` alone: on training data
- `transform` alone: on test data (using medians learned from train)
- `fit_transform`: on training data only (learn + apply in one shot)

**The critical rule: NEVER fit on test data.** That's data leakage. The whole reason `fit` and `transform` are separate methods is so you can fit on train, then transform test with the train-learned params.

---

## PART 2 — Typing: Why Annotations Exist

### 2.1 What typing is (and isn't)

**Is:** documentation that humans, editors, and tools (mypy) read.
**Isn't:** runtime enforcement. Python won't stop you from passing a string where an `int` is annotated.

```python
def add(a: int, b: int) -> int:
    return a + b

add("x", "y")        # runs fine! returns "xy". Types did nothing at runtime.
```

### 2.2 The 5 annotations you actually use

```python
from collections.abc import Callable
import pandas as pd

def f(x: int) -> int: ...                           # 1. simple
def f(name: str | None = None) -> str | None: ...   # 2. optional (str OR None)
def f(rows: list[dict[str, float]]) -> None: ...    # 3. containers
def f(fn: Callable[[int], int], x: int) -> int: ... # 4. callable
def f(df) -> tuple[pd.DataFrame, pd.DataFrame]: ... # 5. tuple of N
```

`str | None` means: "this argument can be a `str` or `None`." Older syntax: `Optional[str]`. Both mean the same.

### 2.3 Why bother

- Editor autocomplete uses types to guess what methods are available
- mypy catches `add("x", "y")` at check time (not runtime)
- Future-you in 3 weeks sees `load(path: Path) -> pd.DataFrame` and knows exactly what to pass and expect — without reading the body

**Rule for the roadmap:** annotate every public function. Skip types on 3-line local helpers. Don't chase `TypeVar`/`Generic`/`Protocol` — you'll see them Day 5+.

---

## PART 3 — Decorators: Wrapping Functions

### 3.1 The one-line definition

A decorator is a function that takes a function and returns a function — with extra behavior layered on top.

```python
@deco
def f(): ...

# is EXACTLY equivalent to:
f = deco(f)
```

### 3.2 The template you memorize

```python
import functools

def decorator_name(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # ← BEFORE behavior (optional)
        result = fn(*args, **kwargs)
        # ← AFTER behavior (optional)
        return result
    return wrapper
```

**Line-by-line logic:**

- `def decorator_name(fn):` — takes the target function
- `@functools.wraps(fn)` — copies `fn.__name__`, `fn.__doc__` onto `wrapper` (see §3.4)
- `def wrapper(*args, **kwargs):` — accepts any arguments, forwards to `fn`
- `result = fn(*args, **kwargs)` — actually calls the original function
- `return result` — pass the result through to the caller
- `return wrapper` — **critical:** without this, the decorator returns `None` and the decorated function becomes uncallable

### 3.3 The flow of `@log_execution_time`

```python
@log_execution_time
def slow():
    time.sleep(1)
```

1. Python sees `@log_execution_time` above `def slow`
2. It reads the `def`, creating `slow` as a normal function
3. Immediately calls `log_execution_time(slow)` and rebinds `slow` to the return value
4. The return value is `wrapper`, so `slow` is now `wrapper`
5. When the user later calls `slow()`, they're calling `wrapper()`, which times and forwards to the original `slow` (captured in closure as `fn`)

### 3.4 Why `@functools.wraps(fn)` — the whole reason

Without it:

```python
def deco(fn):
    def wrapper(): return fn()
    return wrapper

@deco
def my_fn(): ...

print(my_fn.__name__)   # prints "wrapper" — WRONG
print(my_fn.__doc__)    # prints None — WRONG
```

With it:

```python
def deco(fn):
    @functools.wraps(fn)
    def wrapper(): return fn()
    return wrapper

@deco
def my_fn(): ...

print(my_fn.__name__)   # prints "my_fn" — CORRECT
```

Why it matters:

- logging, `help()`, mypy, IDE hover — all read `__name__`/`__doc__`
- If they see `"wrapper"` for every decorated function, debugging becomes hell

**Rule:** if you write a decorator, you use `@functools.wraps(fn)` inside. Non-negotiable.

### 3.5 `*args, **kwargs` — how the wrapper works for any function

```python
def wrapper(*args, **kwargs):
    return fn(*args, **kwargs)
```

- `*args` packs positional args into a tuple
- `**kwargs` packs keyword args into a dict
- `fn(*args, **kwargs)` unpacks them back into positional/keyword form

So the wrapper works whether the wrapped function takes 0, 1, or 10 arguments.

---

## PART 4 — Generators: Lazy Iteration

### 4.1 The core difference

```python
def all_rows(path):
    return [json.loads(l) for l in open(path)]     # ← builds full list in memory NOW

def stream_rows(path):
    for line in open(path):
        yield json.loads(line)                     # ← yields one at a time
```

- `return [...]` → builds the entire list immediately. Memory = O(N).
- `yield` → produces one item per request. Memory = O(1).

### 4.2 How `yield` actually works

```python
def stream():
    print("start")
    yield 1
    print("middle")
    yield 2
    print("end")

it = stream()      # ← nothing printed. No code runs yet. `it` is an iterator.
next(it)           # prints "start", returns 1, PAUSES at yield
next(it)           # prints "middle", returns 2, PAUSES
next(it)           # prints "end", raises StopIteration
```

**The mental model:** `yield` is a pause point. When you call `next()`, the function runs until the next `yield`, hands you that value, and freezes. The next `next()` resumes from the exact line.

### 4.3 The 3 gotchas

**Gotcha 1: You only get one pass.**

```python
it = stream_jsonl(path)
sum(1 for _ in it)      # consumes it fully
list(it)                # → [] — already exhausted!
```

Generators are single-use. If you need to iterate twice, either re-call the function or wrap in `list(...)` (losing laziness).

**Gotcha 2: Nothing runs until you iterate.**

```python
gen = stream_jsonl("missing_file.jsonl")     # ← NO error yet!
next(gen)                                    # ← now LoadError fires
```

This is why `_ensure_file` inside a generator only runs on the first `next()`. Fine for file loading (the caller iterates right after), but surprising if you expect eager validation.

**Gotcha 3: You can't usefully return a value from a generator.** `return value` in a generator attaches the value to `StopIteration`, and a `for` loop silently ignores it — almost never what you want. Use bare `return` to exit early.

### 4.4 The `yield` placement bug (from your loader)

```python
# WRONG
for line in f:
    if not line.strip():
        try:
            record = json.loads(line)
        except ...: ...
    yield record                     # ← yields OLD record for blank lines
```

```python
# RIGHT
for line in f:
    if not line.strip():
        continue                     # skip blank, don't yield
    try:
        yield json.loads(line)       # ← yield only on success, inside try
    except ...: ...
```

**The pattern:** `yield` sits on the happy path — the branch that succeeded. Blank lines are `continue`'d before reaching `yield`.

---

## PART 5 — Exceptions: Loud, Specific, Contextual

### 5.1 The exception hierarchy in your package

```python
class DataError(Exception): ...            # base for ALL data-package errors
class LoadError(DataError): ...            # file can't be read/parsed
class ValidationError(DataError): ...      # data fails a rule
```

**Why the base class exists:** callers can do:

```python
try:
    ...
except DataError as e:      # catches LoadError AND ValidationError
    ...
```

Without a base class, callers write `except (LoadError, ValidationError, SchemaError, ...)` — brittle, breaks when you add a new error type.

### 5.2 Raise: specific, loud, with context

```python
# BAD — swallowed, silent, useless
if col not in df.columns:
    return None

# BAD — bare, no context
raise Exception("bad")

# GOOD — specific subclass, path/line/values in message
raise LoadError(f"malformed JSON in {path!r} line {line_no}: {stripped!r}")
```

The `{x!r}` pattern wraps strings in quotes for the log message — so `" col"` (with a space) vs `"col"` becomes visible.

### 5.3 Catch: only what you can handle

```python
# BAD — catches everything, swallows bugs
try:
    df = load(path)
except Exception:
    pass

# GOOD — catches only what you can recover from
try:
    df = load(path)
except FileNotFoundError:
    logger.warning("missing file %r, using empty df", path)
    df = pd.DataFrame()
```

**Rule:** if you can't recover from this specific type, don't catch it. Let it bubble up — someone else (the caller, the framework) can handle it.

### 5.4 `raise ... from e` — exception chaining

```python
try:
    df = pd.read_csv(path)
except (OSError, pd.errors.ParserError) as e:
    raise LoadError(f"could not load CSV {path!r}: {e}") from e
```

`from e` keeps the original traceback as the "cause." You get both:

- Your clean `LoadError` message at the top
- The low-level `OSError` traceback below it (*"The above exception was the direct cause..."*)

Without `from e`, Python still shows the original as implicit context, but it is labeled *"During handling of the above exception, another exception occurred"* — which reads like a bug in your handler, not a deliberate wrap. **Always use `from e` when wrapping a lower-level error.**

### 5.5 `logger.exception` vs `logger.error`

- `logger.error("...")` — logs just the message
- `logger.exception("...")` — logs the message + current traceback (call inside an `except` block)

Use `exception` when you're in an `except` and want the stack for debugging. Use `error` for logical errors you raise yourself.

---

## PART 6 — Comprehensions: Read-Like-English Builders

### 6.1 The five forms (memorize these shapes)

```python
[expr for x in iterable]                  # list
[expr for x in iterable if cond]          # list with filter
{key: val for x in iterable}              # dict
{expr for x in iterable}                  # set
(expr for x in iterable)                  # generator (lazy)
```

### 6.2 Read them aloud

```python
[c for c in df.columns if df[c].isna().any()]
```

*"A list of `c`, for each `c` in `df.columns`, if `df[c]` has any nulls."*

That's the whole pattern: `[output_expr for item in source if condition]`.

### 6.3 When NOT to use

**Never nest more than one level.**

```python
# DON'T — unreadable
[[x for x in row if x] for row in matrix if row]

# DO
result = []
for row in matrix:
    if not row:
        continue
    result.append([x for x in row if x])
```

**Rule:** if the comprehension needs a comment, write a loop.

### 6.4 Generator expressions (the lazy cousin)

```python
total = sum(x*x for x in range(10**6))       # no list built
```

Same syntax with `()` instead of `[]`. Use when feeding a `sum`/`max`/`any`/`all` — avoids building an intermediate list.

---

## PART 7 — Logging: The Flight Recorder

### 7.1 The pattern

```python
import logging
logger = logging.getLogger(__name__)        # at top of EVERY module

logger.debug("detail %s", x)
logger.info("loaded rows=%d", n)
logger.warning("missing %d values", k)
logger.error("validation failed: %s", err)
```

### 7.2 Why `__name__`

`logging.getLogger(__name__)` inside `data/loader.py` gives you a logger named `"data.loader"`. Now:

- Every log line from that module prints `name=data.loader`
- You can filter in production: "show only warnings from `data.loader`"
- When debugging, you instantly know which file logged it

If you used `logging.getLogger("root")` everywhere, every log line is anonymous. Don't.

### 7.3 Why `%s`-style args, not f-strings

```python
logger.info("loaded rows=%d", n)             # ← GOOD
logger.info(f"loaded rows={n}")              # ← BAD (style)
```

The difference:

- **`%`-style version:** `"loaded rows=%d"` is a constant. `n` is passed separately. Logging formats the message only if the log level is enabled.
- **f-string version:** `f"loaded rows={n}"` is formatted *before* `logger.info` even gets called — even if the log level is disabled and the message is thrown away.

In tight loops, f-string logging wastes cycles. The `%` style is nearly free when logging is off. It's the convention in production Python. Get used to it.

### 7.4 The `if not logger.handlers` guard

```python
def get_logger(name, level=logging.INFO):
    log = logging.getLogger(name)
    log.setLevel(level)
    if not log.handlers:                     # ← THE GUARD
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        log.addHandler(handler)
    return log
```

**Why the guard:** `logging.getLogger("x")` caches loggers by name. Two calls to `get_logger("x")` return the same object. Without the guard, the second call adds a second handler → every log line prints twice → third call prints three times → etc.

**The mental model:** `getLogger(name)` is idempotent (returns the same object). Attaching a handler is not idempotent (appends every time). The guard makes the whole function idempotent.

**Test:** `get_logger("x") is get_logger("x")` → `True`. And you should see each log line once, no matter how many times you call `get_logger("x")`.

### 7.5 The format string decomposed

```text
%(asctime)s | %(levelname)-8s | %(name)s | %(message)s
```

| Field | Meaning | Example |
| --- | --- | --- |
| `%(asctime)s` | timestamp | `2026-10-25 14:30:00,123` |
| `%(levelname)-8s` | level, left-padded to 8 chars | `INFO` / `WARNING` |
| `%(name)s` | logger name | `data.loader` |
| `%(message)s` | the actual message | `loaded CSV 1000 rows` |

Result: aligned columns, easy to grep, easy to read in a log dump.

---

## PART 8 — Virtual Environments & Packages

### 8.1 Venv — the "isolated toolbox" idea

**Problem:** Project A needs `numpy==1.20`, Project B needs `numpy==2.1`. System-wide pip can hold only one version. One project breaks.

**Solution:** Each project gets its own `site-packages` directory inside `.venv/`. `pip install` while activated writes there. Activated Python reads from there. Project A and B can't see each other.

**The workflow (memorize):**

```bash
python -m venv .venv                    # create
source .venv/bin/activate               # activate (Mac/Linux)
.venv\Scripts\activate                  # activate (Windows)
# → prompt shows (.venv)
pip install -r requirements.txt         # install
pip freeze > requirements.txt           # snapshot exact versions
deactivate                              # leave
```

**Never commit `.venv/`.** Add it to `.gitignore`. It's rebuildable from `requirements.txt`.

### 8.2 Packages — the `__init__.py` rule

A folder with `__init__.py` is a package. That's the entire requirement. Even an empty file makes it a package.

`__init__.py` does two jobs:

**Job 1 (structural):** marks the folder as importable.

**Job 2 (API surface):**

```python
# data/__init__.py
from .loader import load, LoadError
from .validator import validate_schema, ValidationReport
from .logger import get_logger

__all__ = ["load", "LoadError", "validate_schema", "ValidationReport", "get_logger"]
```

Now callers do:

```python
from data import load, get_logger       # ← clean
```

Instead of:

```python
from data.loader import load            # ← exposes internal structure
```

Think of `__init__.py` as the **storefront**. The other modules are the **stockroom**. Only the storefront is public.

**Relative vs absolute imports:**

- Inside `data/loader.py`, import siblings: `from .logger import get_logger` (the `.` means "same package")
- From outside, e.g. `scratch.py`: `from data import load` (absolute, uses package name)

---

## PART 9 — The Design Decisions (and why)

| Decision | Your choice | Why it's right |
| --- | --- | --- |
| Where does `DataError` live? | `loader.py` (or `errors.py`) | Base class must be importable by all siblings. `errors.py` is cleanest for a growing package. |
| Module-level logger in `logger.py`? | `logger = logging.getLogger(__name__)` | The `@log_execution_time` decorator needs some logger — using the module's own name (`data.logger`) is consistent with the "logger per module" convention. |
| `.jsonl` in `load()` | Materialize with `list(stream_jsonl(...))` | `load()` is documented to return `pd.DataFrame` — DataFrames are in-memory. Callers who want streaming should call `stream_jsonl` directly. |
| Blank lines in `stream_jsonl` | `continue` (skip silently) | Hand-edited JSONL often has trailing/blank lines. Silent skip is more ergonomic than raising. Add a comment: `# skip blank lines — common in hand-edited files`. |
| Strict or lenient dtype check | Strict (via `is_integer_dtype`, etc.) | Safer. Won't pass `int` when you expected `float` — catches bad pipelines early. |
| NaN handling in `validate_ranges` | Drop NaNs, only check non-null | Missing values are a separate concern (`validate_no_nulls`). Range check shouldn't fail on legitimate nulls. |

These are **choices, not truths.** In a different codebase, some would flip. The point is: you made them, and you can defend them. That's the difference between a code monkey and an engineer.

---

## PART 10 — The Bugs You Fixed (and the pattern behind each)

| Bug | What was wrong | The pattern to remember |
| --- | --- | --- |
| `import logging, StreamHandler` | Imported a non-existent module | After imports, mentally run "does every name I imported exist?" |
| `logger = logging.Logger` | Assigned class, not instance | "Class vs instance" — always ask: is this name holding the class or an object of the class? |
| `name = logging.getLogger(name)` | Rebound the parameter | Never reuse a parameter name for a different-typed value. Use a new name (`log`). |
| `level = logger.level` | Ignored the function's `level` param | Every parameter you accept is a parameter you should use — else delete it. |
| `json.load` vs `json.loads` | Wrong function | `load` takes a file, `loads` takes a string. Same for `dump`/`dumps`. |
| `yield record` outside `if` | Yielded uninitialized or stale value | `yield` goes on the happy path — inside the `try`, after the transformation. |
| `time.process_time()` for elapsed | Measured CPU time, not wall-clock | `perf_counter()` for durations. `process_time()` for CPU profiling. |
| Missing `return wrapper` | Decorator returned `None` | Every function that has a `def inner` must `return inner` — check the last line. |
| `if col not in df[col]` | Checked membership in a Series | `if X not in df.columns` — `df[col]` raises `KeyError` before the `in` even runs. |
| `is_fitted = True` inside `if standardize` | Only set when `standardize` enabled | Set flags at the end of the method, outside all conditionals. |
| `[c for c in out.columns if c not in out.columns]` | Compared a column list to itself | Trace every variable: where does it come from? What does it represent? |
| `from http.cookiejar import LoadError` | Imported unrelated `LoadError` | After imports, verify each name is what you expect — especially for exception classes. |

---


