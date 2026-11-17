"""SAS-inspired, eager-only audit logging for Polars DataFrames.

Designed for Python source files. Python 3.11+ recommended for source positions.
"""
from __future__ import annotations

import ast
import dis
import inspect
import linecache
import logging
import contextvars
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl

_LOG = logging.getLogger("logged_polars")
_LOG.addHandler(logging.NullHandler())
_LOG.propagate = False
_LOG.setLevel(logging.INFO)


def configure_logging(path: str = "processing.log", *, console: bool = False) -> None:
    """Replace only this package's handlers; start a new UTF-8 audit log."""
    for handler in _LOG.handlers[:]:
        _LOG.removeHandler(handler)
        handler.close()
    handlers = [logging.FileHandler(path, mode="w", encoding="utf-8")]
    if console:
        handlers.append(logging.StreamHandler())
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        _LOG.addHandler(handler)
    clear_source_cache()
    _last_call.set(None)


def clear_source_cache() -> None:
    linecache.checkcache()
    _parse_source.cache_clear()
    _instructions.cache_clear()


@lru_cache(maxsize=128)
def _parse_source(filename: str):
    lines = linecache.getlines(filename)
    if not lines:
        return None, []
    try:
        return ast.parse("".join(lines), filename=filename), lines
    except (SyntaxError, ValueError):
        return None, lines


@lru_cache(maxsize=256)
def _instructions(code):
    return {ins.offset: ins for ins in dis.get_instructions(code)}


@dataclass(frozen=True)
class Source:
    filename: str
    start: int
    end: int
    displayed: str
    rank: int
    method_line: int
    columns: tuple[int, int] = (0, 0)
    input_label: str | None = None
    output_label: str | None = None
    other_label: str | None = None

    @property
    def key(self):
        return (self.filename, self.start, self.end, self.columns)


def _site(frame, method_name: str) -> Source:
    """Resolve a CALL instruction to an AST method call and enclosing statement.

    Python 3.11+ bytecode column positions disambiguate multiple statements
    and repeated methods on the same physical source line.
    """
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    tree, lines = _parse_source(filename)
    inst = _instructions(frame.f_code).get(frame.f_lasti)
    pos = inst.positions if inst is not None else None
    if tree is None:
        text = linecache.getline(filename, lineno).rstrip() or "<source unavailable>"
        return Source(filename, lineno, lineno, f"{lineno:5d} | {text}", -1, lineno)

    call_line = pos.lineno if pos and pos.lineno is not None else lineno
    call_end_line = pos.end_lineno if pos and pos.end_lineno is not None else call_line
    call_col = pos.col_offset if pos else None
    call_end_col = pos.end_col_offset if pos else None

    def covers(node, line, col):
        if not (node.lineno <= line <= node.end_lineno):
            return False
        if col is None:
            return True
        if line == node.lineno and col < node.col_offset:
            return False
        if line == node.end_lineno and col >= node.end_col_offset:
            return False
        return True

    # Identify the method call at the actual interpreter position, not just
    # by method name or line number. The fallback supports older runtimes.
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == method_name]

    def call_score(node):
        exact_span = (
            call_col is not None and call_end_col is not None
            and node.lineno == call_line and node.col_offset == call_col
            and node.end_lineno == call_end_line
            and node.end_col_offset == call_end_col
        )
        same_end = (node.end_lineno == call_end_line and
                    call_end_col is not None and node.end_col_offset == call_end_col)
        contains = covers(node, call_line, call_col)
        return (0 if exact_span else 1 if same_end else 2 if contains else 3,
                abs(node.end_lineno - call_end_line),
                abs(node.end_col_offset - (call_end_col or node.end_col_offset)))

    method_call = min(calls, key=call_score) if calls else None
    # Using the actual call node also solves multiple statements on one line.
    statements = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr,
                  ast.Return, ast.Raise, ast.Assert, ast.Delete)
    candidates = [n for n in ast.walk(tree) if isinstance(n, statements)
                  and ((method_call is not None and
                        covers(n, method_call.lineno, method_call.col_offset) and
                        covers(n, method_call.end_lineno,
                               max(0, method_call.end_col_offset - 1)))
                       or (method_call is None and covers(n, call_line, call_col)))]
    if not candidates:
        text = linecache.getline(filename, lineno).rstrip() or "<source unavailable>"
        return Source(filename, lineno, lineno, f"{lineno:5d} | {text}", -1, lineno)

    stmt = min(candidates, key=lambda n: (
        n.end_lineno - n.lineno,
        n.end_col_offset - n.col_offset if n.lineno == n.end_lineno else 0))
    in_stmt = [n for n in ast.walk(stmt)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    # The end of a call corresponds to evaluation order for ordinary method
    # chains and nested arguments (inner calls finish before outer calls).
    in_stmt.sort(key=lambda n: (n.end_lineno, n.end_col_offset,
                                n.lineno, n.col_offset))
    rank = next((i for i, n in enumerate(in_stmt) if n is method_call), -1)
    method_line = method_call.func.end_lineno if method_call else lineno
    displayed = "\n".join(f"{i:5d} | {lines[i-1].rstrip()}"
                           for i in range(stmt.lineno, stmt.end_lineno + 1))
    # These are *source expressions*, not runtime introspection of variables.
    # Only a direct assignment of the outermost call has a definite target.
    def expression_label(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        return None

    input_label = None
    output_label = None
    other_label = None
    if method_call is not None:
        receiver = method_call.func.value
        # For df.group_by(...).agg(...), the input is df, not the
        # unnamed GroupBy object.
        if (isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Attribute)
                and receiver.func.attr in ("group_by", "group_by_dynamic")):
            receiver = receiver.func.value
        input_label = expression_label(receiver)
        if method_name == "join":
            other = next((kw.value for kw in method_call.keywords
                          if kw.arg == "other"), None)
            if other is None and method_call.args:
                other = method_call.args[0]
            other_label = expression_label(other)
        if (isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.value is method_call):
            output_label = stmt.targets[0].id
        elif (isinstance(stmt, ast.AnnAssign)
              and isinstance(stmt.target, ast.Name)
              and stmt.value is method_call):
            output_label = stmt.target.id

    return Source(filename, stmt.lineno, stmt.end_lineno,
                  displayed, rank, method_line,
                  (stmt.col_offset, stmt.end_col_offset),
                  input_label, output_label, other_label)


def _unwrap(obj):
    if isinstance(obj, LoggedDataFrame):
        return obj.native
    if isinstance(obj, list):
        return [_unwrap(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_unwrap(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    return obj


@dataclass
class _Chain:
    key: tuple
    last_rank: int
    execution: int


@dataclass
class _LastCall:
    # Keeping only the most recent caller frame avoids id(frame) reuse and
    # allows unrelated DataFrames in the same statement to share an execution.
    frame: Any
    key: tuple
    rank: int
    offset: int
    execution: int


_last_call = contextvars.ContextVar("polars_audit_last_call", default=None)
_next_execution = 0


def _new_execution():
    global _next_execution
    _next_execution += 1
    return _next_execution


def _track_call(frame, source: Source) -> _Chain:
    previous = _last_call.get()
    # Different call sites may have the same source line. Offset comparison
    # distinguishes progress through the statement from returning in a loop.
    # Bytecode positions increase through separate calls in a single
    # statement, even when AST matching cannot uniquely rank repeated
    # methods (e.g. chained .filter().filter() on the same line).
    # When a loop executes the statement again, its position resets.
    continuation = (
        previous is not None
        and previous.frame is frame
        and previous.key == source.key
        and frame.f_lasti > previous.offset
    )
    execution = previous.execution if continuation else _new_execution()
    if not continuation:
        _LOG.info("\n%s:%s-%s  [execution %s]\n%s",
                  Path(source.filename).name, source.start, source.end,
                  execution, source.displayed)
    _last_call.set(_LastCall(frame, source.key, source.rank,
                            frame.f_lasti, execution))
    return _Chain(source.key, source.rank, execution)


class LoggedDataFrame:
    """Composition-based audit wrapper for an *eager* pl.DataFrame."""

    def __init__(self, df: pl.DataFrame, name: str | None = None, *,
                 _chain: _Chain | None = None, log_initial: bool = True):
        if not isinstance(df, pl.DataFrame):
            raise TypeError("LoggedDataFrame requires a pl.DataFrame")
        self._df = df
        self.name = name
        self._chain = _chain
        if log_initial:
            _LOG.info("NOTE: OPEN | observations=%s | variables=%s\n",
                      df.height, df.width)

    @property
    def native(self) -> pl.DataFrame:
        """Escape hatch: direct operations on this object are NOT logged."""
        return self._df

    def _execute(self, operation: str, method, source: Source, chain: _Chain, *args, **kwargs):
        before = self._df.height
        old_cols = set(self._df.columns)
        inputs = [v for v in list(args) + list(kwargs.values())
                  if isinstance(v, LoggedDataFrame)]
        begin = time.perf_counter()
        try:
            result = method(*_unwrap(args), **_unwrap(kwargs))
        except Exception:
            _LOG.exception("ERROR [%s] %s FAILED (line %d)",
                           chain.execution, operation.upper(), source.method_line)
            raise
        elapsed = time.perf_counter() - begin
        input_name = source.input_label or self.name or "<intermediate>"
        output_name = source.output_label or "<intermediate>"
        _LOG.info("NOTE [%s] %s [line %s] | %.4fs",
                  chain.execution, operation.upper(), source.method_line, elapsed)
        _LOG.info("     INPUT: %s | rows=%s", input_name, before)
        if operation == "join":
            join_how = kwargs.get("how", "inner")
            _LOG.info("     JOIN TYPE: %s", str(join_how).upper())
            for key in ("on", "left_on", "right_on"):
                if key in kwargs:
                    _LOG.info("     %s: %s", key.upper(), kwargs[key])
        for j, item in enumerate(inputs, 1):
            # A source-level variable is more accurate than an inherited
            # wrapper name (particularly after reassignment or aliasing).
            label = (source.other_label if operation == "join" and j == 1
                     else None) or item.name or f"<input {j+1}>"
            _LOG.info("     INPUT %s: %s | rows=%s",
                      j + 1, label, item.native.height)
        if isinstance(result, pl.DataFrame):
            _LOG.info("     OUTPUT: %s | rows=%s | row change=%+d | columns=%s",
                      output_name, result.height, result.height - before, result.width)
            created = sorted(set(result.columns) - old_cols)
            removed = sorted(old_cols - set(result.columns))
            if created:
                _LOG.info("     Columns created: %s", ", ".join(created))
            if removed:
                _LOG.info("     Columns removed: %s", ", ".join(removed))
            return LoggedDataFrame(result, name=source.output_label,
                                   _chain=chain, log_initial=False)
        # Do not mutate this object's chain: the same input DataFrame can
        # be reused in unrelated statements and branches.
        if operation.startswith("write_"):
            _LOG.info("     Write requested from %s for %s rows", input_name, before)
        return result

    def __getattr__(self, name):
        attr = getattr(self._df, name)
        if not callable(attr):
            return attr
        if name in ("group_by", "group_by_dynamic"):
            def grouped(*args, **kwargs):
                frame = inspect.currentframe().f_back
                try:
                    site = _site(frame, name)
                finally:
                    del frame
                # Group creation is not materialized; .agg logs the operation.
                try:
                    group = attr(*_unwrap(args), **_unwrap(kwargs))
                except Exception:
                    _LOG.exception("ERROR: %s failed [line %s]", name, site.method_line)
                    raise
                return LoggedGroupBy(self, group)
            return grouped
        def delegated(*args, **kwargs):
            frame = inspect.currentframe().f_back
            try:
                site = _site(frame, name)
                chain = _track_call(frame, site)
            finally:
                del frame
            return self._execute(name, attr, site, chain, *args, **kwargs)
        return delegated

    def __repr__(self):
        return repr(self._df)


class LoggedGroupBy:
    def __init__(self, parent, group):
        self._parent = parent
        self._group = group

    def __getattr__(self, name):
        attr = getattr(self._group, name)
        if not callable(attr):
            return attr
        def delegated(*args, **kwargs):
            frame = inspect.currentframe().f_back
            try:
                site = _site(frame, name)
                chain = _track_call(frame, site)
            finally:
                del frame
            return self._parent._execute("group_by." + name, attr, site, chain,
                                         *args, **kwargs)
        return delegated
