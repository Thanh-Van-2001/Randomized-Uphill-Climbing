"""Symbolic expression trees for RUC-TS feature synthesis.

An ``ExpressionTree`` is a bounded-depth tree whose internal nodes are
operators drawn from the ``OperatorGrammar`` and whose leaves are either
variable references (column names in the input DataFrame) or numeric
constants.  Trees can be evaluated over a ``pandas.DataFrame``, serialised
to a human-readable string, and randomly generated subject to a maximum
depth constraint.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .operators import (
    Operator,
    OperatorFamily,
    OperatorGrammar,
    ROLLING_WINDOW_SIZES,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DEPTH: int = 4
"""Default maximum tree depth (root = depth 0, so 4 means up to 5 levels)."""

_CONSTANT_RANGE: Tuple[float, float] = (-1.0, 1.0)
"""Range from which random leaf constants are drawn."""

_CLIP_LOWER: float = -1e9
_CLIP_UPPER: float = 1e9
"""Clamp bounds used by the ``clip`` operator and for final output sanity."""

_EPS: float = 1e-10
"""Small epsilon to avoid division-by-zero and log-of-zero."""

_WINSORIZE_LIMITS: Tuple[float, float] = (0.01, 0.99)
"""Default winsorisation quantile limits."""


# ---------------------------------------------------------------------------
# TreeNode
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """A single node in an expression tree.

    A node is either an *operator* node (has ``operator`` set and at least one
    child) or a *leaf* node (has ``variable`` or ``constant`` set).

    Parameters
    ----------
    operator : Operator | None
        The grammar operator at this node (internal nodes only).
    children : list[TreeNode]
        Child sub-trees.  Length must match ``operator.arity`` for internal
        nodes; empty for leaves.
    variable : str | None
        Column name reference (leaf nodes only).
    constant : float | None
        Numeric constant (leaf nodes only).
    """

    operator: Optional[Operator] = None
    children: List["TreeNode"] = field(default_factory=list)
    variable: Optional[str] = None
    constant: Optional[float] = None

    # -- Predicates ---------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        return self.operator is None

    @property
    def is_variable(self) -> bool:
        return self.variable is not None

    @property
    def is_constant(self) -> bool:
        return self.constant is not None

    # -- Traversal helpers --------------------------------------------------

    def depth(self) -> int:
        """Return the depth of the sub-tree rooted at this node."""
        if self.is_leaf:
            return 0
        return 1 + max(child.depth() for child in self.children)

    def size(self) -> int:
        """Total number of nodes in this sub-tree."""
        if self.is_leaf:
            return 1
        return 1 + sum(child.size() for child in self.children)

    def all_nodes(self) -> List["TreeNode"]:
        """Return a flat list of every node (pre-order traversal)."""
        nodes: List["TreeNode"] = [self]
        for child in self.children:
            nodes.extend(child.all_nodes())
        return nodes

    def variables_used(self) -> List[str]:
        """Return de-duplicated list of variable names in this sub-tree."""
        seen: set[str] = set()
        result: List[str] = []
        for node in self.all_nodes():
            if node.is_variable and node.variable not in seen:
                assert node.variable is not None
                seen.add(node.variable)
                result.append(node.variable)
        return result

    # -- Copying ------------------------------------------------------------

    def copy(self) -> "TreeNode":
        """Deep-copy this sub-tree."""
        return copy.deepcopy(self)

    # -- String representation ----------------------------------------------

    def to_string(self) -> str:
        if self.is_variable:
            return str(self.variable)
        if self.is_constant:
            # Round to 4 dp for readability.
            return f"{self.constant:.4g}"
        assert self.operator is not None
        child_strs = [c.to_string() for c in self.children]
        return f"{self.operator.name}({', '.join(child_strs)})"

    def __repr__(self) -> str:  # pragma: no cover
        return f"TreeNode({self.to_string()})"


# ---------------------------------------------------------------------------
# Safe element-wise helpers (numpy)
# ---------------------------------------------------------------------------

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) < _EPS, 0.0, a / b)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_log(a: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(a > _EPS, np.log(a), 0.0)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_sqrt(a: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        result = np.where(a >= 0, np.sqrt(np.maximum(a, 0.0)), 0.0)
    return np.nan_to_num(result, nan=0.0)


def _safe_power(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.power(np.abs(a) + _EPS, np.clip(b, -5.0, 5.0))
    return np.clip(np.nan_to_num(result, nan=0.0, posinf=_CLIP_UPPER, neginf=_CLIP_LOWER),
                   _CLIP_LOWER, _CLIP_UPPER)


def _safe_exp(a: np.ndarray) -> np.ndarray:
    clipped = np.clip(a, -20.0, 20.0)
    return np.exp(clipped)


def _sanitise(arr: np.ndarray) -> np.ndarray:
    """Replace inf/nan with 0 and clamp to sane range."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=_CLIP_UPPER, neginf=_CLIP_LOWER)
    return np.clip(arr, _CLIP_LOWER, _CLIP_UPPER)


# ---------------------------------------------------------------------------
# Rolling-window helpers
# ---------------------------------------------------------------------------

def _rolling_apply_1d(
    series: np.ndarray, window: int, func, default: float = 0.0
) -> np.ndarray:
    """Apply *func* to a rolling window along the first axis of *series*.

    Uses a simple loop; intended for correctness rather than speed in the
    backtesting prototype.  ``func`` receives a 1-D array of length *window*
    and must return a scalar.
    """
    n = len(series)
    out = np.full(n, default, dtype=np.float64)
    for i in range(window - 1, n):
        chunk = series[i - window + 1: i + 1]
        if np.all(np.isnan(chunk)):
            out[i] = default
        else:
            try:
                val = func(chunk)
                out[i] = val if np.isfinite(val) else default
            except Exception:
                out[i] = default
    return out


def _rolling_apply_2d(
    a: np.ndarray, b: np.ndarray, window: int, func, default: float = 0.0
) -> np.ndarray:
    """Binary rolling apply (e.g. rolling correlation)."""
    n = len(a)
    out = np.full(n, default, dtype=np.float64)
    for i in range(window - 1, n):
        ca = a[i - window + 1: i + 1]
        cb = b[i - window + 1: i + 1]
        if np.all(np.isnan(ca)) or np.all(np.isnan(cb)):
            out[i] = default
        else:
            try:
                val = func(ca, cb)
                out[i] = val if np.isfinite(val) else default
            except Exception:
                out[i] = default
    return out


def _decay_linear_weights(w: int) -> np.ndarray:
    weights = np.arange(1, w + 1, dtype=np.float64)
    return weights / weights.sum()


# ---------------------------------------------------------------------------
# Per-column evaluation dispatch
# ---------------------------------------------------------------------------

def _evaluate_node(
    node: TreeNode,
    data: pd.DataFrame,
) -> np.ndarray:
    """Recursively evaluate *node* and return a 1-D numpy array.

    The returned array has the same length as ``len(data)`` (number of rows).
    For leaves referring to missing columns, a zero-array is returned so that
    the tree is always well-defined.
    """

    # -- Leaf nodes ---------------------------------------------------------
    if node.is_variable:
        assert node.variable is not None
        if node.variable in data.columns:
            return data[node.variable].to_numpy(dtype=np.float64, copy=True)
        return np.zeros(len(data), dtype=np.float64)

    if node.is_constant:
        assert node.constant is not None
        return np.full(len(data), node.constant, dtype=np.float64)

    # -- Internal (operator) nodes ------------------------------------------
    assert node.operator is not None
    op = node.operator
    child_vals = [_evaluate_node(c, data) for c in node.children]
    a = child_vals[0]
    b = child_vals[1] if len(child_vals) > 1 else None

    n = len(data)

    # ---- Arithmetic ----
    if op.family == OperatorFamily.ARITHMETIC:
        return _eval_arithmetic(op, a, b, n)

    # ---- Statistical (full-sample) ----
    if op.family == OperatorFamily.STATISTICAL:
        return _eval_statistical(op, a, b, n)

    # ---- Cross-sectional ----
    if op.family == OperatorFamily.CROSS_SECTIONAL:
        return _eval_cross_sectional(op, a, n)

    # ---- Rolling-window ----
    if op.family == OperatorFamily.ROLLING_WINDOW:
        assert op.window is not None
        return _eval_rolling(op, a, b, op.window, n)

    return np.zeros(n, dtype=np.float64)


# ---------------------------------------------------------------------------
# Family-specific evaluators
# ---------------------------------------------------------------------------

def _eval_arithmetic(
    op: Operator, a: np.ndarray, b: Optional[np.ndarray], n: int
) -> np.ndarray:
    name = op.name
    if name == "add":
        assert b is not None
        return a + b
    if name == "sub":
        assert b is not None
        return a - b
    if name == "mul":
        assert b is not None
        return a * b
    if name == "div":
        assert b is not None
        return _safe_div(a, b)
    if name == "log":
        return _safe_log(a)
    if name == "abs":
        return np.abs(a)
    if name == "sign":
        return np.sign(a)
    if name == "power":
        assert b is not None
        return _safe_power(a, b)
    if name == "clip":
        return np.clip(a, _CLIP_LOWER, _CLIP_UPPER)
    if name == "neg":
        return -a
    if name == "sqrt":
        return _safe_sqrt(a)
    if name == "exp":
        return _safe_exp(a)
    return np.zeros(n, dtype=np.float64)


def _eval_statistical(
    op: Operator, a: np.ndarray, b: Optional[np.ndarray], n: int
) -> np.ndarray:
    """Evaluate full-sample statistical operators.

    Most of these broadcast a single summary statistic back to the full
    array length, which is the semantically correct behaviour when a
    statistical operator appears inside an expression tree (the scalar
    result is treated as a constant series).
    """
    name = op.name

    def _broadcast(val: float) -> np.ndarray:
        v = val if np.isfinite(val) else 0.0
        return np.full(n, v, dtype=np.float64)

    # Central tendency
    if name == "mean":
        return _broadcast(np.nanmean(a))
    if name == "median":
        return _broadcast(np.nanmedian(a))
    if name.startswith("quantile_"):
        q = int(name.split("_")[1]) / 100.0
        return _broadcast(float(np.nanquantile(a, q)))

    # Dispersion
    if name == "std":
        return _broadcast(np.nanstd(a))
    if name == "var":
        return _broadcast(np.nanvar(a))
    if name == "mad":
        return _broadcast(float(np.nanmean(np.abs(a - np.nanmean(a)))))
    if name == "iqr":
        return _broadcast(float(
            np.nanquantile(a, 0.75) - np.nanquantile(a, 0.25)
        ))

    # Shape
    if name == "skew":
        m = np.nanmean(a)
        s = np.nanstd(a)
        if s < _EPS:
            return np.zeros(n, dtype=np.float64)
        return _broadcast(float(np.nanmean(((a - m) / s) ** 3)))
    if name == "kurtosis":
        m = np.nanmean(a)
        s = np.nanstd(a)
        if s < _EPS:
            return np.zeros(n, dtype=np.float64)
        return _broadcast(float(np.nanmean(((a - m) / s) ** 4) - 3.0))

    # Normalisation
    if name == "zscore":
        s = np.nanstd(a)
        if s < _EPS:
            return np.zeros(n, dtype=np.float64)
        return (a - np.nanmean(a)) / s
    if name == "min_max_scale":
        lo, hi = np.nanmin(a), np.nanmax(a)
        rng = hi - lo
        if rng < _EPS:
            return np.zeros(n, dtype=np.float64)
        return (a - lo) / rng
    if name == "robust_scale":
        med = np.nanmedian(a)
        iqr_val = np.nanquantile(a, 0.75) - np.nanquantile(a, 0.25)
        if iqr_val < _EPS:
            return np.zeros(n, dtype=np.float64)
        return (a - med) / iqr_val

    # Risk / performance
    if name == "information_ratio":
        assert b is not None
        diff = a - b
        s = np.nanstd(diff)
        return _broadcast(float(np.nanmean(diff) / s)) if s > _EPS else np.zeros(n, dtype=np.float64)
    if name == "sharpe":
        s = np.nanstd(a)
        return _broadcast(float(np.nanmean(a) / s)) if s > _EPS else np.zeros(n, dtype=np.float64)
    if name == "sortino":
        downside = a[a < 0]
        ds = np.nanstd(downside) if len(downside) > 0 else 0.0
        return _broadcast(float(np.nanmean(a) / ds)) if ds > _EPS else np.zeros(n, dtype=np.float64)
    if name == "max_drawdown":
        cummax = np.maximum.accumulate(np.nan_to_num(a, nan=0.0))
        dd = cummax - a
        return dd  # per-element drawdown from running max

    # Bivariate
    if name == "corr":
        assert b is not None
        sa, sb = np.nanstd(a), np.nanstd(b)
        if sa < _EPS or sb < _EPS:
            return np.zeros(n, dtype=np.float64)
        return _broadcast(float(np.nanmean((a - np.nanmean(a)) * (b - np.nanmean(b))) / (sa * sb)))
    if name == "cov":
        assert b is not None
        return _broadcast(float(np.nanmean((a - np.nanmean(a)) * (b - np.nanmean(b)))))
    if name == "beta":
        assert b is not None
        var_b = np.nanvar(b)
        if var_b < _EPS:
            return np.zeros(n, dtype=np.float64)
        cov_ab = np.nanmean((a - np.nanmean(a)) * (b - np.nanmean(b)))
        return _broadcast(float(cov_ab / var_b))
    if name == "residual":
        assert b is not None
        var_b = np.nanvar(b)
        if var_b < _EPS:
            return a.copy()
        beta_val = np.nanmean((a - np.nanmean(a)) * (b - np.nanmean(b))) / var_b
        return a - beta_val * b - (np.nanmean(a) - beta_val * np.nanmean(b))

    # Cumulative
    if name == "cumsum":
        return np.nancumsum(a)
    if name == "cumprod":
        safe = np.nan_to_num(a, nan=1.0)
        return np.cumprod(np.clip(safe, -1e4, 1e4))
    if name == "cummax":
        return np.maximum.accumulate(np.nan_to_num(a, nan=-np.inf))
    if name == "cummin":
        return np.minimum.accumulate(np.nan_to_num(a, nan=np.inf))

    # Difference / return
    if name == "pct_change":
        shifted = np.roll(a, 1)
        shifted[0] = np.nan
        return _safe_div(a - shifted, np.abs(shifted) + _EPS)

    return np.zeros(n, dtype=np.float64)


def _eval_cross_sectional(
    op: Operator, a: np.ndarray, n: int
) -> np.ndarray:
    """Cross-sectional operators.

    In a single-column evaluation context, cross-sectional statistics
    degenerate to simple summary broadcasts.  When ``evaluate()`` is called
    with a full multi-column DataFrame the caller may opt to run a separate
    cross-sectional pass.  Here we implement the *per-column* semantics that
    make sense for expression-tree evaluation (rank/demean/zscore are
    computed over the time axis for a single column).
    """
    name = op.name

    if name == "rank":
        order = np.argsort(np.argsort(np.nan_to_num(a, nan=0.0)))
        return order.astype(np.float64)
    if name == "percentile" or name == "rank_pct":
        order = np.argsort(np.argsort(np.nan_to_num(a, nan=0.0)))
        return order.astype(np.float64) / max(n - 1, 1)
    if name == "demean":
        return a - np.nanmean(a)
    if name == "zscore_cs":
        s = np.nanstd(a)
        return (a - np.nanmean(a)) / s if s > _EPS else np.zeros(n, dtype=np.float64)
    if name == "normalize":
        total = np.nansum(np.abs(a))
        return a / total if total > _EPS else np.zeros(n, dtype=np.float64)
    if name == "winsorize":
        lo = np.nanquantile(a, _WINSORIZE_LIMITS[0])
        hi = np.nanquantile(a, _WINSORIZE_LIMITS[1])
        return np.clip(a, lo, hi)
    if name == "scale":
        total = np.nansum(np.abs(a))
        return a / total if total > _EPS else np.zeros(n, dtype=np.float64)
    if name == "cs_max":
        return np.full(n, np.nanmax(a), dtype=np.float64)
    if name == "cs_min":
        return np.full(n, np.nanmin(a), dtype=np.float64)
    if name == "cs_mean":
        return np.full(n, np.nanmean(a), dtype=np.float64)
    if name == "cs_std":
        return np.full(n, np.nanstd(a), dtype=np.float64)
    if name == "cs_median":
        return np.full(n, np.nanmedian(a), dtype=np.float64)
    if name == "cs_sum":
        return np.full(n, np.nansum(a), dtype=np.float64)
    if name == "cs_count_positive":
        return np.full(n, float(np.nansum(a > 0)), dtype=np.float64)
    return np.zeros(n, dtype=np.float64)


def _eval_rolling(
    op: Operator,
    a: np.ndarray,
    b: Optional[np.ndarray],
    window: int,
    n: int,
) -> np.ndarray:
    """Evaluate a rolling-window operator."""
    base = op.base_name
    assert base is not None
    w = window

    # --- Unary rolling ops ------------------------------------------------

    if base == "ts_mean":
        return _rolling_apply_1d(a, w, np.nanmean)
    if base == "ts_std":
        return _rolling_apply_1d(a, w, lambda c: np.nanstd(c, ddof=1))
    if base == "ts_var":
        return _rolling_apply_1d(a, w, lambda c: np.nanvar(c, ddof=1))
    if base == "ts_rank":
        def _rank_last(c: np.ndarray) -> float:
            return float(np.sum(c <= c[-1]) / len(c))
        return _rolling_apply_1d(a, w, _rank_last)
    if base == "ts_delta":
        out = np.zeros(n, dtype=np.float64)
        out[w:] = a[w:] - a[:-w]
        return out
    if base == "ts_decay_linear":
        weights = _decay_linear_weights(w)
        return _rolling_apply_1d(a, w, lambda c: np.nansum(c * weights))
    if base == "ts_min":
        return _rolling_apply_1d(a, w, np.nanmin)
    if base == "ts_max":
        return _rolling_apply_1d(a, w, np.nanmax)
    if base == "ts_sum":
        return _rolling_apply_1d(a, w, np.nansum)
    if base == "ts_skew":
        def _skew(c: np.ndarray) -> float:
            m, s = np.nanmean(c), np.nanstd(c)
            return float(np.nanmean(((c - m) / (s + _EPS)) ** 3)) if s > _EPS else 0.0
        return _rolling_apply_1d(a, w, _skew)
    if base == "ts_kurt":
        def _kurt(c: np.ndarray) -> float:
            m, s = np.nanmean(c), np.nanstd(c)
            return float(np.nanmean(((c - m) / (s + _EPS)) ** 4) - 3.0) if s > _EPS else 0.0
        return _rolling_apply_1d(a, w, _kurt)
    if base == "ts_zscore":
        def _zscore_last(c: np.ndarray) -> float:
            s = np.nanstd(c)
            return float((c[-1] - np.nanmean(c)) / s) if s > _EPS else 0.0
        return _rolling_apply_1d(a, w, _zscore_last)
    if base == "ts_ir":
        def _ir(c: np.ndarray) -> float:
            s = np.nanstd(c, ddof=1)
            return float(np.nanmean(c) / s) if s > _EPS else 0.0
        return _rolling_apply_1d(a, w, _ir)
    if base == "ts_median":
        return _rolling_apply_1d(a, w, np.nanmedian)
    if base == "ts_argmax":
        return _rolling_apply_1d(a, w, lambda c: float(np.nanargmax(c)))
    if base == "ts_argmin":
        return _rolling_apply_1d(a, w, lambda c: float(np.nanargmin(c)))
    if base == "ts_pct_change":
        out = np.zeros(n, dtype=np.float64)
        for i in range(w, n):
            prev = a[i - w]
            out[i] = (a[i] - prev) / (np.abs(prev) + _EPS) if np.abs(prev) > _EPS else 0.0
        return out
    if base == "ts_return":
        out = np.zeros(n, dtype=np.float64)
        for i in range(w, n):
            out[i] = a[i] - a[i - w]
        return out
    if base == "ts_log_return":
        out = np.zeros(n, dtype=np.float64)
        for i in range(w, n):
            if a[i] > _EPS and a[i - w] > _EPS:
                out[i] = math.log(a[i] / a[i - w])
        return out
    if base == "ts_cumsum":
        return _rolling_apply_1d(a, w, np.nansum)
    if base == "ts_cumprod":
        return _rolling_apply_1d(a, w, lambda c: np.nanprod(np.clip(c, -1e4, 1e4)))
    if base == "ts_cummax":
        return _rolling_apply_1d(a, w, np.nanmax)
    if base == "ts_cummin":
        return _rolling_apply_1d(a, w, np.nanmin)
    if base == "ts_ema":
        alpha = 2.0 / (w + 1)
        out = np.zeros(n, dtype=np.float64)
        out[0] = a[0]
        for i in range(1, n):
            out[i] = alpha * a[i] + (1 - alpha) * out[i - 1]
        return out
    if base == "ts_wma":
        weights = _decay_linear_weights(w)
        return _rolling_apply_1d(a, w, lambda c: np.nansum(c * weights))
    if base == "ts_momentum":
        out = np.zeros(n, dtype=np.float64)
        for i in range(w, n):
            denom = np.abs(a[i - w]) + _EPS
            out[i] = (a[i] / denom) if denom > _EPS else 0.0
        return out
    if base == "ts_rsi":
        out = np.zeros(n, dtype=np.float64)
        for i in range(w, n):
            chunk = np.diff(a[i - w: i + 1])
            gains = np.nansum(chunk[chunk > 0])
            losses = -np.nansum(chunk[chunk < 0])
            out[i] = 100.0 * gains / (gains + losses + _EPS)
        return out
    if base == "ts_maxdrawdown":
        def _mdd(c: np.ndarray) -> float:
            cm = np.maximum.accumulate(c)
            dd = cm - c
            return float(np.nanmax(dd))
        return _rolling_apply_1d(a, w, _mdd)
    if base == "ts_volatility":
        return _rolling_apply_1d(a, w, lambda c: np.nanstd(c, ddof=1))
    if base == "ts_autocorr":
        def _autocorr(c: np.ndarray) -> float:
            if len(c) < 2:
                return 0.0
            m = np.nanmean(c)
            s = np.nanstd(c)
            if s < _EPS:
                return 0.0
            c_centered = c - m
            return float(np.nanmean(c_centered[:-1] * c_centered[1:]) / (s * s))
        return _rolling_apply_1d(a, w, _autocorr)
    if base == "ts_entropy":
        def _entropy(c: np.ndarray) -> float:
            c_pos = np.abs(c) + _EPS
            p = c_pos / np.nansum(c_pos)
            return float(-np.nansum(p * np.log(p + _EPS)))
        return _rolling_apply_1d(a, w, _entropy)
    if base == "ts_mad":
        return _rolling_apply_1d(a, w, lambda c: float(np.nanmean(np.abs(c - np.nanmean(c)))))
    if base == "ts_iqr":
        return _rolling_apply_1d(a, w, lambda c: float(np.nanquantile(c, 0.75) - np.nanquantile(c, 0.25)))
    if base == "ts_count_positive":
        return _rolling_apply_1d(a, w, lambda c: float(np.nansum(c > 0)))
    if base == "ts_count_negative":
        return _rolling_apply_1d(a, w, lambda c: float(np.nansum(c < 0)))
    if base == "ts_quantile_25":
        return _rolling_apply_1d(a, w, lambda c: float(np.nanquantile(c, 0.25)))
    if base == "ts_quantile_75":
        return _rolling_apply_1d(a, w, lambda c: float(np.nanquantile(c, 0.75)))
    if base == "ts_range":
        return _rolling_apply_1d(a, w, lambda c: float(np.nanmax(c) - np.nanmin(c)))
    if base == "ts_cv":
        def _cv(c: np.ndarray) -> float:
            m = np.nanmean(c)
            s = np.nanstd(c, ddof=1)
            return float(s / (np.abs(m) + _EPS))
        return _rolling_apply_1d(a, w, _cv)
    if base == "ts_sharpe":
        def _ts_sharpe(c: np.ndarray) -> float:
            s = np.nanstd(c, ddof=1)
            return float(np.nanmean(c) / s) if s > _EPS else 0.0
        return _rolling_apply_1d(a, w, _ts_sharpe)

    # --- Binary rolling ops -----------------------------------------------

    if base == "ts_corr":
        assert b is not None
        def _corr(ca: np.ndarray, cb: np.ndarray) -> float:
            sa, sb = np.nanstd(ca), np.nanstd(cb)
            if sa < _EPS or sb < _EPS:
                return 0.0
            return float(np.nanmean((ca - np.nanmean(ca)) * (cb - np.nanmean(cb))) / (sa * sb))
        return _rolling_apply_2d(a, b, w, _corr)
    if base == "ts_cov":
        assert b is not None
        def _cov(ca: np.ndarray, cb: np.ndarray) -> float:
            return float(np.nanmean((ca - np.nanmean(ca)) * (cb - np.nanmean(cb))))
        return _rolling_apply_2d(a, b, w, _cov)
    if base == "ts_regression":
        assert b is not None
        def _regression_beta(ca: np.ndarray, cb: np.ndarray) -> float:
            var_b = np.nanvar(cb)
            if var_b < _EPS:
                return 0.0
            return float(np.nanmean((ca - np.nanmean(ca)) * (cb - np.nanmean(cb))) / var_b)
        return _rolling_apply_2d(a, b, w, _regression_beta)
    if base == "ts_beta":
        assert b is not None
        def _beta(ca: np.ndarray, cb: np.ndarray) -> float:
            var_b = np.nanvar(cb)
            if var_b < _EPS:
                return 0.0
            return float(np.nanmean((ca - np.nanmean(ca)) * (cb - np.nanmean(cb))) / var_b)
        return _rolling_apply_2d(a, b, w, _beta)
    if base == "ts_residual":
        assert b is not None
        out = np.zeros(n, dtype=np.float64)
        for i in range(w - 1, n):
            ca = a[i - w + 1: i + 1]
            cb = b[i - w + 1: i + 1]
            var_b = np.nanvar(cb)
            if var_b < _EPS:
                out[i] = ca[-1]
            else:
                beta_val = np.nanmean((ca - np.nanmean(ca)) * (cb - np.nanmean(cb))) / var_b
                alpha_val = np.nanmean(ca) - beta_val * np.nanmean(cb)
                out[i] = ca[-1] - (alpha_val + beta_val * cb[-1])
        return out
    if base == "ts_mutual_info":
        assert b is not None
        def _mi(ca: np.ndarray, cb: np.ndarray) -> float:
            # Simplified: discretise into 5 bins and compute MI
            try:
                bins = 5
                ha, _ = np.histogram(ca, bins=bins)
                hb, _ = np.histogram(cb, bins=bins)
                hab, _, _ = np.histogram2d(ca, cb, bins=bins)
                ha = ha / (ha.sum() + _EPS)
                hb = hb / (hb.sum() + _EPS)
                hab = hab / (hab.sum() + _EPS)
                mi = 0.0
                for ii in range(bins):
                    for jj in range(bins):
                        if hab[ii, jj] > _EPS and ha[ii] > _EPS and hb[jj] > _EPS:
                            mi += hab[ii, jj] * math.log(hab[ii, jj] / (ha[ii] * hb[jj]))
                return mi
            except Exception:
                return 0.0
        return _rolling_apply_2d(a, b, w, _mi)
    if base == "ts_granger":
        # Simplified Granger: correlation of a with lagged b
        assert b is not None
        def _granger(ca: np.ndarray, cb: np.ndarray) -> float:
            if len(ca) < 2:
                return 0.0
            return float(np.corrcoef(ca[1:], cb[:-1])[0, 1])
        return _rolling_apply_2d(a, b, w, _granger)
    if base == "ts_cross_corr":
        assert b is not None
        def _xcorr(ca: np.ndarray, cb: np.ndarray) -> float:
            sa, sb = np.nanstd(ca), np.nanstd(cb)
            if sa < _EPS or sb < _EPS:
                return 0.0
            return float(np.nanmean((ca - np.nanmean(ca)) * (cb - np.nanmean(cb))) / (sa * sb))
        return _rolling_apply_2d(a, b, w, _xcorr)
    if base == "ts_relative_strength":
        assert b is not None
        def _rs(ca: np.ndarray, cb: np.ndarray) -> float:
            mb = np.nanmean(cb)
            return float(np.nanmean(ca) / (mb + _EPS)) if np.abs(mb) > _EPS else 0.0
        return _rolling_apply_2d(a, b, w, _rs)
    if base == "ts_tracking_error":
        assert b is not None
        def _te(ca: np.ndarray, cb: np.ndarray) -> float:
            return float(np.nanstd(ca - cb, ddof=1))
        return _rolling_apply_2d(a, b, w, _te)
    if base == "ts_information_ratio":
        assert b is not None
        def _ir2(ca: np.ndarray, cb: np.ndarray) -> float:
            diff = ca - cb
            s = np.nanstd(diff, ddof=1)
            return float(np.nanmean(diff) / s) if s > _EPS else 0.0
        return _rolling_apply_2d(a, b, w, _ir2)
    if base == "ts_cosine_sim":
        assert b is not None
        def _cos(ca: np.ndarray, cb: np.ndarray) -> float:
            na = np.sqrt(np.nansum(ca ** 2))
            nb = np.sqrt(np.nansum(cb ** 2))
            if na < _EPS or nb < _EPS:
                return 0.0
            return float(np.nansum(ca * cb) / (na * nb))
        return _rolling_apply_2d(a, b, w, _cos)
    if base == "ts_dtw":
        # Simplified DTW: use absolute distance of normalised series as proxy
        assert b is not None
        def _dtw_proxy(ca: np.ndarray, cb: np.ndarray) -> float:
            return float(np.nanmean(np.abs(ca - cb)))
        return _rolling_apply_2d(a, b, w, _dtw_proxy)

    return np.zeros(n, dtype=np.float64)


# ---------------------------------------------------------------------------
# ExpressionTree
# ---------------------------------------------------------------------------

class ExpressionTree:
    """A symbolic expression tree built from an ``OperatorGrammar``.

    Parameters
    ----------
    root : TreeNode
        The root node of the tree.
    grammar : OperatorGrammar | None
        The grammar used to construct this tree (stored for mutation ops).
    max_depth : int
        Maximum allowed depth for this tree.
    """

    def __init__(
        self,
        root: TreeNode,
        grammar: Optional[OperatorGrammar] = None,
        max_depth: int = MAX_DEPTH,
    ) -> None:
        self.root = root
        self.grammar = grammar
        self.max_depth = max_depth

    # -- Core methods -------------------------------------------------------

    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        """Evaluate the tree over *data* and return a 1-D float64 array.

        The returned array has one entry per row in *data*.  All intermediate
        results are sanitised (no inf / NaN in the output).
        """
        result = _evaluate_node(self.root, data)
        return _sanitise(result)

    def to_string(self) -> str:
        """Return a human-readable infix/functional string representation."""
        return self.root.to_string()

    def copy(self) -> "ExpressionTree":
        """Return a deep copy of this tree."""
        return ExpressionTree(
            root=self.root.copy(),
            grammar=self.grammar,
            max_depth=self.max_depth,
        )

    def depth(self) -> int:
        """Return the depth of the tree."""
        return self.root.depth()

    def size(self) -> int:
        """Total number of nodes."""
        return self.root.size()

    def variables_used(self) -> List[str]:
        """List of unique variable names referenced in the tree."""
        return self.root.variables_used()

    # -- Random node selection ----------------------------------------------

    def random_node(self, rng: Optional[random.Random] = None) -> TreeNode:
        """Return a uniformly random node from the tree."""
        _rng = rng or random.Random()
        nodes = self.root.all_nodes()
        return _rng.choice(nodes)

    def random_internal_node(
        self, rng: Optional[random.Random] = None
    ) -> Optional[TreeNode]:
        """Return a random *internal* (operator) node, or ``None``."""
        _rng = rng or random.Random()
        internals = [nd for nd in self.root.all_nodes() if not nd.is_leaf]
        return _rng.choice(internals) if internals else None

    def random_leaf_node(
        self, rng: Optional[random.Random] = None
    ) -> Optional[TreeNode]:
        """Return a random *leaf* node, or ``None``."""
        _rng = rng or random.Random()
        leaves = [nd for nd in self.root.all_nodes() if nd.is_leaf]
        return _rng.choice(leaves) if leaves else None

    # -- Display ------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"ExpressionTree(depth={self.depth()}, '{self.to_string()}')"

    def __str__(self) -> str:
        return self.to_string()


# ---------------------------------------------------------------------------
# Random tree generation
# ---------------------------------------------------------------------------

def generate_random_tree(
    grammar: OperatorGrammar,
    variables: Sequence[str],
    max_depth: int = MAX_DEPTH,
    seed: Optional[int] = None,
    p_constant: float = 0.2,
) -> ExpressionTree:
    """Generate a random expression tree subject to a depth constraint.

    Parameters
    ----------
    grammar : OperatorGrammar
        The operator catalogue to draw from.
    variables : sequence of str
        Names of input variables (DataFrame column names).
    max_depth : int
        Maximum depth of the generated tree (root at depth 0).
    seed : int | None
        RNG seed for reproducibility.
    p_constant : float
        Probability that a leaf is a numeric constant rather than a variable.

    Returns
    -------
    ExpressionTree
        A newly constructed random tree.
    """
    rng = random.Random(seed)

    def _build(depth: int) -> TreeNode:
        # At max depth or with probability that decreases with depth,
        # emit a leaf.
        if depth >= max_depth or (depth > 0 and rng.random() < 0.3):
            return _random_leaf(rng, variables, p_constant)
        # Otherwise pick a random operator and recurse.
        op = grammar.sample()
        children = [_build(depth + 1) for _ in range(op.arity)]
        return TreeNode(operator=op, children=children)

    root = _build(0)
    return ExpressionTree(root=root, grammar=grammar, max_depth=max_depth)


def generate_random_tree_full(
    grammar: OperatorGrammar,
    variables: Sequence[str],
    target_depth: int = MAX_DEPTH,
    seed: Optional[int] = None,
    p_constant: float = 0.2,
) -> ExpressionTree:
    """Generate a *full* random tree (all branches reach *target_depth*).

    This is the "full" initialisation method from genetic programming
    literature.
    """
    rng = random.Random(seed)

    def _build(depth: int) -> TreeNode:
        if depth >= target_depth:
            return _random_leaf(rng, variables, p_constant)
        op = grammar.sample()
        children = [_build(depth + 1) for _ in range(op.arity)]
        return TreeNode(operator=op, children=children)

    root = _build(0)
    return ExpressionTree(root=root, grammar=grammar, max_depth=target_depth)


def _random_leaf(
    rng: random.Random,
    variables: Sequence[str],
    p_constant: float,
) -> TreeNode:
    """Create a random leaf node (variable reference or constant)."""
    if rng.random() < p_constant or len(variables) == 0:
        val = round(rng.uniform(_CONSTANT_RANGE[0], _CONSTANT_RANGE[1]), 4)
        return TreeNode(constant=val)
    var = rng.choice(list(variables))
    return TreeNode(variable=var)
