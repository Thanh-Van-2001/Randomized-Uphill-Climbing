"""Operator grammar for RUC-TS expression trees (Section 3.2).

Defines four operator families -- Arithmetic, Statistical, Cross-sectional,
and Rolling-window -- and an ``OperatorGrammar`` catalogue that stores every
concrete operator and supports random sampling for tree construction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROLLING_WINDOW_SIZES: Tuple[int, ...] = (5, 10, 21, 63, 126, 252)
"""Standard rolling-window look-back periods (trading days)."""


# ---------------------------------------------------------------------------
# Operator family enum
# ---------------------------------------------------------------------------

class OperatorFamily(Enum):
    """High-level classification of operators."""

    ARITHMETIC = auto()
    STATISTICAL = auto()
    CROSS_SECTIONAL = auto()
    ROLLING_WINDOW = auto()


# ---------------------------------------------------------------------------
# Operator dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Operator:
    """A single concrete operator in the grammar.

    Parameters
    ----------
    name : str
        Human-readable operator name (e.g. ``"ts_mean_21"``).
    family : OperatorFamily
        Which family this operator belongs to.
    arity : int
        Number of child sub-expressions (1 = unary, 2 = binary).
    has_window : bool
        Whether the operator carries a look-back window parameter.
    window : int | None
        The look-back window size (only meaningful when *has_window* is True).
    base_name : str | None
        For rolling operators the window-free stem (e.g. ``"ts_mean"``).
        Defaults to *name* when not supplied.
    """

    name: str
    family: OperatorFamily
    arity: int
    has_window: bool = False
    window: Optional[int] = None
    base_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.base_name is None:
            object.__setattr__(self, "base_name", self.name)

    # Convenience -----------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        win = f", w={self.window}" if self.has_window else ""
        return (
            f"Operator({self.name!r}, {self.family.name}, "
            f"arity={self.arity}{win})"
        )


# ---------------------------------------------------------------------------
# Operator catalogues (class-level helpers)
# ---------------------------------------------------------------------------

def _arithmetic_operators() -> List[Operator]:
    """12 element-wise arithmetic / math operators."""
    fam = OperatorFamily.ARITHMETIC
    return [
        Operator("add", fam, arity=2),
        Operator("sub", fam, arity=2),
        Operator("mul", fam, arity=2),
        Operator("div", fam, arity=2),
        Operator("log", fam, arity=1),
        Operator("abs", fam, arity=1),
        Operator("sign", fam, arity=1),
        Operator("power", fam, arity=2),
        Operator("clip", fam, arity=1),
        Operator("neg", fam, arity=1),
        Operator("sqrt", fam, arity=1),
        Operator("exp", fam, arity=1),
    ]


def _statistical_operators() -> List[Operator]:
    """28 full-sample statistical operators."""
    fam = OperatorFamily.STATISTICAL
    return [
        # Central tendency
        Operator("mean", fam, arity=1),
        Operator("median", fam, arity=1),
        Operator("quantile_25", fam, arity=1),
        Operator("quantile_75", fam, arity=1),
        Operator("quantile_10", fam, arity=1),
        Operator("quantile_90", fam, arity=1),
        # Dispersion
        Operator("std", fam, arity=1),
        Operator("var", fam, arity=1),
        Operator("mad", fam, arity=1),           # mean absolute deviation
        Operator("iqr", fam, arity=1),           # inter-quartile range
        # Shape
        Operator("skew", fam, arity=1),
        Operator("kurtosis", fam, arity=1),
        # Normalisation / scaling
        Operator("zscore", fam, arity=1),
        Operator("min_max_scale", fam, arity=1),
        Operator("robust_scale", fam, arity=1),
        # Risk / performance
        Operator("information_ratio", fam, arity=2),
        Operator("sharpe", fam, arity=1),
        Operator("sortino", fam, arity=1),
        Operator("max_drawdown", fam, arity=1),
        # Bivariate
        Operator("corr", fam, arity=2),
        Operator("cov", fam, arity=2),
        Operator("beta", fam, arity=2),
        Operator("residual", fam, arity=2),
        # Cumulative
        Operator("cumsum", fam, arity=1),
        Operator("cumprod", fam, arity=1),
        Operator("cummax", fam, arity=1),
        Operator("cummin", fam, arity=1),
        # Difference / return
        Operator("pct_change", fam, arity=1),
    ]


def _cross_sectional_operators() -> List[Operator]:
    """15 cross-sectional (per-row across assets) operators."""
    fam = OperatorFamily.CROSS_SECTIONAL
    return [
        Operator("rank", fam, arity=1),
        Operator("percentile", fam, arity=1),
        Operator("demean", fam, arity=1),
        Operator("zscore_cs", fam, arity=1),
        Operator("normalize", fam, arity=1),
        Operator("winsorize", fam, arity=1),
        Operator("rank_pct", fam, arity=1),
        Operator("scale", fam, arity=1),          # scale to unit sum-of-abs
        Operator("cs_max", fam, arity=1),
        Operator("cs_min", fam, arity=1),
        Operator("cs_mean", fam, arity=1),
        Operator("cs_std", fam, arity=1),
        Operator("cs_median", fam, arity=1),
        Operator("cs_sum", fam, arity=1),
        Operator("cs_count_positive", fam, arity=1),
    ]


# -- Rolling-window operators are parameterised by window size --------------

_ROLLING_UNARY_BASES: List[Tuple[str, int]] = [
    # (base_name, arity)
    ("ts_mean", 1),
    ("ts_std", 1),
    ("ts_rank", 1),
    ("ts_delta", 1),
    ("ts_decay_linear", 1),
    ("ts_min", 1),
    ("ts_max", 1),
    ("ts_sum", 1),
    ("ts_skew", 1),
    ("ts_kurt", 1),
    ("ts_zscore", 1),
    ("ts_ir", 1),           # information ratio
    ("ts_median", 1),
    ("ts_var", 1),
    ("ts_argmax", 1),
    ("ts_argmin", 1),
    ("ts_pct_change", 1),
    ("ts_return", 1),
    ("ts_log_return", 1),
    ("ts_cumsum", 1),
    ("ts_cumprod", 1),
    ("ts_cummax", 1),
    ("ts_cummin", 1),
    ("ts_ema", 1),           # exponential moving average
    ("ts_wma", 1),           # weighted moving average
    ("ts_momentum", 1),
    ("ts_rsi", 1),           # relative strength index
    ("ts_maxdrawdown", 1),
    ("ts_volatility", 1),
    ("ts_autocorr", 1),
    ("ts_entropy", 1),
    ("ts_mad", 1),
    ("ts_iqr", 1),
    ("ts_count_positive", 1),
    ("ts_count_negative", 1),
    ("ts_quantile_25", 1),
    ("ts_quantile_75", 1),
    ("ts_range", 1),         # max - min
    ("ts_cv", 1),            # coefficient of variation
    ("ts_sharpe", 1),
]

_ROLLING_BINARY_BASES: List[Tuple[str, int]] = [
    ("ts_corr", 2),
    ("ts_cov", 2),
    ("ts_regression", 2),
    ("ts_beta", 2),
    ("ts_residual", 2),
    ("ts_mutual_info", 2),
    ("ts_granger", 2),
    ("ts_cross_corr", 2),
    ("ts_relative_strength", 2),
    ("ts_tracking_error", 2),
    ("ts_information_ratio", 2),
    ("ts_cosine_sim", 2),
    ("ts_dtw", 2),           # dynamic time warping distance
]


def _rolling_window_operators() -> List[Operator]:
    """Generate all (base_op x window_size) rolling-window operators.

    With 53 base operators and 6 window sizes this yields 318 concrete
    operators (315+ as required by the spec).
    """
    fam = OperatorFamily.ROLLING_WINDOW
    ops: List[Operator] = []
    for base_name, arity in _ROLLING_UNARY_BASES + _ROLLING_BINARY_BASES:
        for w in ROLLING_WINDOW_SIZES:
            ops.append(
                Operator(
                    name=f"{base_name}_{w}",
                    family=fam,
                    arity=arity,
                    has_window=True,
                    window=w,
                    base_name=base_name,
                )
            )
    return ops


# ---------------------------------------------------------------------------
# OperatorGrammar
# ---------------------------------------------------------------------------

class OperatorGrammar:
    """Catalogue of every concrete operator available for tree construction.

    The grammar is built once at construction time; subsequent look-ups and
    sampling calls are O(1) amortised.

    Parameters
    ----------
    seed : int | None
        Optional RNG seed for reproducible sampling.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

        # Build the full operator list once.
        self._operators: List[Operator] = (
            _arithmetic_operators()
            + _statistical_operators()
            + _cross_sectional_operators()
            + _rolling_window_operators()
        )

        # Precomputed indices for fast filtered sampling.
        self._by_family: Dict[OperatorFamily, List[Operator]] = {}
        self._by_arity: Dict[int, List[Operator]] = {}
        self._by_base_name: Dict[str, List[Operator]] = {}

        for op in self._operators:
            self._by_family.setdefault(op.family, []).append(op)
            self._by_arity.setdefault(op.arity, []).append(op)
            assert op.base_name is not None
            self._by_base_name.setdefault(op.base_name, []).append(op)

    # -- Properties ---------------------------------------------------------

    @property
    def operators(self) -> List[Operator]:
        """Return a *copy* of the full operator list."""
        return list(self._operators)

    @property
    def families(self) -> FrozenSet[OperatorFamily]:
        return frozenset(self._by_family.keys())

    @property
    def num_operators(self) -> int:
        return len(self._operators)

    # -- Look-up helpers ----------------------------------------------------

    def by_family(self, family: OperatorFamily) -> List[Operator]:
        """All operators that belong to *family*."""
        return list(self._by_family.get(family, []))

    def by_arity(self, arity: int) -> List[Operator]:
        """All operators with a given *arity*."""
        return list(self._by_arity.get(arity, []))

    def by_base_name(self, base_name: str) -> List[Operator]:
        """All window variants for a rolling operator *base_name*."""
        return list(self._by_base_name.get(base_name, []))

    def get(self, name: str) -> Optional[Operator]:
        """Look up a single operator by exact *name*, or ``None``."""
        for op in self._operators:
            if op.name == name:
                return op
        return None

    # -- Sampling -----------------------------------------------------------

    def sample(
        self,
        *,
        family: Optional[OperatorFamily] = None,
        arity: Optional[int] = None,
    ) -> Operator:
        """Return a uniformly random operator, optionally filtered.

        Parameters
        ----------
        family : OperatorFamily | None
            Restrict to a specific family.
        arity : int | None
            Restrict to a specific arity.

        Raises
        ------
        ValueError
            If the filter combination yields an empty candidate set.
        """
        candidates = self._operators
        if family is not None:
            candidates = [op for op in candidates if op.family == family]
        if arity is not None:
            candidates = [op for op in candidates if op.arity == arity]
        if not candidates:
            raise ValueError(
                f"No operators match family={family}, arity={arity}."
            )
        return self._rng.choice(candidates)

    def sample_n(
        self,
        n: int,
        *,
        family: Optional[OperatorFamily] = None,
        arity: Optional[int] = None,
        replace: bool = True,
    ) -> List[Operator]:
        """Sample *n* operators (with or without replacement)."""
        candidates = self._operators
        if family is not None:
            candidates = [op for op in candidates if op.family == family]
        if arity is not None:
            candidates = [op for op in candidates if op.arity == arity]
        if not candidates:
            raise ValueError(
                f"No operators match family={family}, arity={arity}."
            )
        if replace:
            return [self._rng.choice(candidates) for _ in range(n)]
        if n > len(candidates):
            raise ValueError(
                f"Cannot sample {n} operators without replacement from "
                f"{len(candidates)} candidates."
            )
        return self._rng.sample(candidates, n)

    # -- Summary ------------------------------------------------------------

    def summary(self) -> Dict[str, int]:
        """Return per-family operator counts."""
        return {fam.name: len(ops) for fam, ops in self._by_family.items()}

    def __repr__(self) -> str:  # pragma: no cover
        parts = ", ".join(
            f"{fam.name}={len(ops)}"
            for fam, ops in sorted(
                self._by_family.items(), key=lambda kv: kv[0].name
            )
        )
        return f"OperatorGrammar({self.num_operators} ops: {parts})"
