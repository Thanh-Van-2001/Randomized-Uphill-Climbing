"""Stage 1 OLS surrogate scorer (Eq. 1).

    s_OLS = (1 / W) * sum_{w=1}^{W} ( R2_w  -  lambda * BIC_w )

Evaluation uses expanding (growing) windows so that each sub-window *w*
contains all data from the start up to split point *w*, with the test
portion immediately following.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _ols_fit(X: np.ndarray, y: np.ndarray):
    """Ordinary Least Squares via the normal equations.

    Returns
    -------
    beta : np.ndarray
        Coefficient vector including intercept (last element).
    residuals : np.ndarray
        Residual vector ``y - X_aug @ beta``.
    """
    n = X.shape[0]
    X_aug = np.column_stack([X, np.ones(n)])
    # Use lstsq for numerical stability (handles rank-deficient cases).
    beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
    residuals = y - X_aug @ beta
    return beta, residuals


def _r_squared(y_true: np.ndarray, residuals: np.ndarray) -> float:
    """Coefficient of determination."""
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _bic(n: int, k: int, ss_res: float) -> float:
    """Bayesian Information Criterion.

    BIC = n * ln(ss_res / n) + k * ln(n)

    where *n* is sample size and *k* the number of parameters.
    """
    if n <= 0 or ss_res <= 0:
        return 0.0
    return n * np.log(ss_res / n) + k * np.log(n)


class OLSSurrogate:
    """Stage 1 surrogate scorer using expanding-window OLS.

    Parameters
    ----------
    n_windows : int
        Number of expanding windows *W* to evaluate.
    min_train_size : int
        Minimum number of training samples for the first window.
    lam : float
        BIC penalty weight (lambda in Eq. 1).
    bic_normalise : bool
        If ``True``, normalise each BIC term by ``1/n`` so that its scale
        is comparable to R2 across windows of different sizes.
    """

    def __init__(
        self,
        n_windows: int = 5,
        min_train_size: int = 63,
        lam: float = 0.01,
        bic_normalise: bool = True,
    ) -> None:
        if n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {n_windows}")
        self.n_windows = n_windows
        self.min_train_size = min_train_size
        self.lam = lam
        self.bic_normalise = bic_normalise

    def score(
        self,
        feature_values: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Score a single synthesised feature against *target*.

        Parameters
        ----------
        feature_values : np.ndarray
            1-D array of length *T* containing the evaluated feature.
        target : np.ndarray
            1-D target array of length *T*.

        Returns
        -------
        float
            ``s_OLS`` as defined in Eq. 1.
        """
        feature_values = np.asarray(feature_values, dtype=np.float64).ravel()
        target = np.asarray(target, dtype=np.float64).ravel()

        T = len(target)
        if T != len(feature_values):
            raise ValueError(
                f"Length mismatch: feature_values ({len(feature_values)}) "
                f"vs target ({T})"
            )

        splits = self._expanding_splits(T)
        if not splits:
            return float("-inf")

        r2_terms: List[float] = []
        bic_terms: List[float] = []

        for train_end, test_end in splits:
            X_train = feature_values[:train_end].reshape(-1, 1)
            y_train = target[:train_end]
            X_test = feature_values[train_end:test_end].reshape(-1, 1)
            y_test = target[train_end:test_end]

            if len(y_test) == 0 or len(y_train) < 2:
                continue

            beta, _ = _ols_fit(X_train, y_train)
            n_test = len(y_test)
            X_test_aug = np.column_stack([X_test, np.ones(n_test)])
            residuals_test = y_test - X_test_aug @ beta

            r2 = _r_squared(y_test, residuals_test)
            r2_terms.append(r2)

            k = X_train.shape[1] + 1  # features + intercept
            ss_res = np.sum(residuals_test ** 2)
            bic_val = _bic(n_test, k, ss_res)
            if self.bic_normalise and n_test > 0:
                bic_val /= n_test
            bic_terms.append(bic_val)

        if not r2_terms:
            return float("-inf")

        W = len(r2_terms)
        s_ols = (1.0 / W) * sum(
            r2 - self.lam * bic for r2, bic in zip(r2_terms, bic_terms)
        )
        return float(s_ols)

    # -----------------------------------------------------------------
    # Window construction
    # -----------------------------------------------------------------

    def _expanding_splits(self, T: int) -> List[tuple]:
        """Generate expanding-window (train_end, test_end) pairs.

        The training portion grows from ``min_train_size`` to near the end
        of the series, while each test portion covers the gap to the next
        split point (or to *T*).
        """
        usable = T - self.min_train_size
        if usable < self.n_windows:
            # Not enough data -- fall back to a single window.
            if T > self.min_train_size:
                return [(self.min_train_size, T)]
            return []

        step = usable // (self.n_windows + 1)
        if step < 1:
            step = 1

        splits: List[tuple] = []
        for w in range(1, self.n_windows + 1):
            train_end = self.min_train_size + w * step
            test_end = min(train_end + step, T)
            if train_end >= T:
                break
            splits.append((train_end, test_end))
        return splits
