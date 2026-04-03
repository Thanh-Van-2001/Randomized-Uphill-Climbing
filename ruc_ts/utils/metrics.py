"""Evaluation metrics and statistical tests for RUC-TS.

Provides MAE, RMSE, MAPE calculations and the Diebold-Mariano test
for comparing forecast accuracy between two competing models.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def compute_mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        MAE score (non-negative).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.

    Returns
    -------
    float
        RMSE score (non-negative).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: ArrayLike, y_pred: ArrayLike, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    epsilon : float, default 1e-8
        Small constant added to the denominator to avoid division by zero.

    Returns
    -------
    float
        MAPE score as a fraction (not percentage).  Multiply by 100 for %.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))))


def diebold_mariano_test(
    y_true: ArrayLike,
    preds_a: ArrayLike,
    preds_b: ArrayLike,
    loss: str = "squared",
    h: int = 1,
    alpha: float = 0.05,
) -> dict:
    """Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[d_t] = 0  where d_t = L(e_a,t) - L(e_b,t) and L is a loss
    function.  A significant result (p < alpha) indicates that one model's
    forecasts are statistically more accurate than the other's.

    Uses the Harvey, Leybourne & Newbold (1997) small-sample correction.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth values.
    preds_a : array-like of shape (n_samples,)
        Predictions from model A.
    preds_b : array-like of shape (n_samples,)
        Predictions from model B.
    loss : {"squared", "absolute"}, default "squared"
        Loss function used to compute the differential series.
    h : int, default 1
        Forecast horizon.  Controls the Newey-West bandwidth for
        autocovariance estimation.
    alpha : float, default 0.05
        Significance level for the two-sided test.

    Returns
    -------
    dict
        Keys:
        - ``dm_statistic`` (float): The DM test statistic.
        - ``p_value`` (float): Two-sided p-value.
        - ``significant`` (bool): True if p_value < alpha.
        - ``preferred`` (str | None): ``"model_a"`` if model A is
          significantly better, ``"model_b"`` if model B is significantly
          better, ``None`` otherwise.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    preds_a = np.asarray(preds_a, dtype=np.float64)
    preds_b = np.asarray(preds_b, dtype=np.float64)

    if not (y_true.shape == preds_a.shape == preds_b.shape):
        raise ValueError("All input arrays must have the same shape.")

    e_a = y_true - preds_a
    e_b = y_true - preds_b

    if loss == "squared":
        d = e_a**2 - e_b**2
    elif loss == "absolute":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError(f"Unknown loss '{loss}'. Use 'squared' or 'absolute'.")

    n = len(d)
    if n < 3:
        raise ValueError("Need at least 3 observations for the DM test.")

    d_mean = np.mean(d)

    # Newey-West style autocovariance estimation
    gamma_0 = np.mean((d - d_mean) ** 2)
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - d_mean) * (d[:-k] - d_mean))
        gamma_sum += gamma_k

    variance_d = (gamma_0 + 2.0 * gamma_sum) / n
    if variance_d <= 0:
        # Degenerate case: no variation in the loss differential
        return {
            "dm_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "preferred": None,
        }

    dm_stat = d_mean / np.sqrt(variance_d)

    # Harvey, Leybourne & Newbold small-sample correction
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_stat_corrected = dm_stat * correction

    # Two-sided test using t-distribution with n-1 degrees of freedom
    p_value = 2.0 * stats.t.sf(np.abs(dm_stat_corrected), df=n - 1)

    significant = bool(p_value < alpha)
    preferred = None
    if significant:
        # Negative DM stat => model A has smaller loss => model A preferred
        preferred = "model_a" if dm_stat_corrected < 0 else "model_b"

    return {
        "dm_statistic": float(dm_stat_corrected),
        "p_value": float(p_value),
        "significant": significant,
        "preferred": preferred,
    }
