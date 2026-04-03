"""Downstream prediction models for evaluating synthesised features.

Provides a unified ``DownstreamModel`` wrapper around:
- **XGBoost** (fully implemented): 500 trees, max depth 6, learning rate 0.05
- **LSTM** (placeholder): to be implemented with PyTorch / Keras
- **TFT** (placeholder): Temporal Fusion Transformer, to be implemented

All models expose a consistent ``fit`` / ``predict`` / ``evaluate`` interface.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ruc_ts.utils.metrics import compute_mae, compute_mape, compute_rmse

logger = logging.getLogger(__name__)

ModelType = Literal["xgboost", "lstm", "tft"]


class DownstreamModel:
    """Unified wrapper for downstream forecasting models.

    Parameters
    ----------
    model_type : {"xgboost", "lstm", "tft"}
        Which model backend to use.
    xgb_params : dict, optional
        Override default XGBoost hyper-parameters.  Merged on top of the
        defaults (n_estimators=500, max_depth=6, learning_rate=0.05).
    random_state : int
        Seed for reproducibility.
    """

    # Default XGBoost configuration matching the paper
    _XGB_DEFAULTS: dict[str, Any] = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "verbosity": 0,
    }

    def __init__(
        self,
        model_type: ModelType = "xgboost",
        *,
        xgb_params: dict[str, Any] | None = None,
        random_state: int = 42,
    ) -> None:
        self.model_type = model_type
        self.random_state = random_state
        self._model: Any = None
        self._is_fitted = False

        if model_type == "xgboost":
            self._init_xgboost(xgb_params or {})
        elif model_type == "lstm":
            self._init_lstm()
        elif model_type == "tft":
            self._init_tft()
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                "Choose from 'xgboost', 'lstm', 'tft'."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: ArrayLike, y_train: ArrayLike) -> DownstreamModel:
        """Fit the model on training data.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
        y_train : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X_train = self._to_numpy_2d(X_train)
        y_train = np.asarray(y_train, dtype=np.float64).ravel()

        if self.model_type == "xgboost":
            self._model.fit(X_train, y_train)
        elif self.model_type == "lstm":
            self._fit_lstm(X_train, y_train)
        elif self.model_type == "tft":
            self._fit_tft(X_train, y_train)

        self._is_fitted = True
        return self

    def predict(self, X_test: ArrayLike) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        X_test = self._to_numpy_2d(X_test)

        if self.model_type == "xgboost":
            return self._model.predict(X_test)
        elif self.model_type == "lstm":
            return self._predict_lstm(X_test)
        elif self.model_type == "tft":
            return self._predict_tft(X_test)

        raise RuntimeError(f"Unexpected model_type: {self.model_type}")

    def evaluate(
        self,
        X_test: ArrayLike,
        y_test: ArrayLike,
    ) -> dict[str, float]:
        """Predict and compute evaluation metrics.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
        y_test : array-like of shape (n_samples,)

        Returns
        -------
        dict
            ``{"MAE": ..., "RMSE": ..., "MAPE": ...}``
        """
        y_test = np.asarray(y_test, dtype=np.float64).ravel()
        y_pred = self.predict(X_test)

        return {
            "MAE": compute_mae(y_test, y_pred),
            "RMSE": compute_rmse(y_test, y_pred),
            "MAPE": compute_mape(y_test, y_pred),
        }

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    def _init_xgboost(self, overrides: dict[str, Any]) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for model_type='xgboost'. "
                "Install it with: pip install xgboost"
            ) from exc

        params = {**self._XGB_DEFAULTS, **overrides}
        params["random_state"] = self.random_state
        self._model = XGBRegressor(**params)

    # ------------------------------------------------------------------
    # LSTM (placeholder)
    # ------------------------------------------------------------------

    def _init_lstm(self) -> None:
        warnings.warn(
            "LSTM model is a placeholder. fit() will store training mean "
            "and predict() will return that constant.",
            UserWarning,
            stacklevel=3,
        )
        self._train_mean: float = 0.0

    def _fit_lstm(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info(
            "LSTM placeholder: storing training-set mean as baseline "
            "(shape X=%s, y=%s).",
            X_train.shape,
            y_train.shape,
        )
        self._train_mean = float(np.mean(y_train))

    def _predict_lstm(self, X_test: np.ndarray) -> np.ndarray:
        return np.full(X_test.shape[0], self._train_mean)

    # ------------------------------------------------------------------
    # TFT (placeholder)
    # ------------------------------------------------------------------

    def _init_tft(self) -> None:
        warnings.warn(
            "TFT model is a placeholder. fit() will store training mean "
            "and predict() will return that constant.",
            UserWarning,
            stacklevel=3,
        )
        self._train_mean: float = 0.0

    def _fit_tft(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        logger.info(
            "TFT placeholder: storing training-set mean as baseline "
            "(shape X=%s, y=%s).",
            X_train.shape,
            y_train.shape,
        )
        self._train_mean = float(np.mean(y_train))

    def _predict_tft(self, X_test: np.ndarray) -> np.ndarray:
        return np.full(X_test.shape[0], self._train_mean)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy_2d(X: ArrayLike) -> np.ndarray:
        """Coerce input to a 2-D float64 numpy array."""
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float64)
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "unfitted"
        return f"DownstreamModel(model_type='{self.model_type}', {status})"
