"""Stage 2 MLP surrogate scorer (Eq. 2).

    s_MLP = (1 / W) * sum_{w=1}^{W} ( 1  -  MSE_w / Var(y_w) )

Uses a 2-layer MLP (64 -> 32, ReLU, 50 epochs) implemented via
``sklearn.neural_network.MLPRegressor`` for a lightweight, dependency-
light approach.
"""

from __future__ import annotations

import logging
import warnings
from typing import List, Optional

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MLPSurrogate:
    """Stage 2 surrogate scorer using expanding-window MLP evaluation.

    Parameters
    ----------
    n_windows : int
        Number of expanding windows *W*.
    min_train_size : int
        Minimum number of training samples for the first window.
    hidden_layers : tuple[int, ...]
        MLP hidden layer sizes.
    max_epochs : int
        Maximum training epochs per window.
    learning_rate_init : float
        Initial learning rate for Adam optimiser.
    random_state : int | None
        Seed for the MLP and data splitting.
    """

    def __init__(
        self,
        n_windows: int = 5,
        min_train_size: int = 63,
        hidden_layers: tuple = (64, 32),
        max_epochs: int = 50,
        learning_rate_init: float = 1e-3,
        random_state: Optional[int] = None,
    ) -> None:
        if n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {n_windows}")
        self.n_windows = n_windows
        self.min_train_size = min_train_size
        self.hidden_layers = hidden_layers
        self.max_epochs = max_epochs
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state

    def score(
        self,
        feature_values: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Score a synthesised feature vector against *target*.

        Parameters
        ----------
        feature_values : np.ndarray
            1-D or 2-D array.  If 1-D it is reshaped to ``(T, 1)``.
        target : np.ndarray
            1-D target vector of length *T*.

        Returns
        -------
        float
            ``s_MLP`` as defined in Eq. 2.
        """
        feature_values = np.asarray(feature_values, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64).ravel()

        if feature_values.ndim == 1:
            feature_values = feature_values.reshape(-1, 1)

        T = len(target)
        if feature_values.shape[0] != T:
            raise ValueError(
                f"Length mismatch: feature_values ({feature_values.shape[0]}) "
                f"vs target ({T})"
            )

        splits = self._expanding_splits(T)
        if not splits:
            return float("-inf")

        terms: List[float] = []

        for train_end, test_end in splits:
            X_train = feature_values[:train_end]
            y_train = target[:train_end]
            X_test = feature_values[train_end:test_end]
            y_test = target[train_end:test_end]

            if len(y_test) == 0 or len(y_train) < 2:
                continue

            var_y = np.var(y_test)
            if var_y == 0:
                # Constant target in this window -- perfect R2-like score.
                terms.append(1.0)
                continue

            try:
                term = self._fit_and_eval(X_train, y_train, X_test, y_test, var_y)
                terms.append(term)
            except Exception:
                logger.debug(
                    "MLP scoring failed for window (%d, %d)",
                    train_end, test_end,
                    exc_info=True,
                )
                continue

        if not terms:
            return float("-inf")

        return float(np.mean(terms))

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _fit_and_eval(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        var_y: float,
    ) -> float:
        """Train an MLP on one window and return ``1 - MSE / Var(y)``."""
        # Standardise features for stable MLP training.
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        mlp = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation="relu",
            solver="adam",
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_epochs,
            early_stopping=False,
            random_state=self.random_state,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            mlp.fit(X_train_s, y_train)

        y_pred = mlp.predict(X_test_s)
        mse = float(np.mean((y_test - y_pred) ** 2))
        return 1.0 - mse / var_y

    def _expanding_splits(self, T: int) -> List[tuple]:
        """Generate expanding-window (train_end, test_end) pairs."""
        usable = T - self.min_train_size
        if usable < self.n_windows:
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
