"""Section 3.5 -- Overfitting safeguards for synthesised features.

Three complementary filters are applied in sequence:

1. **VIF filtering** -- Remove features whose Variance Inflation Factor
   exceeds a threshold (default 5), reducing multicollinearity.
2. **Stability filtering** -- Discard features whose predictive power
   (rolling R^2) varies by more than 2*sigma across rolling windows,
   ensuring temporal stability.
3. **Nested expanding-window CV** -- Validate features through an
   expanding-window cross-validation scheme with 60/20/20
   train/val/test splits to guard against look-ahead bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


@dataclass
class OverfittingGuard:
    """Composite overfitting filter for synthesised time-series features.

    Parameters
    ----------
    vif_threshold : float
        Maximum acceptable Variance Inflation Factor.  Features with
        VIF above this value are removed.
    stability_sigma_factor : float
        Multiplier on the standard deviation of rolling R^2 values.
        Features whose R^2 range exceeds ``stability_sigma_factor * sigma``
        are discarded.
    rolling_window_size : int
        Number of observations in each rolling-R^2 window.
    cv_train_frac : float
        Fraction of *expanding* data used for training in the nested CV.
    cv_val_frac : float
        Fraction used for validation.
    cv_n_splits : int
        Number of expanding-window splits to evaluate.
    cv_r2_threshold : float
        Minimum mean validation-R^2 across CV folds to keep a feature.
    """

    vif_threshold: float = 5.0
    stability_sigma_factor: float = 2.0
    rolling_window_size: int = 252
    cv_train_frac: float = 0.60
    cv_val_frac: float = 0.20
    cv_n_splits: int = 5
    cv_r2_threshold: float = 0.0
    _removed_log: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
    ) -> pd.Index:
        """Run all three safeguards and return surviving feature indices.

        Parameters
        ----------
        features_df : pd.DataFrame
            Candidate feature matrix.  Rows = time steps, columns = features.
        target : pd.Series
            Prediction target aligned with *features_df*.

        Returns
        -------
        pd.Index
            Column names (indices) of features that passed every filter.
        """
        features_df = features_df.copy()
        target = target.copy()

        # Align and drop missing values
        common_idx = features_df.dropna().index.intersection(target.dropna().index)
        features_df = features_df.loc[common_idx]
        target = target.loc[common_idx]

        if features_df.empty:
            logger.warning("No valid rows after NaN removal.")
            return pd.Index([])

        surviving = list(features_df.columns)
        logger.info("Starting overfitting guard with %d features.", len(surviving))

        # Stage 1 -- VIF
        surviving = self._vif_filter(features_df[surviving])
        logger.info("%d features survived VIF filter.", len(surviving))

        if not surviving:
            return pd.Index([])

        # Stage 2 -- Stability
        surviving = self._stability_filter(features_df[surviving], target)
        logger.info("%d features survived stability filter.", len(surviving))

        if not surviving:
            return pd.Index([])

        # Stage 3 -- Nested expanding-window CV
        surviving = self._nested_cv_filter(features_df[surviving], target)
        logger.info("%d features survived nested CV filter.", len(surviving))

        return pd.Index(surviving)

    # ------------------------------------------------------------------
    # Stage 1: VIF filtering
    # ------------------------------------------------------------------

    def _vif_filter(self, df: pd.DataFrame) -> list[str]:
        """Iteratively remove the feature with the highest VIF until all
        remaining features have VIF <= ``vif_threshold``."""
        remaining = list(df.columns)

        while len(remaining) > 1:
            vifs = self._compute_vifs(df[remaining])
            max_vif_idx = int(np.argmax(vifs))
            max_vif = vifs[max_vif_idx]

            if max_vif <= self.vif_threshold:
                break

            removed = remaining.pop(max_vif_idx)
            logger.debug("VIF removed '%s' (VIF=%.2f).", removed, max_vif)

        return remaining

    @staticmethod
    def _compute_vifs(df: pd.DataFrame) -> np.ndarray:
        """Compute VIF for every column via OLS regression."""
        X = df.values.astype(np.float64)
        n_features = X.shape[1]
        vifs = np.empty(n_features, dtype=np.float64)

        for i in range(n_features):
            y_i = X[:, i]
            X_rest = np.delete(X, i, axis=1)

            if X_rest.shape[1] == 0:
                vifs[i] = 1.0
                continue

            # Add intercept
            X_rest = np.column_stack([np.ones(X_rest.shape[0]), X_rest])
            try:
                beta, *_ = np.linalg.lstsq(X_rest, y_i, rcond=None)
                y_hat = X_rest @ beta
                ss_res = np.sum((y_i - y_hat) ** 2)
                ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
                r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                vifs[i] = 1.0 / (1.0 - r_sq) if r_sq < 1.0 else np.inf
            except np.linalg.LinAlgError:
                vifs[i] = np.inf

        return vifs

    # ------------------------------------------------------------------
    # Stage 2: Rolling-R^2 stability filtering
    # ------------------------------------------------------------------

    def _stability_filter(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
    ) -> list[str]:
        """Discard features whose rolling R^2 varies by more than
        ``stability_sigma_factor * sigma``."""
        surviving: list[str] = []
        n = len(target)
        window = min(self.rolling_window_size, n // 3)

        if window < 10:
            logger.warning(
                "Not enough data for stability filtering (n=%d, window=%d). "
                "Skipping.",
                n,
                window,
            )
            return list(features_df.columns)

        for col in features_df.columns:
            r2_values = self._rolling_r2(
                features_df[col].values, target.values, window
            )
            if len(r2_values) < 2:
                surviving.append(col)
                continue

            sigma = np.std(r2_values, ddof=1)
            r2_range = np.max(r2_values) - np.min(r2_values)

            if r2_range <= self.stability_sigma_factor * sigma:
                surviving.append(col)
            else:
                logger.debug(
                    "Stability removed '%s' (range=%.4f, 2*sigma=%.4f).",
                    col,
                    r2_range,
                    self.stability_sigma_factor * sigma,
                )

        return surviving

    @staticmethod
    def _rolling_r2(
        feature: np.ndarray,
        target: np.ndarray,
        window: int,
    ) -> np.ndarray:
        """Compute R^2 of univariate OLS in rolling windows."""
        n = len(target)
        if n < window:
            return np.array([])

        n_windows = n - window + 1
        r2_values = np.empty(n_windows, dtype=np.float64)

        for start in range(n_windows):
            end = start + window
            x_w = feature[start:end].reshape(-1, 1)
            y_w = target[start:end]

            model = LinearRegression().fit(x_w, y_w)
            y_hat = model.predict(x_w)
            r2_values[start] = r2_score(y_w, y_hat)

        return r2_values

    # ------------------------------------------------------------------
    # Stage 3: Nested expanding-window cross-validation
    # ------------------------------------------------------------------

    def _nested_cv_filter(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
    ) -> list[str]:
        """Keep features whose mean validation R^2 across expanding-window
        folds exceeds ``cv_r2_threshold``."""
        surviving: list[str] = []
        n = len(target)

        # Determine split boundaries for each fold
        folds = self._expanding_window_splits(n)
        if not folds:
            logger.warning("Not enough data for nested CV. Skipping.")
            return list(features_df.columns)

        X = features_df.values
        y = target.values

        for j, col in enumerate(features_df.columns):
            fold_r2s: list[float] = []

            for train_end, val_end in folds:
                x_train = X[:train_end, j].reshape(-1, 1)
                y_train = y[:train_end]
                x_val = X[train_end:val_end, j].reshape(-1, 1)
                y_val = y[train_end:val_end]

                if len(y_val) < 2:
                    continue

                model = LinearRegression().fit(x_train, y_train)
                y_hat = model.predict(x_val)
                fold_r2s.append(r2_score(y_val, y_hat))

            if not fold_r2s:
                continue

            mean_r2 = float(np.mean(fold_r2s))
            if mean_r2 >= self.cv_r2_threshold:
                surviving.append(col)
            else:
                logger.debug(
                    "Nested CV removed '%s' (mean R^2=%.4f).", col, mean_r2
                )

        return surviving

    def _expanding_window_splits(
        self, n: int
    ) -> list[tuple[int, int]]:
        """Generate expanding-window fold boundaries.

        Each fold produces ``(train_end, val_end)`` indices such that:
        - train = [0, train_end)
        - val   = [train_end, val_end)
        - test  = [val_end, n)   (reserved but not used for filtering)

        The training window *expands* with each fold while the validation
        and test windows remain proportionally sized (60/20/20).
        """
        min_train = max(30, int(n * 0.1))
        test_size = max(1, int(n * (1.0 - self.cv_train_frac - self.cv_val_frac)))
        val_size = max(1, int(n * self.cv_val_frac))

        available = n - test_size - val_size
        if available < min_train:
            return []

        step = max(1, (available - min_train) // max(1, self.cv_n_splits - 1))
        folds: list[tuple[int, int]] = []

        for i in range(self.cv_n_splits):
            train_end = min_train + i * step
            if train_end > available:
                break
            val_end = train_end + val_size
            folds.append((train_end, val_end))

        return folds
