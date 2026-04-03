"""Tests for end-to-end pipeline."""

import numpy as np
import pytest

from ruc_ts.data.datasets import DatasetLoader
from ruc_ts.utils.metrics import compute_mae, compute_rmse, diebold_mariano_test


class TestDatasetLoader:
    def test_sp500_synthetic_shape(self):
        loader = DatasetLoader("sp500")
        X, y = loader.generate_synthetic("sp500")
        assert X.shape[0] == 5034
        assert X.shape[1] == 25
        assert len(y) == 5034

    def test_uci_synthetic_shape(self):
        loader = DatasetLoader("uci_appliances")
        X, y = loader.generate_synthetic("uci_appliances")
        assert X.shape[0] == 19735
        assert X.shape[1] == 28

    def test_jena_synthetic_shape(self):
        loader = DatasetLoader("jena_climate")
        X, y = loader.generate_synthetic("jena_climate", scale=0.02)
        assert X.shape[1] == 14


class TestMetrics:
    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mae(y, y) == 0.0

    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(y, y) == 0.0

    def test_dm_test_same_model(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(0, 1, 100)
        pred = actual + rng.normal(0, 0.1, 100)
        result = diebold_mariano_test(actual, pred, pred)
        assert result["p_value"] > 0.05
