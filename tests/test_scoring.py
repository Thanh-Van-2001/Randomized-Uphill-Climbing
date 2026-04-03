"""Tests for dual surrogate scoring (Section 3.4)."""

import numpy as np
import pytest

from ruc_ts.scoring.ols_surrogate import OLSSurrogate
from ruc_ts.scoring.mlp_surrogate import MLPSurrogate
from ruc_ts.scoring.dual_scorer import DualScorer


class TestOLSSurrogate:
    def test_perfect_linear_feature(self):
        """A perfectly correlated feature should score high."""
        rng = np.random.default_rng(42)
        n = 500
        feature = rng.normal(0, 1, n)
        target = 2.0 * feature + 0.1 * rng.normal(0, 1, n)
        scorer = OLSSurrogate()
        score = scorer.score(feature, target)
        assert score > 0.5

    def test_noise_feature_scores_low(self):
        """Random noise should score near zero or negative."""
        rng = np.random.default_rng(42)
        n = 500
        feature = rng.normal(0, 1, n)
        target = rng.normal(0, 1, n)
        scorer = OLSSurrogate()
        score = scorer.score(feature, target)
        assert score < 0.3

    def test_score_is_finite(self):
        rng = np.random.default_rng(42)
        feature = rng.normal(0, 1, 200)
        target = rng.normal(0, 1, 200)
        score = OLSSurrogate().score(feature, target)
        assert np.isfinite(score)


class TestMLPSurrogate:
    def test_nonlinear_feature(self):
        """MLP should capture nonlinear relationship."""
        rng = np.random.default_rng(42)
        n = 500
        feature = rng.normal(0, 1, n)
        target = feature ** 2 + 0.1 * rng.normal(0, 1, n)
        scorer = MLPSurrogate()
        score = scorer.score(feature, target)
        assert score > 0.0  # should capture some signal

    def test_score_is_finite(self):
        rng = np.random.default_rng(42)
        feature = rng.normal(0, 1, 200)
        target = rng.normal(0, 1, 200)
        score = MLPSurrogate().score(feature, target)
        assert np.isfinite(score)


class TestDualScorer:
    def test_alpha_default(self):
        """Paper: alpha = 0.6."""
        scorer = DualScorer()
        assert scorer.alpha == 0.6

    def test_combined_score_in_range(self):
        rng = np.random.default_rng(42)
        feature = rng.normal(0, 1, 300)
        target = 0.5 * feature + rng.normal(0, 1, 300)
        scorer = DualScorer(alpha=0.6)
        score = scorer.score(feature, target)
        assert np.isfinite(score)
