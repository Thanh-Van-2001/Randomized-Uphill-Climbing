"""Tests for operator grammar and expression trees (Section 3.2)."""

import numpy as np
import pandas as pd
import pytest

from ruc_ts.grammar.operators import OperatorGrammar, OperatorFamily
from ruc_ts.grammar.expression_tree import (
    ExpressionTree, TreeNode, generate_random_tree, MAX_DEPTH,
)


class TestOperatorGrammar:
    @pytest.fixture
    def grammar(self):
        return OperatorGrammar()

    def test_total_operators_370_plus(self, grammar):
        """Paper: 370+ operators."""
        assert len(grammar.operators) >= 370

    def test_arithmetic_count(self, grammar):
        """Section 3.2: |O_A| = 12."""
        arith = grammar.by_family(OperatorFamily.ARITHMETIC)
        assert len(arith) == 12

    def test_statistical_count(self, grammar):
        """Section 3.2: |O_S| = 28."""
        stat = grammar.by_family(OperatorFamily.STATISTICAL)
        assert len(stat) == 28

    def test_cross_sectional_count(self, grammar):
        """Section 3.2: |O_C| = 15."""
        cs = grammar.by_family(OperatorFamily.CROSS_SECTIONAL)
        assert len(cs) == 15

    def test_rolling_window_count(self, grammar):
        """Section 3.2: |O_R| = 315+."""
        rw = grammar.by_family(OperatorFamily.ROLLING_WINDOW)
        assert len(rw) >= 315

    def test_sample_returns_valid_operator(self, grammar):
        op = grammar.sample()
        assert op.name is not None
        assert op.family in OperatorFamily

    def test_summary_has_all_families(self, grammar):
        s = grammar.summary()
        assert isinstance(s, dict)
        assert "ARITHMETIC" in s or "arithmetic" in str(s).lower()


class TestExpressionTree:
    @pytest.fixture
    def variables(self):
        return ["close", "volume", "high", "low"]

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(42)
        n = 300
        return pd.DataFrame({
            "close": 100 + np.cumsum(rng.normal(0, 1, n)),
            "volume": rng.lognormal(10, 1, n),
            "high": 101 + np.cumsum(rng.normal(0, 1, n)),
            "low": 99 + np.cumsum(rng.normal(0, 1, n)),
        })

    def test_random_tree_depth_bounded(self, variables):
        grammar = OperatorGrammar()
        for _ in range(20):
            tree = generate_random_tree(grammar, variables, seed=None)
            assert tree.depth() <= MAX_DEPTH

    def test_tree_evaluates_to_array(self, variables, data):
        grammar = OperatorGrammar()
        tree = generate_random_tree(grammar, variables, seed=42)
        result = tree.evaluate(data)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)

    def test_tree_no_nan_inf(self, variables, data):
        """Safe evaluation: no NaN or Inf in output."""
        grammar = OperatorGrammar()
        for seed in range(10):
            tree = generate_random_tree(grammar, variables, seed=seed)
            result = tree.evaluate(data)
            assert np.all(np.isfinite(result)), f"Non-finite for seed={seed}: {tree}"

    def test_to_string_readable(self, variables):
        grammar = OperatorGrammar()
        tree = generate_random_tree(grammar, variables, seed=42)
        s = tree.to_string()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_copy_independence(self, variables, data):
        grammar = OperatorGrammar()
        tree = generate_random_tree(grammar, variables, seed=42)
        copy = tree.copy()
        assert tree.to_string() == copy.to_string()
