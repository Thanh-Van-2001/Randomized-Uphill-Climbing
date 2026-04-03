"""Grammar module for RUC-TS expression tree construction.

Public API
----------
OperatorGrammar
    Catalogue of all concrete operators; supports look-up and random sampling.
ExpressionTree
    Symbolic expression tree that can be evaluated over a DataFrame.
Operator
    Immutable descriptor for a single operator (name, family, arity, window).
"""

from .operators import Operator, OperatorFamily, OperatorGrammar, ROLLING_WINDOW_SIZES
from .expression_tree import (
    ExpressionTree,
    TreeNode,
    generate_random_tree,
    generate_random_tree_full,
    MAX_DEPTH,
)

__all__ = [
    "Operator",
    "OperatorFamily",
    "OperatorGrammar",
    "ExpressionTree",
    "TreeNode",
    "generate_random_tree",
    "generate_random_tree_full",
    "ROLLING_WINDOW_SIZES",
    "MAX_DEPTH",
]
