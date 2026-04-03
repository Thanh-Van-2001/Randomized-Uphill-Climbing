"""Perturbation operators for RUC-TS tree mutation (Section 3.3).

Four mutation operators, each modifying exactly ONE component of an
expression tree. PerturbationEngine.perturb picks one uniformly at
random and returns a mutated copy (original is never modified).
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional

from ruc_ts.grammar.expression_tree import (
    ExpressionTree, TreeNode, generate_random_tree, MAX_DEPTH,
)
from ruc_ts.grammar.operators import (
    OperatorGrammar, OperatorFamily, ROLLING_WINDOW_SIZES,
)

VALID_WINDOW_SIZES: List[int] = list(ROLLING_WINDOW_SIZES)


def _collect_nodes(node: TreeNode) -> List[TreeNode]:
    """Depth-first collection of all nodes."""
    results: List[TreeNode] = []
    stack = [node]
    while stack:
        current = stack.pop()
        results.append(current)
        for child in current.children:
            stack.append(child)
    return results


def _node_depth(root: TreeNode, target: TreeNode) -> int:
    """Return depth of target inside tree rooted at root."""
    stack: list[tuple[TreeNode, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node is target:
            return depth
        for child in node.children:
            stack.append((child, depth + 1))
    return 0


def _find_parent(root: TreeNode, target: TreeNode) -> Optional[TreeNode]:
    """Return parent of target in the subtree."""
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node.children:
            if child is target:
                return node
            stack.append(child)
    return None


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

def swap_operator(tree: ExpressionTree, grammar: OperatorGrammar) -> ExpressionTree:
    """Replace an internal operator with a type-compatible alternative."""
    mutated = tree.copy()
    internal = [n for n in _collect_nodes(mutated.root) if not n.is_leaf]
    if not internal:
        return mutated

    node = random.choice(internal)
    arity = node.operator.arity
    family = node.operator.family
    alternatives = [
        op for op in grammar.by_family(family)
        if op.arity == arity and op.name != node.operator.name
    ]
    if not alternatives:
        # Try any family with same arity
        alternatives = [
            op for op in grammar.operators
            if op.arity == arity and op.name != node.operator.name
        ]
    if alternatives:
        node.operator = random.choice(alternatives)
    return mutated


def resize_window(tree: ExpressionTree, grammar: OperatorGrammar) -> ExpressionTree:
    """Change a rolling-window operator to a different window size."""
    mutated = tree.copy()
    rolling = [
        n for n in _collect_nodes(mutated.root)
        if not n.is_leaf and n.operator.has_window
    ]
    if not rolling:
        return mutated

    node = random.choice(rolling)
    current_window = node.operator.window
    alternatives = [w for w in VALID_WINDOW_SIZES if w != current_window]
    if not alternatives:
        return mutated

    new_window = random.choice(alternatives)
    # Find operator with same base name but different window
    base = node.operator.base_name
    new_ops = [
        op for op in grammar.operators
        if op.base_name == base and op.window == new_window
    ]
    if new_ops:
        node.operator = new_ops[0]
    return mutated


def swap_variable(tree: ExpressionTree, grammar: OperatorGrammar) -> ExpressionTree:
    """Replace a leaf variable with another from the tree's variable set."""
    mutated = tree.copy()
    leaves = [n for n in _collect_nodes(mutated.root) if n.is_variable]
    if not leaves:
        return mutated

    node = random.choice(leaves)
    all_vars = list(mutated.root.variables_used())
    # If we know the available variables from the grammar, use those
    # Otherwise use variables already in the tree plus some defaults
    alternatives = [v for v in all_vars if v != node.variable]
    if alternatives:
        node.variable = random.choice(alternatives)
    return mutated


def graft_subtree(tree: ExpressionTree, grammar: OperatorGrammar) -> ExpressionTree:
    """Replace a random node with a freshly generated subtree."""
    mutated = tree.copy()
    all_nodes = _collect_nodes(mutated.root)
    if not all_nodes:
        return mutated

    target = random.choice(all_nodes)
    depth = _node_depth(mutated.root, target)
    remaining = max(1, MAX_DEPTH - depth)

    # Get variables from existing tree
    variables = list(mutated.root.variables_used()) or ["x"]
    new_tree = generate_random_tree(grammar, variables, max_depth=remaining)

    parent = _find_parent(mutated.root, target)
    if parent is None:
        mutated.root = new_tree.root
    else:
        idx = parent.children.index(target)
        parent.children[idx] = new_tree.root

    return mutated


MUTATION_OPERATORS = [swap_operator, resize_window, swap_variable, graft_subtree]


class PerturbationEngine:
    """Applies a single random mutation to an expression tree (Section 3.3).

    Each mutation changes exactly one component, keeping the search local.
    """

    def __init__(self, seed: int | None = None):
        self._rng = random.Random(seed)

    def perturb(
        self, tree: ExpressionTree, grammar: OperatorGrammar
    ) -> ExpressionTree:
        """Return a mutated copy of tree using one random operator."""
        op = self._rng.choice(MUTATION_OPERATORS)
        old_state = random.getstate()
        random.setstate(self._rng.getstate())
        try:
            result = op(tree, grammar)
        finally:
            self._rng.setstate(random.getstate())
            random.setstate(old_state)
        return result
