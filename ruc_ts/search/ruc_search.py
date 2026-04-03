"""RUC-TS main search loop -- Algorithm 1 from the paper.

Randomised Uphill Climbing with simulated-annealing acceptance,
elite-based perturbation, and periodic diversity injection.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, List, Optional

from ruc_ts.search.annealing import AnnealingSchedule
from ruc_ts.search.perturbation import PerturbationEngine

if TYPE_CHECKING:
    from ruc_ts.grammar.grammar import Grammar
    from ruc_ts.grammar.tree import ExpressionTree

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class _ScoredTree:
    """Lightweight wrapper pairing a tree with its cached score."""

    tree: "ExpressionTree"
    score: float = float("-inf")


@dataclass
class SearchHistory:
    """Convergence diagnostics recorded during a search run."""

    best_scores: List[float] = field(default_factory=list)
    mean_scores: List[float] = field(default_factory=list)
    pool_sizes: List[int] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)


class RUCSearch:
    """Randomised Uphill Climbing search for time-series expression trees.

    Parameters
    ----------
    grammar : Grammar
        Grammar object used to generate and validate expression trees.
    pool_size : int
        Number of candidate trees maintained each iteration.
    elite_count : int
        Number of top trees selected as elites for perturbation.
    iterations : int
        Total number of search iterations (*I*).
    tau_0 : float
        Initial annealing temperature.
    diversity_rate : float
        Fraction of worst pool members replaced with fresh random trees
        each iteration.
    rng_seed : int | None
        Master seed for reproducibility.
    """

    def __init__(
        self,
        grammar: "Grammar",
        pool_size: int = 500,
        elite_count: int = 50,
        iterations: int = 200,
        tau_0: float = 1.0,
        diversity_rate: float = 0.10,
        rng_seed: Optional[int] = None,
    ) -> None:
        if elite_count > pool_size:
            raise ValueError(
                f"elite_count ({elite_count}) must be <= pool_size ({pool_size})"
            )
        self.grammar = grammar
        self.pool_size = pool_size
        self.elite_count = elite_count
        self.iterations = iterations
        self.tau_0 = tau_0
        self.diversity_rate = diversity_rate
        self.rng_seed = rng_seed

        self._rng = random.Random(rng_seed)
        self._perturber = PerturbationEngine(
            rng_seed=rng_seed + 1 if rng_seed is not None else None
        )
        self._schedule = AnnealingSchedule(
            tau_0=tau_0,
            total_iterations=iterations,
            rng_seed=rng_seed + 2 if rng_seed is not None else None,
        )
        self.history: SearchHistory = SearchHistory()

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scorer: Callable[["ExpressionTree", np.ndarray, np.ndarray], float],
        top_k: int = 50,
    ) -> List["ExpressionTree"]:
        """Run Algorithm 1 and return the *top_k* best expression trees.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape ``(T, D)`` where *T* is the
            number of time steps and *D* the number of raw variables.
        y : np.ndarray
            Target vector of shape ``(T,)``.
        scorer : callable
            ``scorer(tree, X, y) -> float``.  Higher is better.
        top_k : int
            Number of best trees to return.

        Returns
        -------
        list[ExpressionTree]
            Top-*K* trees ordered by descending score.
        """
        self.history = SearchHistory()

        # ------ Step 1: initialise random pool ------
        pool = self._init_pool(X, y, scorer)

        # ------ Step 2: main loop ------
        for t in range(self.iterations):
            tau_t = self._schedule.temperature(t)

            # Select elites (top elite_count by score).
            pool.sort(key=lambda s: s.score, reverse=True)
            elites = pool[: self.elite_count]

            # Perturb each elite to produce candidates.
            candidates: List[_ScoredTree] = []
            for elite in elites:
                mutated_tree = self._perturber.perturb(
                    elite.tree, self.grammar
                )
                mutated_score = self._safe_score(scorer, mutated_tree, X, y)
                delta = mutated_score - elite.score

                if self._schedule.metropolis_accept(delta, tau_t):
                    candidates.append(
                        _ScoredTree(tree=mutated_tree, score=mutated_score)
                    )
                else:
                    # Keep the original elite.
                    candidates.append(elite)

            # Rebuild pool: accepted candidates + remaining non-elites.
            non_elites = pool[self.elite_count :]
            pool = candidates + non_elites

            # Ensure pool is exactly pool_size (pad if needed).
            while len(pool) < self.pool_size:
                t_new = self.grammar.random_tree()
                s_new = self._safe_score(scorer, t_new, X, y)
                pool.append(_ScoredTree(tree=t_new, score=s_new))

            # ------ Diversity injection ------
            pool.sort(key=lambda s: s.score, reverse=True)
            n_replace = max(1, int(self.diversity_rate * self.pool_size))
            for i in range(n_replace):
                idx = len(pool) - 1 - i
                if idx < 0:
                    break
                fresh_tree = self.grammar.random_tree()
                fresh_score = self._safe_score(scorer, fresh_tree, X, y)
                pool[idx] = _ScoredTree(tree=fresh_tree, score=fresh_score)

            # ------ Record convergence diagnostics ------
            scores = [s.score for s in pool]
            self.history.best_scores.append(max(scores))
            self.history.mean_scores.append(float(np.mean(scores)))
            self.history.pool_sizes.append(len(pool))
            self.history.temperatures.append(tau_t)

            if (t + 1) % 20 == 0 or t == 0:
                logger.info(
                    "Iter %4d/%d | best=%.6f  mean=%.6f  tau=%.4f",
                    t + 1,
                    self.iterations,
                    self.history.best_scores[-1],
                    self.history.mean_scores[-1],
                    tau_t,
                )

        # ------ Step 3: filter and return top-K ------
        pool.sort(key=lambda s: s.score, reverse=True)
        top = pool[:top_k]
        return [s.tree for s in top]

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _init_pool(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scorer: Callable,
    ) -> List[_ScoredTree]:
        """Generate the initial random pool and score every member."""
        pool: List[_ScoredTree] = []
        for _ in range(self.pool_size):
            tree = self.grammar.random_tree()
            score = self._safe_score(scorer, tree, X, y)
            pool.append(_ScoredTree(tree=tree, score=score))
        return pool

    @staticmethod
    def _safe_score(
        scorer: Callable,
        tree: "ExpressionTree",
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Score a tree, returning ``-inf`` on any exception."""
        try:
            val = scorer(tree, X, y)
            if not np.isfinite(val):
                return float("-inf")
            return float(val)
        except Exception:
            logger.debug("Scoring failed for tree %s", tree, exc_info=True)
            return float("-inf")
