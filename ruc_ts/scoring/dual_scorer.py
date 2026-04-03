"""Dual-stage scorer combining OLS and MLP surrogates.

    s = alpha * s_OLS  +  (1 - alpha) * s_MLP

Only the top-*M* candidates from the OLS stage are advanced to the
(more expensive) MLP stage, making the two-stage pipeline efficient.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from ruc_ts.scoring.mlp_surrogate import MLPSurrogate
from ruc_ts.scoring.ols_surrogate import OLSSurrogate

if TYPE_CHECKING:
    from ruc_ts.grammar.tree import ExpressionTree

logger = logging.getLogger(__name__)


class DualScorer:
    """Two-stage scorer: OLS screening followed by MLP refinement.

    Parameters
    ----------
    alpha : float
        Blending weight.  ``s = alpha * s_OLS + (1 - alpha) * s_MLP``.
    top_m : int
        Number of top OLS candidates promoted to the MLP stage.
    ols : OLSSurrogate | None
        Custom OLS surrogate instance.  A default is created if ``None``.
    mlp : MLPSurrogate | None
        Custom MLP surrogate instance.  A default is created if ``None``.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        top_m: int = 100,
        ols: Optional[OLSSurrogate] = None,
        mlp: Optional[MLPSurrogate] = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if top_m < 1:
            raise ValueError(f"top_m must be >= 1, got {top_m}")

        self.alpha = alpha
        self.top_m = top_m
        self.ols = ols if ols is not None else OLSSurrogate()
        self.mlp = mlp if mlp is not None else MLPSurrogate()

    # -----------------------------------------------------------------
    # Single-feature scoring  (used by RUCSearch scorer callback)
    # -----------------------------------------------------------------

    def score(
        self,
        feature_values: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Score a single feature using OLS only (fast path).

        This is the scorer interface expected by
        ``RUCSearch.search(scorer=...)``.  During the search loop every
        candidate is evaluated with the cheap OLS surrogate; the full
        dual score is computed only in :meth:`score_batch` after the
        search terminates.
        """
        return self.ols.score(feature_values, target)

    # -----------------------------------------------------------------
    # Batch scoring  (two-stage pipeline)
    # -----------------------------------------------------------------

    def score_batch(
        self,
        trees: List["ExpressionTree"],
        X: np.ndarray,
        y: np.ndarray,
        evaluator=None,
    ) -> List[Tuple["ExpressionTree", float]]:
        """Score a batch of trees through the full dual-stage pipeline.

        Parameters
        ----------
        trees : list[ExpressionTree]
            Candidate expression trees.
        X : np.ndarray
            Raw input data ``(T, D)``.
        y : np.ndarray
            Target vector ``(T,)``.
        evaluator : callable, optional
            ``evaluator(tree, X) -> np.ndarray`` returning the
            materialised feature vector.  If ``None`` the tree's own
            ``evaluate`` method is used.

        Returns
        -------
        list[tuple[ExpressionTree, float]]
            ``(tree, dual_score)`` pairs sorted by descending score.
        """
        eval_fn = evaluator or (lambda tree, X_: tree.evaluate(X_))

        # --- Stage 1: OLS screening ---
        ols_scored: List[Tuple["ExpressionTree", np.ndarray, float]] = []
        for tree in trees:
            try:
                fv = np.asarray(eval_fn(tree, X), dtype=np.float64).ravel()
                if not np.all(np.isfinite(fv)):
                    continue
                s_ols = self.ols.score(fv, y)
                if not np.isfinite(s_ols):
                    continue
                ols_scored.append((tree, fv, s_ols))
            except Exception:
                logger.debug("OLS eval failed for %s", tree, exc_info=True)
                continue

        # Sort descending by OLS score and keep top-M.
        ols_scored.sort(key=lambda x: x[2], reverse=True)
        promoted = ols_scored[: self.top_m]

        # --- Stage 2: MLP refinement ---
        results: List[Tuple["ExpressionTree", float]] = []
        for tree, fv, s_ols in promoted:
            try:
                s_mlp = self.mlp.score(fv, y)
                if not np.isfinite(s_mlp):
                    s_mlp = 0.0
            except Exception:
                logger.debug("MLP eval failed for %s", tree, exc_info=True)
                s_mlp = 0.0

            dual = self.alpha * s_ols + (1.0 - self.alpha) * s_mlp
            results.append((tree, float(dual)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
