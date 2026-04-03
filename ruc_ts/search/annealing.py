"""Linear annealing schedule with Metropolis acceptance for RUC-TS.

Implements the temperature schedule:

    tau_t = tau_0 * (1 - t / I)

and the Metropolis criterion:

    accept if delta_score > 0
    else accept with probability  exp(delta_score / tau_t)
"""

from __future__ import annotations

import math
import random
from typing import Optional


class AnnealingSchedule:
    """Linear annealing schedule with Metropolis acceptance.

    Parameters
    ----------
    tau_0 : float
        Initial temperature.
    total_iterations : int
        Total number of search iterations (*I*).
    rng_seed : int | None, optional
        Seed for reproducible accept/reject decisions.
    """

    def __init__(
        self,
        tau_0: float = 1.0,
        total_iterations: int = 200,
        rng_seed: Optional[int] = None,
    ) -> None:
        if tau_0 <= 0:
            raise ValueError(f"tau_0 must be positive, got {tau_0}")
        if total_iterations < 1:
            raise ValueError(
                f"total_iterations must be >= 1, got {total_iterations}"
            )
        self.tau_0 = tau_0
        self.total_iterations = total_iterations
        self._rng = random.Random(rng_seed)

    # -----------------------------------------------------------------
    # Temperature
    # -----------------------------------------------------------------

    def temperature(self, t: int) -> float:
        """Compute the temperature at iteration *t*.

        Returns ``tau_0 * (1 - t / I)``, clamped to a small positive
        floor to avoid division-by-zero at the final iteration.
        """
        ratio = t / self.total_iterations
        tau = self.tau_0 * (1.0 - ratio)
        # Clamp to a small positive value so exp() stays finite.
        return max(tau, 1e-12)

    # -----------------------------------------------------------------
    # Metropolis criterion
    # -----------------------------------------------------------------

    def metropolis_accept(
        self,
        delta_score: float,
        temperature: float,
    ) -> bool:
        """Decide whether to accept a candidate solution.

        Parameters
        ----------
        delta_score : float
            ``score(candidate) - score(current)``.  Positive means the
            candidate is *better*.
        temperature : float
            Current annealing temperature (``tau_t``).

        Returns
        -------
        bool
            ``True`` if the candidate should be accepted.
        """
        if delta_score > 0:
            return True

        if temperature <= 0:
            # Frozen schedule -- only accept strict improvements.
            return False

        acceptance_prob = math.exp(delta_score / temperature)
        return self._rng.random() < acceptance_prob
