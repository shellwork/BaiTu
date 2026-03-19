"""
Active Learning Belief Model for Battleship
============================================
Maintains P(ship | observations) for each cell via Bayesian probability
density over all valid ship placements.

Active Learning Analogy
-----------------------
  - prob_map      ↔  model prediction / posterior mean
  - entropy_map   ↔  model uncertainty (used for uncertainty sampling)
  - select_query  ↔  acquisition function

Strategies
----------
  random       : uniform random baseline (no model)
  prob         : query argmax P(hit)  — pure exploitation
  entropy      : query argmax H(p)    — pure uncertainty sampling
  hunt_target  : classic heuristic (no probabilistic model needed)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


class BeliefModel:
    """
    Bayesian belief model using probability density over ship placements.

    After each observation we recount how many *consistent* ship placements
    cover each unqueried cell.  Normalising gives P(ship at cell | obs).

    This is an independent-ships approximation (exact per-ship, approximate
    for joint) but works very well in practice.
    """

    def __init__(self, board_size: int = 10, ship_sizes: Optional[List[int]] = None):
        self.board_size = board_size
        self.ship_sizes = sorted(ship_sizes or [5, 4, 3, 3, 2], reverse=True)

        self.hits: Set[Tuple[int, int]] = set()
        self.misses: Set[Tuple[int, int]] = set()
        self.sunk_cells: Set[Tuple[int, int]] = set()   # cells of confirmed-sunk ships
        self.remaining_sizes: List[int] = list(self.ship_sizes)

        self.prob_map: np.ndarray = np.zeros((board_size, board_size))
        self.n_observations: int = 0
        self._update_prob_map()

    # ------------------------------------------------------------------
    # Update (Bayesian posterior update)
    # ------------------------------------------------------------------

    def update(self, row: int, col: int, is_hit: bool, sunk_ship=None):
        """Incorporate a new oracle observation into the belief."""
        self.n_observations += 1
        if is_hit:
            self.hits.add((row, col))
        else:
            self.misses.add((row, col))

        if sunk_ship is not None:
            # Remove ship size from remaining fleet
            self.remaining_sizes.remove(sunk_ship.size)
            # Move all its cells from 'hits' to 'sunk_cells'
            for pos in sunk_ship.positions:
                self.hits.discard(pos)
                self.sunk_cells.add(pos)

        self._update_prob_map()

    # ------------------------------------------------------------------
    # Internal: probability density map
    # ------------------------------------------------------------------

    def _queried(self) -> Set[Tuple[int, int]]:
        return self.hits | self.misses | self.sunk_cells

    def _is_valid_placement(self, positions: List[Tuple[int, int]]) -> bool:
        """Return True iff the placement is consistent with all observations."""
        n = self.board_size
        for r, c in positions:
            if not (0 <= r < n and 0 <= c < n):
                return False
            if (r, c) in self.misses:
                return False
            if (r, c) in self.sunk_cells:
                return False
        return True

    def _update_prob_map(self):
        """
        Recompute probability density map.

        For each distinct ship size s still remaining:
          density[r,c] += (number of valid placements of size s covering (r,c))

        Cells that have already been queried are zeroed out (not selectable).

        The density is then L1-normalised to produce a proper probability map.
        """
        n = self.board_size
        density = np.zeros((n, n), dtype=float)
        queried = self._queried()

        for size in set(self.remaining_sizes):
            multiplier = self.remaining_sizes.count(size)
            size_density = np.zeros((n, n), dtype=float)
            n_valid_placements = 0

            # Horizontal
            for r in range(n):
                for c in range(n - size + 1):
                    positions = [(r, c + i) for i in range(size)]
                    if self._is_valid_placement(positions):
                        for pr, pc in positions:
                            if (pr, pc) not in queried:
                                size_density[pr, pc] += 1.0
                        n_valid_placements += 1

            # Vertical
            for c in range(n):
                for r in range(n - size + 1):
                    positions = [(r + i, c) for i in range(size)]
                    if self._is_valid_placement(positions):
                        for pr, pc in positions:
                            if (pr, pc) not in queried:
                                size_density[pr, pc] += 1.0
                        n_valid_placements += 1

            if n_valid_placements > 0:
                density += multiplier * size_density / n_valid_placements

        total = density.sum()
        self.prob_map = density / total if total > 0 else density

    # ------------------------------------------------------------------
    # Acquisition functions
    # ------------------------------------------------------------------

    def get_entropy_map(self) -> np.ndarray:
        """
        Binary entropy H(p) = -p log2 p - (1-p) log2 (1-p).

        High entropy ≈ 0.5 probability → maximum uncertainty.
        This is the standard *uncertainty sampling* score in active learning.
        """
        p = np.clip(self.prob_map, 1e-9, 1 - 1e-9)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        # Zero out queried cells so they are never selected
        for r, c in self._queried():
            entropy[r, c] = 0.0
        return entropy

    def select_query(self, strategy: str = "prob") -> Optional[Tuple[int, int]]:
        """
        Acquisition function: select which cell to query next.

        Parameters
        ----------
        strategy : one of {'random', 'prob', 'entropy', 'hunt_target'}

        Returns
        -------
        (row, col) or None if no cells available.
        """
        n = self.board_size
        queried = self._queried()
        available = [(r, c) for r in range(n) for c in range(n) if (r, c) not in queried]

        if not available:
            return None

        # ── Random baseline ────────────────────────────────────────────
        if strategy == "random":
            return random.choice(available)

        # ── Exploitation: query highest P(hit) ─────────────────────────
        elif strategy == "prob":
            scores = self.prob_map.copy()
            for r, c in queried:
                scores[r, c] = -np.inf
            r, c = np.unravel_index(np.argmax(scores), scores.shape)
            return (int(r), int(c))

        # ── Uncertainty sampling: query highest entropy ─────────────────
        elif strategy == "entropy":
            scores = self.get_entropy_map()
            for r, c in queried:
                scores[r, c] = -np.inf
            r, c = np.unravel_index(np.argmax(scores), scores.shape)
            return (int(r), int(c))

        # ── Classic heuristic: Hunt (checkerboard) → Target (neighbors) ─
        elif strategy == "hunt_target":
            unaccounted_hits = self.hits - self.sunk_cells
            if unaccounted_hits:
                candidates = []
                for hr, hc in unaccounted_hits:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = hr + dr, hc + dc
                        if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in queried:
                            candidates.append((nr, nc))
                if candidates:
                    return random.choice(candidates)
            # Hunt: checkerboard pattern reduces search space by half
            checkers = [(r, c) for r, c in available if (r + c) % 2 == 0]
            pool = checkers if checkers else available
            return random.choice(pool)

        else:
            raise ValueError(f"Unknown strategy '{strategy}'. "
                             f"Choose from: random, prob, entropy, hunt_target")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        return {
            "n_observations": self.n_observations,
            "n_hits": len(self.hits),
            "n_misses": len(self.misses),
            "remaining_ships": len(self.remaining_sizes),
            "max_prob": float(self.prob_map.max()),
            "total_entropy": float(self.get_entropy_map().sum()),
        }
