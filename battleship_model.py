"""
Active Learning Belief Model
==============================
Two operating modes, same interface:

  ship mode  (default)
      Bayesian probability density over all valid ship placements.
      P(ship at cell | obs) ∝ #consistent placements covering that cell.
      Requires ship_sizes to be provided.

  plate mode  (ship_sizes=[] or plate_mode=True)
      Independent Beta(α, β) posteriors per cell with spatial Gaussian
      spreading.  Used for 96-well plates where there are no shape
      constraints on positive wells.

Both modes expose the same public API:
  .prob_map          – P(positive) for every cell
  .get_entropy_map() – uncertainty map for uncertainty sampling
  .update(r, c, is_hit, sunk_ship=None)
  .select_query(strategy, grid_order=None)

Strategies
----------
  random       : uniform random baseline
  prob         : argmax P(hit)      — exploitation
  entropy      : argmax H(p)        — uncertainty sampling
  hunt_target  : battleship heuristic (ship mode only)
  grid         : row-by-row scan    (plate mode)

``Game`` is a thin subclass that uses ``board_rows`` / ``board_cols`` keyword
names (merge compatibility with scripts that imported ``Game``).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


class BeliefModel:
    """
    Unified belief model for battleship and 96-well plate active learning.

    Parameters
    ----------
    board_size   : int – used when rows == cols (legacy, default 10)
    rows, cols   : override for non-square boards (e.g. 8×10 plate area)
    ship_sizes   : list of ship sizes; pass [] for plate mode
    plate_mode   : force plate mode regardless of ship_sizes
    prior_purple : prior P(positive) for plate mode (default 0.25)
    spatial_sigma: Gaussian spread radius in plate mode (default 1.5 cells)
    """

    def __init__(
        self,
        board_size:     int              = 10,
        rows:           Optional[int]    = None,
        cols:           Optional[int]    = None,
        ship_sizes:     Optional[List[int]] = None,
        plate_mode:     bool             = False,
        prior_purple:   float            = 0.25,
        spatial_sigma:  float            = 1.5,
        spatial_strength: float          = 0.25,
    ):
        self.rows = rows if rows is not None else board_size
        self.cols = cols if cols is not None else board_size

        # Determine operating mode
        if plate_mode or (ship_sizes is not None and len(ship_sizes) == 0):
            self._plate_mode = True
            self.ship_sizes     = []
            self.remaining_sizes: List[int] = []
        else:
            self._plate_mode = False
            self.ship_sizes     = sorted(ship_sizes or [5, 4, 3, 3, 2], reverse=True)
            self.remaining_sizes = list(self.ship_sizes)

        # Observations
        self.hits:       Set[Tuple[int, int]] = set()
        self.misses:     Set[Tuple[int, int]] = set()
        self.sunk_cells: Set[Tuple[int, int]] = set()
        self.n_observations: int = 0

        # Plate-mode Beta posterior parameters
        self._prior_purple  = prior_purple
        self._spatial_sigma = spatial_sigma
        self._spatial_strength = spatial_strength
        alpha0 = 4.0 * prior_purple
        beta0  = 4.0 * (1.0 - prior_purple)
        self._alpha = np.full((self.rows, self.cols), alpha0)
        self._beta  = np.full((self.rows, self.cols), beta0)

        self.prob_map: np.ndarray = np.zeros((self.rows, self.cols))
        self._update_prob_map()

    # ------------------------------------------------------------------
    # Update (Bayesian posterior update)
    # ------------------------------------------------------------------

    def update(
        self,
        row:       int,
        col:       int,
        is_hit:    bool,
        sunk_ship  = None,
    ):
        """Incorporate a new oracle observation."""
        self.n_observations += 1

        if is_hit:
            self.hits.add((row, col))
        else:
            self.misses.add((row, col))

        if sunk_ship is not None and not self._plate_mode:
            self.remaining_sizes.remove(sunk_ship.size)
            for pos in sunk_ship.positions:
                self.hits.discard(pos)
                self.sunk_cells.add(pos)

        # Plate mode: update Beta posteriors with spatial spreading
        if self._plate_mode:
            delta = np.zeros((self.rows, self.cols))
            delta[row, col] = 1.0
            spread = gaussian_filter(delta, sigma=self._spatial_sigma, mode="constant")
            spread[row, col] = 0.0
            peak = spread.max()
            if peak > 0:
                spread = spread / peak * self._spatial_strength

            if is_hit:
                self._alpha[row, col] += 1.0
                self._alpha += spread
            else:
                self._beta[row, col]  += 1.0
                self._beta  += spread

        self._update_prob_map()

    # ------------------------------------------------------------------
    # Probability map
    # ------------------------------------------------------------------

    def _queried(self) -> Set[Tuple[int, int]]:
        return self.hits | self.misses | self.sunk_cells

    def _update_prob_map(self):
        if self._plate_mode:
            self._update_prob_map_plate()
        else:
            self._update_prob_map_ship()

    def _update_prob_map_plate(self):
        """Posterior mean of Beta(α, β) per cell."""
        density = self._alpha / (self._alpha + self._beta)
        for r, c in self._queried():
            density[r, c] = 0.0
        self.prob_map = density

    def _update_prob_map_ship(self):
        """
        Placement-density map for ship mode.

        For each ship size s remaining, count valid placements covering each
        unqueried cell; normalise to probability.
        """
        nr, nc  = self.rows, self.cols
        density = np.zeros((nr, nc), dtype=float)
        queried = self._queried()

        for size in set(self.remaining_sizes):
            multiplier    = self.remaining_sizes.count(size)
            size_density  = np.zeros((nr, nc), dtype=float)
            n_valid       = 0

            # Horizontal
            for r in range(nr):
                for c in range(nc - size + 1):
                    positions = [(r, c + i) for i in range(size)]
                    if self._is_valid_placement(positions):
                        for pr, pc in positions:
                            if (pr, pc) not in queried:
                                size_density[pr, pc] += 1.0
                        n_valid += 1

            # Vertical
            for c in range(nc):
                for r in range(nr - size + 1):
                    positions = [(r + i, c) for i in range(size)]
                    if self._is_valid_placement(positions):
                        for pr, pc in positions:
                            if (pr, pc) not in queried:
                                size_density[pr, pc] += 1.0
                        n_valid += 1

            if n_valid > 0:
                density += multiplier * size_density / n_valid

        total = density.sum()
        self.prob_map = density / total if total > 0 else density

    def _is_valid_placement(self, positions: List[Tuple[int, int]]) -> bool:
        for r, c in positions:
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                return False
            if (r, c) in self.misses:
                return False
            if (r, c) in self.sunk_cells:
                return False
        return True

    # ------------------------------------------------------------------
    # Acquisition functions
    # ------------------------------------------------------------------

    def get_entropy_map(self) -> np.ndarray:
        """
        Binary entropy H(p) = -p log2 p - (1-p) log2 (1-p).

        High entropy (p≈0.5) → maximum uncertainty → uncertainty sampling.
        Queried cells are zeroed so they are never re-selected.
        """
        p = np.clip(self.prob_map, 1e-9, 1 - 1e-9)
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
        for r, c in self._queried():
            entropy[r, c] = 0.0
        return entropy

    def select_query(
        self,
        strategy:   str = "prob",
        grid_order: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Acquisition function: select the next cell to query.

        Parameters
        ----------
        strategy   : 'random' | 'prob' | 'entropy' | 'hunt_target' | 'grid'
        grid_order : precomputed scan order for the 'grid' strategy

        Returns
        -------
        (row, col) or None if all cells already queried.
        """
        nr, nc  = self.rows, self.cols
        queried = self._queried()
        available = [
            (r, c) for r in range(nr) for c in range(nc)
            if (r, c) not in queried
        ]

        if not available:
            return None

        if strategy == "random":
            return random.choice(available)

        elif strategy == "prob":
            scores = self.prob_map.copy()
            for r, c in queried:
                scores[r, c] = -np.inf
            r, c = np.unravel_index(np.argmax(scores), scores.shape)
            return (int(r), int(c))

        elif strategy == "entropy":
            scores = self.get_entropy_map()
            for r, c in queried:
                scores[r, c] = -np.inf
            r, c = np.unravel_index(np.argmax(scores), scores.shape)
            return (int(r), int(c))

        elif strategy == "grid":
            order = grid_order or [
                (r, c) for r in range(nr) for c in range(nc)
            ]
            for pos in order:
                if pos not in queried:
                    return pos
            return available[0]

        elif strategy == "hunt_target":
            unaccounted = self.hits - self.sunk_cells
            if unaccounted:
                candidates = []
                for hr, hc in unaccounted:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr2, nc2 = hr + dr, hc + dc
                        if (0 <= nr2 < nr and 0 <= nc2 < nc
                                and (nr2, nc2) not in queried):
                            candidates.append((nr2, nc2))
                if candidates:
                    return random.choice(candidates)
            checkers = [(r, c) for r, c in available if (r + c) % 2 == 0]
            return random.choice(checkers if checkers else available)

        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: random, prob, entropy, hunt_target, grid"
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict:
        return {
            "n_observations":  self.n_observations,
            "n_hits":          len(self.hits),
            "n_misses":        len(self.misses),
            "remaining_ships": len(self.remaining_sizes),
            "plate_mode":      self._plate_mode,
            "max_prob":        float(self.prob_map.max()),
            "total_entropy":   float(self.get_entropy_map().sum()),
        }


class Game(BeliefModel):
    """
    Same as ``BeliefModel`` but with ``board_rows`` / ``board_cols`` parameters
    for scripts that used the merged-in ``Game`` API.
    """

    def __init__(
        self,
        board_rows: int = 10,
        board_cols: int = 10,
        ship_sizes: Optional[List[int]] = None,
        **kwargs,
    ):
        super().__init__(rows=board_rows, cols=board_cols, ship_sizes=ship_sizes, **kwargs)
