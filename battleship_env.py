"""
Battleship Game Environment
===========================
Provides the ground-truth oracle for active learning experiments.
Ships are placed randomly; each cell query returns hit/miss (the label).
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Optional


@dataclass
class Ship:
    size: int
    positions: frozenset  # set of (row, col) tuples
    hits: Set[Tuple[int, int]] = field(default_factory=set)

    def register_hit(self, pos: Tuple[int, int]) -> bool:
        if pos in self.positions:
            self.hits.add(pos)
            return True
        return False

    def is_sunk(self) -> bool:
        return len(self.hits) == self.size


class BattleshipBoard:
    """
    Standard 10x10 Battleship board.

    Analogy to active learning:
        pool        = all unqueried cells
        label       = hit (1) or miss (0)
        oracle cost = 1 query per cell
    """

    DEFAULT_SHIPS = [5, 4, 3, 3, 2]   # Carrier, Battleship, Cruiser, Sub, Destroyer

    def __init__(
        self,
        size: int = 10,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        ship_sizes: Optional[List[int]] = None,
        seed: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        size : int
            Square board side length when ``rows``/``cols`` are omitted.
        rows, cols : optional int
            Rectangular board (e.g. 8×12 for a 96-well plate). If only one
            of ``rows``/``cols`` is given, the other defaults to ``size``.
        """
        if rows is not None or cols is not None:
            self.rows = int(rows if rows is not None else size)
            self.cols = int(cols if cols is not None else size)
        else:
            self.rows = self.cols = int(size)
        self.size = max(self.rows, self.cols)  # legacy alias for square callers

        self.ship_sizes = ship_sizes or self.DEFAULT_SHIPS
        self._rng = random.Random(seed)

        # Ground-truth grid (hidden from learner)
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        # Observation grid visible to learner: -1=unknown, 0=miss, 1=hit
        self.observed = np.full((self.rows, self.cols), -1, dtype=int)

        self.ships: List[Ship] = []
        self.n_queries: int = 0
        self.query_history: List[Tuple[int, int, bool]] = []

        self._place_ships()
        self.total_ship_cells: int = int(np.sum(self.grid))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _place_ships(self):
        for size in self.ship_sizes:
            can_h = self.cols >= size
            can_v = self.rows >= size
            if not (can_h or can_v):
                raise RuntimeError(
                    f"Ship of size {size} cannot fit on a {self.rows}×{self.cols} board"
                )
            placed = False
            for _ in range(100_000):
                horizontal = self._rng.choice([True, False]) if (can_h and can_v) else can_h
                if horizontal:
                    r = self._rng.randint(0, self.rows - 1)
                    c = self._rng.randint(0, self.cols - size)
                    positions = [(r, c + i) for i in range(size)]
                else:
                    r = self._rng.randint(0, self.rows - size)
                    c = self._rng.randint(0, self.cols - 1)
                    positions = [(r + i, c) for i in range(size)]

                if all(self.grid[p[0], p[1]] == 0 for p in positions):
                    for p in positions:
                        self.grid[p[0], p[1]] = 1
                    self.ships.append(Ship(size=size, positions=frozenset(positions)))
                    placed = True
                    break

            if not placed:
                raise RuntimeError(f"Could not place ship of size {size}")

    # ------------------------------------------------------------------
    # Oracle
    # ------------------------------------------------------------------

    def query(self, row: int, col: int) -> Tuple[bool, Optional[Ship]]:
        """
        Query a cell – this is the *labelling oracle* in active learning.

        Returns
        -------
        is_hit   : bool
        sunk_ship: Ship | None  – the ship object if this query sank it
        """
        assert 0 <= row < self.rows and 0 <= col < self.cols
        assert self.observed[row, col] == -1, f"Cell ({row},{col}) already queried"

        self.n_queries += 1
        is_hit = bool(self.grid[row, col])
        self.observed[row, col] = int(is_hit)

        sunk_ship: Optional[Ship] = None
        if is_hit:
            for ship in self.ships:
                if ship.register_hit((row, col)) and ship.is_sunk():
                    sunk_ship = ship

        self.query_history.append((row, col, is_hit))
        return is_hit, sunk_ship

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def is_game_over(self) -> bool:
        return all(s.is_sunk() for s in self.ships)

    def get_sunk_ships(self) -> List[Ship]:
        return [s for s in self.ships if s.is_sunk()]

    def get_unsunk_ships(self) -> List[Ship]:
        return [s for s in self.ships if not s.is_sunk()]

    def get_remaining_ship_cells(self) -> int:
        return sum(s.size - len(s.hits) for s in self.ships if not s.is_sunk())

    def get_available_cells(self) -> List[Tuple[int, int]]:
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.observed[r, c] == -1
        ]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        sym = {-1: "·", 0: "O", 1: "X"}
        header = "   " + " ".join(str(c) for c in range(self.cols))
        rows = [header]
        for r in range(self.rows):
            row_str = f"{r:2d} " + " ".join(sym[self.observed[r, c]] for c in range(self.cols))
            rows.append(row_str)
        return "\n".join(rows)
