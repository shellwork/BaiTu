"""
Unified matrix oracle for Battleship active learning.

Both sources below are normalised to the same active-area matrix:
- direct Battleship board generation
- plate image readout via fixed-ROI RGB classification
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from battleship_env import BattleshipBoard, Ship
from plate.battleship_plate_readout import (
    ACTIVE_COLS,
    SHIP_LIQUID_RGB,
    WATER_LIQUID_RGB,
    mean_bgr_to_mean_rgb,
    query_well_fixed_geometry_rgb,
)
from plate.battleship_plate_simulation import get_fixed_well_geometry, simulate_photo_from_board


@dataclass
class ActiveMatrixReadout:
    matrix: np.ndarray
    unknown_mask: np.ndarray
    confidence: np.ndarray


def board_to_active_matrix(board: BattleshipBoard) -> np.ndarray:
    """Return the active 8x10 matrix used by the learner."""
    return board.grid.astype(np.int8).copy()


def image_to_active_matrix_readout(
    image_bgr: np.ndarray,
    geometry: Dict,
    *,
    active_cols: int = ACTIVE_COLS,
    inner_fraction: float = 0.60,
    **rgb_classify_kw,
) -> ActiveMatrixReadout:
    """
    Decode a plate image once into the Battleship-style active matrix.

    The output matrix is what downstream models query by table lookup.
    """
    rows = int(geometry["rows"])
    cols = min(active_cols, int(geometry["cols"]))
    matrix = np.zeros((rows, cols), dtype=np.int8)
    unknown_mask = np.zeros((rows, cols), dtype=bool)
    confidence = np.zeros((rows, cols), dtype=np.float32)

    for row in range(rows):
        for col in range(cols):
            label, mean_bgr, conf = query_well_fixed_geometry_rgb(
                image_bgr,
                row,
                col,
                geometry,
                inner_fraction=inner_fraction,
                **rgb_classify_kw,
            )
            if label == "ship":
                value = 1
            elif label == "water":
                value = 0
            else:
                unknown_mask[row, col] = True
                mean_rgb = mean_bgr_to_mean_rgb(mean_bgr)
                ds = float(np.linalg.norm(mean_rgb - SHIP_LIQUID_RGB))
                dw = float(np.linalg.norm(mean_rgb - WATER_LIQUID_RGB))
                value = 1 if ds < dw else 0
            matrix[row, col] = value
            confidence[row, col] = float(conf)

    return ActiveMatrixReadout(
        matrix=matrix,
        unknown_mask=unknown_mask,
        confidence=confidence,
    )


class BattleshipMatrixOracle:
    """
    Query a precomputed active matrix while keeping Battleship ship bookkeeping.

    The learner only sees `observed_hit`, which comes from the decoded matrix.
    The board is still advanced to preserve sunk-ship semantics and QC metrics.
    """

    def __init__(
        self,
        board: BattleshipBoard,
        observed_matrix: np.ndarray,
        *,
        unknown_mask: Optional[np.ndarray] = None,
    ):
        if observed_matrix.shape != board.grid.shape:
            raise ValueError(
                f"Observed matrix shape {observed_matrix.shape} does not match board shape {board.grid.shape}"
            )
        self.board = board
        self.observed_matrix = observed_matrix.astype(np.int8, copy=True)
        self.unknown_mask = (
            unknown_mask.astype(bool, copy=True)
            if unknown_mask is not None
            else np.zeros_like(self.observed_matrix, dtype=bool)
        )
        self.n_calls = 0
        self.n_unknown = 0
        self.n_cv_errors = 0

    @classmethod
    def from_board(cls, board: BattleshipBoard) -> "BattleshipMatrixOracle":
        return cls(board=board, observed_matrix=board_to_active_matrix(board))

    @classmethod
    def from_image(
        cls,
        board: BattleshipBoard,
        image_bgr: np.ndarray,
        geometry: Dict,
        *,
        inner_fraction: float = 0.60,
        **rgb_classify_kw,
    ) -> "BattleshipMatrixOracle":
        readout = image_to_active_matrix_readout(
            image_bgr,
            geometry,
            active_cols=board.cols,
            inner_fraction=inner_fraction,
            **rgb_classify_kw,
        )
        return cls(
            board=board,
            observed_matrix=readout.matrix,
            unknown_mask=readout.unknown_mask,
        )

    def query(self, row: int, col: int) -> Tuple[bool, Optional[Ship], bool]:
        self.n_calls += 1
        observed_hit = bool(self.observed_matrix[row, col])
        if self.unknown_mask[row, col]:
            self.n_unknown += 1

        actual_hit, actual_sunk_ship = self.board.query(row, col)
        if observed_hit != bool(actual_hit):
            self.n_cv_errors += 1

        sunk_for_model = actual_sunk_ship if (observed_hit and actual_hit) else None
        return observed_hit, sunk_for_model, bool(actual_hit)

    @property
    def cv_error_rate(self) -> float:
        return self.n_cv_errors / max(1, self.n_calls)

    @property
    def unknown_rate(self) -> float:
        return self.n_unknown / max(1, self.n_calls)


def make_battleship_oracle(
    board: BattleshipBoard,
    seed: int,
    *,
    oracle_mode: str = "board",
    rgb_l2_max: Optional[float] = None,
    rgb_per_channel_delta: Optional[float] = None,
) -> BattleshipMatrixOracle:
    """
    Build the shared oracle used by both experiment and campaign flows.

    `board` mode: matrix comes directly from the simulated Battleship grid.
    `image` mode: board -> synthetic image -> decoded matrix -> query-by-table.
    """
    if oracle_mode == "board":
        return BattleshipMatrixOracle.from_board(board)
    if oracle_mode == "image":
        image = simulate_photo_from_board(board, seed=seed)
        geometry = get_fixed_well_geometry(seed=seed)
        rgb_kw: Dict[str, float] = {}
        if rgb_l2_max is not None:
            rgb_kw["l2_max"] = float(rgb_l2_max)
        if rgb_per_channel_delta is not None:
            rgb_kw["per_channel_delta"] = float(rgb_per_channel_delta)
        return BattleshipMatrixOracle.from_image(
            board,
            image,
            geometry,
            **rgb_kw,
        )
    raise ValueError(f"Unsupported oracle_mode: {oracle_mode}")
