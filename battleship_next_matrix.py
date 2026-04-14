"""
8×12 实验结果矩阵 → 下一步检测位置矩阵
========================================
输入：当前读数结果，形状 (8, 12)。

编码约定（与 demo 里 observed 板一致）::

    np.nan 或 -1  : 尚未检测
    0             : 阴性 / 未命中（水）
    1             : 阳性 / 命中（船或 purple）

输出：同形状 8×12，在模型选中的「下一个孔」处为 1，其余为 0；
若可测位置已全部测完，则全 0。

支持两种信念模型（见 ``mode``）::

    battleship  — 默认舰艇约束（``Game``，8×12 棋盘）
    plate       — 96 孔板独立 Beta 模型（``BeliefModel(..., ship_sizes=[])``）
"""

from __future__ import annotations

from typing import Literal, Optional, Set, Tuple

import numpy as np

from battleship_model import BeliefModel, Game

Mode = Literal["battleship", "plate"]


def _is_untested(x, sentinel: int = -1) -> bool:
    if isinstance(x, (float, np.floating)) and np.isnan(x):
        return True
    try:
        return int(x) == sentinel
    except (ValueError, TypeError):
        return False


def _build_model_from_results(
    results: np.ndarray,
    *,
    mode: Mode,
    untested_sentinel: int = -1,
    ship_sizes: Optional[list] = None,
    prior_purple: float = 0.25,
    spatial_sigma: float = 1.5,
) -> BeliefModel:
    """Replay observations into a fresh belief model (order: row-major)."""
    if results.shape != (8, 12):
        raise ValueError(f"Expected results shape (8, 12), got {results.shape}")

    if mode == "plate":
        model: BeliefModel = BeliefModel(
            rows=8,
            cols=12,
            ship_sizes=[],
            prior_purple=prior_purple,
            spatial_sigma=spatial_sigma,
        )
    else:
        model = Game(
            board_rows=8,
            board_cols=12,
            ship_sizes=ship_sizes if ship_sizes is not None else [5, 4, 3, 3, 2],
        )

    for r in range(8):
        for c in range(12):
            v = results[r, c]
            if _is_untested(v, untested_sentinel):
                continue
            is_hit = bool(int(v))
            model.update(r, c, is_hit=is_hit, sunk_ship=None)

    return model


def next_test_matrix(
    results: np.ndarray,
    *,
    strategy: str = "prob",
    mode: Mode = "battleship",
    untested_sentinel: int = -1,
    ship_sizes: Optional[list] = None,
    prior_purple: float = 0.25,
    spatial_sigma: float = 1.5,
    selectable_mask: Optional[np.ndarray] = None,
    rng_seed: Optional[int] = None,
) -> np.ndarray:
    """
    根据当前结果矩阵，返回下一步应检测位置的 0/1 矩阵（单孔为 1）。

    Parameters
    ----------
    results
        (8, 12) 实验结果；未测孔为 ``nan`` 或整数 ``untested_sentinel``（默认 -1）。
    strategy
        与 ``BeliefModel.select_query`` 相同（plate 模式勿用 ``hunt_target`` / ``pro_solver``）。
    mode
        ``battleship`` 或 ``plate``。
    selectable_mask
        可选 (8, 12) bool；仅 ``True`` 的位置允许作为下一步（例如仅前 10 列为游戏区）。
        未提供时全部孔位均可选。
    rng_seed
        仅对随机策略生效；``None`` 时不改全局随机状态。

    Returns
    -------
    (8, 12) ``float64`` 矩阵，至多一个元素为 1.0。
    """
    if selectable_mask is not None and selectable_mask.shape != (8, 12):
        raise ValueError("selectable_mask must have shape (8, 12)")

    if mode == "plate" and strategy in ("hunt_target", "pro_solver"):
        raise ValueError("Plate mode does not support hunt_target / pro_solver; use prob, entropy, grid, or random.")

    results = np.asarray(results)
    if results.shape != (8, 12):
        raise ValueError(f"Expected results shape (8, 12), got {results.shape}")

    if rng_seed is not None:
        np.random.seed(rng_seed)

    model = _build_model_from_results(
        results,
        mode=mode,
        untested_sentinel=untested_sentinel,
        ship_sizes=ship_sizes,
        prior_purple=prior_purple,
        spatial_sigma=spatial_sigma,
    )

    allowed: Optional[Set[Tuple[int, int]]] = None
    if selectable_mask is not None:
        allowed = {
            (r, c)
            for r in range(8)
            for c in range(12)
            if bool(selectable_mask[r, c])
        }

    grid_order = [(r, c) for r in range(8) for c in range(12)]
    pos = model.select_query(strategy, grid_order=grid_order, allowed_cells=allowed)

    out = np.zeros((8, 12), dtype=np.float64)
    if pos is not None:
        r, c = pos
        out[r, c] = 1.0
    return out


if __name__ == "__main__":
    # 示例：全未测时 plate / prob 会选一个高先验孔；battleship 会按策略选一格
    blank = np.full((8, 12), -1, dtype=int)
    m_plate = next_test_matrix(blank, strategy="prob", mode="plate", rng_seed=0)
    m_ship = next_test_matrix(blank, strategy="prob", mode="battleship", rng_seed=0)
    print("blank plate next (1 cell):", int(m_plate.sum()), "argmax", np.argwhere(m_plate == 1))
    print("blank battleship next (1 cell):", int(m_ship.sum()), "argmax", np.argwhere(m_ship == 1))

    partial = blank.copy()
    partial[0, 0] = 0
    partial[0, 1] = 1
    m2 = next_test_matrix(partial, strategy="entropy", mode="plate", rng_seed=1)
    print("after 2 reads:", np.argwhere(m2 == 1))
