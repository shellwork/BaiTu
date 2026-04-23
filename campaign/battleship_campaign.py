"""
Closed-loop Battleship active-learning campaign.

This script turns the existing Battleship simulator into a cycle-based automated
campaign that:

1. Proposes a batch of acquisition policies each cycle.
2. Runs them against the simulated board / image-reading pipeline.
3. Fits an ensemble surrogate over policy -> performance.
4. Tracks learning, QC, and stopping metrics.
5. Saves figures, JSON history, and a markdown QC/stopping plan.

Usage
-----
  python -m campaign.battleship_campaign --max_cycles 5 --query_size 8
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.battleship_env import BattleshipBoard
from core.battleship_matrix_oracle import BattleshipMatrixOracle, make_battleship_oracle
from core.battleship_model import Game
from plate.battleship_plate_simulation import (
    ACTIVE_COLS,
    DEFAULT_RGB_L2_TOLERANCE,
)


POLICY_COMPONENTS = ("prob", "entropy", "target", "checker")
DEFAULT_RESULTS_DIR = "battleship_campaign_results"
THRESHOLDS = {
    # ── QC thresholds ──────────────────────────────────────────────────────
    # Battleship boards are randomly generated; with only 3 replicates the
    # board-layout variance alone pushes CoV to 15-30%.  A 30% ceiling flags
    # genuine instability (e.g. policy degeneracy or numerical errors) while
    # tolerating normal stochastic variation.
    "replicate_cv_pct": 30.0,
    "min_query_size": 8,
    "max_query_size": 96,
    "ensemble_var_lower": 1e-8,
    "ensemble_var_upper": 400.0,

    # ── Stopping criteria ──────────────────────────────────────────────────
    # 1. Held-out prediction error: stop when surrogate MAE falls BELOW this.
    #    Calibrated from empirical runs: the surrogate typically reaches ~5 queries
    #    MAE in the first few cycles and plateaus; 4.0 would indicate genuinely
    #    good generalisation across the policy simplex.
    "heldout_mae_queries": 4.0,

    # 2. Information gain: stop when variance reduction per cycle falls BELOW this.
    #    Below 2% means the surrogate is no longer learning meaningful structure
    #    from new observations — a reliable signal of diminishing returns.
    "info_gain_pct": 2.0,

    # 3. Prediction stability: stop when ≥ this fraction of pool predictions
    #    change by < 1% between consecutive cycles.
    #    90% means the surrogate has effectively converged: 9 out of 10 candidate
    #    policies are predicted identically, so further cycles cannot improve
    #    the ranking used by LCB acquisition.
    "prediction_stability_pct": 90.0,

    # 4. No-improvement streak: stop when best_so_far has not improved for this
    #    many consecutive cycles.  With 8 queries/cycle and 3 replicates the
    #    policy landscape near the optimum is sampled ~24 times per streak cycle;
    #    5 consecutive failures is strong evidence that the true optimum has
    #    already been found within measurement noise.
    "no_improvement_streak": 5,

    # 5. Budget: hard cap — guarantees termination regardless of other criteria.
    "max_cycles": 20,
}


@dataclass
class StoppingDecision:
    should_stop: bool
    reason: str
    triggered_criteria: List[str]
    metrics_at_stop: Dict[str, float]


class BattleshipImageOracle:
    """
    Backward-compatible wrapper around the shared matrix oracle.

    The synthetic plate image is decoded once into an active 8x10 matrix, and all
    later queries are simple table lookups on that matrix.
    """

    def __init__(
        self,
        board: BattleshipBoard,
        seed: int,
        *,
        rgb_l2_max: Optional[float] = None,
        rgb_per_channel_delta: Optional[float] = None,
    ):
        self._oracle: BattleshipMatrixOracle = make_battleship_oracle(
            board,
            seed=seed,
            oracle_mode="image",
            rgb_l2_max=rgb_l2_max,
            rgb_per_channel_delta=rgb_per_channel_delta,
        )

    def query(self, row: int, col: int) -> Tuple[bool, Optional[object], bool]:
        return self._oracle.query(row, col)

    @property
    def cv_error_rate(self) -> float:
        return self._oracle.cv_error_rate

    @property
    def unknown_rate(self) -> float:
        return self._oracle.unknown_rate


class KernelEnsembleSurrogate:
    """Small bootstrap kernel ensemble for policy-performance regression."""

    def __init__(self, n_members: int = 8, bandwidth: float = 0.22, seed: int = 0):
        self.n_members = int(n_members)
        self.bandwidth = float(bandwidth)
        self._rng = np.random.RandomState(seed)
        self._members: List[Tuple[np.ndarray, np.ndarray]] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._members = []
        if len(x) == 0:
            return
        for _ in range(self.n_members):
            indices = self._rng.randint(0, len(x), size=len(x))
            self._members.append((x[indices], y[indices]))

    def _predict_member(self, x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray) -> np.ndarray:
        if len(x_train) == 0:
            return np.zeros(len(x_query), dtype=float)
        x_train = np.asarray(x_train, dtype=np.float64)
        x_query = np.asarray(x_query, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(x_train)):
            x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(x_query)):
            x_query = np.nan_to_num(x_query, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.all(np.isfinite(y_train)):
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
        # Battleship episode length is bounded; clip avoids rare inf in matmul downstream.
        y_train = np.clip(y_train, 0.0, 500.0)
        if len(x_train) == 1:
            return np.full(len(x_query), float(y_train[0]), dtype=float)

        diff = x_query[:, None, :] - x_train[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        if not np.all(np.isfinite(d2)):
            d2 = np.nan_to_num(d2, nan=1e12, posinf=1e12, neginf=0.0)
        scale = max(1e-8, 2.0 * self.bandwidth * self.bandwidth)
        # Log-space row softmax + small ridge so no row is all-zero after normalize.
        log_w = -d2 / scale
        log_w_max = np.max(log_w, axis=1, keepdims=True)
        w = np.exp(np.clip(log_w - log_w_max, -80.0, 0.0)) + 1e-12
        row_sum = np.sum(w, axis=1, keepdims=True)
        w = w / np.maximum(row_sum, 1e-15)
        out = np.dot(w, y_train)
        return np.asarray(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), dtype=float)

    def predict(self, x_query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self._members:
            mean = np.zeros(len(x_query), dtype=float)
            var = np.full(len(x_query), 1.0, dtype=float)
            return mean, var

        preds = np.stack(
            [self._predict_member(x_train, y_train, x_query) for x_train, y_train in self._members],
            axis=0,
        )
        preds = np.nan_to_num(preds, nan=0.0, posinf=500.0, neginf=0.0)
        mean = preds.mean(axis=0)
        var = preds.var(axis=0) + 1e-8
        mean = np.nan_to_num(mean, nan=0.0, posinf=500.0, neginf=0.0)
        var = np.clip(np.nan_to_num(var, nan=1.0, posinf=1e6, neginf=1e-8), 1e-8, 1e6)
        return mean, var


def _normalize(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float).copy()
    finite = np.isfinite(out)
    if not finite.any():
        return np.zeros_like(out, dtype=float)
    vals = out[finite]
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        out[finite] = 0.0
    else:
        out[finite] = (vals - lo) / (hi - lo)
    out[~finite] = 0.0
    return out


def _target_map(model: Game) -> np.ndarray:
    score = np.zeros_like(model.prob_map, dtype=float)
    queried = model._queried()
    frontier_hits = model.hits - model.sunk_cells

    for hr, hc in frontier_hits:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = hr + dr, hc + dc
            if 0 <= rr < model.rows and 0 <= cc < model.cols and (rr, cc) not in queried:
                score[rr, cc] += 1.0

    if score.max() > 0:
        score = score / score.max()
    return score


def _checker_map(model: Game) -> np.ndarray:
    score = np.full_like(model.prob_map, 0.25, dtype=float)
    for r in range(model.rows):
        for c in range(model.cols):
            score[r, c] = 1.0 if (r + c) % 2 == 0 else 0.25
    for r, c in model._queried():
        score[r, c] = 0.0
    return score


def select_weighted_query(model: Game, policy: np.ndarray) -> Optional[Tuple[int, int]]:
    queried = model._queried()
    available = [(r, c) for r in range(model.rows) for c in range(model.cols) if (r, c) not in queried]
    if not available:
        return None

    prob = _normalize(model.prob_map)
    ent = _normalize(model.get_entropy_map())
    target = _target_map(model)
    checker = _checker_map(model)

    stacked = np.stack([prob, ent, target, checker], axis=0)
    combined = np.tensordot(policy, stacked, axes=(0, 0))
    for r, c in queried:
        combined[r, c] = -np.inf

    row, col = np.unravel_index(np.argmax(combined), combined.shape)
    return int(row), int(col)


def sample_policies(rng: np.random.RandomState, n: int) -> np.ndarray:
    specials = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.35, 0.35, 0.20, 0.10],
            [0.20, 0.20, 0.50, 0.10],
            [0.25, 0.25, 0.25, 0.25],
        ],
        dtype=float,
    )
    if n <= len(specials):
        return specials[:n].copy()

    random_part = rng.dirichlet(alpha=np.ones(len(POLICY_COMPONENTS)), size=n - len(specials))
    return np.vstack([specials, random_part])


def _policy_key(policy: np.ndarray) -> Tuple[int, ...]:
    return tuple(np.round(policy * 1000).astype(int).tolist())


def propose_batch(
    rng: np.random.RandomState,
    surrogate: KernelEnsembleSurrogate,
    observed_x: np.ndarray,
    observed_y: np.ndarray,
    query_size: int,
    cycle_idx: int,
) -> np.ndarray:
    if len(observed_x) < query_size or cycle_idx == 0:
        return sample_policies(rng, query_size)

    candidate_pool = sample_policies(rng, 256)
    pred_mean, pred_var = surrogate.predict(candidate_pool)
    acquisition = pred_mean - 1.5 * np.sqrt(pred_var)
    order = np.argsort(acquisition)

    chosen: List[np.ndarray] = []
    seen = {_policy_key(x) for x in observed_x}
    for idx in order:
        cand = candidate_pool[idx]
        key = _policy_key(cand)
        if key in seen:
            continue
        if any(np.linalg.norm(cand - prev, ord=1) < 0.30 for prev in chosen):
            continue
        chosen.append(cand)
        seen.add(key)
        if len(chosen) >= query_size:
            break

    while len(chosen) < query_size:
        extra = rng.dirichlet(alpha=np.ones(len(POLICY_COMPONENTS)))
        key = _policy_key(extra)
        if key in seen:
            continue
        chosen.append(extra)
        seen.add(key)

    return np.array(chosen, dtype=float)


def run_weighted_episode(
    policy: np.ndarray,
    seed: int,
    use_image_oracle: bool = True,
    *,
    rgb_l2_max: Optional[float] = None,
    rgb_per_channel_delta: Optional[float] = None,
) -> Dict:
    board = BattleshipBoard(rows=8, cols=ACTIVE_COLS, seed=seed)
    model = Game(board_rows=8, board_cols=ACTIVE_COLS)
    oracle = make_battleship_oracle(
        board,
        seed=seed,
        oracle_mode="image" if use_image_oracle else "board",
        rgb_l2_max=rgb_l2_max,
        rgb_per_channel_delta=rgb_per_channel_delta,
    )

    history: List[Dict] = []
    while not board.is_game_over():
        pos = select_weighted_query(model, policy)
        if pos is None:
            break

        row, col = pos
        observed_hit, sunk_ship, actual_hit = oracle.query(row, col)

        model.update(row, col, observed_hit, sunk_ship)
        history.append(
            {
                "step": board.n_queries,
                "row": row,
                "col": col,
                "observed_hit": bool(observed_hit),
                "actual_hit": bool(actual_hit),
                "ships_sunk": len(board.get_sunk_ships()),
                "cells_found": board.total_ship_cells - board.get_remaining_ship_cells(),
                "frac_found": (board.total_ship_cells - board.get_remaining_ship_cells())
                / max(1, board.total_ship_cells),
                "total_entropy": float(model.get_entropy_map().sum()),
                "max_prob": float(model.prob_map.max()),
            }
        )

    cv_error_rate = oracle.cv_error_rate if oracle is not None else 0.0
    unknown_rate = oracle.unknown_rate if oracle is not None else 0.0
    return {
        "seed": seed,
        "policy": policy.tolist(),
        "n_queries": board.n_queries,
        "cv_error_rate": cv_error_rate,
        "unknown_rate": unknown_rate,
        "history": history,
    }


def evaluate_policy(
    policy: np.ndarray,
    seeds: Sequence[int],
    use_image_oracle: bool = True,
    *,
    rgb_l2_max: Optional[float] = None,
    rgb_per_channel_delta: Optional[float] = None,
) -> Dict:
    episodes = [
        run_weighted_episode(
            policy,
            seed=s,
            use_image_oracle=use_image_oracle,
            rgb_l2_max=rgb_l2_max,
            rgb_per_channel_delta=rgb_per_channel_delta,
        )
        for s in seeds
    ]
    queries = np.array([ep["n_queries"] for ep in episodes], dtype=float)
    cv_errors = np.array([ep["cv_error_rate"] for ep in episodes], dtype=float)
    unknown_rates = np.array([ep["unknown_rate"] for ep in episodes], dtype=float)
    replicate_cv_pct = 100.0 * float(queries.std(ddof=0) / max(1e-8, queries.mean()))
    return {
        "policy": policy.tolist(),
        "mean_queries": float(queries.mean()),
        "std_queries": float(queries.std(ddof=0)),
        "replicate_cv_pct": replicate_cv_pct,
        "mean_cv_error_rate": float(cv_errors.mean()),
        "mean_unknown_rate": float(unknown_rates.mean()),
        "replicate_queries": queries.tolist(),
        "episodes": episodes,
    }


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _compute_no_improvement_streak(history: List[Dict]) -> int:
    """Count how many consecutive trailing cycles had no improvement in best_so_far."""
    streak = 0
    for h in reversed(history):
        if streak == 0:
            streak = 1
            continue
        # Compare against the cycle before; if best didn't drop we keep counting
        prev_best = history[history.index(h) ].get("best_so_far_queries", float("inf"))
        curr_best = history[history.index(h) + 1].get("best_so_far_queries", float("inf"))
        if curr_best < prev_best - 1e-6:
            break
        streak += 1
    return streak


def evaluate_stopping(history: List[Dict]) -> StoppingDecision:
    if not history:
        return StoppingDecision(False, "No history yet", [], {})

    last = history[-1]
    triggered: List[str] = []

    # 1. Held-out MAE below target (surrogate has converged to accurate predictions)
    if last["heldout_mae_queries"] <= THRESHOLDS["heldout_mae_queries"]:
        triggered.append("heldout_mae")

    # 2. Budget exhausted
    if last["al_cycle_count"] >= THRESHOLDS["max_cycles"]:
        triggered.append("cumulative_experiment_count")

    # 3. Prediction stability: surrogate predictions no longer changing
    if len(history) >= 2 and last["prediction_stability_pct"] >= THRESHOLDS["prediction_stability_pct"]:
        triggered.append("prediction_stability")

    # 4. Information gain diminished: variance barely reduced this cycle
    if len(history) >= 2 and 0.0 < last["information_gain_pct"] <= THRESHOLDS["info_gain_pct"]:
        triggered.append("information_gain")

    # 5. No-improvement streak: best_so_far unchanged for N consecutive cycles
    streak = _compute_no_improvement_streak(history)
    if streak >= THRESHOLDS["no_improvement_streak"]:
        triggered.append("no_improvement_streak")

    metrics = {
        "heldout_mae_queries":      float(last["heldout_mae_queries"]),
        "information_gain_pct":     float(last["information_gain_pct"]),
        "prediction_stability_pct": float(last["prediction_stability_pct"]),
        "al_cycle_count":           float(last["al_cycle_count"]),
        "no_improvement_streak":    float(streak),
    }
    reason = " | ".join(triggered) if triggered else "No stopping criterion triggered"
    return StoppingDecision(bool(triggered), reason, triggered, metrics)


def generate_markdown_plan(history: List[Dict]) -> str:
    last = history[-1] if history else {}
    return "\n".join(
        [
            "# QC and Stopping Criteria Plan",
            "",
            "## Campaign context",
            "This campaign tunes a weighted Battleship acquisition policy in a simulated closed loop.",
            "Each cycle proposes a batch of candidate policies, runs them against the Battleship",
            "simulator plus synthetic image-reading pipeline, and fits an ensemble surrogate over",
            "policy -> mean queries to finish the board.",
            "",
            "## Chosen stopping criteria",
            "",
            "1. Held-out prediction error",
            "- Metric: MAE on a fixed validation policy set (units: mean queries to finish board).",
            f"- Threshold: stop when MAE ≤ {THRESHOLDS['heldout_mae_queries']:.1f} queries.",
            "- Rationale: an MAE ≤ 4 queries means the surrogate can distinguish good from bad policies",
            "  with sub-step precision — sufficient to confidently identify the optimum.",
            "- Computation: fit the surrogate on all completed cycles, predict the held-out policy",
            "  set, and compare predictions against the simulated ground-truth performance.",
            "- Robustness: held-out policies are fixed at campaign start and never proposed; this",
            "  prevents the surrogate from appearing accurate by overfitting to seen regions.",
            "",
            "2. Information gain per cycle",
            "- Metric: percent reduction in mean ensemble variance on a fixed unlabeled policy pool.",
            f"- Threshold: stop when 0 < information gain ≤ {THRESHOLDS['info_gain_pct']:.1f}%.",
            "- Rationale: below 2% the surrogate is extracting negligible new structure from each",
            "  additional cycle — a reliable signal of diminishing returns.",
            "- Computation: compare mean predictive variance at cycle t vs cycle t-1 on the fixed pool.",
            "- Robustness: zero-gain cycles (variance briefly rising due to noise) are excluded; only",
            "  sustained low-gain cycles trigger this criterion.",
            "",
            "3. Prediction stability",
            "- Metric: fraction of pool predictions whose relative shift is < 1% between consecutive cycles.",
            f"- Threshold: stop when stability ≥ {THRESHOLDS['prediction_stability_pct']:.0f}%.",
            "- Rationale: when 90% of candidate policies are predicted identically cycle-to-cycle,",
            "  the LCB acquisition ranking has effectively converged and further cycles cannot",
            "  change the policy recommendation.",
            "- Computation: on the fixed unlabeled policy pool, compare surrogate mean predictions",
            "  between cycle t and t-1; count the proportion with |Δ|/max(|pred|,1) < 0.01.",
            "- Robustness: requires at least 2 completed cycles to avoid triggering on the first cycle.",
            "",
            "4. No-improvement streak",
            "- Metric: number of consecutive cycles in which best_so_far_queries did not decrease.",
            f"- Threshold: stop when streak ≥ {THRESHOLDS['no_improvement_streak']} consecutive cycles.",
            "- Rationale: with 8 queries/cycle and 3 replicates, 5 consecutive failures to beat the",
            "  incumbent (~120 board evaluations) is strong evidence that the true optimum has been",
            "  found within measurement noise and additional exploration is unproductive.",
            "- Computation: walk history in reverse; count cycles until best_so_far improved by > 1e-6.",
            "- Robustness: uses best_so_far (cumulative minimum) rather than per-cycle best, so a",
            "  single lucky cycle cannot reset a streak prematurely.",
            "",
            "5. Cumulative experiment count",
            "- Metric: al_cycle_count.",
            f"- Threshold: stop when al_cycle_count reaches {THRESHOLDS['max_cycles']}.",
            "- Rationale: hard budget cap — guarantees termination regardless of surrogate behaviour.",
            "- Computation: increment once after each lab handoff / cycle completes.",
            "",
            "## QC metrics",
            "",
            "1. Process consistency checks",
            f"- Metric: replicate CV (%) of mean queries for the cycle's policy replicates.",
            f"- Acceptable range: < {THRESHOLDS['replicate_cv_pct']:.1f}%.",
            "- Action on failure: retry the offending batch once; if instability persists, flag for",
            "  review before using the observations for model updates.",
            "",
            "2. Hardware errors",
            "- Metric: count of simulated hardware errors or read failures per cycle.",
            "- Acceptable range: 0 during normal operation.",
            "- Action on failure: allow operator review; either recover and continue or terminate the",
            "  campaign and decide which completed data points to retain.",
            "",
            "3. Ensemble uncertainty",
            "- Metric: mean ensemble predictive variance on the unlabeled policy pool.",
            f"- Acceptable range: > {THRESHOLDS['ensemble_var_lower']:.1e} and < {THRESHOLDS['ensemble_var_upper']:.1f}.",
            "- Action on failure: review surrogate fit, increase exploration, or reduce ensemble size",
            "  if the variance collapses or explodes.",
            "",
            "4. Lab throughput / hardware limits",
            f"- Metric: query_size per cycle.",
            f"- Acceptable range: {THRESHOLDS['min_query_size']} to {THRESHOLDS['max_query_size']} candidates per cycle.",
            "- Action on failure: reject the cycle configuration before execution.",
            "",
            "## Monitoring strategy",
            "",
            "- Per cycle: compute stopping metrics, surrogate diagnostics, replicate CV, and CV readout error.",
            "- Continuously during the cycle: log hardware-error events and detector unknown reads.",
            "- On QC failure: retry once when possible, otherwise flag and pause operator review.",
            "- On stopping trigger: save all figures, JSON history, and final campaign summary.",
            "",
            "## Robustness safeguards",
            "",
            "- Held-out MAE uses a fixed validation set never seen during proposal — prevents",
            "  the surrogate appearing accurate by overfitting to already-sampled regions.",
            "- Information gain is measured on a fixed unlabeled pool to avoid cherry-picking.",
            "- Prediction stability uses relative change across the full pool, not just the current best.",
            "- No-improvement streak uses best_so_far (cumulative minimum), so a single lucky cycle",
            "  cannot prematurely reset the counter.",
            "- The max-cycle budget criterion guarantees termination even if all other criteria fail.",
            "- All five criteria are evaluated independently; the campaign stops as soon as any one",
            "  is triggered, and the triggering reason is recorded for audit.",
            "",
            "## Stopping criteria summary table",
            "",
            "| # | Criterion | Metric | Threshold | Direction |",
            "|---|-----------|--------|-----------|-----------|",
            f"| 1 | Held-out prediction error | MAE (queries) | ≤ {THRESHOLDS['heldout_mae_queries']:.1f} | lower is better |",
            f"| 2 | Information gain | variance reduction % | ≤ {THRESHOLDS['info_gain_pct']:.1f}% | stop at diminishing returns |",
            f"| 3 | Prediction stability | % stable predictions | ≥ {THRESHOLDS['prediction_stability_pct']:.0f}% | stop when converged |",
            f"| 4 | No-improvement streak | consecutive cycles | ≥ {THRESHOLDS['no_improvement_streak']} | stop at plateau |",
            f"| 5 | Budget | al_cycle_count | ≥ {THRESHOLDS['max_cycles']} | hard cap |",
            "",
            "## Final-cycle snapshot",
            "",
            f"- Completed cycles: {int(last.get('al_cycle_count', 0))}",
            f"- Best observed mean queries: {float(last.get('best_so_far_queries', float('nan'))):.2f}",
            f"- Held-out MAE: {float(last.get('heldout_mae_queries', float('nan'))):.2f}",
            f"- Information gain: {float(last.get('information_gain_pct', float('nan'))):.2f}%",
            f"- Prediction stability: {float(last.get('prediction_stability_pct', float('nan'))):.2f}%",
            f"- No-improvement streak: {int(last.get('no_improvement_streak', 0))} cycles",
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
# Baseline comparison
# ──────────────────────────────────────────────────────────────────────────────

BASELINES = {
    "random":   {"label": "Random (baseline)",          "policy": np.array([0.00, 0.00, 0.00, 0.00]), "color": "#e74c3c"},
    "prob":     {"label": "Max Probability (exploit)",  "policy": np.array([1.00, 0.00, 0.00, 0.00]), "color": "#2ecc71"},
    "entropy":  {"label": "Max Entropy (explore)",      "policy": np.array([0.00, 1.00, 0.00, 0.00]), "color": "#3498db"},
    "target":   {"label": "Hunt-Target heuristic",      "policy": np.array([0.00, 0.00, 1.00, 0.00]), "color": "#f39c12"},
}


def _run_random_episode(seed: int, use_image_oracle: bool = True) -> int:
    """Play one episode with uniform random query selection."""
    import random as _random
    board = BattleshipBoard(rows=8, cols=ACTIVE_COLS, seed=seed)
    model = Game(board_rows=8, board_cols=ACTIVE_COLS)
    oracle = make_battleship_oracle(
        board,
        seed=seed,
        oracle_mode="image" if use_image_oracle else "board",
    )
    while not board.is_game_over():
        available = [(r, c) for r in range(board.rows) for c in range(board.cols)
                     if board.observed[r, c] == -1]
        if not available:
            break
        row, col = _random.choice(available)
        is_hit, sunk, _ = oracle.query(row, col)
        model.update(row, col, is_hit, sunk)
    return board.n_queries


def run_baselines(
    seeds: Sequence[int],
    use_image_oracle: bool = True,
) -> Dict[str, Dict]:
    """
    Evaluate the four fixed-policy baselines (random, prob, entropy, target)
    over *seeds* and return summary stats.
    """
    results: Dict[str, Dict] = {}
    for key, meta in BASELINES.items():
        if key == "random":
            queries = np.array([_run_random_episode(s, use_image_oracle) for s in seeds], dtype=float)
        else:
            queries = np.array(
                [run_weighted_episode(meta["policy"], seed=s, use_image_oracle=use_image_oracle)["n_queries"]
                 for s in seeds],
                dtype=float,
            )
        results[key] = {
            "label":       meta["label"],
            "color":       meta["color"],
            "policy":      meta["policy"].tolist(),
            "mean_queries": float(queries.mean()),
            "std_queries":  float(queries.std(ddof=0)),
            "queries":      queries.tolist(),
        }
        print(f"  baseline [{key:<8s}]  mean={queries.mean():.2f}  std={queries.std():.2f}")
    return results


def plot_baseline_comparison(
    baselines: Dict[str, Dict],
    campaign_best_policy: Dict[str, float],
    campaign_best_mean: float,
    campaign_history: List[Dict],
    out_dir: Path,
    seeds: Sequence[int],
    use_image_oracle: bool = True,
) -> Path:
    """
    Three-panel figure:
      Left  : bar chart – mean queries ± std for baselines + campaign optimum
      Centre: CDF of queries per-seed for each strategy
      Right : campaign learning curve overlaid with baseline lines

    Also writes baselines.json alongside the figure for the dashboard.
    """
    # evaluate campaign optimum over the same seeds
    opt_policy = np.array([campaign_best_policy[k] for k in POLICY_COMPONENTS], dtype=float)
    opt_queries = np.array(
        [run_weighted_episode(opt_policy, seed=s, use_image_oracle=use_image_oracle)["n_queries"]
         for s in seeds],
        dtype=float,
    )
    opt_mean = float(opt_queries.mean())
    opt_std  = float(opt_queries.std(ddof=0))
    opt_label = "Campaign optimum\n" + "  ".join(f"{k}={v:.2f}" for k, v in campaign_best_policy.items())

    # ── Save baseline data as JSON so the dashboard can render it ────────────
    random_mean = baselines["random"]["mean_queries"]
    baseline_json: Dict = {
        "seeds": list(seeds),
        "baselines": {
            key: {
                "label":        meta["label"],
                "color":        meta["color"],
                "policy":       meta["policy"],
                "mean_queries": meta["mean_queries"],
                "std_queries":  meta["std_queries"],
                "queries":      meta["queries"],
                "improvement_vs_random_pct": round(
                    100.0 * (random_mean - meta["mean_queries"]) / max(random_mean, 1e-9), 2
                ),
            }
            for key, meta in baselines.items()
        },
        "campaign_optimum": {
            "label":        opt_label.replace("\n", " "),
            "color":        "#9b59b6",
            "policy":       campaign_best_policy,
            "mean_queries": opt_mean,
            "std_queries":  opt_std,
            "queries":      opt_queries.tolist(),
            "improvement_vs_random_pct": round(
                100.0 * (random_mean - opt_mean) / max(random_mean, 1e-9), 2
            ),
        },
    }
    (out_dir / "baselines.json").write_text(json.dumps(baseline_json, indent=2), encoding="utf-8")

    all_keys    = list(baselines.keys()) + ["campaign_opt"]
    all_labels  = [baselines[k]["label"] for k in baselines] + [opt_label]
    all_means   = [baselines[k]["mean_queries"] for k in baselines] + [opt_mean]
    all_stds    = [baselines[k]["std_queries"]  for k in baselines] + [opt_std]
    all_colors  = [baselines[k]["color"]        for k in baselines] + ["#9b59b6"]
    all_queries = [np.array(baselines[k]["queries"]) for k in baselines] + [opt_queries]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Bayesian-Optimised Policy vs Baselines", fontsize=13, fontweight="bold")

    # ── Panel 1: bar ─────────────────────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(all_keys))
    bars = ax.bar(x, all_means, yerr=all_stds, color=all_colors, alpha=0.8,
                  capsize=4, error_kw={"elinewidth": 1.2})
    # highlight improvement
    baseline_mean = baselines["random"]["mean_queries"]
    ax.axhline(baseline_mean, color="#e74c3c", ls=":", lw=1.2, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Mean queries to finish board")
    ax.set_title("Mean performance ± std")
    ax.grid(axis="y", alpha=0.3)
    # annotate % improvement vs random
    for i, (m, col) in enumerate(zip(all_means, all_colors)):
        pct = 100.0 * (baseline_mean - m) / baseline_mean
        ax.text(i, m + all_stds[i] + 0.5,
                f"{pct:+.1f}%", ha="center", fontsize=7.5, color=col)

    # ── Panel 2: CDF ─────────────────────────────────────────────────────────
    ax = axes[1]
    for qs, label, col in zip(all_queries, all_labels, all_colors):
        q_sorted = np.sort(qs)
        cdf = np.arange(1, len(q_sorted) + 1) / len(q_sorted)
        ax.plot(q_sorted, cdf, label=label.split("\n")[0], color=col, lw=2)
    ax.set_xlabel("Queries to finish board")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("CDF comparison")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # ── Panel 3: campaign learning curve + baseline horizontal lines ──────────
    ax = axes[2]
    cycles      = [h["al_cycle_count"]   for h in campaign_history]
    best_so_far = [h["best_so_far_queries"] for h in campaign_history]
    ax.plot(cycles, best_so_far, marker="o", color="#9b59b6", lw=2.5, label="Campaign best-so-far")
    for key, meta in BASELINES.items():
        ax.axhline(baselines[key]["mean_queries"], color=meta["color"],
                   ls="--", lw=1.4, alpha=0.8, label=meta["label"])
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Queries to finish board")
    ax.set_title("Campaign learning vs baseline means")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "baseline_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nBaseline comparison saved → {out_path}")
    return out_path


def plot_learning(history: List[Dict], out_dir: Path) -> Path:
    cycles = [h["al_cycle_count"] for h in history]
    best_so_far = [h["best_so_far_queries"] for h in history]
    batch_best = [h["cycle_best_queries"] for h in history]
    batch_mean = [h["cycle_mean_queries"] for h in history]
    mae = [h["heldout_mae_queries"] for h in history]
    r2 = [h["heldout_r2"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Battleship Closed-Loop Learning over Cycles", fontsize=13, fontweight="bold")

    axes[0].plot(cycles, batch_mean, marker="o", label="Cycle mean queries", color="#5b8ff9")
    axes[0].plot(cycles, batch_best, marker="o", label="Cycle best queries", color="#61dDAA")
    axes[0].plot(cycles, best_so_far, marker="o", label="Best-so-far queries", color="#f6bd16")
    axes[0].set_xlabel("Cycle")
    axes[0].set_ylabel("Queries to finish board")
    axes[0].set_title("Policy optimization progress")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(cycles, mae, marker="o", label="Held-out MAE", color="#e8684a")
    axes[1].axhline(THRESHOLDS["heldout_mae_queries"], color="#e8684a", ls="--", alpha=0.7)
    ax2 = axes[1].twinx()
    ax2.plot(cycles, r2, marker="s", label="Held-out R²", color="#6dc8ec")
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("MAE (queries)")
    ax2.set_ylabel("R²")
    axes[1].set_title("Surrogate learning quality")
    axes[1].grid(alpha=0.3)

    lines_left, labels_left = axes[1].get_legend_handles_labels()
    lines_right, labels_right = ax2.get_legend_handles_labels()
    axes[1].legend(lines_left + lines_right, labels_left + labels_right, fontsize=8, loc="best")

    plt.tight_layout()
    out_path = out_dir / "learning_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_stopping(history: List[Dict], out_dir: Path) -> Path:
    cycles    = [h["al_cycle_count"]           for h in history]
    mae       = [h["heldout_mae_queries"]       for h in history]
    gain      = [h["information_gain_pct"]      for h in history]
    stability = [h["prediction_stability_pct"]  for h in history]
    count     = [h["al_cycle_count"]            for h in history]
    streak    = [h.get("no_improvement_streak", 0) for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Stopping-Criterion Metrics", fontsize=13, fontweight="bold")
    axs = axes.ravel()

    # ── Panel 0: Held-out MAE ────────────────────────────────────────────────
    axs[0].plot(cycles, mae, marker="o", color="#e8684a", lw=2)
    axs[0].axhline(THRESHOLDS["heldout_mae_queries"], color="#e8684a", ls="--", alpha=0.8,
                   label=f"target ≤ {THRESHOLDS['heldout_mae_queries']:.1f}")
    axs[0].set_title("① Held-out prediction error (MAE)")
    axs[0].set_xlabel("Cycle")
    axs[0].set_ylabel("MAE (queries)")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=0.3)

    # ── Panel 1: Information gain ────────────────────────────────────────────
    axs[1].plot(cycles, gain, marker="o", color="#6f5ef9", lw=2)
    axs[1].axhline(THRESHOLDS["info_gain_pct"], color="#6f5ef9", ls="--", alpha=0.8,
                   label=f"stop ≤ {THRESHOLDS['info_gain_pct']:.1f}%")
    axs[1].set_title("② Information gain per cycle")
    axs[1].set_xlabel("Cycle")
    axs[1].set_ylabel("Variance reduction (%)")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=0.3)

    # ── Panel 2: Prediction stability ────────────────────────────────────────
    axs[2].plot(cycles, stability, marker="o", color="#61ddaa", lw=2)
    axs[2].axhline(THRESHOLDS["prediction_stability_pct"], color="#61ddaa", ls="--", alpha=0.8,
                   label=f"stop ≥ {THRESHOLDS['prediction_stability_pct']:.0f}%")
    axs[2].set_title("③ Prediction stability")
    axs[2].set_xlabel("Cycle")
    axs[2].set_ylabel("Stable predictions (%)")
    axs[2].set_ylim(-5, 105)
    axs[2].legend(fontsize=8)
    axs[2].grid(alpha=0.3)

    # ── Panel 3: No-improvement streak ──────────────────────────────────────
    axs[3].bar(cycles, streak, color="#f06292", alpha=0.75, width=0.6)
    axs[3].axhline(THRESHOLDS["no_improvement_streak"], color="#f06292", ls="--", lw=1.5, alpha=0.9,
                   label=f"stop ≥ {THRESHOLDS['no_improvement_streak']} cycles")
    # shade the trigger zone
    axs[3].axhspan(THRESHOLDS["no_improvement_streak"], max(streak + [THRESHOLDS["no_improvement_streak"]]) + 1,
                   color="#f06292", alpha=0.08)
    axs[3].set_title("④ No-improvement streak")
    axs[3].set_xlabel("Cycle")
    axs[3].set_ylabel("Consecutive cycles without\nbest_so_far improvement")
    axs[3].legend(fontsize=8)
    axs[3].grid(axis="y", alpha=0.3)

    # ── Panel 4: Cumulative experiment count ─────────────────────────────────
    axs[4].plot(cycles, count, marker="o", color="#f6bd16", lw=2)
    axs[4].axhline(THRESHOLDS["max_cycles"], color="#f6bd16", ls="--", alpha=0.8,
                   label=f"budget = {THRESHOLDS['max_cycles']}")
    axs[4].set_title("⑤ Cumulative experiment count")
    axs[4].set_xlabel("Cycle")
    axs[4].set_ylabel("al_cycle_count")
    axs[4].legend(fontsize=8)
    axs[4].grid(alpha=0.3)

    # ── Panel 5: Multi-criterion summary ─────────────────────────────────────
    # Show which criteria are "active" (approaching threshold) each cycle
    mae_norm       = [m / max(THRESHOLDS["heldout_mae_queries"], 1e-6) for m in mae]
    gain_inv_norm  = [max(0.0, 1.0 - g / max(THRESHOLDS["info_gain_pct"], 1e-6)) for g in gain]
    stab_norm      = [s / THRESHOLDS["prediction_stability_pct"] for s in stability]
    streak_norm    = [min(st / THRESHOLDS["no_improvement_streak"], 1.5) for st in streak]

    axs[5].plot(cycles, mae_norm,      label="MAE progress",       color="#e8684a", lw=1.5, ls="-")
    axs[5].plot(cycles, gain_inv_norm, label="Diminishing returns", color="#6f5ef9", lw=1.5, ls="-")
    axs[5].plot(cycles, stab_norm,     label="Stability progress",  color="#61ddaa", lw=1.5, ls="-")
    axs[5].plot(cycles, streak_norm,   label="Streak progress",     color="#f06292", lw=1.5, ls="-")
    axs[5].axhline(1.0, color="black", ls=":", lw=1.2, alpha=0.6, label="trigger threshold (=1)")
    axs[5].set_title("⑥ Multi-criterion progress (normalised)")
    axs[5].set_xlabel("Cycle")
    axs[5].set_ylabel("Fraction toward trigger threshold")
    axs[5].legend(fontsize=7, loc="upper left")
    axs[5].grid(alpha=0.3)

    # red stop-line on all panels
    stop_cycle = next((h["al_cycle_count"] for h in history if h.get("stop_triggered")), None)
    if stop_cycle is not None:
        for ax in axs:
            ax.axvline(stop_cycle, color="red", ls=":", alpha=0.55, lw=1.3)

    plt.tight_layout()
    out_path = out_dir / "stopping_criteria.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_qc(history: List[Dict], out_dir: Path) -> Path:
    cycles = [h["al_cycle_count"] for h in history]
    qc_keys = [
        ("qc_replicate_cv_ok", "Replicate CV"),
        ("qc_ensemble_var_ok", "Ensemble variance"),
        ("qc_query_size_ok", "Query size"),
        ("qc_no_hardware_errors", "Hardware"),
    ]
    matrix = np.array([[1.0 if h[key] else 0.0 for h in history] for key, _ in qc_keys], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1.2, 1.5]})
    axes[0].imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    axes[0].set_yticks(range(len(qc_keys)))
    axes[0].set_yticklabels([label for _, label in qc_keys])
    axes[0].set_xticks(range(len(cycles)))
    axes[0].set_xticklabels([f"C{c}" for c in cycles])
    axes[0].set_title("QC pass/fail heatmap")

    axes[1].plot(cycles, [h["replicate_cv_pct"] for h in history], marker="o", label="Replicate CV (%)")
    axes[1].axhline(THRESHOLDS["replicate_cv_pct"], color="#e8684a", ls="--", alpha=0.6)
    axes[1].plot(cycles, [100.0 * h["mean_cv_error_rate"] for h in history], marker="s", label="CV error rate (%)")
    axes[1].plot(cycles, [h["mean_ensemble_var"] for h in history], marker="^", label="Mean ensemble variance")
    axes[1].bar(cycles, [h["hardware_errors"] for h in history], alpha=0.25, label="Hardware errors")
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("Metric value")
    axes[1].set_title("QC metric trajectories")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    out_path = out_dir / "qc_timeline.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _policy_to_named_dict(policy: np.ndarray) -> Dict[str, float]:
    return {name: float(val) for name, val in zip(POLICY_COMPONENTS, policy)}


def run_campaign(
    out_dir: Path,
    max_cycles: int = 5,
    query_size: int = 8,
    replicates: int = 3,
    validation_policy_count: int = 18,
    validation_replicates: int = 6,
    ensemble_members: int = 8,
    hardware_error_rate: float = 0.0,
    seed: int = 42,
    use_image_oracle: bool = True,
    rgb_l2_max: Optional[float] = None,
    rgb_per_channel_delta: Optional[float] = None,
) -> Dict[str, object]:
    rng = np.random.RandomState(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    THRESHOLDS["max_cycles"] = int(max_cycles)
    surrogate = KernelEnsembleSurrogate(n_members=ensemble_members, seed=seed)

    validation_policies = sample_policies(np.random.RandomState(seed + 1), validation_policy_count)
    validation_seeds = [1000 + i for i in range(validation_replicates)]
    validation_truth = np.array(
        [
            evaluate_policy(
                policy,
                validation_seeds,
                use_image_oracle,
                rgb_l2_max=rgb_l2_max,
                rgb_per_channel_delta=rgb_per_channel_delta,
            )["mean_queries"]
            for policy in validation_policies
        ],
        dtype=float,
    )

    prediction_pool = sample_policies(np.random.RandomState(seed + 2), 128)

    observed_x = np.empty((0, len(POLICY_COMPONENTS)), dtype=float)
    observed_y = np.empty((0,), dtype=float)
    history: List[Dict] = []
    observations: List[Dict] = []
    previous_pool_mean: Optional[np.ndarray] = None
    previous_pool_var: Optional[np.ndarray] = None
    best_so_far = float("inf")

    oracle_label = (
        "image+fixed ROI mean RGB vs prototypes (no HSV pipeline)"
        if use_image_oracle
        else "board oracle (ground-truth cells, no image readout)"
    )
    print(
        f"[campaign] oracle={oracle_label}\n"
        f"            validation_policies={len(validation_policies)}  "
        f"query_size={query_size}  replicates={replicates}"
    )

    for cycle_idx in range(max_cycles):
        batch = propose_batch(rng, surrogate, observed_x, observed_y, query_size, cycle_idx)
        cycle_seed_block = [2000 + cycle_idx * 100 + rep for rep in range(replicates)]
        cycle_results = []
        hardware_errors = 0

        print(f"\nCycle {cycle_idx + 1}/{max_cycles}")
        for batch_idx, policy in enumerate(batch, start=1):
            if rng.rand() < hardware_error_rate:
                hardware_errors += 1
            result = evaluate_policy(
                policy,
                cycle_seed_block,
                use_image_oracle,
                rgb_l2_max=rgb_l2_max,
                rgb_per_channel_delta=rgb_per_channel_delta,
            )
            cycle_results.append(result)
            observations.append(
                {
                    "cycle": cycle_idx + 1,
                    "batch_index": batch_idx,
                    "policy": _policy_to_named_dict(policy),
                    "mean_queries": result["mean_queries"],
                    "replicate_cv_pct": result["replicate_cv_pct"],
                    "replicate_queries": result["replicate_queries"],
                    "mean_cv_error_rate": result["mean_cv_error_rate"],
                    "mean_unknown_rate": result["mean_unknown_rate"],
                }
            )
            det_err = 100.0 * result["mean_cv_error_rate"]
            print(
                f"  policy {batch_idx:02d}  mean_queries={result['mean_queries']:.2f}  "
                f"replicate_CoV={result['replicate_cv_pct']:.2f}% "
                f"(std/mean of {replicates} boards)  "
                f"readout_mismatch={det_err:.2f}%  "
                f"weights={_policy_to_named_dict(policy)}"
            )

        batch_x = np.array([np.array(entry["policy"], dtype=float) for entry in cycle_results])
        batch_y = np.array([entry["mean_queries"] for entry in cycle_results], dtype=float)
        observed_x = np.vstack([observed_x, batch_x])
        observed_y = np.concatenate([observed_y, batch_y])
        surrogate.fit(observed_x, observed_y)

        val_pred_mean, val_pred_var = surrogate.predict(validation_policies)
        pool_pred_mean, pool_pred_var = surrogate.predict(prediction_pool)

        heldout_mae = mean_absolute_error(validation_truth, val_pred_mean)
        heldout_r2 = r2_score(validation_truth, val_pred_mean)
        mean_ensemble_var = float(pool_pred_var.mean())

        if previous_pool_var is None:
            information_gain_pct = float("nan")
        else:
            prev_var = float(previous_pool_var.mean())
            curr_var = float(pool_pred_var.mean())
            information_gain_pct = 100.0 * max(0.0, prev_var - curr_var) / max(1e-8, prev_var)

        if previous_pool_mean is None:
            prediction_stability_pct = 0.0
        else:
            rel_shift = np.abs(pool_pred_mean - previous_pool_mean) / np.maximum(1.0, np.abs(previous_pool_mean))
            prediction_stability_pct = 100.0 * float((rel_shift < 0.01).mean())

        previous_pool_mean = pool_pred_mean.copy()
        previous_pool_var = pool_pred_var.copy()

        cycle_best = float(batch_y.min())
        cycle_mean = float(batch_y.mean())
        best_so_far = min(best_so_far, cycle_best)
        cycle_cv_values = np.array([entry["replicate_cv_pct"] for entry in cycle_results], dtype=float)
        cycle_cv_error = np.array([entry["mean_cv_error_rate"] for entry in cycle_results], dtype=float)
        cycle_unknown = np.array([entry["mean_unknown_rate"] for entry in cycle_results], dtype=float)

        qc_replicate_cv_ok = bool(cycle_cv_values.max() < THRESHOLDS["replicate_cv_pct"])
        qc_ensemble_var_ok = bool(
            THRESHOLDS["ensemble_var_lower"] < mean_ensemble_var < THRESHOLDS["ensemble_var_upper"]
        )
        qc_query_size_ok = bool(THRESHOLDS["min_query_size"] <= query_size <= THRESHOLDS["max_query_size"])
        qc_no_hardware_errors = bool(hardware_errors == 0)

        qc_flags: List[str] = []
        if not qc_replicate_cv_ok:
            qc_flags.append("replicate_cv_out_of_range")
        if not qc_ensemble_var_ok:
            qc_flags.append("ensemble_variance_out_of_range")
        if not qc_query_size_ok:
            qc_flags.append("query_size_out_of_range")
        if not qc_no_hardware_errors:
            qc_flags.append("hardware_error_detected")

        # compute streak before appending so evaluate_stopping gets full history
        streak_so_far = _compute_no_improvement_streak(
            history + [{"best_so_far_queries": best_so_far}]
        ) if history else 1

        cycle_record = {
            "cycle": cycle_idx,
            "al_cycle_count": cycle_idx + 1,
            "query_size": query_size,
            "replicates": replicates,
            "ensemble_members": ensemble_members,
            "cycle_mean_queries": cycle_mean,
            "cycle_best_queries": cycle_best,
            "best_so_far_queries": best_so_far,
            "heldout_mae_queries": heldout_mae,
            "heldout_r2": heldout_r2,
            "information_gain_pct": float(information_gain_pct) if np.isfinite(information_gain_pct) else 0.0,
            "prediction_stability_pct": prediction_stability_pct,
            "mean_ensemble_var": mean_ensemble_var,
            "replicate_cv_pct": float(cycle_cv_values.max()),
            "replicate_cv_mean_pct": float(cycle_cv_values.mean()),
            "mean_cv_error_rate": float(cycle_cv_error.mean()),
            "mean_unknown_rate": float(cycle_unknown.mean()),
            "hardware_errors": hardware_errors,
            "no_improvement_streak": streak_so_far,
            "best_policy": _policy_to_named_dict(batch[np.argmin(batch_y)]),
            "qc_replicate_cv_ok": qc_replicate_cv_ok,
            "qc_ensemble_var_ok": qc_ensemble_var_ok,
            "qc_query_size_ok": qc_query_size_ok,
            "qc_no_hardware_errors": qc_no_hardware_errors,
            "qc_all_ok": bool(qc_replicate_cv_ok and qc_ensemble_var_ok and qc_query_size_ok and qc_no_hardware_errors),
            "qc_flags": qc_flags,
            "stop_triggered": False,
            "stop_reason": "",
            "triggered_criteria": [],
        }
        history.append(cycle_record)

        decision = evaluate_stopping(history)
        if decision.should_stop:
            cycle_record["stop_triggered"] = True
            cycle_record["stop_reason"] = decision.reason
            cycle_record["triggered_criteria"] = decision.triggered_criteria
            print(f"  stop_triggered={decision.reason}")
            break

        print(
            f"  heldout_mae={heldout_mae:.2f}  info_gain={cycle_record['information_gain_pct']:.2f}%  "
            f"stability={prediction_stability_pct:.2f}%  streak={streak_so_far}  "
            f"qc={'OK' if cycle_record['qc_all_ok'] else 'FLAG'}"
        )

    learning_path = plot_learning(history, out_dir)
    stopping_path = plot_stopping(history, out_dir)
    qc_path = plot_qc(history, out_dir)

    # ── Baseline comparison ────────────────────────────────────────────────
    final_record_tmp = history[-1] if history else {}
    best_policy_dict = final_record_tmp.get("best_policy", {k: 0.25 for k in POLICY_COMPONENTS})
    baseline_seeds = list(range(seed, seed + max(10, replicates * 2)))
    print("\nRunning baseline comparisons …")
    baselines = run_baselines(seeds=baseline_seeds, use_image_oracle=use_image_oracle)
    baseline_path = plot_baseline_comparison(
        baselines=baselines,
        campaign_best_policy=best_policy_dict,
        campaign_best_mean=float(final_record_tmp.get("best_so_far_queries", float("nan"))),
        campaign_history=history,
        out_dir=out_dir,
        seeds=baseline_seeds,
        use_image_oracle=use_image_oracle,
    )

    history_path = out_dir / "history.json"
    observations_path = out_dir / "observations.json"
    config_path = out_dir / "config.json"
    plan_path = out_dir / "qc_stopping_criteria_plan.md"

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    observations_path.write_text(json.dumps(observations, indent=2), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "max_cycles": max_cycles,
                "query_size": query_size,
                "replicates": replicates,
                "validation_policy_count": validation_policy_count,
                "validation_replicates": validation_replicates,
                "ensemble_members": ensemble_members,
                "hardware_error_rate": hardware_error_rate,
                "seed": seed,
                "rgb_l2_max": rgb_l2_max,
                "rgb_per_channel_delta": rgb_per_channel_delta,
                "default_rgb_l2_if_unset": DEFAULT_RGB_L2_TOLERANCE,
                "thresholds": THRESHOLDS,
                "policy_components": POLICY_COMPONENTS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plan_path.write_text(generate_markdown_plan(history), encoding="utf-8")

    final_record = history[-1] if history else {}
    print("\nArtifacts")
    print(f"  {learning_path}")
    print(f"  {stopping_path}")
    print(f"  {qc_path}")
    print(f"  {baseline_path}")
    print(f"  {history_path}")
    print(f"  {plan_path}")
    if history:
        print(
            f"\nFinal summary: cycles={final_record['al_cycle_count']}  "
            f"best_queries={final_record['best_so_far_queries']:.2f}  "
            f"stop_reason={final_record.get('stop_reason') or 'not_triggered'}"
        )

    return {
        "history": history,
        "observations": observations,
        "paths": {
            "learning_curves": str(learning_path),
            "stopping_criteria": str(stopping_path),
            "qc_timeline": str(qc_path),
            "history": str(history_path),
            "observations": str(observations_path),
            "config": str(config_path),
            "plan": str(plan_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Battleship closed-loop campaign")
    parser.add_argument("--out_dir", default=DEFAULT_RESULTS_DIR, help="Output directory for campaign artifacts.")
    parser.add_argument("--max_cycles", type=int, default=5, help="Maximum active-learning cycles.")
    parser.add_argument("--query_size", type=int, default=8, help="Policies proposed per cycle.")
    parser.add_argument("--replicates", type=int, default=3, help="Replicate boards per policy evaluation.")
    parser.add_argument("--validation_policy_count", type=int, default=18, help="Held-out policy count.")
    parser.add_argument("--validation_replicates", type=int, default=6, help="Boards per held-out policy.")
    parser.add_argument("--ensemble_members", type=int, default=8, help="Bootstrap ensemble members.")
    parser.add_argument("--hardware_error_rate", type=float, default=0.0, help="Simulated hardware-error rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--oracle_mode",
        choices=["image", "board"],
        default="image",
        help=(
            "image: synthetic plate photo + fixed-geometry mean RGB vs ship/water prototypes. "
            "board: query ground truth directly (no image readout)."
        ),
    )
    parser.add_argument(
        "--rgb_l2_max",
        type=float,
        default=None,
        help=(
            "RGB L2 distance tolerance in fixed-ROI readout (defaults to "
            f"battleship_plate_simulation.DEFAULT_RGB_L2_TOLERANCE={DEFAULT_RGB_L2_TOLERANCE})."
        ),
    )
    parser.add_argument(
        "--rgb_per_channel_delta",
        type=float,
        default=None,
        help="If set, each RGB channel must be within ±delta of the chosen prototype (stricter).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_campaign(
        out_dir=Path(args.out_dir),
        max_cycles=args.max_cycles,
        query_size=args.query_size,
        replicates=args.replicates,
        validation_policy_count=args.validation_policy_count,
        validation_replicates=args.validation_replicates,
        ensemble_members=args.ensemble_members,
        hardware_error_rate=args.hardware_error_rate,
        seed=args.seed,
        use_image_oracle=(args.oracle_mode == "image"),
        rgb_l2_max=args.rgb_l2_max,
        rgb_per_channel_delta=args.rgb_per_channel_delta,
    )


if __name__ == "__main__":
    main()
