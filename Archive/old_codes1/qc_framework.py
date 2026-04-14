"""
QC Framework for the BaiTu Active Learning Campaign.

Implements:
  - QCReport: per-cycle quality control summary
  - StoppingDecision: result of evaluating stopping criteria
  - QCMonitor: evaluates QC checks and stopping criteria each cycle
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
THRESHOLDS: Dict[str, float] = {
    # Biological plausibility bounds
    "kcat_min": 1e-4,   # s^-1
    "kcat_max": 1e8,    # s^-1
    "km_min":   1e-7,   # M
    "km_max":   1e-1,   # M
    # Lab process quality
    "replicate_cv":      0.10,   # 10 % coefficient of variation
    # Model health
    "ensemble_var_floor": 0.0,   # variance must be strictly > 0
    # Hardware / throughput
    "query_size_min":  8,
    "query_size_max": 96,
    # Stopping criteria
    "max_cycles":              5,
    "mae_log_kcat_threshold":  0.3,   # log-scale MAE (log units, natural log)
    "mae_log_km_threshold":    0.3,
    "prediction_stability":    0.95,  # fraction of pool predictions changing <1 %
    "stability_consecutive":   2,     # must hold for this many consecutive cycles
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QCReport:
    cycle: int
    replicate_cv_mean: float         # mean CV across queried samples
    replicate_cv_ok: bool            # True if < 10 %
    kcat_bounds_ok: bool             # all predicted kcat in [1e-4, 1e8] s^-1
    km_bounds_ok: bool               # all predicted km in [1e-7, 1e-1] M
    ensemble_var_gt_zero: bool       # variance > 0 (no mode collapse)
    query_size_ok: bool              # 8 ≤ query_size ≤ 96
    hardware_errors: int             # count of simulated hardware errors
    flags: List[str] = field(default_factory=list)

    @property
    def all_ok(self) -> bool:
        return (
            self.replicate_cv_ok
            and self.kcat_bounds_ok
            and self.km_bounds_ok
            and self.ensemble_var_gt_zero
            and self.query_size_ok
            and self.hardware_errors == 0
        )


@dataclass
class StoppingDecision:
    should_stop: bool
    reason: str
    triggered_criteria: List[str] = field(default_factory=list)
    metrics_at_stop: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# QCMonitor
# ---------------------------------------------------------------------------

class QCMonitor:
    """
    Evaluates per-cycle quality control checks and stopping criteria.

    Usage
    -----
    monitor = QCMonitor()
    report  = monitor.run_cycle_checks(replicate_cv, kcat_preds, km_preds,
                                       mean_ensemble_var, query_size, hardware_errors)
    decision = monitor.should_stop(history)
    """

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.thresholds = thresholds or THRESHOLDS

    # ------------------------------------------------------------------
    # Per-cycle checks
    # ------------------------------------------------------------------

    def run_cycle_checks(
        self,
        cycle: int,
        replicate_cv: np.ndarray,
        kcat_preds: np.ndarray,
        km_preds: np.ndarray,
        mean_ensemble_var: float,
        query_size: int,
        hardware_errors: int = 0,
    ) -> QCReport:
        """
        Run all QC checks for one active-learning cycle.

        Parameters
        ----------
        cycle            : current cycle index (0-based)
        replicate_cv     : CV values for each queried sample (from simulator.get_replicate_cv)
        kcat_preds       : predicted kcat values [s^-1] for queried samples
        km_preds         : predicted km values [M] for queried samples
        mean_ensemble_var: mean ensemble variance over the pool
        query_size       : number of samples selected this cycle
        hardware_errors  : count of simulated hardware / instrument errors
        """
        t = self.thresholds
        flags: List[str] = []

        # Replicate CV
        cv_mean = float(np.mean(replicate_cv)) if len(replicate_cv) > 0 else 0.0
        cv_ok = cv_mean < t["replicate_cv"]
        if not cv_ok:
            flags.append(
                f"Replicate CV {cv_mean:.1%} exceeds threshold {t['replicate_cv']:.0%}"
            )

        # Biological bounds — kcat
        kcat_ok = bool(
            np.all(kcat_preds >= t["kcat_min"]) and np.all(kcat_preds <= t["kcat_max"])
        ) if len(kcat_preds) > 0 else True
        if not kcat_ok:
            n_out = int(np.sum((kcat_preds < t["kcat_min"]) | (kcat_preds > t["kcat_max"])))
            flags.append(f"{n_out} kcat predictions outside biological bounds")

        # Biological bounds — km
        km_ok = bool(
            np.all(km_preds >= t["km_min"]) and np.all(km_preds <= t["km_max"])
        ) if len(km_preds) > 0 else True
        if not km_ok:
            n_out = int(np.sum((km_preds < t["km_min"]) | (km_preds > t["km_max"])))
            flags.append(f"{n_out} km predictions outside biological bounds")

        # Ensemble health (mode collapse guard)
        var_ok = mean_ensemble_var > t["ensemble_var_floor"]
        if not var_ok:
            flags.append("Ensemble variance = 0 — possible mode collapse")

        # Query size within hardware range
        qs_ok = t["query_size_min"] <= query_size <= t["query_size_max"]
        if not qs_ok:
            flags.append(
                f"Query size {query_size} outside hardware range "
                f"[{t['query_size_min']}, {t['query_size_max']}]"
            )

        # Hardware errors
        if hardware_errors > 0:
            flags.append(f"{hardware_errors} hardware error(s) detected — requires user review")

        return QCReport(
            cycle=cycle,
            replicate_cv_mean=cv_mean,
            replicate_cv_ok=cv_ok,
            kcat_bounds_ok=kcat_ok,
            km_bounds_ok=km_ok,
            ensemble_var_gt_zero=var_ok,
            query_size_ok=qs_ok,
            hardware_errors=hardware_errors,
            flags=flags,
        )

    # ------------------------------------------------------------------
    # Stopping criteria
    # ------------------------------------------------------------------

    def should_stop(self, history: List[Dict]) -> StoppingDecision:
        """
        Evaluate stopping criteria against the full campaign history.

        Criteria (checked in order; first triggered wins):
        1. Cycle budget exhausted (cycle_count >= max_cycles).
        2. Held-out log-scale MAE below threshold for BOTH kcat and km.
        3. Prediction stability >= 95 % for two consecutive cycles.

        Parameters
        ----------
        history : list of per-cycle dicts, each containing at minimum:
                  'cycle', 'mae_log_kcat', 'mae_log_km', 'prediction_stability_pct'
        """
        if not history:
            return StoppingDecision(should_stop=False, reason="No history yet")

        t = self.thresholds
        triggered: List[str] = []
        last = history[-1]
        current_cycle = last["cycle"]  # 0-based

        # --- Criterion 1: Budget ---
        if current_cycle + 1 >= t["max_cycles"]:
            triggered.append("budget_exhausted")

        # --- Criterion 2: MAE threshold ---
        mae_kcat = last.get("mae_log_kcat", math.inf)
        mae_km   = last.get("mae_log_km",   math.inf)
        if mae_kcat < t["mae_log_kcat_threshold"] and mae_km < t["mae_log_km_threshold"]:
            triggered.append("mae_below_threshold")

        # --- Criterion 3: Prediction stability (requires 2 consecutive cycles) ---
        if len(history) >= t["stability_consecutive"]:
            recent = history[-int(t["stability_consecutive"]):]
            if all(
                h.get("prediction_stability_pct", 0.0) >= t["prediction_stability"]
                for h in recent
            ):
                triggered.append("prediction_stable")

        should_stop = len(triggered) > 0
        reason = " | ".join(triggered) if triggered else "No stopping criterion triggered"

        return StoppingDecision(
            should_stop=should_stop,
            reason=reason,
            triggered_criteria=triggered,
            metrics_at_stop={
                "cycle": current_cycle,
                "mae_log_kcat": mae_kcat,
                "mae_log_km":   mae_km,
                "prediction_stability_pct": last.get("prediction_stability_pct", 0.0),
                "n_labeled": last.get("n_labeled", 0),
            },
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def generate_markdown_report(self, history: List[Dict]) -> str:
        """
        Produce a ~2-page markdown QC & Stopping-Criteria plan from campaign history.
        """
        lines = [
            "# QC and Stopping-Criteria Plan",
            "",
            "## Overview",
            "This document describes the quality control (QC) metrics, acceptable operating",
            "ranges, stopping criteria, and monitoring strategy for the BaiTu automated",
            "enzyme kinetics active learning campaign.",
            "",
            "---",
            "",
            "## Chosen QC Metrics",
            "",
            "### 1. Process Consistency — Replicate Coefficient of Variation (CV)",
            f"- **Definition**: Standard deviation / mean of replicate v₀ measurements × 100 %",
            f"- **Threshold**: < {THRESHOLDS['replicate_cv']:.0%}",
            "- **Monitoring**: Continuously, per queried batch, before lab handoff",
            "- **Action on failure**: Flag batch for human review; optionally re-run assay",
            "",
            "### 2. Biological Plausibility Bounds",
            f"- **k_cat range**: [{THRESHOLDS['kcat_min']:.0e}, {THRESHOLDS['kcat_max']:.0e}] s⁻¹",
            f"- **K_m range**: [{THRESHOLDS['km_min']:.0e}, {THRESHOLDS['km_max']:.0e}] M",
            "- **Monitoring**: Per cycle, on model predictions over the full pool",
            "- **Action on failure**: Flag out-of-bounds predictions; investigate data quality",
            "",
            "### 3. Ensemble Health — Variance > 0",
            "- **Definition**: Mean ensemble variance over the pool must be strictly positive",
            "- **Threshold**: variance > 0 (mode collapse guard)",
            "- **Monitoring**: Per cycle, after ensemble training",
            "- **Action on failure**: Re-initialise ensemble with different random seeds",
            "",
            "### 4. Lab Throughput — Query Size",
            f"- **Range**: [{THRESHOLDS['query_size_min']}, {THRESHOLDS['query_size_max']}] samples per cycle",
            "- **Monitoring**: Before each lab handoff",
            "- **Action on failure**: Adjust query_size to match hardware capacity",
            "",
            "### 5. Hardware Errors",
            "- **Definition**: Count of instrument/robot errors per cycle",
            "- **Threshold**: 0 (any error triggers review)",
            "- **Monitoring**: Continuously during experiment execution",
            "- **Action on failure**: Allow user to recover protocol or terminate experiment;",
            "  user selects which data points (if any) to add to the training set",
            "",
            "---",
            "",
            "## Stopping Criteria",
            "",
            "### Criterion 1: Budget Exhausted",
            f"- **Condition**: Cycle count ≥ {THRESHOLDS['max_cycles']}",
            "- **Metric**: `al_cycle_count` — incremented after each lab handoff",
            "- **Rationale**: Fixed resource budget prevents runaway experiments",
            "",
            "### Criterion 2: Held-Out Prediction Error Below Threshold",
            f"- **Condition**: log-scale MAE(k_cat) < {THRESHOLDS['mae_log_kcat_threshold']} AND",
            f"  log-scale MAE(K_m) < {THRESHOLDS['mae_log_km_threshold']} (natural log units)",
            "- **Metric**: Computed on fixed held-out validation set at end of each cycle",
            "- **Rationale**: Model has reached sufficient predictive accuracy for the scientific goal",
            "",
            "### Criterion 3: Prediction Stability",
            f"- **Condition**: ≥ {THRESHOLDS['prediction_stability']:.0%} of pool predictions",
            f"  shift by < 1 % between consecutive cycles, for {int(THRESHOLDS['stability_consecutive'])}",
            "  consecutive cycles",
            "- **Metric**: Fraction of pool samples with |Δpred / pred_prev| < 0.01",
            "- **Rationale**: Model has converged — additional experiments yield diminishing returns",
            "",
            "---",
            "",
            "## Acceptable Operating Ranges (Summary)",
            "",
            "| Metric | Acceptable Range | Action if Violated |",
            "|--------|----------------|--------------------|",
            f"| Replicate CV | < {THRESHOLDS['replicate_cv']:.0%} | Re-run assay / flag for review |",
            f"| k_cat prediction | [{THRESHOLDS['kcat_min']:.0e}, {THRESHOLDS['kcat_max']:.0e}] s⁻¹ | Flag out-of-bounds samples |",
            f"| K_m prediction | [{THRESHOLDS['km_min']:.0e}, {THRESHOLDS['km_max']:.0e}] M | Flag out-of-bounds samples |",
            "| Ensemble variance | > 0 | Re-initialise ensemble |",
            f"| Query size | [{THRESHOLDS['query_size_min']}, {THRESHOLDS['query_size_max']}] | Adjust to hardware capacity |",
            "| Hardware errors | 0 | Halt; user intervention |",
            "",
            "---",
            "",
            "## Monitoring Strategy",
            "",
            "- **Per-cycle**: QC report generated after each cycle; sent to campaign dashboard.",
            "- **Continuous**: Hardware errors monitored in real time during experiment execution.",
            "- **Rolling**: Prediction stability evaluated over a sliding window of 2 cycles.",
            "",
            "## Robustness Safeguards",
            "",
            "- The prediction-stability criterion requires **2 consecutive** cycles above the",
            "  threshold to prevent premature stopping from a single lucky cycle.",
            "- The MAE criterion requires **both** k_cat AND K_m to be below threshold",
            "  simultaneously, preventing a one-sided fit from triggering early stop.",
            "- Replicate CV is checked **before** lab handoff so that noisy batches are",
            "  caught before they contaminate the training set.",
            "",
            "---",
            "",
        ]

        if history:
            lines += [
                "## Campaign History Summary",
                "",
                "| Cycle | n_labeled | MAE log k_cat | MAE log K_m | Stability % | QC OK |",
                "|-------|-----------|---------------|-------------|-------------|-------|",
            ]
            for h in history:
                lines.append(
                    f"| {h['cycle']+1} | {h.get('n_labeled','?')} "
                    f"| {h.get('mae_log_kcat', float('nan')):.3f} "
                    f"| {h.get('mae_log_km', float('nan')):.3f} "
                    f"| {h.get('prediction_stability_pct', 0.0):.1%} "
                    f"| {'✓' if h.get('qc_all_ok', False) else '✗'} |"
                )
            lines.append("")

        return "\n".join(lines)
