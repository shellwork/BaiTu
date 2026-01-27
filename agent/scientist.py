from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from schemas.plan import ExperimentPlan
from schemas.result import ExperimentResult


@dataclass
class ScientistAgent:
    def decide_next_step(self, plan: ExperimentPlan, result: ExperimentResult) -> Dict[str, Any]:
        return {
            "should_iterate": True,
            "rationale": "结果未达到预设阈值",
            "next_hypothesis": "调整pH区间与孵育时间",

        }
    def analyze(self, raw_data: Dict[str, Any]) -> ExperimentResult:
        summary = "analysis_complete"
        metrics = {"sample_metric": 0.0}
        return ExperimentResult(summary=summary, metrics=metrics, artifacts={"raw": raw_data})
