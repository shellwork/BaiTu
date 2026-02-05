from __future__ import annotations

from dataclasses import dataclass

from schemas.plan import ExperimentPlan
from tools.hardware.momentum_api import MomentumAPI
from typing import Dict, Any

@dataclass
class ExecutorAgent:
    hardware_api: MomentumAPI = MomentumAPI()

    def generate_code(self, plan: ExperimentPlan) -> str:
        header = "# Auto-generated Momentum protocol\n"
        body = self.hardware_api.render_protocol(plan)
        return header + body

    def simulate(self, code: str) -> None:
        if "syntax_error" in code:
            raise ValueError("Dry run detected syntax_error marker")
        return None

    def execute(self, code: str) -> str:
        return "run_001"

    def collect_data(self, run_id: str) -> Dict[str, Any]:
        return {"run_id": run_id, "raw": "mock_data"}