from __future__ import annotations

from dataclasses import dataclass

from schemas.plan import ExperimentPlan


@dataclass
class OpentronsAPI:
    def render_protocol(self, plan: ExperimentPlan) -> str:
        lines = [
            "def run(protocol):",
            "    # TODO: map steps to hardware commands",
            f"    # Plan title: {plan.title}",
            "    pass",
        ]
        return "\n".join(lines) + "\n"
