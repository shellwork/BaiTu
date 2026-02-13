from __future__ import annotations

from dataclasses import dataclass

from schemas.plan import ExperimentPlan


@dataclass
class InstrumentClient:
    def validate(self, plan: ExperimentPlan) -> bool:
        return True
