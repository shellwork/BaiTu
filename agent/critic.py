from __future__ import annotations

from dataclasses import dataclass

from schemas.plan import ExperimentPlan
from tools.rag.inventory_client import InventoryClient
from tools.rag.instrument_client import InstrumentClient


@dataclass
class CriticAgent:
    inventory: InventoryClient = InventoryClient()
    instruments: InstrumentClient = InstrumentClient()

    def verify_plan(self, plan: ExperimentPlan) -> ExperimentPlan:
        inventory_ok = self.inventory.check(plan)
        instrument_ok = self.instruments.validate(plan)

        if not inventory_ok:
            plan.constraints["inventory_warning"] = ""
        if not instrument_ok:
            plan.constraints["instrument_warning"] = ""
        return plan
