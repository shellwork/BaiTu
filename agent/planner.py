from __future__ import annotations

from dataclasses import dataclass

from schemas.plan import ExperimentPlan
from tools.rag.sop_client import SOPRAGClient


@dataclass
class PlannerAgent:
    rag_client: SOPRAGClient = SOPRAGClient()

    def create_plan(self, user_intent: str) -> ExperimentPlan:
        references = self.rag_client.search(user_intent)
        steps = [
            
        ]
        return ExperimentPlan(
            title=f"Plan for: {user_intent}",
            steps=steps,
            resources={"references": references},
            constraints={"source": "rag"},
        )
