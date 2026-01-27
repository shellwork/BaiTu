from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

from planner import PlannerAgent
from critic import CriticAgent
from executor import ExecutorAgent
from scientist import ScientistAgent
from schemas.plan import ExperimentPlan
from schemas.state import OrchestratorState
from schemas.result import ExperimentResult


@dataclass
class CentralOrchestrator:
    state: OrchestratorState = field(default_factory=OrchestratorState)
    planner: PlannerAgent = field(default_factory=PlannerAgent)
    critic: CriticAgent = field(default_factory=CriticAgent)
    executor: ExecutorAgent = field(default_factory=ExecutorAgent)
    scientist: ScientistAgent = field(default_factory=ScientistAgent)

    def handle_user_request(self, user_intent: str) -> Dict[str, Any]:
        self.state.last_user_intent = user_intent
        plan: ExperimentPlan = self.planner.create_plan(user_intent)
        verified_plan: ExperimentPlan = self.critic.verify_plan(plan)

        code = self.executor.generate_code(verified_plan)
        self.executor.simulate(code)

        run_id = self.executor.execute(code)
        raw_data = self.executor.collect_data(run_id)

        result: ExperimentResult = self.scientist.analyze(raw_data)
        next_step = self.scientist.decide_next_step(verified_plan, result)

        self.state.last_result = result
        self.state.loop_count += 1

        return {
            "plan": verified_plan.to_dict(),
            "code": code,
            "result": result.to_dict(),
            "next_step": next_step,
        }
