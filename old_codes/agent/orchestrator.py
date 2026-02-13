from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from agent.planner import PlannerAgent
from agent.critic import CriticAgent
from agent.executor import ExecutorAgent
from agent.scientist import ScientistAgent
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
    llm_provider: Optional[Any] = None

    def __post_init__(self) -> None:
        """
        Inject the LLM provider into agents that need it during initialization.
        """
        if self.llm_provider:
            self.planner.llm_provider = self.llm_provider
            # Can also be injected into Critic or Scientist later

    def run_experiment_design(self, user_intent: str) -> Dict[str, Any]:
        """
        Run ONLY the design phase: Natural Language -> Text Protocol.
        """
        self.state.last_user_intent = user_intent
        
        # Call the planner's design method
        plan: ExperimentPlan = self.planner.design_experiment(user_intent)
        
        return {
            "status": "design_complete",
            "plan_title": plan.title,
            "protocol_text": plan.protocol_text,
            "references": plan.resources.get("references", [])
        }

    def handle_user_request(self, user_intent: str) -> Dict[str, Any]:
        """
        Main multi-agent workflow:
        1. Planner: natural language -> structured plan (LLM-driven)
        2. Critic: plan verification (inventory/equipment/safety)
        3. Executor: plan -> Momentum code -> execution
        4. Scientist: result analysis -> next-step suggestion
        """
        self.state.last_user_intent = user_intent
        
        # 1. Planning stage
        plan: ExperimentPlan = self.planner.create_plan(user_intent)
        
        # 2. Review stage
        verified_plan: ExperimentPlan = self.critic.verify_plan(plan)
        if verified_plan.constraints.get("error"):
            return {"error": "Plan verification failed", "details": verified_plan.constraints}

        # 3. Execution stage
        code = self.executor.generate_code(verified_plan)
        try:
            self.executor.simulate(code)
            run_id = self.executor.execute(code)
            raw_data = self.executor.collect_data(run_id)
        except Exception as e:
            return {"error": "Execution failed", "message": str(e), "code": code}

        # 4. Analysis stage
        result: ExperimentResult = self.scientist.analyze(raw_data)
        next_step = self.scientist.decide_next_step(verified_plan, result)

        self.state.last_result = result
        self.state.loop_count += 1

        return {
            "plan": verified_plan.to_dict(),
            "code": code,
            "result": result.to_dict(),
            "next_step": next_step,
            "status": "success"
        }
