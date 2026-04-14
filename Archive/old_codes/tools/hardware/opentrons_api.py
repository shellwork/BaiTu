from __future__ import annotations

from dataclasses import dataclass
from typing import List

from schemas.plan import ExperimentPlan


@dataclass
class MomentumAPI:
    """
    Convert a structured ExperimentPlan into Momentum-executable Python scripts or API call sequences.
    """

    def render_protocol(self, plan: ExperimentPlan) -> str:
        lines = [
            "import momentum_api",
            "from momentum_api import Workflow, Device",
            "",
            f"# Generated Protocol: {plan.title}",
            "workflow = Workflow(name='Generated_Workflow')",
            "",
        ]

        for i, step in enumerate(plan.steps):
            lines.append(f"# Step {i+1}: {step.description}")
            
            # Map actions to specific Momentum/Felix instructions
            if step.action == "liquid_transfer":
                vol = step.parameters.get("volume", "50")
                unit = step.parameters.get("unit", "ul")
                lines.append(f"workflow.add_task(device='{step.device}', action='pipette', params={{'volume': {vol}, 'unit': '{unit}'}})")
            
            elif step.action == "incubation":
                temp = step.parameters.get("temperature", "37")
                dur = step.parameters.get("duration", "10")
                lines.append(f"workflow.add_task(device='{step.device}', action='incubate', params={{'temp': {temp}, 'duration': {dur}}})")
            
            elif step.action == "plate_reading":
                lines.append(f"workflow.add_task(device='{step.device}', action='read_plate')")
            
            else:
                lines.append(f"workflow.add_task(device='{step.device}', action='{step.action}', params={step.parameters})")
            
            lines.append("")

        lines.append("if __name__ == '__main__':")
        lines.append("    workflow.run()")
        
        return "\n".join(lines) + "\n"
