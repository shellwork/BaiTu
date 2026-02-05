from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from schemas.plan import ExperimentPlan, ProtocolStep
from tools.rag.sop_client import SOPRAGClient


@dataclass
class PlannerAgent:
    rag_client: SOPRAGClient = SOPRAGClient()
    llm_provider: Optional[Any] = None  # Injected by Orchestrator

    def create_plan(self, user_intent: str) -> ExperimentPlan:
        """
        Use the LLM to convert a user's natural-language intent into a structured experiment plan.
        """
        # 1. Retrieve relevant SOP references
        references = self.rag_client.search(user_intent)
        
        # 2. Build the prompt and call the LLM
        if self.llm_provider:
            steps = self._llm_extract_steps(user_intent, references)
        else:
            # Fall back to heuristic parsing (safety net)
            steps = self._heuristic_extract_steps(user_intent)
        
        return ExperimentPlan(
            title=f"Momentum Workflow: {user_intent[:50]}",
            steps=steps,
            resources={
                "references": references,
                "platform": "Momentum + Felix",
                "engine": "LLM-Powered" if self.llm_provider else "Heuristic"
            },
            constraints={"source": "llm_structured_extraction"},
        )

    def _llm_extract_steps(self, intent: str, references: List[str]) -> List[ProtocolStep]:
        """
        Obtain structured steps using LLM prompt engineering.
        """
        prompt = self._build_planner_prompt(intent, references)
        
        # Call the LLM
        response = self.llm_provider.chat([
            {"role": "system", "content": "You are an expert lab automation scientist specializing in Momentum and Felix platforms."},
            {"role": "user", "content": prompt}
        ])
        
        # Parse the JSON content returned by the LLM
        try:
            # Assume the LLM returns a structured JSON string
            content = response.get("output", "")
            # Light cleanup for possible markdown markers
            json_str = re.search(r'\[.*\]', content, re.DOTALL).group()
            raw_steps = json.loads(json_str)
            
            steps = []
            for s in raw_steps:
                steps.append(ProtocolStep(
                    action=s.get("action", "generic_task"),
                    device=s.get("device", "Momentum_Scheduler"),
                    parameters=s.get("parameters", {}),
                    description=s.get("description", "")
                ))
            return steps
        except Exception as e:
            print(f"Error parsing LLM response: {e}. Falling back to heuristics.")
            return self._heuristic_extract_steps(intent)

    def _build_planner_prompt(self, intent: str, references: List[str]) -> str:
        return f"""
Translate the following laboratory experiment intent into a structured sequence of steps for an automated platform (Momentum scheduler and Felix liquid handler).

User Intent: {intent}
Reference SOPs: {", ".join(references)}

Available Devices:
- Felix: Liquid handling (pipetting, transfer, mixing)
- Inheco_Incubator: Heating, shaking, incubation
- Agilent_Centrifuge: Centrifugation
- Brooks_Sealer: Plate sealing
- Brooks_Peeler: Plate peeling
- BMG_Reader: Fluorescence/Absorbance reading

Output Format (JSON array only):
[
  {{
    "action": "liquid_transfer",
    "device": "Felix",
    "parameters": {{"volume": 50, "unit": "ul"}},
    "description": "Transfer 50ul of buffer to plate A"
  }},
  ...
]
"""

    def _heuristic_extract_steps(self, intent: str) -> List[ProtocolStep]:
        # Keep heuristic logic as a safety net
        steps = []
        rules = {
            r"(移液|加样|分液|transfer|pipette)": ("liquid_transfer", "Felix"),
            r"(孵育|加热|震荡|incubate|shake)": ("incubation", "Inheco_Incubator"),
            r"(检测|读板|read|measure)": ("plate_reading", "BMG_Reader"),
        }
        raw_sentences = re.split(r'[;；\n]', intent)
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence: continue
            matched = False
            for pattern, (action, default_device) in rules.items():
                if re.search(pattern, sentence, re.IGNORECASE):
                    steps.append(ProtocolStep(action=action, device=default_device, parameters={}, description=sentence))
                    matched = True
                    break
            if not matched:
                steps.append(ProtocolStep(action="generic_task", device="Momentum_Scheduler", parameters={}, description=sentence))
        return steps
