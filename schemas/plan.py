from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ProtocolStep:
    action: str
    device: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "device": self.device,
            "parameters": self.parameters,
            "description": self.description,
        }


@dataclass
class ExperimentPlan:
    title: str
    steps: List[ProtocolStep] = field(default_factory=list)
    resources: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "steps": [step.to_dict() for step in self.steps],
            "resources": dict(self.resources),
            "constraints": dict(self.constraints),
        }
