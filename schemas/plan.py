from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ExperimentPlan:
    title: str
    steps: List[str]
    resources: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "steps": list(self.steps),
            "resources": dict(self.resources),
            "constraints": dict(self.constraints),
        }
