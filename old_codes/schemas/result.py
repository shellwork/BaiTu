from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ExperimentResult:
    summary: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "metrics": dict(self.metrics),
            "artifacts": dict(self.artifacts),
        }
