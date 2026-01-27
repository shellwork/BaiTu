from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from schemas.result import ExperimentResult


@dataclass
class OrchestratorState:
    last_user_intent: Optional[str] = None
    last_result: Optional[ExperimentResult] = None
    loop_count: int = 0
