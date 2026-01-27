from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SOPRAGClient:
    def search(self, query: str) -> List[str]:
        return ["SOP-001", "SOP-002"]
