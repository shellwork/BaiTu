from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any

from core.orchestrator import CentralOrchestrator


@dataclass
class AgentAPIServer:
    """
    统一的API入口（框架占位）。
    后续可替换为FastAPI/Flask/GRPC等实现。
    """

    orchestrator: CentralOrchestrator = field(default_factory=CentralOrchestrator)

    def handle_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_intent = payload.get("user_intent", "")
        if not user_intent:
            return {"error": "user_intent is required"}
        return self.orchestrator.handle_user_request(user_intent)


def main() -> None:
    server = AgentAPIServer()
    demo_payload = {"user_intent": "测试这10种突变体在不同pH值下的荧光强度"}
    print(server.handle_request(demo_payload))


if __name__ == "__main__":
    main()
