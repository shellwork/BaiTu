from __future__ import annotations

from core.orchestrator import CentralOrchestrator


def main() -> None:
    orchestrator = CentralOrchestrator()
    user_intent = "测试这10种突变体在不同pH值下的荧光强度"
    output = orchestrator.handle_user_request(user_intent)
    print(output)


if __name__ == "__main__":
    main()
