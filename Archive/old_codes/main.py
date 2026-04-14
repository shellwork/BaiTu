from __future__ import annotations

from agent.orchestrator import CentralOrchestrator


def main() -> None:
    orchestrator = CentralOrchestrator()
    user_intent = ""
    output = orchestrator.handle_user_request(user_intent)
    print(output)


if __name__ == "__main__":
    main()
