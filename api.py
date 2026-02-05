from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Protocol, Optional

from agent.orchestrator import CentralOrchestrator


class LLMProvider(Protocol):
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        ...


@dataclass
class OpenAIProvider:
    api_key: str = ""
    base_url: str = ""
    model: str = "gpt-4o-mini"

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        # Framework placeholder: replace with real OpenAI/compatible API calls later
        return {
            "provider": "openai",
            "model": self.model,
            "messages": messages,
            "output": "stub_response",
        }


@dataclass
class ProviderRegistry:
    providers: Dict[str, LLMProvider] = field(default_factory=dict)

    def register(self, name: str, provider: LLMProvider) -> None:
        self.providers[name] = provider

    def get(self, name: str) -> Optional[LLMProvider]:
        return self.providers.get(name)


@dataclass
class AgentAPIServer:
    """
    Unified API entrypoint (framework placeholder).
    Can be replaced later with FastAPI/Flask/GRPC implementations.
    """

    orchestrator: CentralOrchestrator = field(default_factory=CentralOrchestrator)
    registry: ProviderRegistry = field(default_factory=ProviderRegistry)

    def handle_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        user_intent = payload.get("user_intent", "")
        if not user_intent:
            return {"error": "user_intent is required"}
        
        provider_name = payload.get("provider", "openai")
        provider = self.registry.get(provider_name)
        
        # Inject the LLM provider into the Orchestrator
        self.orchestrator.llm_provider = provider
        self.orchestrator.__post_init__() # Ensure the injection takes effect

        result = self.orchestrator.handle_user_request(user_intent)
        return result


def main() -> None:
    server = AgentAPIServer()
    server.registry.register("openai", OpenAIProvider())
    demo_payload = {"user_intent": "测试这10种突变体在不同pH值下的荧光强度"}
    print(server.handle_request(demo_payload))


if __name__ == "__main__":
    main()
