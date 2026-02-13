from __future__ import annotations

import os
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Dict, Any, List, Protocol, Optional

from agent.orchestrator import CentralOrchestrator


class LLMProvider(Protocol):
    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        ...


@dataclass
class OpenAIProvider:
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    model: str = "gpt-4o"
    client: OpenAI = field(init=False)

    def __post_init__(self):
        if not self.api_key:
            # Try fallbacks or warn
            print("Warning: OPENAI_API_KEY not found in environment variables.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> Dict[str, Any]:
        """
        Execute chat completion using the official OpenAI client.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            content = response.choices[0].message.content
            return {
                "provider": "openai",
                "model": self.model,
                "messages": messages,
                "output": content,
            }
        except Exception as e:
            return {
                "error": str(e),
                "provider": "openai",
                "output": f"Error calling LLM: {str(e)}"
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
        # Instantiate provider on demand if not registered, to pick up env vars
        if provider_name == "openai" and "openai" not in self.registry.providers:
            # Allow overriding via payload or env vars
            api_key = payload.get("api_key") or os.getenv("OPENAI_API_KEY")
            base_url = payload.get("base_url") or os.getenv("OPENAI_BASE_URL")
            model = payload.get("model") or "gpt-4o"
            
            self.registry.register("openai", OpenAIProvider(
                api_key=api_key if api_key else "",
                base_url=base_url if base_url else "https://api.openai.com/v1",
                model=model
            ))
            
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
