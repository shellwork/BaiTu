import sys
import os

# Ensure the project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()
from agent.orchestrator import CentralOrchestrator
from api import OpenAIProvider


def main():
    # Load .env next to this script to support local development.
    # 1. Initialize the LLM Provider
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it via 'export OPENAI_API_KEY=sk-...' before running.")
        return

    print(f"Initializing LLM Provider (Model: {model})...")
    llm = OpenAIProvider(api_key=api_key, base_url=base_url, model=model)
    
    # 2. Initialize the Orchestrator with the real LLM
    orchestrator = CentralOrchestrator(llm_provider=llm)
    
    # 3. Define the user request
    user_request = "design a experiment to measure the kinetic of a proteinase, we need to get km and vmax"
    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
        
    print(f"User Request: {user_request}\n")
    
    # 4. Run the Design Workflow
    print("--- Starting Workflow with Real Agent ---")
    try:
        result = orchestrator.run_experiment_design(user_request)
        
        # 5. Output the result
        print("\n--- Workflow Complete ---")
        print(f"Status: {result['status']}")
        print(f"Plan Title: {result['plan_title']}")
        print("\nGenerated Protocol:\n")
        print(result['protocol_text'])
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
