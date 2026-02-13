# BaiTu
An AI agent for automated experiment protocol design(02-762)

## Project Structure
- `core/`: central orchestrator and state
- `agent/brain/`: planner + critic
- `agent/translator/`: code generation + dry run
- `agent/hands/`: execution and monitoring
- `agent/eyes/`: analysis + scientist loop
- `schemas/`: shared data models
- `tools/`: RAG + hardware adapters

## Quick Start
```
python main.py
```
new project workflow:
Input (natural language)-> model1 -> output_m1(numerical parameters) -> model2 -> output_m2(predicted kinetics curve)
