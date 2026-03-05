# AI Tool Mesh Platform

A microservice-based AI platform that enables an LLM agent to interact with enterprise systems using guardrailed tools.

## Key Features

- Azure OpenAI powered agent
- Tool Mesh dynamic discovery
- Guardrailed JSON processing
- Safe SQL Server access
- Containerized microservices

## Services

| Service | Port | Purpose |
|-------|------|------|
| agent-api | 8000 | LLM orchestration |
| json-fixer | 8010 | JSON validation & repair |
| sql-tool | 8020 | Safe SQL execution |
| tool-registry | 8030 | Tool discovery |

## Start Platform

```bash
docker compose up -d --build
