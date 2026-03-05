# AI Tool Mesh Platform -- Architecture & Implementation

## Overview

This project implements a containerized AI platform that enables an
LLM-powered agent to interact with enterprise systems through
guardrailed microservice tools.

Key technologies: - FastAPI microservices - Docker Compose - Azure
OpenAI - LangGraph agents - Tool Mesh discovery pattern - SQL Server
integration - JSON validation and repair

The system separates LLM reasoning from operational capabilities, which
is a recommended enterprise architecture for AI systems.

------------------------------------------------------------------------

# Architecture

## High-Level Architecture

            +-----------------------+
            | Tool Registry         |
            | (8030)                |
            +----------+------------+
                       |
                       v
            +-----------------------+
            | Agent API             |
            | (8000)                |
            | LangGraph ReAct Agent |
            +----------+------------+
                       |
     +-----------------+-------------------+
     |                                     |
     v                                     v

+----------------------+ +------------------------+
| JSON Guardrail Tool | | SQL Tool |
| (8010) | | (8020) |
| validate/autofix | | query/schema |
+----------+-----------+ +-----------+------------+
| |
v v
Azure OpenAI SQL Server
AdventureWorks

------------------------------------------------------------------------

# Services

## Agent API

Responsible for orchestrating the LLM and tools.

Endpoint: POST /invoke

Example:

{ "input": "Show the top 5 rows from Sales.SalesOrderHeader" }

The agent dynamically discovers tools from the registry and invokes them
as needed.

------------------------------------------------------------------------

## Tool Registry

Provides a catalog of available tools.

Endpoint: GET /tools

Purpose: Allow agents to discover tools dynamically instead of
hardcoding endpoints.

------------------------------------------------------------------------

## JSON Guardrail Service

Ensures structured JSON outputs remain valid.

Endpoints: POST /validate POST /autofix

Pattern:

Validate → Fix → Revalidate

This prevents malformed structured outputs from breaking downstream
automation.

------------------------------------------------------------------------

## SQL Tool Service

Provides safe database access.

Features: - SELECT-only queries - Row limits - Query timeouts - Schema
discovery

Endpoints: POST /query GET /schema

Example request:

{ "sql": "SELECT TOP (5) SalesOrderID, OrderDate FROM
Sales.SalesOrderHeader", "params": {} }

------------------------------------------------------------------------

# Tool Mesh Pattern

Traditional architecture:

LLM → hardcoded tools

Tool Mesh architecture:

Agent ↓ Tool Registry ↓ Dynamic tool wrappers ↓ Microservice tools

Benefits: - modular architecture - runtime tool discovery - easier
platform expansion

------------------------------------------------------------------------

# Docker Deployment

Services:

  Service         Port
  --------------- ------
  agent-api       8000
  json-fixer      8010
  sql-tool        8020
  tool-registry   8030

Start stack:

docker compose up -d --build

------------------------------------------------------------------------

# Example Request Flow

User request:

"Show me the latest sales orders"

Execution:

User ↓ Agent API ↓ Tool discovery ↓ sql_schema ↓ sql_query ↓ SQL Server
↓ Results returned

------------------------------------------------------------------------

# Capabilities

The platform provides:

-   AI tool orchestration
-   schema-aware SQL generation
-   guardrailed JSON outputs
-   microservice discovery
-   containerized deployment
-   integration with automation tools

------------------------------------------------------------------------

# Future Enhancements

-   Redis caching for schema memory
-   Tool health monitoring
-   Role-based access control
-   Multi-agent orchestration
-   Query cost analysis

------------------------------------------------------------------------

# Conclusion

This platform demonstrates how to build production-ready AI systems by
combining LLM reasoning with microservice tools, dynamic discovery, and
deterministic guardrails.
