import os
import json
import time
import asyncio
from typing import Optional, Any

import httpx
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# =========================================================
# Config (env)
# =========================================================
# Azure OpenAI (required)
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", AZURE_OPENAI_DEPLOYMENT)

# Agent API auth (optional; if empty/unset => no auth)
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()
security = HTTPBearer(auto_error=False)

# Request controls
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "35"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Tool Registry (Tool Mesh)
TOOL_REGISTRY_URL = os.getenv("TOOL_REGISTRY_URL", "http://tool-registry:8030").rstrip("/")
TOOL_REGISTRY_TTL_SECONDS = int(os.getenv("TOOL_REGISTRY_TTL_SECONDS", "300"))  # cache registry for 5 minutes


# =========================================================
# FastAPI app
# =========================================================
app = FastAPI(title="Agent API (Azure + LangGraph + Tool Mesh)", version="2.0")


class InvokeRequest(BaseModel):
    input: str


class InvokeResponse(BaseModel):
    output: str


def _check_auth(credentials: Optional[HTTPAuthorizationCredentials]) -> None:
    if not API_BEARER_TOKEN:
        return  # auth disabled
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


@app.get("/health")
def health():
    return {"ok": True}


# =========================================================
# Local tool(s)
# =========================================================
@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression (e.g., '12*(3+4)'). Demo only."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# =========================================================
# Azure OpenAI + Agent
# =========================================================
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    model=AZURE_OPENAI_MODEL,
    temperature=0,
    timeout=30,
    max_retries=2,
)


# =========================================================
# Tool Mesh (dynamic tool discovery)
# =========================================================
_tools_cache: dict[str, Any] = {"ts": 0.0, "tools": None}


def _resolve_bearer_token_from_env(env_name: Optional[str]) -> str:
    if not env_name:
        return ""
    return os.getenv(env_name, "").strip()


def _registry_fetch_tools() -> list[dict]:
    with httpx.Client(timeout=10.0) as client:
        r = client.get(f"{TOOL_REGISTRY_URL}/tools")
    r.raise_for_status()
    return r.json().get("tools", [])


def _get_registry_tools_cached() -> list[dict]:
    now = time.time()
    if _tools_cache["tools"] is not None and (now - _tools_cache["ts"] < TOOL_REGISTRY_TTL_SECONDS):
        return _tools_cache["tools"]
    tools = _registry_fetch_tools()
    _tools_cache["tools"] = tools
    _tools_cache["ts"] = now
    return tools


def _find_tool(reg_tools: list[dict], name: str) -> dict:
    for t in reg_tools:
        if t.get("name", "").lower() == name.lower():
            return t
    raise RuntimeError(f"Tool not found in registry: {name}")


def _make_headers(auth_block: Optional[dict]) -> dict:
    headers = {"Content-Type": "application/json"}
    if auth_block and auth_block.get("type") == "bearer":
        token = _resolve_bearer_token_from_env(auth_block.get("env"))
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def _build_mesh_tools():
    """
    Builds tool wrappers based on tool-registry definitions.

    Expected tools in registry:
      - json_validate  -> base_url http://json-fixer:8010, POST /validate
      - json_autofix   -> base_url http://json-fixer:8010, POST /autofix
      - sql_query      -> base_url http://sql-tool:8020, POST /query and GET /schema
    """
    reg = _get_registry_tools_cached()

    json_validate_tool = _find_tool(reg, "json_validate")
    json_autofix_tool = _find_tool(reg, "json_autofix")
    sql_tool = _find_tool(reg, "sql_query")

    json_validate_url = json_validate_tool["base_url"].rstrip("/") + "/validate"
    json_autofix_url = json_autofix_tool["base_url"].rstrip("/") + "/autofix"
    sql_query_url = sql_tool["base_url"].rstrip("/") + "/query"
    sql_schema_url = sql_tool["base_url"].rstrip("/") + "/schema"

    json_validate_headers = _make_headers(json_validate_tool.get("auth"))
    json_autofix_headers = _make_headers(json_autofix_tool.get("auth"))
    sql_headers = _make_headers(sql_tool.get("auth"))

    @tool
    def json_validate(payload_json: str, schema_json: str) -> str:
        """Validate payload against schema via Tool Mesh (json-fixer /validate). Inputs are JSON strings."""
        try:
            payload = json.loads(payload_json)
            schema = json.loads(schema_json)
        except Exception as e:
            return f"Invalid JSON input: {e}"

        try:
            with httpx.Client(timeout=15.0) as client:
                r = client.post(
                    json_validate_url,
                    headers=json_validate_headers,
                    json={"payload": payload, "schema": schema},
                )
            r.raise_for_status()
            return json.dumps(r.json(), ensure_ascii=False)
        except Exception as e:
            return f"json_validate call failed: {e}"

    @tool
    def json_autofix(payload_json: str, schema_json: str, max_attempts: int = 3) -> str:
        """Auto-fix payload to satisfy schema via Tool Mesh (json-fixer /autofix). Inputs are JSON strings."""
        try:
            payload = json.loads(payload_json)
            schema = json.loads(schema_json)
        except Exception as e:
            return f"Invalid JSON input: {e}"

        try:
            with httpx.Client(timeout=30.0) as client:
                r = client.post(
                    json_autofix_url,
                    headers=json_autofix_headers,
                    json={"payload": payload, "schema": schema, "max_attempts": int(max_attempts)},
                )
            r.raise_for_status()
            return json.dumps(r.json(), ensure_ascii=False)
        except Exception as e:
            return f"json_autofix call failed: {e}"

    @tool
    def sql_query(sql: str, params_json: str = "{}") -> str:
        """Run a read-only SQL query via Tool Mesh (sql-tool /query). params_json is a JSON string."""
        try:
            params = json.loads(params_json) if params_json else {}
        except Exception as e:
            return f"Invalid params_json: {e}"

        try:
            with httpx.Client(timeout=20.0) as client:
                r = client.post(
                    sql_query_url,
                    headers=sql_headers,
                    json={"sql": sql, "params": params},
                )
            r.raise_for_status()
            return json.dumps(r.json(), ensure_ascii=False)
        except Exception as e:
            return f"sql_query call failed: {e}"

    @tool
    def sql_schema(schema: str = "", table: str = "", include_system: bool = False) -> str:
        """Fetch DB schema info via Tool Mesh (sql-tool /schema)."""
        params: dict[str, str] = {}
        if schema:
            params["schema"] = schema
        if table:
            params["table"] = table
        if include_system:
            params["include_system"] = "true"

        try:
            with httpx.Client(timeout=15.0) as client:
                r = client.get(sql_schema_url, headers=sql_headers, params=params)
            r.raise_for_status()
            return json.dumps(r.json(), ensure_ascii=False)
        except Exception as e:
            return f"sql_schema call failed: {e}"

    return [json_validate, json_autofix, sql_query, sql_schema]


# Build mesh tools once at startup
mesh_tools = _build_mesh_tools()

# Create agent with local + mesh tools
agent = create_react_agent(llm, tools=[calculator, *mesh_tools])


def _run_agent_sync(user_text: str) -> str:
    result = agent.invoke({"messages": [HumanMessage(content=user_text)]})
    return result["messages"][-1].content


# =========================================================
# API endpoint
# =========================================================
@app.post("/invoke", response_model=InvokeResponse)
async def invoke(
    req: InvokeRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    _check_auth(credentials)

    async with _semaphore:
        try:
            output = await asyncio.wait_for(
                asyncio.to_thread(_run_agent_sync, req.input),
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return {"output": output}
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timeout")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# Local run
# =========================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)