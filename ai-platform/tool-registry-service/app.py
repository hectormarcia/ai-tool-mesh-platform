import json
import os
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


TOOLS_FILE = os.getenv("TOOLS_FILE", "/app/tools.json")


class ToolRecord(BaseModel):
    name: str
    description: str
    base_url: str
    endpoints: list[dict[str, Any]]
    auth: Optional[dict[str, Any]] = None


class ToolsResponse(BaseModel):
    tools: list[ToolRecord]


app = FastAPI(title="Tool Registry", version="1.0")


def _load_tools() -> list[ToolRecord]:
    if not os.path.exists(TOOLS_FILE):
        raise HTTPException(status_code=500, detail=f"Tools file not found: {TOOLS_FILE}")

    try:
        with open(TOOLS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        tools = data.get("tools", [])
        return [ToolRecord(**t) for t in tools]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load tools registry: {e}")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/tools", response_model=ToolsResponse)
def list_tools():
    return {"tools": _load_tools()}


@app.get("/tools/{tool_name}", response_model=ToolRecord)
def get_tool(tool_name: str):
    tools = _load_tools()
    for t in tools:
        if t.name.lower() == tool_name.lower():
            return t
    raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")