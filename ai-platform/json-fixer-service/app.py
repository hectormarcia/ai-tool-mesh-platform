import os
import json
import asyncio
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from jsonschema import Draft202012Validator

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ----------------------------
# Config
# ----------------------------
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()
security = HTTPBearer(auto_error=False)

REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "35"))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Azure OpenAI config (required)
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL", AZURE_OPENAI_DEPLOYMENT)

# LLM used only for "autofix"
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


# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(title="JSON Schema Validator + AutoFix", version="1.0")


class ValidateRequest(BaseModel):
    payload: Any
    schema: dict


class ValidateResponse(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)


class AutoFixRequest(BaseModel):
    payload: Any
    schema: dict
    max_attempts: int = 3


class AutoFixResponse(BaseModel):
    valid: bool
    attempts_used: int
    fixed_payload: Any = None
    errors: list[str] = Field(default_factory=list)
    last_model_output: Optional[str] = None


def _check_auth(credentials: Optional[HTTPAuthorizationCredentials]) -> None:
    if not API_BEARER_TOKEN:
        return
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


def _validate(payload: Any, schema: dict) -> tuple[bool, list[str]]:
    try:
        validator = Draft202012Validator(schema)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON Schema: {e}")

    errors = sorted(validator.iter_errors(payload), key=lambda e: list(e.path))
    if not errors:
        return True, []

    lines: list[str] = []
    for err in errors[:40]:
        # Build a readable JSONPath-ish pointer
        path = "$"
        for p in err.path:
            if isinstance(p, int):
                path += f"[{p}]"
            else:
                path += f".{p}"
        lines.append(f"{path}: {err.message}")

    if len(errors) > 40:
        lines.append(f"...and {len(errors) - 40} more error(s).")

    return False, lines


def _extract_json(text: str) -> Any:
    """
    Tries to parse the model output as JSON.
    If the model wrapped JSON in code fences, strip them.
    """
    t = text.strip()
    if t.startswith("```"):
        # remove leading ```lang and trailing ```
        t = t.split("\n", 1)[-1]
        if t.endswith("```"):
            t = t[:-3]
        t = t.strip()
    return json.loads(t)


async def _autofix_once(payload: Any, schema: dict, errors: list[str]) -> tuple[Any, str]:
    """
    Ask LLM to produce corrected JSON ONLY (no prose).
    Returns (fixed_payload, raw_model_output).
    """
    system = SystemMessage(content=(
        "You are a strict JSON transformer.\n"
        "Return ONLY valid JSON (no explanation, no markdown).\n"
        "Goal: transform the given payload so it validates against the provided JSON Schema.\n"
        "Do NOT add extra keys unless schema allows it.\n"
        "If a field has wrong type, convert it.\n"
        "If required fields missing, add them with sensible defaults.\n"
        "If additionalProperties is false, remove extra fields.\n"
    ))

    human = HumanMessage(content=json.dumps({
        "schema": schema,
        "payload": payload,
        "validation_errors": errors
    }, ensure_ascii=False))

    msg = await llm.ainvoke([system, human])
    raw = msg.content

    fixed = _extract_json(raw)
    return fixed, raw


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/validate", response_model=ValidateResponse)
def validate(
    req: ValidateRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    _check_auth(credentials)
    valid, errors = _validate(req.payload, req.schema)
    return {"valid": valid, "errors": errors}


@app.post("/autofix", response_model=AutoFixResponse)
async def autofix(
    req: AutoFixRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    _check_auth(credentials)

    if req.max_attempts < 1 or req.max_attempts > 10:
        raise HTTPException(status_code=400, detail="max_attempts must be between 1 and 10")

    async with _semaphore:
        try:
            # Whole request hard timeout
            return await asyncio.wait_for(_autofix_flow(req), timeout=REQUEST_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="Request timeout")


async def _autofix_flow(req: AutoFixRequest) -> AutoFixResponse:
    # 1) Validate initial payload
    valid, errors = _validate(req.payload, req.schema)
    if valid:
        return AutoFixResponse(
            valid=True,
            attempts_used=0,
            fixed_payload=req.payload,
            errors=[],
        )

    current = req.payload
    last_raw: Optional[str] = None

    # 2) Try to fix + revalidate up to max_attempts
    for attempt in range(1, req.max_attempts + 1):
        fixed, raw = await _autofix_once(current, req.schema, errors)
        last_raw = raw

        valid2, errors2 = _validate(fixed, req.schema)
        if valid2:
            return AutoFixResponse(
                valid=True,
                attempts_used=attempt,
                fixed_payload=fixed,
                errors=[],
                last_model_output=last_raw,
            )

        # continue loop using latest candidate + errors
        current = fixed
        errors = errors2

    # 3) Failed
    return AutoFixResponse(
        valid=False,
        attempts_used=req.max_attempts,
        fixed_payload=current,
        errors=errors,
        last_model_output=last_raw,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8010, reload=True)