import os
import re
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError


# ----------------------------
# Config
# ----------------------------
DB_URL = os.environ["DB_URL"]  # SQLAlchemy URL using pyodbc
API_BEARER_TOKEN = os.getenv("API_BEARER_TOKEN", "").strip()

MAX_ROWS = int(os.getenv("MAX_ROWS", "500"))
QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "15"))

# Optional allowlist (simple substring check)
TABLE_ALLOWLIST = [x.strip() for x in os.getenv("TABLE_ALLOWLIST", "").split(",") if x.strip()]

security = HTTPBearer(auto_error=False)

engine = create_engine(DB_URL, pool_pre_ping=True)


app = FastAPI(title="SQL Tool Microservice (SQL Server, Read-only)", version="1.0")


class QueryRequest(BaseModel):
    sql: str = Field(..., description="Read-only SELECT statement")
    params: dict[str, Any] = Field(default_factory=dict, description="Named parameters for the SQL query")


class QueryResponse(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    elapsed_ms: int

class SchemaRequest(BaseModel):
    schema: Optional[str] = Field(default=None, description="Schema name, e.g. dbo")
    table: Optional[str] = Field(default=None, description="Table name, e.g. Customers")
    include_system: bool = Field(default=False, description="Include system schemas like sys, INFORMATION_SCHEMA")


class ColumnInfo(BaseModel):
    schema: str
    table: str
    column: str
    data_type: str
    is_nullable: bool
    ordinal_position: int


class SchemaResponse(BaseModel):
    server: str = "sqlserver"
    database: Optional[str] = None
    tables: list[str] = Field(default_factory=list)
    columns: list[ColumnInfo] = Field(default_factory=list)

def _check_auth(credentials: Optional[HTTPAuthorizationCredentials]) -> None:
    if not API_BEARER_TOKEN:
        return
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    if credentials.credentials != API_BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


# Guardrails
_SELECT_ONLY = re.compile(r"^\s*(with\s+[\s\S]+?\)\s*)?select\b", re.IGNORECASE)
_DISALLOWED = re.compile(r"\b(insert|update|delete|merge|drop|alter|create|truncate|exec|execute|grant|revoke)\b", re.IGNORECASE)

def _enforce_guardrails(sql: str) -> str:
    s = sql.strip().rstrip(";")

    # block stacked statements quickly
    if ";" in s:
        raise HTTPException(status_code=400, detail="Only a single SELECT statement is allowed (no semicolons).")

    if not _SELECT_ONLY.search(s):
        raise HTTPException(status_code=400, detail="Only SELECT queries are allowed.")

    if _DISALLOWED.search(s):
        raise HTTPException(status_code=400, detail="Disallowed SQL keyword detected.")

    if TABLE_ALLOWLIST:
        lower = s.lower()
        if not any(t.lower() in lower for t in TABLE_ALLOWLIST):
            raise HTTPException(status_code=403, detail="Query does not reference an allowed table/view.")

    return s


def _apply_top(sql: str, max_rows: int) -> str:
    """
    SQL Server row limiting:
    - If the query already has TOP, leave it.
    - Otherwise inject TOP (max_rows) after SELECT (or SELECT DISTINCT).
    """
    # Already has TOP
    if re.search(r"^\s*select\s+top\s*\(", sql, re.IGNORECASE) or re.search(r"^\s*select\s+top\s+\d+", sql, re.IGNORECASE):
        return sql

    # SELECT DISTINCT ...
    m = re.match(r"^\s*select\s+distinct\s+", sql, re.IGNORECASE)
    if m:
        return re.sub(r"^\s*select\s+distinct\s+", f"SELECT DISTINCT TOP ({max_rows}) ", sql, flags=re.IGNORECASE)

    # Plain SELECT ...
    return re.sub(r"^\s*select\s+", f"SELECT TOP ({max_rows}) ", sql, flags=re.IGNORECASE)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/schema", response_model=SchemaResponse)
def schema(
    schema: Optional[str] = None,
    table: Optional[str] = None,
    include_system: bool = False,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    _check_auth(credentials)

    # Basic input hygiene
    if schema and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", schema):
        raise HTTPException(status_code=400, detail="Invalid schema name.")
    if table and not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        raise HTTPException(status_code=400, detail="Invalid table name.")

    # Optional allowlist enforcement:
    # If TABLE_ALLOWLIST is set, only allow schema/table that match it.
    # TABLE_ALLOWLIST entries like "dbo.Customers"
    if TABLE_ALLOWLIST:
        if schema and table:
            needle = f"{schema}.{table}".lower()
            if not any(x.lower() == needle for x in TABLE_ALLOWLIST):
                raise HTTPException(status_code=403, detail="Schema/table not in allowlist.")
        elif schema or table:
            # If they only ask for schema or table, still restrict to allowlist entries
            requested = (schema or "").lower()
            if schema and not any(x.lower().startswith(requested + ".") for x in TABLE_ALLOWLIST):
                raise HTTPException(status_code=403, detail="Schema not in allowlist.")
            if table and not any(x.lower().endswith("." + table.lower()) for x in TABLE_ALLOWLIST):
                raise HTTPException(status_code=403, detail="Table not in allowlist.")

    # Build filters (parameterized)
    where = []
    params: dict[str, Any] = {}

    if not include_system:
        where.append("c.TABLE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA')")

    if schema:
        where.append("c.TABLE_SCHEMA = :schema")
        params["schema"] = schema

    if table:
        where.append("c.TABLE_NAME = :table")
        params["table"] = table

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    sql_columns = f"""
        SELECT
            c.TABLE_SCHEMA AS [schema],
            c.TABLE_NAME   AS [table],
            c.COLUMN_NAME  AS [column],
            c.DATA_TYPE    AS [data_type],
            CASE WHEN c.IS_NULLABLE = 'YES' THEN 1 ELSE 0 END AS [is_nullable],
            c.ORDINAL_POSITION AS [ordinal_position]
        FROM INFORMATION_SCHEMA.COLUMNS c
        {where_sql}
        ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME, c.ORDINAL_POSITION
    """

    sql_tables = f"""
        SELECT DISTINCT
            c.TABLE_SCHEMA AS [schema],
            c.TABLE_NAME   AS [table]
        FROM INFORMATION_SCHEMA.COLUMNS c
        {where_sql}
        ORDER BY c.TABLE_SCHEMA, c.TABLE_NAME
    """

    try:
        with engine.connect() as conn:
            conn = conn.execution_options(timeout=QUERY_TIMEOUT_SECONDS)

            # Fetch tables
            trows = conn.execute(text(sql_tables), params).fetchall()
            tables = [f"{r.schema}.{r.table}" for r in trows]

            # Fetch columns (cap output defensively)
            crows = conn.execute(text(sql_columns), params).fetchmany(5000)
            columns = [
                ColumnInfo(
                    schema=r.schema,
                    table=r.table,
                    column=r.column,
                    data_type=r.data_type,
                    is_nullable=bool(r.is_nullable),
                    ordinal_position=int(r.ordinal_position),
                )
                for r in crows
            ]

            # Try to get DB name (optional)
            dbname = conn.execute(text("SELECT DB_NAME() AS db")).scalar()

    except SQLAlchemyError as e:
        raise HTTPException(status_code=400, detail=f"SQL error: {e}")

    return SchemaResponse(database=dbname, tables=tables, columns=columns)




@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    _check_auth(credentials)
    clean_sql = _enforce_guardrails(req.sql)
    limited_sql = _apply_top(clean_sql, MAX_ROWS)

    t0 = time.time()
    try:
        with engine.connect() as conn:
            # Driver-dependent timeout; still useful
            conn = conn.execution_options(timeout=QUERY_TIMEOUT_SECONDS)

            # NOCOUNT reduces extra rowcount messages
            final_sql = "SET NOCOUNT ON; " + limited_sql

            result = conn.execute(text(final_sql), req.params)
            cols = list(result.keys())
            rows = [list(r) for r in result.fetchall()]
    except SQLAlchemyError as e:
        raise HTTPException(status_code=400, detail=f"SQL error: {e}")
    finally:
        elapsed = int((time.time() - t0) * 1000)

    return QueryResponse(columns=cols, rows=rows, row_count=len(rows), elapsed_ms=elapsed)