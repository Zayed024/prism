"""Prism - Multi-Agent Productivity Assistant API Server."""

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from core.mcp_client import MCPClient
from core.tool_manager import ToolManager
from models import PrismRequest
from orchestrator import Orchestrator

load_dotenv()

# Globals set during lifespan
_mcp_clients: dict[str, MCPClient] = {}
_tool_manager: ToolManager | None = None
_orchestrator: Orchestrator | None = None
_exit_stack: AsyncExitStack | None = None
_db_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _mcp_clients, _tool_manager, _orchestrator, _exit_stack, _db_pool

    stack = AsyncExitStack()
    _exit_stack = stack

    database_url = os.environ.get("DATABASE_URL", "")
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    # Detect the Python command (for Cloud Run vs local)
    python_cmd = sys.executable

    # Environment for MCP server subprocesses
    env = {**os.environ}

    # Start MCP servers
    print("[Prism] Starting MCP servers...")

    _mcp_clients["tasks"] = await stack.enter_async_context(
        MCPClient(command=python_cmd, args=["mcp_servers/tasks_server.py"], env=env)
    )
    print("[Prism] Tasks MCP connected")

    _mcp_clients["notes"] = await stack.enter_async_context(
        MCPClient(command=python_cmd, args=["mcp_servers/notes_server.py"], env=env)
    )
    print("[Prism] Notes MCP connected")

    _mcp_clients["calendar"] = await stack.enter_async_context(
        MCPClient(command=python_cmd, args=["mcp_servers/calendar_server.py"], env=env)
    )
    print("[Prism] Calendar MCP connected")

    _mcp_clients["email"] = await stack.enter_async_context(
        MCPClient(command=python_cmd, args=["mcp_servers/email_server.py"], env=env)
    )
    print("[Prism] Email MCP connected")

    # Create tool manager and orchestrator
    _tool_manager = ToolManager(_mcp_clients)
    _orchestrator = Orchestrator(_tool_manager, model=gemini_model)
    print(f"[Prism] Orchestrator ready (model: {gemini_model})")

    # Direct DB pool for REST endpoints
    if database_url:
        try:
            _db_pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
            print("[Prism] Database pool ready")
        except Exception as e:
            print(f"[Prism] DB pool failed (REST endpoints will use MCP): {e}")
    else:
        print("[Prism] No DATABASE_URL, REST endpoints will query via MCP")

    yield

    # Cleanup
    if _db_pool:
        await _db_pool.close()
    await stack.aclose()
    print("[Prism] Shutdown complete")


app = FastAPI(title="Prism", version="1.0.0", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Frontend ──────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ── Main Prism Endpoint (SSE) ─────────────────────────────────

@app.get("/api/prism")
async def prism_sse(query: str):
    """Run Prism via SSE (GET for EventSource compatibility)."""
    queue: asyncio.Queue = asyncio.Queue()

    async def callback(event: dict):
        await queue.put(event)

    async def run_orchestrator():
        try:
            result = await _orchestrator.run(query, callback=callback)
            # Save session to DB
            if _db_pool:
                try:
                    await _db_pool.execute(
                        """INSERT INTO prism_sessions (user_request, red_response, blue_response, green_response, merged_result)
                           VALUES ($1, $2, $3, $4, $5)""",
                        query,
                        json.dumps(result["agents"].get("red", {})),
                        json.dumps(result["agents"].get("blue", {})),
                        json.dumps(result["agents"].get("green", {})),
                        json.dumps(result["merged"]),
                    )
                except Exception as e:
                    print(f"[Prism] Failed to save session: {e}")
        except Exception as e:
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(None)  # Signal end

    async def event_generator():
        task = asyncio.create_task(run_orchestrator())
        while True:
            event = await queue.get()
            if event is None:
                break
            yield {
                "event": event["type"],
                "data": json.dumps(event),
            }
        await task

    return EventSourceResponse(event_generator())


@app.post("/api/prism")
async def prism_post(request: PrismRequest):
    """Run Prism synchronously (for non-SSE clients)."""
    result = await _orchestrator.run(request.query)

    # Save session
    session_id = None
    if _db_pool:
        try:
            row = await _db_pool.fetchrow(
                """INSERT INTO prism_sessions (user_request, red_response, blue_response, green_response, merged_result)
                   VALUES ($1, $2, $3, $4, $5) RETURNING id""",
                request.query,
                json.dumps(result["agents"].get("red", {})),
                json.dumps(result["agents"].get("blue", {})),
                json.dumps(result["agents"].get("green", {})),
                json.dumps(result["merged"]),
            )
            session_id = row["id"]
        except Exception as e:
            print(f"[Prism] Failed to save session: {e}")

    return {
        "session_id": session_id,
        "user_request": request.query,
        "agents": result["agents"],
        "merged": result["merged"],
    }


# ── REST Endpoints ────────────────────────────────────────────

@app.get("/api/tasks")
async def get_tasks(status: str = "", priority: str = ""):
    if _db_pool:
        query = "SELECT * FROM tasks WHERE 1=1"
        params = []
        idx = 1
        if status:
            query += f" AND status = ${idx}"; params.append(status); idx += 1
        if priority:
            query += f" AND priority = ${idx}"; params.append(priority); idx += 1
        query += " ORDER BY created_at DESC"
        rows = await _db_pool.fetch(query, *params)
        return [_row_to_dict(r) for r in rows]
    # Fallback: query via MCP
    args = {}
    if status: args["status"] = status
    if priority: args["priority"] = priority
    result = await _mcp_clients["tasks"].call_tool("list_tasks", args)
    if result and result.content:
        return json.loads(result.content[0].text)
    return []


@app.get("/api/notes")
async def get_notes():
    if _db_pool:
        rows = await _db_pool.fetch("SELECT * FROM notes ORDER BY created_at DESC")
        return [_row_to_dict(r) for r in rows]
    result = await _mcp_clients["notes"].call_tool("list_notes", {})
    if result and result.content:
        return json.loads(result.content[0].text)
    return []


@app.get("/api/history")
async def get_history(limit: int = 20):
    if not _db_pool:
        return []  # No history without DB
    rows = await _db_pool.fetch(
        "SELECT id, user_request, merged_result, created_at FROM prism_sessions ORDER BY created_at DESC LIMIT $1",
        limit,
    )
    return [_row_to_dict(r) for r in rows]


@app.get("/api/stats")
async def get_stats():
    if _db_pool:
        tasks = await _db_pool.fetchval("SELECT COUNT(*) FROM tasks")
        notes = await _db_pool.fetchval("SELECT COUNT(*) FROM notes")
        sessions = await _db_pool.fetchval("SELECT COUNT(*) FROM prism_sessions")
        return {"tasks": tasks, "notes": notes, "sessions": sessions}
    # Fallback: count via MCP
    try:
        t = await _mcp_clients["tasks"].call_tool("list_tasks", {})
        n = await _mcp_clients["notes"].call_tool("list_notes", {})
        tc = len(json.loads(t.content[0].text)) if t and t.content else 0
        nc = len(json.loads(n.content[0].text)) if n and n.content else 0
        return {"tasks": tc, "notes": nc, "sessions": 0}
    except Exception:
        return {"tasks": 0, "notes": 0, "sessions": 0}


# ── Helpers ───────────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


if __name__ == "__main__":
    import uvicorn
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
