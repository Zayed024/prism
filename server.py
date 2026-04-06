"""Prism - Multi-Agent Productivity Assistant API Server.

Built with Google ADK (Agent Development Kit), MCP, and AlloyDB.
"""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime

import asyncpg
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from core.mcp_client import MCPClient
import math
from models import PrismRequest, CreateTaskRequest, CreateNoteRequest, UpdateTaskRequest
from orchestrator import Orchestrator

load_dotenv()

# Globals
_orchestrator: Orchestrator | None = None
_db_pool: asyncpg.Pool | None = None
_mcp_clients: dict[str, MCPClient] = {}  # For REST endpoint fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _db_pool, _mcp_clients

    database_url = os.environ.get("DATABASE_URL", "")
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    python_cmd = sys.executable
    env = {**os.environ}

    # Initialize ADK orchestrator (manages MCP toolsets internally)
    print("[Prism] Initializing ADK orchestrator...")
    _orchestrator = Orchestrator(model=gemini_model)
    await _orchestrator.initialize(python_cmd=python_cmd, env=env)

    # MCP clients for REST endpoints (separate from ADK's toolsets)
    from contextlib import AsyncExitStack
    stack = AsyncExitStack()

    _mcp_clients["tasks"] = await stack.enter_async_context(
        MCPClient(command=python_cmd, args=["mcp_servers/tasks_server.py"], env=env)
    )
    _mcp_clients["notes"] = await stack.enter_async_context(
        MCPClient(command=python_cmd, args=["mcp_servers/notes_server.py"], env=env)
    )
    print("[Prism] REST MCP clients connected")

    # Share MCP clients with orchestrator for smart context (AlloyDB AI semantic search)
    _orchestrator._mcp_clients = _mcp_clients

    # Direct DB pool for REST endpoints
    if database_url:
        try:
            _db_pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
            print("[Prism] Database pool ready")
        except Exception as e:
            print(f"[Prism] DB pool failed: {e}")
    else:
        print("[Prism] No DATABASE_URL, REST endpoints use MCP fallback")

    yield

    if _db_pool:
        await _db_pool.close()
    await stack.aclose()
    print("[Prism] Shutdown complete")


app = FastAPI(
    title="Prism",
    version="1.0.0",
    description="Multi-Agent Productivity Assistant — 3 AI agents with different cognitive styles collaborate via MCP tools, negotiate, and merge results. Built with Google ADK, Gemini 2.5 Flash, AlloyDB AI, and MCP.",
    lifespan=lifespan,
)
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


# ── Pre-built Workflows ──────────────────────────────────────

WORKFLOWS = {
    "morning_briefing": {
        "name": "Morning Briefing",
        "description": "Check emails, review overdue tasks, scan calendar, get an action plan",
        "icon": "sunrise",
        "steps": [
            {"label": "Scan inbox", "query": "Check my unread emails and summarize anything that needs attention today"},
            {"label": "Review overdue", "query": "What tasks are overdue or due today? List them by priority"},
            {"label": "Check calendar", "query": "What's on my calendar today? Flag any conflicts or tight scheduling"},
            {"label": "Action plan", "query": "Based on my emails, tasks, and calendar, create a prioritized action plan for today. Flag anything that violates human performance limits."},
        ],
    },
    "weekly_review": {
        "name": "Weekly Review",
        "description": "Summarize the week, identify patterns, plan next week",
        "icon": "chart",
        "steps": [
            {"label": "Task status", "query": "List all tasks grouped by status (done, in_progress, todo). How many were completed vs created this week?"},
            {"label": "Find patterns", "query": "Search my notes and tasks for recurring themes or blockers. What keeps coming up?"},
            {"label": "Plan next week", "query": "Based on my current tasks, notes, and calendar, suggest a focus plan for next week with the top 3 priorities"},
        ],
    },
    "meeting_prep": {
        "name": "Meeting Prep",
        "description": "Gather context, check related tasks, draft an agenda",
        "icon": "users",
        "steps": [
            {"label": "Find context", "query": "Search my notes and emails for anything related to upcoming client meetings or presentations"},
            {"label": "Check tasks", "query": "What tasks are related to client work or presentations? What's their status?"},
            {"label": "Draft agenda", "query": "Based on the context and tasks, create a meeting agenda note with key discussion points and action items to review"},
        ],
    },
}


@app.get("/api/workflows")
async def list_workflows():
    """List available pre-built workflows."""
    return {k: {"name": v["name"], "description": v["description"], "icon": v["icon"], "steps": len(v["steps"])} for k, v in WORKFLOWS.items()}


@app.get("/api/workflows/{workflow_id}/run")
async def run_workflow(workflow_id: str):
    """Run a multi-step workflow via SSE — chains Prism queries sequentially."""
    if workflow_id not in WORKFLOWS:
        return {"error": f"Workflow '{workflow_id}' not found"}

    workflow = WORKFLOWS[workflow_id]
    queue: asyncio.Queue = asyncio.Queue()

    async def run_steps():
        try:
            step_results = []
            for i, step in enumerate(workflow["steps"]):
                await queue.put({
                    "type": "workflow_step_start",
                    "step": i,
                    "label": step["label"],
                    "query": step["query"],
                    "total_steps": len(workflow["steps"]),
                })

                # Build enriched query with previous step context
                enriched_query = step["query"]
                if step_results:
                    prev_context = "\n".join(
                        f"[Previous step '{s['label']}']: {s['summary'][:300]}"
                        for s in step_results
                    )
                    enriched_query = f"Context from previous steps:\n{prev_context}\n\nCurrent task: {step['query']}"

                # Run Prism for this step
                result = await _orchestrator.run(enriched_query)
                merged = result.get("merged", {})
                summary = merged.get("merged_response", "No result")

                step_results.append({"label": step["label"], "summary": summary})

                await queue.put({
                    "type": "workflow_step_done",
                    "step": i,
                    "label": step["label"],
                    "result": {
                        "agents": result.get("agents", {}),
                        "merged": merged,
                    },
                })

                # Brief pause between steps
                if i < len(workflow["steps"]) - 1:
                    await asyncio.sleep(2)

            await queue.put({
                "type": "workflow_complete",
                "workflow": workflow["name"],
                "steps_completed": len(step_results),
                "results": step_results,
            })
        except Exception as e:
            await queue.put({"type": "workflow_error", "message": str(e)})
        finally:
            await queue.put(None)

    async def event_generator():
        task = asyncio.create_task(run_steps())
        while True:
            event = await queue.get()
            if event is None:
                break
            yield {"event": event["type"], "data": json.dumps(event)}
        await task

    return EventSourceResponse(event_generator())


# ── Main Prism Endpoint (SSE) ─────────────────────────────────

@app.get("/api/prism")
async def prism_sse(query: str, deep: bool = False):
    """Run Prism via SSE. Add ?deep=true for negotiation + contrarian analysis."""
    queue: asyncio.Queue = asyncio.Queue()

    async def callback(event: dict):
        await queue.put(event)

    async def run_orchestrator():
        try:
            result = await _orchestrator.run(query, callback=callback, deep_analysis=deep)
            # Save session + performance to DB
            if _db_pool:
                try:
                    row = await _db_pool.fetchrow(
                        """INSERT INTO prism_sessions (user_request, red_response, blue_response, green_response, merged_result)
                           VALUES ($1, $2, $3, $4, $5) RETURNING id""",
                        query,
                        json.dumps(result["agents"].get("red", {})),
                        json.dumps(result["agents"].get("blue", {})),
                        json.dumps(result["agents"].get("green", {})),
                        json.dumps(result["merged"]),
                    )
                    if row:
                        await Orchestrator.log_performance(
                            _db_pool, row["id"], result["agents"], result["merged"]
                        )
                except Exception as e:
                    print(f"[Prism] Failed to save session: {e}")
        except Exception as e:
            await queue.put({"type": "error", "message": str(e)})
        finally:
            await queue.put(None)

    async def event_generator():
        task = asyncio.create_task(run_orchestrator())
        while True:
            event = await queue.get()
            if event is None:
                break
            yield {"event": event["type"], "data": json.dumps(event)}
        await task

    return EventSourceResponse(event_generator())


@app.post("/api/prism")
async def prism_post(request: PrismRequest):
    """Run Prism synchronously (for non-SSE clients)."""
    result = await _orchestrator.run(request.query)
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
        return [_add_momentum(_row_to_dict(r)) for r in rows]
    args = {}
    if status: args["status"] = status
    if priority: args["priority"] = priority
    result = await _mcp_clients["tasks"].call_tool("list_tasks", args)
    if result and result.content:
        tasks = json.loads(result.content[0].text)
        return [_add_momentum(t) for t in tasks]
    return []


@app.post("/api/tasks")
async def create_task(req: CreateTaskRequest):
    args = {"title": req.title, "description": req.description, "priority": req.priority, "tags": req.tags}
    if req.due_date:
        args["due_date"] = req.due_date
    result = await _mcp_clients["tasks"].call_tool("create_task", args)
    if result and result.content:
        return json.loads(result.content[0].text)
    return {"error": "Failed to create task"}


@app.patch("/api/tasks/{task_id}")
async def update_task(task_id: int, req: UpdateTaskRequest):
    args = {"task_id": task_id}
    if req.status: args["status"] = req.status
    if req.priority: args["priority"] = req.priority
    if req.title: args["title"] = req.title
    if req.description: args["description"] = req.description
    result = await _mcp_clients["tasks"].call_tool("update_task", args)
    if result and result.content:
        return json.loads(result.content[0].text)
    return {"error": "Failed to update task"}


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: int):
    result = await _mcp_clients["tasks"].call_tool("delete_task", {"task_id": task_id})
    if result and result.content:
        return json.loads(result.content[0].text)
    return {"error": "Failed to delete task"}


@app.get("/api/notes")
async def get_notes():
    if _db_pool:
        rows = await _db_pool.fetch("SELECT * FROM notes ORDER BY created_at DESC")
        return [_row_to_dict(r) for r in rows]
    result = await _mcp_clients["notes"].call_tool("list_notes", {})
    if result and result.content:
        return json.loads(result.content[0].text)
    return []


@app.post("/api/notes")
async def create_note(req: CreateNoteRequest):
    args = {"title": req.title, "content": req.content, "tags": req.tags}
    if req.linked_task_id:
        args["linked_task_id"] = req.linked_task_id
    result = await _mcp_clients["notes"].call_tool("create_note", args)
    if result and result.content:
        return json.loads(result.content[0].text)
    return {"error": "Failed to create note"}


@app.get("/api/history")
async def get_history(limit: int = 20):
    if not _db_pool:
        return []
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
    try:
        t = await _mcp_clients["tasks"].call_tool("list_tasks", {})
        n = await _mcp_clients["notes"].call_tool("list_notes", {})
        tc = len(json.loads(t.content[0].text)) if t and t.content else 0
        nc = len(json.loads(n.content[0].text)) if n and n.content else 0
        return {"tasks": tc, "notes": nc, "sessions": 0}
    except Exception:
        return {"tasks": 0, "notes": 0, "sessions": 0}


@app.get("/api/connections")
async def get_connections():
    """Show which data sources are live vs demo."""
    db_connected = _db_pool is not None
    return {
        "tasks": {"status": "live", "backend": "AlloyDB"} if db_connected else {"status": "demo", "backend": "In-memory mock"},
        "notes": {"status": "live", "backend": "AlloyDB"} if db_connected else {"status": "demo", "backend": "In-memory mock"},
        "calendar": {"status": "demo", "backend": "Mock data (Google Calendar ready)"},
        "email": {"status": "demo", "backend": "Mock data (Gmail ready)"},
    }


@app.get("/api/agent-stats")
async def get_agent_stats():
    """Agent performance analytics — which agent gets selected most."""
    if not _db_pool:
        return {"red": {"selected": 0, "total": 0}, "blue": {"selected": 0, "total": 0}, "green": {"selected": 0, "total": 0}}
    rows = await _db_pool.fetch(
        """SELECT agent_name,
                  COUNT(*) as total,
                  SUM(CASE WHEN was_selected THEN 1 ELSE 0 END) as selected,
                  AVG(execution_time_ms) as avg_time_ms
           FROM agent_performance
           GROUP BY agent_name"""
    )
    stats = {}
    for r in rows:
        stats[r["agent_name"]] = {
            "selected": int(r["selected"]),
            "total": int(r["total"]),
            "avg_time_ms": int(r["avg_time_ms"] or 0),
            "win_rate": round(int(r["selected"]) / max(int(r["total"]), 1) * 100, 1),
        }
    return stats


# ── Helpers ───────────────────────────────────────────────────

def _row_to_dict(row) -> dict:
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def _add_momentum(task: dict) -> dict:
    """Add momentum score — decays 5% per day of inactivity (Akasha physics model)."""
    if task.get("status") == "done":
        task["momentum"] = 100
        task["momentum_trend"] = "completed"
        return task
    updated = task.get("updated_at") or task.get("created_at", "")
    if not updated:
        task["momentum"] = 50
        task["momentum_trend"] = "unknown"
        return task
    try:
        last_touch = datetime.fromisoformat(updated.replace("Z", "+00:00")).replace(tzinfo=None)
        days_idle = (datetime.now() - last_touch).total_seconds() / 86400
        # Exponential decay: M(t) = 100 * e^(-0.05 * days)
        momentum = max(5, min(100, int(100 * math.exp(-0.05 * days_idle))))
        if days_idle < 1:
            trend = "active"
        elif days_idle < 3:
            trend = "stable"
        elif days_idle < 7:
            trend = "declining"
        else:
            trend = "stale"
        task["momentum"] = momentum
        task["momentum_trend"] = trend
    except Exception:
        task["momentum"] = 50
        task["momentum_trend"] = "unknown"
    return task


if __name__ == "__main__":
    import uvicorn
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
