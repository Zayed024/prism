"""MCP Server for Task Management - PostgreSQL/AlloyDB with in-memory fallback."""

import asyncio
import json
import math
import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from pydantic import Field

STOPWORDS = {"a","an","the","and","or","but","is","are","was","were","be","been","being",
             "have","has","had","do","does","did","will","would","could","should","may",
             "might","can","this","that","these","those","i","me","my","mine","you","your",
             "we","our","they","their","what","when","where","why","how","which","who",
             "for","of","in","on","at","to","from","with","by","about","as","into","than",
             "then","so","if","some","any","all","each","every","no","not","need","needs",
             "want","wants","get","got","make","made","tell","show","find","help","please",
             "show","list","plan","prepare","check","review","summarize"}


def _extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a natural language query."""
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]+\b", query.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def _score_text(text: str, keywords: list[str]) -> float:
    """Score how well text matches keywords (0-1)."""
    if not keywords:
        return 0
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches / len(keywords)


def _enrich_with_momentum(task: dict) -> dict:
    """Add momentum score and human-readable trend to a task.
    Momentum decays exponentially with inactivity (Akasha physics model).
    Agents see this and can prioritize stale items."""
    if task.get("status") == "done":
        task["momentum"] = 100
        task["momentum_label"] = "completed"
        return task
    updated = task.get("updated_at") or task.get("created_at", "")
    if not updated:
        task["momentum"] = 50
        task["momentum_label"] = "unknown"
        return task
    try:
        last = datetime.fromisoformat(updated.replace("Z", "+00:00")).replace(tzinfo=None)
        days_idle = (datetime.now() - last).total_seconds() / 86400
        momentum = max(5, min(100, int(100 * math.exp(-0.05 * days_idle))))
        if days_idle < 1:
            label = f"active ({momentum}% momentum)"
        elif days_idle < 3:
            label = f"stable ({momentum}% momentum, {int(days_idle)}d idle)"
        elif days_idle < 7:
            label = f"DECLINING ({momentum}% momentum, {int(days_idle)}d untouched - needs attention)"
        else:
            label = f"STALE ({momentum}% momentum, {int(days_idle)}d untouched - consider archiving or reviving)"
        task["momentum"] = momentum
        task["momentum_label"] = label
    except Exception:
        task["momentum"] = 50
        task["momentum_label"] = "unknown"
    return task

# ── In-memory store (fallback when no DB) ──
MOCK_TASKS = [
    {"id": 1, "title": "Finish Q2 quarterly report", "description": "Compile data from all departments and create executive summary", "status": "in_progress", "priority": "high", "due_date": "2026-04-10T17:00:00", "tags": ["work", "report", "Q2"], "created_by": "user", "created_at": "2026-04-01T09:00:00", "updated_at": "2026-04-01T09:00:00"},
    {"id": 2, "title": "Review API documentation", "description": "Go through the updated API docs and flag inconsistencies", "status": "todo", "priority": "medium", "due_date": "2026-04-07T12:00:00", "tags": ["work", "docs", "api"], "created_by": "user", "created_at": "2026-04-01T09:00:00", "updated_at": "2026-04-01T09:00:00"},
    {"id": 3, "title": "Prepare client presentation", "description": "Create slide deck for Friday client meeting", "status": "todo", "priority": "high", "due_date": "2026-04-04T09:00:00", "tags": ["work", "client", "presentation"], "created_by": "user", "created_at": "2026-04-01T09:00:00", "updated_at": "2026-04-01T09:00:00"},
    {"id": 4, "title": "Grocery shopping", "description": "Buy vegetables, milk, eggs, and bread", "status": "todo", "priority": "low", "due_date": "2026-04-05T18:00:00", "tags": ["personal", "errands"], "created_by": "user", "created_at": "2026-04-01T09:00:00", "updated_at": "2026-04-01T09:00:00"},
    {"id": 5, "title": "Code review: auth module", "description": "Review PR #247 for the new authentication flow", "status": "todo", "priority": "high", "due_date": "2026-04-04T15:00:00", "tags": ["work", "code-review", "auth"], "created_by": "user", "created_at": "2026-04-01T09:00:00", "updated_at": "2026-04-01T09:00:00"},
    {"id": 6, "title": "Schedule dentist appointment", "description": "Call Dr. Patel office for a cleaning", "status": "todo", "priority": "low", "due_date": None, "tags": ["personal", "health"], "created_by": "user", "created_at": "2026-04-01T09:00:00", "updated_at": "2026-04-01T09:00:00"},
    {"id": 7, "title": "Write blog post on MCP", "description": "Draft a technical blog about Model Context Protocol integration patterns", "status": "in_progress", "priority": "medium", "due_date": "2026-04-12T12:00:00", "tags": ["work", "writing", "tech"], "created_by": "user", "created_at": "2026-04-01T09:00:00", "updated_at": "2026-04-01T09:00:00"},
    {"id": 8, "title": "Update team onboarding guide", "description": "Add new sections about CI/CD pipeline and testing", "status": "todo", "priority": "medium", "due_date": "2026-04-15T17:00:00", "tags": ["work", "docs", "onboarding"], "created_by": "user", "created_at": "2026-04-01T09:00:00", "updated_at": "2026-04-01T09:00:00"},
    {"id": 9, "title": "Plan weekend trip", "description": "Research destinations and book accommodation for Apr 18-20", "status": "todo", "priority": "low", "due_date": "2026-04-11T20:00:00", "tags": ["personal", "travel"], "created_by": "user", "created_at": "2026-04-01T09:00:00", "updated_at": "2026-04-01T09:00:00"},
    {"id": 10, "title": "Fix deployment pipeline bug", "description": "Cloud Build fails intermittently on the test stage", "status": "in_progress", "priority": "high", "due_date": "2026-04-05T12:00:00", "tags": ["work", "devops", "bug"], "created_by": "user", "created_at": "2026-04-01T09:00:00", "updated_at": "2026-04-01T09:00:00"},
]
_next_id = 11
_use_db = False
_pool = None


def _row_to_dict(row):
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


@asynccontextmanager
async def lifespan(server: FastMCP):
    global _use_db, _pool
    database_url = os.environ.get("DATABASE_URL", "")
    if database_url:
        try:
            import asyncpg
            _pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
            _use_db = True
            print("[TasksMCP] Connected to database", file=sys.stderr)
        except Exception as e:
            print(f"[TasksMCP] DB unavailable, using mock data: {e}", file=sys.stderr)
    else:
        print("[TasksMCP] No DATABASE_URL, using mock data", file=sys.stderr)
    try:
        yield {"pool": _pool}
    finally:
        if _pool:
            await _pool.close()


mcp = FastMCP("TasksMCP", log_level="ERROR", lifespan=lifespan)


@mcp.tool(name="create_task", description="Create a new task with title, description, priority, due date, and tags.")
async def create_task(
    title: str = Field(description="Task title"),
    description: str = Field(default="", description="Task description"),
    priority: str = Field(default="medium", description="Priority: high, medium, or low"),
    due_date: str = Field(default="", description="Due date in ISO format (YYYY-MM-DDTHH:MM:SS), or empty"),
    tags: str = Field(default="", description="Comma-separated tags"),
    ctx=None,
) -> str:
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    due = due_date if due_date else None

    if _use_db:
        pool = _pool
        due_dt = datetime.fromisoformat(due_date) if due_date else None
        row = await pool.fetchrow(
            """INSERT INTO tasks (title, description, priority, due_date, tags, created_by)
               VALUES ($1, $2, $3, $4, $5, 'agent') RETURNING *""",
            title, description, priority, due_dt, tag_list,
        )
        return json.dumps(_row_to_dict(row), default=str)

    global _next_id
    task = {"id": _next_id, "title": title, "description": description, "status": "todo",
            "priority": priority, "due_date": due, "tags": tag_list, "created_by": "agent",
            "created_at": datetime.now().isoformat(), "updated_at": datetime.now().isoformat()}
    _next_id += 1
    MOCK_TASKS.append(task)
    return json.dumps(task)


@mcp.tool(name="list_tasks", description="List tasks with momentum scores. Each task includes 'momentum' (0-100, lower=stale) and 'momentum_label' (e.g. 'STALE - 14d untouched'). Use these to identify tasks that need attention or revival.")
async def list_tasks(
    status: str = Field(default="", description="Filter by status: todo, in_progress, done. Empty for all."),
    priority: str = Field(default="", description="Filter by priority: high, medium, low. Empty for all."),
    limit: int = Field(default=20, description="Max number of tasks to return"),
    ctx=None,
) -> str:
    if _use_db:
        pool = _pool
        query = "SELECT * FROM tasks WHERE 1=1"
        params = []
        idx = 1
        if status:
            query += f" AND status = ${idx}"; params.append(status); idx += 1
        if priority:
            query += f" AND priority = ${idx}"; params.append(priority); idx += 1
        query += f" ORDER BY created_at DESC LIMIT ${idx}"; params.append(limit)
        rows = await pool.fetch(query, *params)
        return json.dumps([_enrich_with_momentum(_row_to_dict(r)) for r in rows], default=str)

    filtered = MOCK_TASKS
    if status:
        filtered = [t for t in filtered if t["status"] == status]
    if priority:
        filtered = [t for t in filtered if t["priority"] == priority]
    return json.dumps([_enrich_with_momentum(dict(t)) for t in filtered[:limit]])


@mcp.tool(name="get_task", description="Get a single task by its ID.")
async def get_task(
    task_id: int = Field(description="The task ID"),
    ctx=None,
) -> str:
    if _use_db:
        pool = _pool
        row = await pool.fetchrow("SELECT * FROM tasks WHERE id = $1", task_id)
        if not row: return json.dumps({"error": f"Task {task_id} not found"})
        return json.dumps(_enrich_with_momentum(_row_to_dict(row)), default=str)

    task = next((t for t in MOCK_TASKS if t["id"] == task_id), None)
    if not task: return json.dumps({"error": f"Task {task_id} not found"})
    return json.dumps(_enrich_with_momentum(dict(task)))


@mcp.tool(name="update_task", description="Update a task's status, priority, title, or description.")
async def update_task(
    task_id: int = Field(description="The task ID to update"),
    status: str = Field(default="", description="New status: todo, in_progress, done"),
    priority: str = Field(default="", description="New priority: high, medium, low"),
    title: str = Field(default="", description="New title"),
    description: str = Field(default="", description="New description"),
    ctx=None,
) -> str:
    if _use_db:
        pool = _pool
        sets, params, idx = [], [], 1
        if status: sets.append(f"status = ${idx}"); params.append(status); idx += 1
        if priority: sets.append(f"priority = ${idx}"); params.append(priority); idx += 1
        if title: sets.append(f"title = ${idx}"); params.append(title); idx += 1
        if description: sets.append(f"description = ${idx}"); params.append(description); idx += 1
        if not sets: return json.dumps({"error": "No fields to update"})
        sets.append("updated_at = NOW()")
        query = f"UPDATE tasks SET {', '.join(sets)} WHERE id = ${idx} RETURNING *"; params.append(task_id)
        row = await pool.fetchrow(query, *params)
        if not row: return json.dumps({"error": f"Task {task_id} not found"})
        return json.dumps(_row_to_dict(row), default=str)

    task = next((t for t in MOCK_TASKS if t["id"] == task_id), None)
    if not task: return json.dumps({"error": f"Task {task_id} not found"})
    if status: task["status"] = status
    if priority: task["priority"] = priority
    if title: task["title"] = title
    if description: task["description"] = description
    task["updated_at"] = datetime.now().isoformat()
    return json.dumps(_enrich_with_momentum(dict(task)))


@mcp.tool(name="delete_task", description="DESTRUCTIVE: Permanently delete a task by ID. Only call this if the user EXPLICITLY says 'delete'. To mark a task done, use update_task with status='done' instead. Requires confirm=True parameter as a safety check.")
async def delete_task(
    task_id: int = Field(description="The task ID to delete"),
    confirm: bool = Field(default=False, description="Must be True to actually delete. Defaults to False as a safety check."),
    ctx=None,
) -> str:
    if not confirm:
        return json.dumps({"error": "Delete blocked: confirm=True required. Did the user explicitly ask to delete? If they want to mark as done, use update_task with status='done' instead."})
    if _use_db:
        pool = _pool
        row = await pool.fetchrow("DELETE FROM tasks WHERE id = $1 RETURNING id, title", task_id)
        if not row: return json.dumps({"error": f"Task {task_id} not found"})
        return json.dumps({"deleted": _row_to_dict(row)}, default=str)

    global MOCK_TASKS
    before = len(MOCK_TASKS)
    MOCK_TASKS = [t for t in MOCK_TASKS if t["id"] != task_id]
    if len(MOCK_TASKS) < before: return json.dumps({"deleted": task_id})
    return json.dumps({"error": f"Task {task_id} not found"})


@mcp.tool(name="search_tasks", description="Search tasks by keyword in title or description.")
async def search_tasks(
    query: str = Field(description="Search keyword"),
    ctx=None,
) -> str:
    if _use_db:
        pool = _pool
        rows = await pool.fetch(
            "SELECT * FROM tasks WHERE title ILIKE $1 OR description ILIKE $1 ORDER BY created_at DESC LIMIT 20",
            f"%{query}%",
        )
        return json.dumps([_enrich_with_momentum(_row_to_dict(r)) for r in rows], default=str)

    q = query.lower()
    matched = [t for t in MOCK_TASKS if q in t["title"].lower() or q in t.get("description", "").lower()]
    return json.dumps([_enrich_with_momentum(dict(t)) for t in matched[:20]])


@mcp.tool(name="semantic_search_tasks", description="Hybrid semantic search for tasks. Tries AlloyDB AI vector search first (when embeddings available), falls back to smart keyword scoring with stopword filtering. Always returns relevant matches even for natural language queries.")
async def semantic_search_tasks(
    query: str = Field(description="Natural language search query"),
    limit: int = Field(default=5, description="Max results to return"),
    ctx=None,
) -> str:
    keywords = _extract_keywords(query)

    # Phase 1: Try AlloyDB AI vector search
    if _use_db:
        pool = _pool
        try:
            rows = await pool.fetch("SELECT * FROM semantic_search_tasks($1, $2)", query, limit)
            if rows:
                return json.dumps([_enrich_with_momentum(_row_to_dict(r)) for r in rows], default=str)
        except Exception:
            pass  # Embeddings not set up, fall through to keyword search

        # Phase 2: Smart keyword fallback on database
        if keywords:
            ilike_clauses = " OR ".join([f"title ILIKE ${i+1} OR description ILIKE ${i+1} OR ${i+1} = ANY(tags)" for i in range(len(keywords))])
            params = [f"%{kw}%" for kw in keywords]
            params.append(limit)
            query_sql = f"SELECT * FROM tasks WHERE {ilike_clauses} ORDER BY updated_at DESC LIMIT ${len(keywords)+1}"
            try:
                rows = await pool.fetch(query_sql, *params)
                # Re-score and sort by relevance
                scored = []
                for r in rows:
                    d = _row_to_dict(r)
                    text = f"{d.get('title','')} {d.get('description','')} {' '.join(d.get('tags',[]))}"
                    score = _score_text(text, keywords)
                    scored.append({**d, "similarity": round(score, 2)})
                scored.sort(key=lambda x: x["similarity"], reverse=True)
                return json.dumps([_enrich_with_momentum(t) for t in scored[:limit]], default=str)
            except Exception:
                return json.dumps([])

    # Mock: smart keyword scoring
    if not keywords:
        return json.dumps([])
    scored = []
    for t in MOCK_TASKS:
        text = f"{t['title']} {t.get('description', '')} {' '.join(t.get('tags', []))}"
        score = _score_text(text, keywords)
        if score > 0:
            scored.append({**t, "similarity": round(score, 2)})
    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return json.dumps([_enrich_with_momentum(dict(t)) for t in scored[:limit]])


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    mcp.run(transport="stdio")
