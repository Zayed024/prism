"""MCP Server for Notes Management - PostgreSQL/AlloyDB with in-memory fallback."""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# ── In-memory store (fallback when no DB) ──
MOCK_NOTES = [
    {"id": 1, "title": "Q2 Report Data Sources", "content": "## Data Sources\n- Finance: revenue_q2.xlsx (shared drive)\n- Engineering: velocity metrics from Jira\n- Marketing: campaign performance from Analytics\n- Sales: pipeline report from Salesforce\n\n## Key Deadlines\n- Draft due: Apr 8\n- Review: Apr 9\n- Final submission: Apr 10", "tags": ["report", "Q2", "data"], "linked_task_id": 1, "created_by": "user", "created_at": "2026-04-01T09:00:00"},
    {"id": 2, "title": "Client Meeting Agenda", "content": "## Friday Client Meeting\n- Project status update (15 min)\n- Demo new features (20 min)\n- Roadmap discussion (15 min)\n- Q&A (10 min)\n\n## Prep needed\n- Update demo environment\n- Prepare backup slides for technical questions", "tags": ["client", "meeting", "agenda"], "linked_task_id": 3, "created_by": "user", "created_at": "2026-04-01T09:00:00"},
    {"id": 3, "title": "Auth Module Notes", "content": "## Current Issues with PR #247\n- Token refresh logic needs edge case handling\n- Missing rate limiting on login endpoint\n- Session invalidation not tested\n\n## Suggested Changes\n- Add retry with exponential backoff\n- Implement sliding window rate limiter\n- Add integration tests for session lifecycle", "tags": ["code-review", "auth", "security"], "linked_task_id": 5, "created_by": "user", "created_at": "2026-04-01T09:00:00"},
    {"id": 4, "title": "MCP Blog Outline", "content": "## Title: Building Multi-Agent Systems with MCP\n\n1. What is Model Context Protocol?\n2. Why MCP matters for AI applications\n3. Building your first MCP server (code walkthrough)\n4. Connecting multiple MCP servers to an orchestrator\n5. Real-world patterns: tool routing, error handling\n6. Performance considerations\n7. Conclusion + resources", "tags": ["blog", "MCP", "draft"], "linked_task_id": 7, "created_by": "user", "created_at": "2026-04-01T09:00:00"},
    {"id": 5, "title": "Weekly Standup Notes - Mar 30", "content": "## What I did\n- Completed API endpoint refactoring\n- Fixed 3 critical bugs in notification service\n- Started Q2 report data collection\n\n## What I plan to do\n- Finish Q2 report draft\n- Review auth module PR\n- Client presentation prep\n\n## Blockers\n- Waiting on finance data for Q2 report\n- CI/CD pipeline intermittent failures", "tags": ["standup", "weekly", "status"], "linked_task_id": None, "created_by": "user", "created_at": "2026-04-01T09:00:00"},
]
_next_id = 6
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
            print("[NotesMCP] Connected to database", file=sys.stderr)
        except Exception as e:
            print(f"[NotesMCP] DB unavailable, using mock data: {e}", file=sys.stderr)
    else:
        print("[NotesMCP] No DATABASE_URL, using mock data", file=sys.stderr)
    try:
        yield {"pool": _pool}
    finally:
        if _pool:
            await _pool.close()


mcp = FastMCP("NotesMCP", log_level="ERROR", lifespan=lifespan)


@mcp.tool(name="create_note", description="Create a new note with title, content, tags, and optional task link.")
async def create_note(
    title: str = Field(description="Note title"),
    content: str = Field(default="", description="Note content (supports markdown)"),
    tags: str = Field(default="", description="Comma-separated tags"),
    linked_task_id: int = Field(default=0, description="Task ID to link this note to, or 0 for none"),
    ctx=None,
) -> str:
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    link_id = linked_task_id if linked_task_id > 0 else None

    if _use_db:
        pool = ctx.request_context.lifespan_context["pool"]
        row = await pool.fetchrow(
            """INSERT INTO notes (title, content, tags, linked_task_id, created_by)
               VALUES ($1, $2, $3, $4, 'agent') RETURNING *""",
            title, content, tag_list, link_id,
        )
        return json.dumps(_row_to_dict(row), default=str)

    global _next_id
    note = {"id": _next_id, "title": title, "content": content, "tags": tag_list,
            "linked_task_id": link_id, "created_by": "agent", "created_at": datetime.now().isoformat()}
    _next_id += 1
    MOCK_NOTES.append(note)
    return json.dumps(note)


@mcp.tool(name="list_notes", description="List notes, optionally filtered by linked task.")
async def list_notes(
    linked_task_id: int = Field(default=0, description="Filter by linked task ID, or 0 for all"),
    limit: int = Field(default=20, description="Max number of notes to return"),
    ctx=None,
) -> str:
    if _use_db:
        pool = ctx.request_context.lifespan_context["pool"]
        if linked_task_id > 0:
            rows = await pool.fetch(
                "SELECT * FROM notes WHERE linked_task_id = $1 ORDER BY created_at DESC LIMIT $2",
                linked_task_id, limit,
            )
        else:
            rows = await pool.fetch("SELECT * FROM notes ORDER BY created_at DESC LIMIT $1", limit)
        return json.dumps([_row_to_dict(r) for r in rows], default=str)

    filtered = MOCK_NOTES
    if linked_task_id > 0:
        filtered = [n for n in filtered if n.get("linked_task_id") == linked_task_id]
    return json.dumps(filtered[:limit])


@mcp.tool(name="get_note", description="Get a single note by its ID.")
async def get_note(
    note_id: int = Field(description="The note ID"),
    ctx=None,
) -> str:
    if _use_db:
        pool = ctx.request_context.lifespan_context["pool"]
        row = await pool.fetchrow("SELECT * FROM notes WHERE id = $1", note_id)
        if not row: return json.dumps({"error": f"Note {note_id} not found"})
        return json.dumps(_row_to_dict(row), default=str)

    note = next((n for n in MOCK_NOTES if n["id"] == note_id), None)
    if not note: return json.dumps({"error": f"Note {note_id} not found"})
    return json.dumps(note)


@mcp.tool(name="search_notes", description="Search notes by keyword in title or content.")
async def search_notes(
    query: str = Field(description="Search keyword"),
    ctx=None,
) -> str:
    if _use_db:
        pool = ctx.request_context.lifespan_context["pool"]
        rows = await pool.fetch(
            "SELECT * FROM notes WHERE title ILIKE $1 OR content ILIKE $1 ORDER BY created_at DESC LIMIT 20",
            f"%{query}%",
        )
        return json.dumps([_row_to_dict(r) for r in rows], default=str)

    q = query.lower()
    matched = [n for n in MOCK_NOTES if q in n["title"].lower() or q in n.get("content", "").lower()]
    return json.dumps(matched[:20])


@mcp.tool(name="link_note_to_task", description="Link an existing note to a task.")
async def link_note_to_task(
    note_id: int = Field(description="The note ID"),
    task_id: int = Field(description="The task ID to link to"),
    ctx=None,
) -> str:
    if _use_db:
        pool = ctx.request_context.lifespan_context["pool"]
        row = await pool.fetchrow(
            "UPDATE notes SET linked_task_id = $1 WHERE id = $2 RETURNING *",
            task_id, note_id,
        )
        if not row: return json.dumps({"error": f"Note {note_id} not found"})
        return json.dumps(_row_to_dict(row), default=str)

    note = next((n for n in MOCK_NOTES if n["id"] == note_id), None)
    if not note: return json.dumps({"error": f"Note {note_id} not found"})
    note["linked_task_id"] = task_id
    return json.dumps(note)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    mcp.run(transport="stdio")
