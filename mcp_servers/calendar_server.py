"""MCP Server for Calendar Management - mock implementation with realistic data."""

import asyncio
import json
import sys
from datetime import datetime, timedelta

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("CalendarMCP", log_level="ERROR")

# Mock calendar events (realistic data for demo)
EVENTS = [
    {"id": "evt-1", "title": "Daily Standup", "start": "2026-04-03T09:00:00", "end": "2026-04-03T09:30:00", "attendees": ["team"], "recurring": True, "description": "Daily sync with engineering team"},
    {"id": "evt-2", "title": "Sprint Planning", "start": "2026-04-03T14:00:00", "end": "2026-04-03T15:30:00", "attendees": ["team", "PM"], "recurring": False, "description": "Plan sprint 24 work items"},
    {"id": "evt-3", "title": "1:1 with Manager", "start": "2026-04-04T10:00:00", "end": "2026-04-04T10:30:00", "attendees": ["manager"], "recurring": True, "description": "Weekly sync"},
    {"id": "evt-4", "title": "Client Presentation", "start": "2026-04-04T14:00:00", "end": "2026-04-04T15:00:00", "attendees": ["client-team", "sales"], "recurring": False, "description": "Q2 progress review with client"},
    {"id": "evt-5", "title": "Team Lunch", "start": "2026-04-04T12:00:00", "end": "2026-04-04T13:00:00", "attendees": ["team"], "recurring": False, "description": "Monthly team lunch at Italian place"},
    {"id": "evt-6", "title": "Code Review Session", "start": "2026-04-05T11:00:00", "end": "2026-04-05T12:00:00", "attendees": ["backend-team"], "recurring": True, "description": "Weekly code review"},
    {"id": "evt-7", "title": "Design Review", "start": "2026-04-07T15:00:00", "end": "2026-04-07T16:00:00", "attendees": ["design", "frontend"], "recurring": False, "description": "Review new dashboard mockups"},
    {"id": "evt-8", "title": "All Hands", "start": "2026-04-07T11:00:00", "end": "2026-04-07T12:00:00", "attendees": ["company"], "recurring": True, "description": "Monthly all-hands meeting"},
    {"id": "evt-9", "title": "Dentist Appointment", "start": "2026-04-08T09:00:00", "end": "2026-04-08T10:00:00", "attendees": [], "recurring": False, "description": "Regular cleaning"},
    {"id": "evt-10", "title": "Deep Work Block", "start": "2026-04-03T10:00:00", "end": "2026-04-03T12:00:00", "attendees": [], "recurring": True, "description": "Protected focus time - no meetings"},
]

_next_id = 11


@mcp.tool(name="list_events", description="List calendar events for a specific date or upcoming days.")
async def list_events(
    date: str = Field(default="", description="Specific date (YYYY-MM-DD) or empty for today"),
    days_ahead: int = Field(default=1, description="Number of days to look ahead (1-14)"),
) -> str:
    target = datetime.fromisoformat(date) if date else datetime.now()
    target_start = target.replace(hour=0, minute=0, second=0)
    target_end = target_start + timedelta(days=max(1, min(days_ahead, 14)))

    matching = []
    for evt in EVENTS:
        evt_start = datetime.fromisoformat(evt["start"])
        if target_start <= evt_start < target_end:
            matching.append(evt)
    matching.sort(key=lambda e: e["start"])
    return json.dumps(matching)


@mcp.tool(name="create_event", description="Create a new calendar event.")
async def create_event(
    title: str = Field(description="Event title"),
    start: str = Field(description="Start time in ISO format (YYYY-MM-DDTHH:MM:SS)"),
    end: str = Field(description="End time in ISO format (YYYY-MM-DDTHH:MM:SS)"),
    description: str = Field(default="", description="Event description"),
    attendees: str = Field(default="", description="Comma-separated attendee names"),
) -> str:
    global _next_id
    evt = {
        "id": f"evt-{_next_id}",
        "title": title,
        "start": start,
        "end": end,
        "attendees": [a.strip() for a in attendees.split(",") if a.strip()] if attendees else [],
        "recurring": False,
        "description": description,
    }
    _next_id += 1
    EVENTS.append(evt)
    return json.dumps({"created": evt})


@mcp.tool(name="find_free_slots", description="Find available time slots on a given date.")
async def find_free_slots(
    date: str = Field(description="Date to check (YYYY-MM-DD)"),
    duration_minutes: int = Field(default=60, description="Desired slot duration in minutes"),
) -> str:
    target = datetime.fromisoformat(date)
    day_start = target.replace(hour=9, minute=0, second=0)
    day_end = target.replace(hour=18, minute=0, second=0)

    busy = []
    for evt in EVENTS:
        evt_start = datetime.fromisoformat(evt["start"])
        evt_end = datetime.fromisoformat(evt["end"])
        if evt_start.date() == target.date():
            busy.append((evt_start, evt_end))
    busy.sort(key=lambda x: x[0])

    free = []
    current = day_start
    for start, end in busy:
        if (start - current).total_seconds() >= duration_minutes * 60:
            free.append({"start": current.isoformat(), "end": start.isoformat()})
        current = max(current, end)
    if (day_end - current).total_seconds() >= duration_minutes * 60:
        free.append({"start": current.isoformat(), "end": day_end.isoformat()})

    return json.dumps({"date": date, "duration_minutes": duration_minutes, "free_slots": free})


@mcp.tool(name="delete_event", description="DESTRUCTIVE: Delete a calendar event. Only call if user EXPLICITLY says 'delete' or 'cancel'. Requires confirm=True as a safety check.")
async def delete_event(
    event_id: str = Field(description="The event ID to delete"),
    confirm: bool = Field(default=False, description="Must be True to actually delete. Defaults to False as a safety check."),
) -> str:
    if not confirm:
        return json.dumps({"error": "Delete blocked: confirm=True required. Did the user explicitly ask to cancel this event?"})
    global EVENTS
    before = len(EVENTS)
    EVENTS = [e for e in EVENTS if e["id"] != event_id]
    if len(EVENTS) < before:
        return json.dumps({"deleted": event_id})
    return json.dumps({"error": f"Event {event_id} not found"})


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    mcp.run(transport="stdio")
