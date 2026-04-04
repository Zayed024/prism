"""MCP Server for Email - mock implementation with realistic data."""

import asyncio
import json
import sys
from datetime import datetime

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("EmailMCP", log_level="ERROR")

SENT_LOG = []

INBOX = [
    {
        "id": "mail-1",
        "from": "sarah.chen@company.com",
        "to": "you@company.com",
        "subject": "Q2 Report - Finance Data Ready",
        "body": "Hi, the finance data for Q2 is ready in the shared drive. File: revenue_q2.xlsx. Let me know if you need anything else. The board wants to see the report by April 10th.",
        "date": "2026-04-02T16:30:00",
        "read": True,
        "important": True,
        "labels": ["work", "Q2-report"],
    },
    {
        "id": "mail-2",
        "from": "raj.patel@company.com",
        "to": "you@company.com",
        "subject": "PR #247 - Auth Module Review Needed",
        "body": "Hey, the auth module PR has been open for 3 days now. Could you take a look when you get a chance? There are some security considerations I'd like your input on, especially around the token refresh logic.",
        "date": "2026-04-02T14:15:00",
        "read": True,
        "important": True,
        "labels": ["work", "code-review"],
    },
    {
        "id": "mail-3",
        "from": "manager@company.com",
        "to": "you@company.com",
        "subject": "Re: Client Meeting Friday",
        "body": "The client confirmed Friday 2pm. Please make sure the demo environment is updated with the latest features. Also, can you prepare a 1-page summary of our Q2 progress? They specifically asked about the API improvements.",
        "date": "2026-04-03T08:45:00",
        "read": False,
        "important": True,
        "labels": ["work", "client", "urgent"],
    },
    {
        "id": "mail-4",
        "from": "devops-alerts@company.com",
        "to": "team@company.com",
        "subject": "[Alert] Cloud Build Pipeline Failed - Run #1847",
        "body": "Build failed at test stage. Error: Connection timeout to test database. This is the 3rd failure this week. Logs: https://console.cloud.google.com/cloud-build/builds/...",
        "date": "2026-04-03T07:20:00",
        "read": False,
        "important": False,
        "labels": ["devops", "alert"],
    },
    {
        "id": "mail-5",
        "from": "newsletter@techweekly.com",
        "to": "you@company.com",
        "subject": "This Week in AI: MCP Protocol Gains Traction",
        "body": "Model Context Protocol (MCP) is seeing rapid adoption across the industry. Major players including Anthropic, Google, and OpenAI are converging on tool-use standards...",
        "date": "2026-04-03T06:00:00",
        "read": False,
        "important": False,
        "labels": ["newsletter"],
    },
    {
        "id": "mail-6",
        "from": "hr@company.com",
        "to": "all@company.com",
        "subject": "Reminder: Submit Timesheet by Friday",
        "body": "Please submit your timesheets for the week ending April 4th by end of day Friday. Late submissions will be processed in the next pay cycle.",
        "date": "2026-04-02T09:00:00",
        "read": True,
        "important": False,
        "labels": ["hr", "admin"],
    },
    {
        "id": "mail-7",
        "from": "alex.kim@company.com",
        "to": "you@company.com",
        "subject": "Blog Post Draft - Can You Review?",
        "body": "Hey! I wrote a draft for the engineering blog about our microservices migration. Would love your feedback, especially on the architecture diagrams. No rush, but before next Wednesday would be great. Link: docs.google.com/...",
        "date": "2026-04-01T15:30:00",
        "read": True,
        "important": False,
        "labels": ["work", "blog", "review"],
    },
]


@mcp.tool(name="list_emails", description="List emails from inbox, optionally filtered by read status or importance.")
async def list_emails(
    unread_only: bool = Field(default=False, description="Only show unread emails"),
    important_only: bool = Field(default=False, description="Only show important emails"),
    limit: int = Field(default=10, description="Max number of emails to return"),
) -> str:
    filtered = INBOX
    if unread_only:
        filtered = [e for e in filtered if not e["read"]]
    if important_only:
        filtered = [e for e in filtered if e["important"]]
    filtered = sorted(filtered, key=lambda e: e["date"], reverse=True)[:limit]
    return json.dumps(filtered)


@mcp.tool(name="search_emails", description="Search emails by keyword in subject or body.")
async def search_emails(
    query: str = Field(description="Search keyword"),
) -> str:
    q = query.lower()
    matching = [
        e for e in INBOX
        if q in e["subject"].lower() or q in e["body"].lower() or q in e["from"].lower()
    ]
    matching.sort(key=lambda e: e["date"], reverse=True)
    return json.dumps(matching)


@mcp.tool(name="send_email", description="Send an email (compose and send).")
async def send_email(
    to: str = Field(description="Recipient email address"),
    subject: str = Field(description="Email subject line"),
    body: str = Field(description="Email body text"),
) -> str:
    email = {
        "id": f"sent-{len(SENT_LOG) + 1}",
        "to": to,
        "subject": subject,
        "body": body,
        "sent_at": datetime.now().isoformat(),
    }
    SENT_LOG.append(email)
    return json.dumps({"status": "sent", "email": email})


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    mcp.run(transport="stdio")
