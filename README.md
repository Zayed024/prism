# Prism — Multi-Agent Productivity Assistant

> Three AI agents. One request. The best of each.

Prism splits every natural language request across three AI agents with different cognitive styles — **Red (Speed)**, **Blue (Depth)**, and **Green (Creative)** — each interacting with your tools via MCP. An orchestrator merges the best parts into a single result.

**Live Demo**: _[deployed URL]_

---

## How It Works

```
User: "Plan my week — I need to finish the Q2 report and prepare for the client meeting"

        ┌──────────────────────────────────────┐
        │         Orchestrator (Gemini)         │
        └────────┬──────────┬──────────┬───────┘
                 │          │          │
          ┌──────▼──┐ ┌────▼────┐ ┌───▼──────┐
          │ 🔴 Red  │ │ 🔵 Blue │ │ 🟢 Green │
          │ Speed   │ │ Depth   │ │ Creative │
          └────┬────┘ └────┬────┘ └────┬─────┘
               │           │           │
          Creates 3    Searches     Finds your
          tasks,       existing     dentist appt
          blocks       notes,       conflicts with
          calendar     builds       prep time,
          time         full plan    reschedules it
               │           │           │
               └───────────┴───────────┘
                           │
                    ┌──────▼──────┐
                    │   MERGED    │
                    │   RESULT    │
                    └─────────────┘
```

Each agent independently calls tools via **MCP (Model Context Protocol)** — creating tasks, searching notes, checking calendars, sending emails — then the orchestrator synthesizes the best contributions from each.

---

## Architecture

| Component | Technology |
|-----------|-----------|
| **AI Model** | Gemini 3 Flash (function calling) |
| **Backend** | FastAPI + Uvicorn |
| **Database** | AlloyDB (PostgreSQL) |
| **Tool Protocol** | MCP (Model Context Protocol) |
| **Frontend** | HTML + Tailwind CSS + SSE |
| **Deployment** | Google Cloud Run |

### MCP Servers (18 tools)

| Server | Tools | Backend |
|--------|-------|---------|
| **Tasks MCP** | create, list, get, update, delete, search | AlloyDB |
| **Notes MCP** | create, list, get, search, link_to_task | AlloyDB |
| **Calendar MCP** | list_events, create, find_free_slots, delete | Google Calendar |
| **Email MCP** | list, search, send | Gmail |

### Agent Personalities

| Agent | Style | Max Iterations | Approach |
|-------|-------|---------------|----------|
| 🔴 **Red** | Speed | 3 | Minimum viable action, act first |
| 🔵 **Blue** | Depth | 5 | Research first, structured plan |
| 🟢 **Green** | Creative | 5 | Lateral thinking, find hidden connections |

---

## Project Structure

```
prism/
├── server.py                  # FastAPI + SSE streaming
├── orchestrator.py            # Parallel agents + merge
├── models.py                  # Pydantic models
├── schema.sql                 # DB schema + seed data
├── agents/
│   ├── base.py               # Agentic tool loop (Gemini function calling)
│   ├── red.py                # Speed agent
│   ├── blue.py               # Depth agent
│   └── green.py              # Creative agent
├── mcp_servers/
│   ├── tasks_server.py       # Task CRUD via MCP
│   ├── notes_server.py       # Notes CRUD via MCP
│   ├── calendar_server.py    # Calendar operations via MCP
│   └── email_server.py       # Email operations via MCP
├── core/
│   ├── gemini.py             # Gemini wrapper + function calling
│   ├── mcp_client.py         # MCP client (stdio transport)
│   ├── tool_manager.py       # Tool discovery + routing
│   └── database.py           # asyncpg connection pool
├── templates/
│   └── index.html            # Frontend SPA
├── Dockerfile
├── deploy.sh
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.11+
- Gemini API key ([get one](https://aistudio.google.com/apikey))
- AlloyDB instance (or run locally with mock data)

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# Run (uses mock data without DATABASE_URL)
python -m uvicorn server:app --host 127.0.0.1 --port 5000 --reload

# Open http://127.0.0.1:5000
```

### Database Setup (AlloyDB)

```bash
# Connect to AlloyDB and run schema
psql -h <ALLOYDB_IP> -U postgres -f schema.sql
```

### Deploy to Cloud Run

```bash
# Edit deploy.sh with your credentials
bash deploy.sh
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/prism?query=...` | Run Prism (SSE streaming) |
| `POST` | `/api/prism` | Run Prism (JSON response) |
| `GET` | `/api/tasks` | List tasks |
| `GET` | `/api/notes` | List notes |
| `GET` | `/api/history` | Session history |
| `GET` | `/api/stats` | Dashboard stats |

### SSE Events

```
agent_start  → Agent begins working
tool_call    → Agent calls an MCP tool
tool_result  → Tool returns result
agent_done   → Agent finished (response + timing)
merge_start  → Orchestrator begins merging
merge_done   → Final merged result
```

---

## Example Queries

- *"Plan my week — I need to finish the Q2 report and prepare for the client meeting on Friday"*
- *"Check my unread emails and create tasks for anything urgent"*
- *"What tasks are overdue? Summarize and suggest next steps"*
- *"Find free time tomorrow and schedule a 2-hour deep work block"*

---

## Tech Stack

Built for **Gen AI Academy APAC Edition**

- **Google Gemini 3 Flash** — AI reasoning + function calling
- **MCP (Model Context Protocol)** — Standardized tool integration
- **AlloyDB** — Structured data persistence
- **FastAPI** — Async API server
- **Google Cloud Run** — Serverless deployment
