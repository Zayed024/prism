"""Orchestrator - ADK-powered parallel agent execution with smart context,
agent negotiation, and merge. The core of the Prism multi-agent system."""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta

from google.adk.agents import LlmAgent
from google.adk.sessions import InMemorySessionService
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioConnectionParams
from mcp import StdioServerParameters

from agents.base import AgentResult, run_adk_agent
from agents.red import create_red_agent
from agents.blue import create_blue_agent
from agents.green import create_green_agent
from core.audit import PrismAudit
from core.gemini import GeminiAgent


# ── Prompts ───────────────────────────────────────────────────

NEGOTIATION_PROMPT = """You are the {agent_name} agent ({agent_style}). You just completed a task and now see what the other two agents did.

## Original Request
{request}

## Your Response
{own_response}

## Other Agent Responses
{other_responses}

In 2-3 sentences, respond with:
1. One thing another agent did BETTER than you (give credit)
2. One thing YOU did that the others missed
3. One specific suggestion to IMPROVE the final merged result

Be concise and direct. No fluff."""


MERGE_PROMPT = """You are the Prism Orchestrator. Three AI agents analyzed the same user request with different cognitive styles. After their initial work, they reviewed each other's output and provided feedback in a negotiation round.

## User's Original Request
{request}

## Relevant Context (from AlloyDB AI semantic search)
{smart_context}

## RED AGENT (Speed-focused)
Response: {red_response}
Negotiation feedback: {red_negotiation}

## BLUE AGENT (Depth-focused)
Response: {blue_response}
Negotiation feedback: {blue_negotiation}

## GREEN AGENT (Creative)
Response: {green_response}
Negotiation feedback: {green_negotiation}

## Your Task
1. Consider BOTH the agent responses AND their negotiation feedback
2. Merge the best parts into a single coherent result
3. Note which agent contributed what
4. Highlight any conflicts that were resolved through negotiation

Respond in this JSON format:
{{
    "merged_response": "Your merged summary — what was done and recommended",
    "contributions": [
        {{"agent": "red", "kept": "what you kept from red", "reason": "why"}},
        {{"agent": "blue", "kept": "what you kept from blue", "reason": "why"}},
        {{"agent": "green", "kept": "what you kept from green", "reason": "why"}}
    ],
    "conflicts_resolved": "Any disagreements between agents and how they were resolved",
    "verdict": "One sentence: which agent performed best and why"
}}"""


CONTRARIAN_PROMPT = """You are a critical thinking advisor. Given this merged action plan, provide a brief contrarian perspective — what could go WRONG, what assumptions might be flawed, or what alternative approach was overlooked.

## Merged Result
{merged_response}

Respond in 2-3 sentences. Be specific and constructive, not generically negative. Start with "Consider this:" """


class Orchestrator:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self.session_service = InMemorySessionService()
        self.merger = GeminiAgent(
            model=model,
            system_prompt="You are an expert at synthesizing multiple agent perspectives. Always respond with valid JSON.",
        )
        self.negotiator = GeminiAgent(
            model=model,
            system_prompt="You are a collaborative AI agent reviewing peer work. Be concise and constructive.",
        )
        self._toolsets: list[McpToolset] = []
        self._mcp_clients = None  # Set by server for smart context
        self._red: LlmAgent | None = None
        self._blue: LlmAgent | None = None
        self._green: LlmAgent | None = None

    async def initialize(self, python_cmd: str, env: dict):
        """Initialize MCP toolsets and ADK agents."""
        tasks_toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=python_cmd,
                    args=["mcp_servers/tasks_server.py"],
                    env=env,
                )
            )
        )
        notes_toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=python_cmd,
                    args=["mcp_servers/notes_server.py"],
                    env=env,
                )
            )
        )
        calendar_toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=python_cmd,
                    args=["mcp_servers/calendar_server.py"],
                    env=env,
                )
            )
        )
        email_toolset = McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=python_cmd,
                    args=["mcp_servers/email_server.py"],
                    env=env,
                )
            )
        )

        self._toolsets = [tasks_toolset, notes_toolset, calendar_toolset, email_toolset]

        self._red = create_red_agent(tools=list(self._toolsets), model=self.model)
        self._blue = create_blue_agent(tools=list(self._toolsets), model=self.model)
        self._green = create_green_agent(tools=list(self._toolsets), model=self.model)

        print(f"[Prism] ADK agents initialized (model: {self.model})")

    async def run(self, request: str, callback=None, deep_analysis: bool = False) -> dict:
        """Prism pipeline. Default: Context → Agents → Merge (fast).
        With deep_analysis=True: adds Negotiation + Contrarian (thorough but slower)."""
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        audit = PrismAudit(model=self.model)

        # ── Phase 1: Smart Context (AlloyDB AI) ──────────────
        if callback:
            await callback({"type": "context_gathering", "message": "Searching for relevant context via AlloyDB AI..."})

        import time as _time
        ctx_start = _time.time()
        smart_context = await self._gather_context(request)
        audit.log("context", "orchestrator", "semantic_search", f"Found context: {smart_context[:100]}",
                  input_text=request, output_text=smart_context, latency_ms=int((_time.time()-ctx_start)*1000))

        if callback:
            await callback({"type": "context_done", "context": smart_context})

        # Build enriched prompt with date + context
        enriched = f"[Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')}). Tomorrow: {tomorrow.strftime('%Y-%m-%d')} ({tomorrow.strftime('%A')}).]"
        if smart_context:
            enriched += f"\n\n[Relevant context from your data:\n{smart_context}]"
        enriched += f"\n\n{request}"

        # ── Phase 2: Parallel Agent Execution ─────────────────
        if callback:
            for name, color in [("red", "#EF4444"), ("blue", "#3B82F6"), ("green", "#10B981")]:
                await callback({"type": "agent_start", "agent": name, "color": color})

        async def launch_red():
            return await self._run_agent(self._red, "red", "#EF4444", enriched, callback)
        async def launch_blue():
            await asyncio.sleep(0.5)
            return await self._run_agent(self._blue, "blue", "#3B82F6", enriched, callback)
        async def launch_green():
            await asyncio.sleep(1.0)
            return await self._run_agent(self._green, "green", "#10B981", enriched, callback)

        results: list[AgentResult] = await asyncio.gather(
            launch_red(), launch_blue(), launch_green(),
        )
        red_result, blue_result, green_result = results

        for result in results:
            audit.log("agent", result.agent_name, "llm_generate",
                      f"Response: {result.response[:80]}", input_text=enriched,
                      output_text=result.response, latency_ms=result.execution_time_ms,
                      status="error" if result.error else "success")
            for tc in result.tool_calls:
                audit.log("agent", result.agent_name, "tool_call",
                          f"{tc.tool_name}({json.dumps(tc.args)[:60]})")

        if callback:
            for result in results:
                await callback({
                    "type": "agent_done",
                    "agent": result.agent_name,
                    "color": result.color,
                    "response": result.response,
                    "tool_calls": [{"tool": tc.tool_name, "args": tc.args} for tc in result.tool_calls],
                    "execution_time_ms": result.execution_time_ms,
                    "error": result.error,
                })

        # ── Phase 3 (optional): Agent Negotiation ─────────────
        negotiations = {}
        if deep_analysis:
            await asyncio.sleep(2)
            if callback:
                await callback({"type": "negotiation_start", "message": "Agents reviewing each other's work..."})
            neg_start = _time.time()
            negotiations = await self._negotiate(request, red_result, blue_result, green_result)
            audit.log("negotiation", "orchestrator", "llm_generate", "3 agents exchanged feedback",
                      input_text=request, output_text=json.dumps(negotiations)[:200],
                      latency_ms=int((_time.time()-neg_start)*1000))
            if callback:
                await callback({"type": "negotiation_done", "negotiations": negotiations})

        # ── Phase 4: Merge ────────────────────────────────────
        await asyncio.sleep(2)

        if callback:
            await callback({"type": "merge_start"})

        merge_start = _time.time()
        merged = await self._merge(request, smart_context, red_result, blue_result, green_result, negotiations)
        audit.log("merge", "orchestrator", "llm_generate", f"Merged: {merged.get('merged_response', '')[:80]}",
                  input_text=request, output_text=json.dumps(merged)[:300],
                  latency_ms=int((_time.time()-merge_start)*1000))

        # ── Phase 5 (optional): Contrarian View ──────────────
        contrarian = ""
        if deep_analysis:
            merged_text = merged.get("merged_response", "")
            if merged_text and not merged_text.startswith("Merge failed"):
                try:
                    await asyncio.sleep(1)
                    if callback:
                        await callback({"type": "contrarian_start"})
                    ct_start = _time.time()
                    prompt = CONTRARIAN_PROMPT.format(merged_response=merged_text[:800])
                    response = await self.merger.generate([{"role": "user", "parts": [prompt]}])
                    contrarian = GeminiAgent.extract_text(response)
                    audit.log("contrarian", "orchestrator", "llm_generate", f"Contrarian: {contrarian[:80]}",
                              input_text=prompt, output_text=contrarian,
                              latency_ms=int((_time.time()-ct_start)*1000))
                except Exception as e:
                    contrarian = ""

        merged["contrarian_view"] = contrarian

        if callback:
            await callback({"type": "merge_done", "result": merged})
            await callback({"type": "audit_summary", "audit": audit.summary(), "entries": audit.to_list()})

        return {
            "agents": {
                "red": self._result_to_dict(red_result),
                "blue": self._result_to_dict(blue_result),
                "green": self._result_to_dict(green_result),
            },
            "negotiations": negotiations,
            "smart_context": smart_context,
            "merged": merged,
            "audit": audit.summary(),
        }

    async def run_lite(self, request: str) -> dict:
        """Lightweight single-agent run for workflows. Uses Blue (Depth) agent only.
        Avoids parallel execution, negotiation, and merge to minimize API calls."""
        now = datetime.now()
        tomorrow = now + timedelta(days=1)

        enriched = f"[Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')}). Tomorrow: {tomorrow.strftime('%Y-%m-%d')} ({tomorrow.strftime('%A')}).]"

        # Quick context gather
        smart_context = await self._gather_context(request)
        if smart_context:
            enriched += f"\n\n[Context:\n{smart_context}]"
        enriched += f"\n\n{request}"

        # Run single agent (Blue — most thorough for multi-step workflows)
        result = await self._run_agent(self._blue, "blue", "#3B82F6", enriched)

        return {
            "merged": {"merged_response": result.response or result.error or "No result"},
            "agents": {"blue": self._result_to_dict(result)},
        }

    async def _gather_context(self, request: str) -> str:
        """Use MCP semantic search to find relevant tasks and notes."""
        if not self._mcp_clients:
            return ""

        context_parts = []
        try:
            # Semantic search tasks
            result = await self._mcp_clients["tasks"].call_tool(
                "semantic_search_tasks", {"query": request, "limit": 3}
            )
            if result and result.content:
                tasks = json.loads(result.content[0].text)
                if tasks:
                    context_parts.append("Related tasks: " + "; ".join(
                        f"'{t.get('title', '')}' ({t.get('status', '')}, {t.get('priority', '')} priority)"
                        for t in tasks[:3]
                    ))
        except Exception as e:
            print(f"[Prism] Context search (tasks) failed: {e}")

        try:
            # Semantic search notes
            result = await self._mcp_clients["notes"].call_tool(
                "semantic_search_notes", {"query": request, "limit": 3}
            )
            if result and result.content:
                notes = json.loads(result.content[0].text)
                if notes:
                    context_parts.append("Related notes: " + "; ".join(
                        f"'{n.get('title', '')}'"
                        for n in notes[:3]
                    ))
        except Exception as e:
            print(f"[Prism] Context search (notes) failed: {e}")

        return "\n".join(context_parts) if context_parts else ""

    async def _negotiate(self, request: str, red: AgentResult, blue: AgentResult, green: AgentResult) -> dict:
        """Run negotiation round — each agent reviews the other two."""
        agents_data = [
            ("Red", "Speed-focused, quick action", red, [blue, green]),
            ("Blue", "Depth-focused, thorough analysis", blue, [red, green]),
            ("Green", "Creative, lateral thinking", green, [red, blue]),
        ]

        async def get_feedback(name, style, own, others):
            other_text = "\n\n".join(
                f"**{o.agent_name.upper()} AGENT:** {o.response[:500]}"
                for o in others if o.response
            )
            prompt = NEGOTIATION_PROMPT.format(
                agent_name=name, agent_style=style,
                request=request,
                own_response=own.response[:500] if own.response else "[No response]",
                other_responses=other_text or "[No responses from other agents]",
            )
            try:
                response = await self.negotiator.generate(
                    [{"role": "user", "parts": [prompt]}]
                )
                return GeminiAgent.extract_text(response)
            except Exception as e:
                return f"[Negotiation skipped: {str(e)[:80]}]"

        # Run negotiations in parallel
        feedbacks = await asyncio.gather(
            get_feedback(*agents_data[0]),
            get_feedback(*agents_data[1]),
            get_feedback(*agents_data[2]),
        )

        return {"red": feedbacks[0], "blue": feedbacks[1], "green": feedbacks[2]}

    async def _run_agent(self, agent: LlmAgent, name: str, color: str, request: str, callback=None) -> AgentResult:
        """Run a single ADK agent with timeout."""
        try:
            return await asyncio.wait_for(
                run_adk_agent(
                    agent=agent, agent_name=name, color=color,
                    query=request, session_service=self.session_service,
                    callback=callback,
                ),
                timeout=60,
            )
        except asyncio.TimeoutError:
            return AgentResult(agent_name=name, color=color, response="", error="Agent timed out after 60 seconds")
        except Exception as e:
            return AgentResult(agent_name=name, color=color, response="", error=str(e))

    async def _merge(self, request: str, smart_context: str, red: AgentResult, blue: AgentResult, green: AgentResult, negotiations: dict) -> dict:
        """Merge with negotiation context."""
        prompt = MERGE_PROMPT.format(
            request=request,
            smart_context=smart_context or "No additional context found.",
            red_response=red.response or f"[Error: {red.error}]",
            blue_response=blue.response or f"[Error: {blue.error}]",
            green_response=green.response or f"[Error: {green.error}]",
            red_negotiation=negotiations.get("red", "N/A"),
            blue_negotiation=negotiations.get("blue", "N/A"),
            green_negotiation=negotiations.get("green", "N/A"),
        )

        try:
            response = await self.merger.generate(
                [{"role": "user", "parts": [prompt]}]
            )
            text = GeminiAgent.extract_text(response)
            clean = text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                return {"merged_response": text, "contributions": [], "conflicts_resolved": "", "verdict": ""}
        except Exception as e:
            return {"merged_response": f"Merge failed: {str(e)}", "contributions": [], "conflicts_resolved": "", "verdict": "Error during merge"}

    @staticmethod
    def _result_to_dict(result: AgentResult) -> dict:
        return {
            "agent_name": result.agent_name,
            "color": result.color,
            "response": result.response,
            "tool_calls": [
                {"tool": tc.tool_name, "args": tc.args, "result": tc.result}
                for tc in result.tool_calls
            ],
            "execution_time_ms": result.execution_time_ms,
            "error": result.error,
        }

    @staticmethod
    async def log_performance(db_pool, session_id: int, agents: dict, merged: dict):
        """Log agent performance metrics to database."""
        if not db_pool or not session_id:
            return
        selected_agents = set()
        for c in merged.get("contributions", []):
            if c.get("kept") and c["kept"] != "None":
                selected_agents.add(c.get("agent", ""))

        for name, data in agents.items():
            try:
                tools = [tc["tool"] for tc in data.get("tool_calls", [])]
                await db_pool.execute(
                    """INSERT INTO agent_performance (session_id, agent_name, was_selected, tools_used, execution_time_ms)
                       VALUES ($1, $2, $3, $4, $5)""",
                    session_id, name, name in selected_agents, tools, data.get("execution_time_ms", 0),
                )
            except Exception as e:
                print(f"[Prism] Failed to log performance for {name}: {e}")
