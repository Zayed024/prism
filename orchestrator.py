"""Orchestrator - ADK-powered parallel agent execution with merge."""

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
from core.gemini import GeminiAgent


MERGE_PROMPT = """You are the Prism Orchestrator. Three AI agents analyzed the same user request, each with a different cognitive style. Your job is to merge the best parts of each into a single coherent response.

## User's Original Request
{request}

## RED AGENT (Speed-focused, quick action)
{red_response}

## BLUE AGENT (Depth-focused, thorough analysis)
{blue_response}

## GREEN AGENT (Creative, lateral thinking)
{green_response}

## Your Task
1. Identify the BEST contributions from each agent
2. Merge them into a single, coherent action plan
3. Note which agent contributed what (for transparency)
4. If agents took conflicting actions, explain which you'd keep and why

Respond in this JSON format:
{{
    "merged_response": "Your merged summary here - what was done and what's recommended",
    "contributions": [
        {{"agent": "red", "kept": "what you kept from red", "reason": "why"}},
        {{"agent": "blue", "kept": "what you kept from blue", "reason": "why"}},
        {{"agent": "green", "kept": "what you kept from green", "reason": "why"}}
    ],
    "verdict": "One sentence: which agent performed best for this specific request and why"
}}"""


class Orchestrator:
    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = model
        self.session_service = InMemorySessionService()
        self.merger = GeminiAgent(
            model=model,
            system_prompt="You are an expert at synthesizing multiple perspectives. Always respond with valid JSON.",
        )
        self._toolsets: list[McpToolset] = []
        self._red: LlmAgent | None = None
        self._blue: LlmAgent | None = None
        self._green: LlmAgent | None = None

    async def initialize(self, python_cmd: str, env: dict):
        """Initialize MCP toolsets and ADK agents."""
        # Create MCP toolsets via ADK's McpToolset
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

        # Create ADK agents with MCP tools
        self._red = create_red_agent(tools=list(self._toolsets), model=self.model)
        self._blue = create_blue_agent(tools=list(self._toolsets), model=self.model)
        self._green = create_green_agent(tools=list(self._toolsets), model=self.model)

        print(f"[Prism] ADK agents initialized (model: {self.model})")

    async def run(self, request: str, callback=None) -> dict:
        """Run all 3 ADK agents in parallel, then merge results."""
        # Inject current date context so agents know what "today"/"tomorrow" means
        now = datetime.now()
        tomorrow = now + timedelta(days=1)
        context = f"[Current date: {now.strftime('%Y-%m-%d')} ({now.strftime('%A')}). Tomorrow: {tomorrow.strftime('%Y-%m-%d')} ({tomorrow.strftime('%A')}).]\n\n{request}"

        # Signal agent start
        if callback:
            for name, color in [("red", "#EF4444"), ("blue", "#3B82F6"), ("green", "#10B981")]:
                await callback({"type": "agent_start", "agent": name, "color": color})

        # Run ADK agents in parallel
        results: list[AgentResult] = await asyncio.gather(
            self._run_agent(self._red, "red", "#EF4444", context, callback),
            self._run_agent(self._blue, "blue", "#3B82F6", context, callback),
            self._run_agent(self._green, "green", "#10B981", context, callback),
        )

        red_result, blue_result, green_result = results

        # Emit agent_done events
        if callback:
            for result in results:
                await callback({
                    "type": "agent_done",
                    "agent": result.agent_name,
                    "color": result.color,
                    "response": result.response,
                    "tool_calls": [
                        {"tool": tc.tool_name, "args": tc.args}
                        for tc in result.tool_calls
                    ],
                    "execution_time_ms": result.execution_time_ms,
                    "error": result.error,
                })

        # Brief pause before merge to avoid rate limits after parallel agent calls
        await asyncio.sleep(3)

        # Merge phase
        if callback:
            await callback({"type": "merge_start"})

        merged = await self._merge(request, red_result, blue_result, green_result)

        if callback:
            await callback({"type": "merge_done", "result": merged})

        return {
            "agents": {
                "red": self._result_to_dict(red_result),
                "blue": self._result_to_dict(blue_result),
                "green": self._result_to_dict(green_result),
            },
            "merged": merged,
        }

    async def _run_agent(self, agent: LlmAgent, name: str, color: str, request: str, callback=None) -> AgentResult:
        """Run a single ADK agent with timeout."""
        try:
            return await asyncio.wait_for(
                run_adk_agent(
                    agent=agent,
                    agent_name=name,
                    color=color,
                    query=request,
                    session_service=self.session_service,
                    callback=callback,
                ),
                timeout=60,
            )
        except asyncio.TimeoutError:
            return AgentResult(
                agent_name=name, color=color, response="",
                error="Agent timed out after 60 seconds",
            )
        except Exception as e:
            return AgentResult(
                agent_name=name, color=color, response="",
                error=str(e),
            )

    async def _merge(self, request: str, red: AgentResult, blue: AgentResult, green: AgentResult) -> dict:
        """Use Gemini to merge the three agent results."""
        prompt = MERGE_PROMPT.format(
            request=request,
            red_response=red.response or f"[Error: {red.error}]",
            blue_response=blue.response or f"[Error: {blue.error}]",
            green_response=green.response or f"[Error: {green.error}]",
        )

        try:
            response = await self.merger.generate(
                [{"role": "user", "parts": [prompt]}]
            )
            text = GeminiAgent.extract_text(response)

            # Strip markdown code fences
            clean = text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()

            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                return {"merged_response": text, "contributions": [], "verdict": ""}
        except Exception as e:
            return {"merged_response": f"Merge failed: {str(e)}", "contributions": [], "verdict": "Error during merge"}

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
        # Determine which agents were selected based on merge contributions
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
                    session_id,
                    name,
                    name in selected_agents,
                    tools,
                    data.get("execution_time_ms", 0),
                )
            except Exception as e:
                print(f"[Prism] Failed to log performance for {name}: {e}")
