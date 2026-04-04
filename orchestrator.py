"""Orchestrator - splits request to 3 agents, runs in parallel, merges results."""

import asyncio
import json
import time

from agents.base import AgentResult
from agents.red import RedAgent
from agents.blue import BlueAgent
from agents.green import GreenAgent
from core.gemini import GeminiAgent
from core.tool_manager import ToolManager


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
    def __init__(self, tool_manager: ToolManager, model: str = "gemini-2.5-flash"):
        self.tool_manager = tool_manager
        self.model = model
        self.red = RedAgent(tool_manager, model=model)
        self.blue = BlueAgent(tool_manager, model=model)
        self.green = GreenAgent(tool_manager, model=model)
        self.merger = GeminiAgent(model=model, system_prompt="You are an expert at synthesizing multiple perspectives into the best combined result. Always respond with valid JSON.")

    async def run(self, request: str, callback=None) -> dict:
        """
        Run all 3 agents in parallel, then merge results.
        callback: async function(event_dict) for SSE streaming.
        """
        # Signal agent start
        if callback:
            for agent_name, color in [("red", "#EF4444"), ("blue", "#3B82F6"), ("green", "#10B981")]:
                await callback({"type": "agent_start", "agent": agent_name, "color": color})

        # Run agents in parallel
        results: list[AgentResult] = await asyncio.gather(
            self._run_agent(self.red, request, callback),
            self._run_agent(self.blue, request, callback),
            self._run_agent(self.green, request, callback),
        )

        red_result, blue_result, green_result = results

        # Signal agents done
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

    async def _run_agent(self, agent, request: str, callback=None) -> AgentResult:
        """Run a single agent with error handling."""
        try:
            return await asyncio.wait_for(
                agent.execute(request, callback=callback),
                timeout=60,
            )
        except asyncio.TimeoutError:
            return AgentResult(
                agent_name=agent.name,
                color=agent.color,
                response="",
                error="Agent timed out after 60 seconds",
            )
        except Exception as e:
            return AgentResult(
                agent_name=agent.name,
                color=agent.color,
                response="",
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

            # Try to parse as JSON
            # Strip markdown code fences if present
            clean = text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1]
                if clean.endswith("```"):
                    clean = clean[:-3]
                clean = clean.strip()

            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                return {
                    "merged_response": text,
                    "contributions": [],
                    "verdict": "Could not parse structured merge result",
                }
        except Exception as e:
            return {
                "merged_response": f"Merge failed: {str(e)}",
                "contributions": [],
                "verdict": "Error during merge",
            }

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
