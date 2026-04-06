"""Agent definitions using Google ADK (Agent Development Kit)."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


@dataclass
class ToolCallLog:
    tool_name: str
    args: dict
    result: str = ""


@dataclass
class AgentResult:
    agent_name: str
    color: str
    response: str
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    execution_time_ms: int = 0
    error: str = ""


async def run_adk_agent(
    agent: LlmAgent,
    agent_name: str,
    color: str,
    query: str,
    session_service: InMemorySessionService,
    callback=None,
) -> AgentResult:
    """Run an ADK LlmAgent and collect results with SSE callbacks."""
    start = time.time()
    tool_calls_log = []
    response_text = ""

    try:
        runner = Runner(
            app_name=f"prism_{agent_name}",
            agent=agent,
            session_service=session_service,
        )

        session = await session_service.create_session(
            app_name=f"prism_{agent_name}",
            user_id="prism_user",
        )

        content = types.Content(
            role="user",
            parts=[types.Part(text=query)],
        )

        async for event in runner.run_async(
            user_id="prism_user",
            session_id=session.id,
            new_message=content,
        ):
            # Extract function calls from event
            fc_list = event.get_function_calls()
            if fc_list:
                for fc in fc_list:
                    args = dict(fc.args) if fc.args else {}
                    tool_calls_log.append(ToolCallLog(tool_name=fc.name, args=args))
                    if callback:
                        await callback({
                            "type": "tool_call",
                            "agent": agent_name,
                            "color": color,
                            "tool": fc.name,
                            "args": args,
                        })

            # Extract function responses (tool results)
            fn_responses = event.get_function_responses()
            if fn_responses and callback:
                for fr in fn_responses:
                    result_str = json.dumps(dict(fr.response)) if fr.response else ""
                    await callback({
                        "type": "tool_result",
                        "agent": agent_name,
                        "color": color,
                        "tool": fr.name,
                        "result_preview": result_str[:200],
                    })

            # Extract final text response
            if event.is_final_response() and event.content and event.content.parts:
                texts = [p.text for p in event.content.parts if p.text]
                if texts:
                    response_text = "\n".join(texts)

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        return AgentResult(
            agent_name=agent_name,
            color=color,
            response="",
            tool_calls=tool_calls_log,
            execution_time_ms=elapsed,
            error=str(e),
        )

    elapsed = int((time.time() - start) * 1000)
    return AgentResult(
        agent_name=agent_name,
        color=color,
        response=response_text or "Agent completed but produced no text response.",
        tool_calls=tool_calls_log,
        execution_time_ms=elapsed,
    )
