"""Base Agent - implements the Gemini function calling loop with MCP tools."""

import json
import time
from dataclasses import dataclass, field

from core.gemini import GeminiAgent
from core.tool_manager import ToolManager


@dataclass
class ToolCallLog:
    tool_name: str
    args: dict
    result: str


@dataclass
class AgentResult:
    agent_name: str
    color: str
    response: str
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    execution_time_ms: int = 0
    error: str = ""


class BaseAgent:
    """Base agent that runs a Gemini function calling loop over MCP tools."""

    def __init__(
        self,
        name: str,
        color: str,
        system_prompt: str,
        tool_manager: ToolManager,
        model: str = "gemini-2.5-flash",
        max_iterations: int = 5,
    ):
        self.name = name
        self.color = color
        self.system_prompt = system_prompt
        self.tool_manager = tool_manager
        self.model = model
        self.max_iterations = max_iterations
        self.gemini = GeminiAgent(model=model, system_prompt=system_prompt)

    async def execute(self, user_request: str, callback=None) -> AgentResult:
        """
        Run the agentic loop.
        callback: async function(event_dict) called on each tool call for SSE streaming.
        """
        start = time.time()
        tool_calls_log = []
        response_text = ""

        try:
            # Discover tools from MCP servers
            tools = await self.tool_manager.discover_tools()

            # Build initial conversation
            contents = [{"role": "user", "parts": [user_request]}]

            for iteration in range(self.max_iterations):
                # Call Gemini
                response = await self.gemini.generate(contents, tools=tools)

                # Check for function calls
                function_calls = GeminiAgent.extract_function_calls(response)

                if not function_calls:
                    # No more tool calls — extract final text
                    response_text = GeminiAgent.extract_text(response)
                    break

                # Add model's response (with function calls) to history
                contents.append(response.candidates[0].content)

                # Execute each function call
                function_responses = []
                for fc in function_calls:
                    args = dict(fc.args) if fc.args else {}

                    # Emit SSE event
                    if callback:
                        await callback({
                            "type": "tool_call",
                            "agent": self.name,
                            "color": self.color,
                            "tool": fc.name,
                            "args": args,
                        })

                    # Execute via MCP
                    result_text = await self.tool_manager.execute(fc.name, args)
                    tool_calls_log.append(ToolCallLog(
                        tool_name=fc.name, args=args, result=result_text
                    ))

                    # Emit result event
                    if callback:
                        await callback({
                            "type": "tool_result",
                            "agent": self.name,
                            "color": self.color,
                            "tool": fc.name,
                            "result_preview": result_text[:200],
                        })

                    # Build function response for Gemini (must be a dict, not list)
                    try:
                        parsed = json.loads(result_text)
                        if isinstance(parsed, list):
                            result_obj = {"results": parsed}
                        elif isinstance(parsed, dict):
                            result_obj = parsed
                        else:
                            result_obj = {"result": str(parsed)}
                    except (json.JSONDecodeError, TypeError):
                        result_obj = {"result": result_text}

                    function_responses.append(
                        GeminiAgent.build_function_response(fc.name, result_obj)
                    )

                # Add function responses to history
                contents.append({"role": "user", "parts": function_responses})

            if not response_text:
                response_text = "Agent completed tool calls but produced no final summary."

        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            return AgentResult(
                agent_name=self.name,
                color=self.color,
                response="",
                tool_calls=tool_calls_log,
                execution_time_ms=elapsed,
                error=str(e),
            )

        elapsed = int((time.time() - start) * 1000)
        return AgentResult(
            agent_name=self.name,
            color=self.color,
            response=response_text,
            tool_calls=tool_calls_log,
            execution_time_ms=elapsed,
        )
