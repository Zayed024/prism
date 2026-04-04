"""Tool Manager - aggregates MCP tools and routes execution."""

import copy
import json
from typing import Optional

from google.generativeai import protos
from google.generativeai.types import FunctionDeclaration, Tool

from core.mcp_client import MCPClient
from mcp.types import TextContent

# Fields Gemini's Schema proto does NOT support
_UNSUPPORTED_FIELDS = {"title", "default", "additionalProperties", "$schema", "examples"}


def _clean_schema(schema: dict) -> dict:
    """Recursively strip fields that Gemini's Schema proto doesn't support."""
    cleaned = {}
    for k, v in schema.items():
        if k in _UNSUPPORTED_FIELDS:
            continue
        if k == "properties" and isinstance(v, dict):
            cleaned[k] = {pk: _clean_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            cleaned[k] = _clean_schema(v)
        elif k == "anyOf" and isinstance(v, list):
            cleaned["any_of"] = [_clean_schema(item) for item in v]
        else:
            cleaned[k] = v
    return cleaned


class ToolManager:
    """Manages tool discovery and execution across multiple MCP clients."""

    def __init__(self, clients: dict[str, MCPClient]):
        self.clients = clients
        self._tool_to_client: dict[str, MCPClient] = {}

    async def discover_tools(self) -> list:
        """Discover all tools from all MCP servers and return as Gemini Tool objects."""
        declarations = []
        for name, client in self.clients.items():
            mcp_tools = await client.list_tools()
            for t in mcp_tools:
                self._tool_to_client[t.name] = client
                # Convert MCP tool schema to Gemini FunctionDeclaration
                params = copy.deepcopy(t.inputSchema) if t.inputSchema else {}
                # Remove 'ctx' from parameters if present (injected by FastMCP)
                if "properties" in params and "ctx" in params["properties"]:
                    del params["properties"]["ctx"]
                if "required" in params and "ctx" in params["required"]:
                    params["required"].remove("ctx")

                # Clean schema for Gemini compatibility
                params = _clean_schema(params)

                declarations.append(FunctionDeclaration(
                    name=t.name,
                    description=t.description or "",
                    parameters=params,
                ))
        return [Tool(function_declarations=declarations)] if declarations else []

    async def execute(self, tool_name: str, args: dict) -> str:
        """Execute a tool call via the appropriate MCP client."""
        client = self._tool_to_client.get(tool_name)
        if not client:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})

        try:
            result = await client.call_tool(tool_name, args)
            if result and result.content:
                texts = [
                    item.text for item in result.content
                    if isinstance(item, TextContent)
                ]
                return "\n".join(texts) if texts else "Tool executed successfully (no text output)"
            return "Tool executed successfully"
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})
