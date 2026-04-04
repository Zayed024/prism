"""Gemini wrapper with function calling support for the agentic tool loop."""

import asyncio
import os

from google import generativeai as genai
from google.generativeai import protos


_configured = False


def _ensure_configured():
    global _configured
    if not _configured:
        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        genai.configure(api_key=api_key)
        _configured = True


class GeminiAgent:
    """A Gemini-powered agent that supports function calling."""

    def __init__(self, model: str = "gemini-2.5-flash", system_prompt: str = ""):
        _ensure_configured()
        self.model_name = model
        self.system_prompt = system_prompt
        self.model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt if system_prompt else None,
        )

    async def generate(self, contents: list, tools=None) -> protos.GenerateContentResponse:
        """Call Gemini with contents and optional tools. Returns raw response."""
        kwargs = {"contents": contents}
        if tools:
            kwargs["tools"] = tools
        response = await asyncio.to_thread(
            self.model.generate_content, **kwargs
        )
        return response

    @staticmethod
    def extract_function_calls(response) -> list:
        """Extract function calls from a Gemini response."""
        calls = []
        if not response.candidates:
            return calls
        for part in response.candidates[0].content.parts:
            if part.function_call and part.function_call.name:
                calls.append(part.function_call)
        return calls

    @staticmethod
    def extract_text(response) -> str:
        """Extract text from a Gemini response."""
        if not response.candidates:
            return ""
        texts = []
        for part in response.candidates[0].content.parts:
            if part.text:
                texts.append(part.text)
        return "\n".join(texts)

    @staticmethod
    def build_function_response(name: str, result: dict) -> protos.Part:
        """Build a FunctionResponse Part to send back to Gemini."""
        return protos.Part(
            function_response=protos.FunctionResponse(
                name=name,
                response=result,
            )
        )
