"""Gemini wrapper — supports both API key and Vertex AI (via google.genai)."""

import asyncio
import os

from google import genai
from google.genai import types


_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        # If GOOGLE_GENAI_USE_VERTEXAI is set, uses Vertex AI (no API key needed)
        # Otherwise falls back to API key
        if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE":
            _client = genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
                location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
        else:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")
            if not api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required (or set GOOGLE_GENAI_USE_VERTEXAI=TRUE)")
            _client = genai.Client(api_key=api_key)
    return _client


class GeminiAgent:
    """A Gemini-powered agent using google.genai (supports Vertex AI)."""

    def __init__(self, model: str = "gemini-3-flash-preview", system_prompt: str = ""):
        self.client = _get_client()
        self.model_name = model
        self.system_prompt = system_prompt

    async def generate(self, contents: list, tools=None, max_retries: int = 5):
        """Call Gemini with contents and optional tools. Retries on rate limit."""
        # Convert our simple format to genai Content objects
        genai_contents = []
        for item in contents:
            if isinstance(item, dict):
                role = item.get("role", "user")
                parts_raw = item.get("parts", [])
                parts = []
                for p in parts_raw:
                    if isinstance(p, str):
                        parts.append(types.Part(text=p))
                    else:
                        parts.append(p)
                genai_contents.append(types.Content(role=role, parts=parts))
            else:
                genai_contents.append(item)

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt if self.system_prompt else None,
        )

        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=genai_contents,
                    config=config,
                )
                return response
            except Exception as e:
                if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < max_retries - 1:
                    wait = 3 * (attempt + 1)  # 3, 6, 9, 12, 15 seconds
                    print(f"[Gemini] Rate limited, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait)
                else:
                    raise

    @staticmethod
    def extract_text(response) -> str:
        """Extract text from a Gemini response."""
        if not response.candidates:
            return ""
        texts = []
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                texts.append(part.text)
        return "\n".join(texts)
