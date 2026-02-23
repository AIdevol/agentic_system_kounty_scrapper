from __future__ import annotations

import asyncio
import os
from typing import Any
from dotenv import load_dotenv

from groq import Groq

load_dotenv()

# Default Groq model (Llama 3.1 8B Instant).
DEFAULT_MODEL = "llama-3.1-8b-instant"


class LLMClient:
    """
    Thin async wrapper around the underlying LLM provider.

    This implementation uses Groq's API with Llama 3 8B
    via the `groq` Python client.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GROQ_API_KEY environment variable. "
                "Set it in .env or your environment for the Groq API."
            )
        self._client = Groq(api_key=api_key)

    async def complete(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        **kwargs: Any,
    ) -> str:
        """
        Simple chat completion that returns the assistant's message content.
        """

        def _generate() -> str:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            params: dict[str, Any] = {
                "model": model,
                "messages": messages,
            }
            if "temperature" in kwargs and kwargs["temperature"] is not None:
                params["temperature"] = kwargs.pop("temperature")
            params.update(kwargs)

            response = self._client.chat.completions.create(**params)
            if not response.choices:
                return ""
            return (response.choices[0].message.content or "").strip()

        return await asyncio.to_thread(_generate)
