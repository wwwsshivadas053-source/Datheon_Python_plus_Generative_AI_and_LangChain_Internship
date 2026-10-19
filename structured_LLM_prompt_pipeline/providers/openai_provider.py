"""OpenAI LLM Provider.

Supports GPT-4o, GPT-4o-mini, and other chat models with json_object mode.
"""

from typing import Optional
from .base import BaseLLMProvider
from core.config import settings


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API integration."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError(
                "OpenAI API Key missing. Please set OPENAI_API_KEY in .env or pass it to OpenAIProvider."
            )
        self.model_name = model_name or settings.default_openai_model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        json_mode: bool = True
    ) -> str:
        """Call OpenAI chat completion API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if content:
            return content
        raise RuntimeError("OpenAI returned empty message content.")

    def get_provider_name(self) -> str:
        return "OpenAI"

    def get_model_name(self) -> str:
        return self.model_name
