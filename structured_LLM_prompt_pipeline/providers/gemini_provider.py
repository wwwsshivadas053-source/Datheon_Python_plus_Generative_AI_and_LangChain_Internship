"""Google Gemini LLM Provider.

Supports both modern 'google-genai' (2025/2026 standard) and legacy 'google-generativeai'
with native JSON mode (application/json) enforcement.
"""

from typing import Optional
from .base import BaseLLMProvider
from core.config import settings


class GeminiProvider(BaseLLMProvider):
    """Google Gemini integration."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.gemini_api_key
        if not self.api_key:
            raise ValueError(
                "Gemini API Key missing. Please set GEMINI_API_KEY in .env or pass it to GeminiProvider."
            )
        self.model_name = model_name or settings.default_gemini_model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        json_mode: bool = True
    ) -> str:
        """Call Gemini API with structured response MIME type."""
        # Try modern google-genai first
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            mime_type = "application/json" if json_mode else "text/plain"

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                response_mime_type=mime_type,
            )

            response = client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=config,
            )
            if response and response.text:
                return response.text
            raise RuntimeError("Gemini returned empty response.")

        except ImportError:
            # Fallback to google.generativeai
            import google.generativeai as gai

            gai.configure(api_key=self.api_key)
            generation_config = {
                "temperature": temperature,
            }
            if json_mode:
                generation_config["response_mime_type"] = "application/json"

            model = gai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
                generation_config=generation_config,
            )

            response = model.generate_content(user_prompt)
            if response and response.text:
                return response.text
            raise RuntimeError("Gemini returned empty response via google.generativeai.")

    def get_provider_name(self) -> str:
        return "Google Gemini"

    def get_model_name(self) -> str:
        return self.model_name
