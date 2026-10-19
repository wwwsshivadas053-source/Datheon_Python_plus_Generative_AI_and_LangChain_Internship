"""Abstract Base Class for LLM Providers."""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLMProvider(ABC):
    """Unified interface for LLM model inference."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        json_mode: bool = True
    ) -> str:
        """Execute text completion and return raw response string."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return provider identifier (e.g., 'gemini', 'openai', 'mock')."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return specific model name (e.g., 'gemini-2.5-flash', 'gpt-4o-mini')."""
        pass
