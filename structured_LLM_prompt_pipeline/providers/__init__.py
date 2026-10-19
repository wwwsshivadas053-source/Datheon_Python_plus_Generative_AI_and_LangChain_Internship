"""Provider factory and registry for structured LLM prompt pipeline."""

from typing import Optional
from .base import BaseLLMProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .langchain_provider import LangChainProvider
from .mock_provider import MockProvider
from core.config import settings


def get_llm_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseLLMProvider:
    """Factory function to instantiate the requested LLM provider.

    Supported names: 'mock', 'gemini', 'openai', 'langchain_gemini', 'langchain_openai'.
    Falls back gracefully to MockProvider if API keys are missing.
    """
    selected = (provider_name or settings.default_provider).lower().strip()

    if selected == "gemini":
        key = api_key or settings.gemini_api_key
        if not key:
            print("[WARNING] Gemini API key not found. Falling back to Offline Mock Provider.")
            return MockProvider(**kwargs)
        return GeminiProvider(api_key=key, model_name=model_name)

    elif selected == "openai":
        key = api_key or settings.openai_api_key
        if not key:
            print("[WARNING] OpenAI API key not found. Falling back to Offline Mock Provider.")
            return MockProvider(**kwargs)
        return OpenAIProvider(api_key=key, model_name=model_name)

    elif selected in ("langchain", "langchain_gemini"):
        key = api_key or settings.gemini_api_key
        if not key:
            print("[WARNING] Gemini API key not found for LangChain. Falling back to Offline Mock Provider.")
            return MockProvider(**kwargs)
        return LangChainProvider(backend="gemini", api_key=key, model_name=model_name)

    elif selected == "langchain_openai":
        key = api_key or settings.openai_api_key
        if not key:
            print("[WARNING] OpenAI API key not found for LangChain. Falling back to Offline Mock Provider.")
            return MockProvider(**kwargs)
        return LangChainProvider(backend="openai", api_key=key, model_name=model_name)

    elif selected == "mock":
        return MockProvider(**kwargs)

    else:
        raise ValueError(
            f"Unsupported provider '{provider_name}'. Choose from: 'mock', 'gemini', 'openai', 'langchain_gemini', 'langchain_openai'."
        )


__all__ = [
    "BaseLLMProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "LangChainProvider",
    "MockProvider",
    "get_llm_provider",
]
