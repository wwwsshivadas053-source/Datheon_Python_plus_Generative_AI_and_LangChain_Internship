"""
agent/llm_factory.py
====================
Factory that creates the appropriate LangChain LLM based on
the LLM_PROVIDER environment variable.

Supported providers:
  - google    → ChatGoogleGenerativeAI  (GOOGLE_API_KEY)
  - openai    → ChatOpenAI              (OPENAI_API_KEY)
  - anthropic → ChatAnthropic           (ANTHROPIC_API_KEY)
"""

import os
from langchain_core.language_models import BaseChatModel


def create_llm() -> BaseChatModel:
    """
    Instantiate and return the configured LangChain chat model.

    Provider is controlled by LLM_PROVIDER in .env.
    Model name is controlled by LLM_MODEL in .env.
    Temperature is controlled by TEMPERATURE in .env.
    """
    provider = os.getenv("LLM_PROVIDER", "google").lower()
    temperature = float(os.getenv("TEMPERATURE", "0.1"))

    # ------------------------------------------------------------------ #
    #  Google Gemini                                                        #
    # ------------------------------------------------------------------ #
    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY is missing.\n"
                "Set it in your .env file. Get a free key at: "
                "https://aistudio.google.com/app/apikey"
            )
        model_name = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai is not installed.\n"
                "Run: pip install langchain-google-genai"
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True,
        )

    # ------------------------------------------------------------------ #
    #  OpenAI GPT                                                           #
    # ------------------------------------------------------------------ #
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is missing.\n"
                "Set it in your .env file. Get a key at: https://platform.openai.com/api-keys"
            )
        model_name = os.getenv("LLM_MODEL", "gpt-4o")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai is not installed.\n"
                "Run: pip install langchain-openai"
            )
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=temperature,
        )

    # ------------------------------------------------------------------ #
    #  Anthropic Claude                                                     #
    # ------------------------------------------------------------------ #
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is missing.\n"
                "Set it in your .env file. Get a key at: https://console.anthropic.com/"
            )
        model_name = os.getenv("LLM_MODEL", "claude-3-5-sonnet-20241022")
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError(
                "langchain-anthropic is not installed.\n"
                "Run: pip install langchain-anthropic"
            )
        return ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'.\n"
            "Valid options: 'google', 'openai', 'anthropic'"
        )

