"""LangChain LCEL Pipeline Provider.

Leverages LangChain Expression Language (LCEL) with ChatGoogleGenerativeAI
and ChatOpenAI, demonstrating modern LangChain chain composition.
"""

from typing import Optional, Literal
from .base import BaseLLMProvider
from core.config import settings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class LangChainProvider(BaseLLMProvider):
    """LangChain LCEL integration supporting multiple chat model backends."""

    def __init__(
        self,
        backend: Literal["gemini", "openai"] = "gemini",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.backend = backend.lower()
        self.api_key = api_key
        self.model_name = model_name

        if self.backend == "gemini":
            self.api_key = self.api_key or settings.gemini_api_key
            self.model_name = self.model_name or settings.default_gemini_model
            if not self.api_key:
                raise ValueError("Gemini API key missing for LangChain Gemini provider.")
        elif self.backend == "openai":
            self.api_key = self.api_key or settings.openai_api_key
            self.model_name = self.model_name or settings.default_openai_model
            if not self.api_key:
                raise ValueError("OpenAI API key missing for LangChain OpenAI provider.")
        else:
            raise ValueError(f"Unsupported LangChain backend '{self.backend}'. Use 'gemini' or 'openai'.")

    def _get_chat_model(self, temperature: float = 0.1, json_mode: bool = True):
        """Instantiate underlying LangChain ChatModel with JSON settings."""
        if self.backend == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=temperature,
            )
        else:
            from langchain_openai import ChatOpenAI
            model_kwargs = {}
            if json_mode:
                model_kwargs["response_format"] = {"type": "json_object"}
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=temperature,
                model_kwargs=model_kwargs,
            )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        json_mode: bool = True
    ) -> str:
        """Execute extraction chain using LangChain LCEL (Prompt | Model | StrOutputParser)."""
        chat_model = self._get_chat_model(temperature=temperature, json_mode=json_mode)

        # Assemble LCEL prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_instruction}"),
            ("human", "{user_instruction}"),
        ])

        # Compose runnable LCEL chain
        chain = prompt | chat_model | StrOutputParser()

        # Run invocation
        result = chain.invoke({
            "system_instruction": system_prompt,
            "user_instruction": user_prompt,
        })

        return str(result)

    def get_provider_name(self) -> str:
        return f"LangChain ({self.backend.capitalize()})"

    def get_model_name(self) -> str:
        return self.model_name
