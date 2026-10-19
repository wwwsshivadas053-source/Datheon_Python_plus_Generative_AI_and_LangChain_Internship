"""Configuration management for Structured LLM Prompt Pipeline.

Loads settings from .env file or environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file if available
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings(BaseModel):
    """Application-wide settings and parameters."""

    base_dir: Path = Field(default=BASE_DIR)

    # API Keys
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )

    # Default Provider & Models
    default_provider: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_PROVIDER", "mock")
    )
    default_gemini_model: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_GEMINI_MODEL", "gemini-2.5-flash")
    )
    default_openai_model: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_OPENAI_MODEL", "gpt-4o-mini")
    )

    # Pipeline Tuning
    default_temperature: float = Field(
        default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
    )
    max_self_correction_retries: int = Field(
        default_factory=lambda: int(os.getenv("MAX_SELF_CORRECTION_RETRIES", "2"))
    )
    enable_few_shot: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_FEW_SHOT", "true").lower() in ("true", "1", "yes")
    )
    strict_validation_mode: bool = Field(
        default_factory=lambda: os.getenv("STRICT_VALIDATION_MODE", "true").lower() in ("true", "1", "yes")
    )

    # Web Server Config
    port: int = Field(
        default_factory=lambda: int(os.getenv("PORT", "5000"))
    )
    host: str = Field(
        default_factory=lambda: os.getenv("HOST", "127.0.0.1")
    )
    debug: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    )

    def has_gemini_key(self) -> bool:
        """Check if a Gemini API key is available."""
        return bool(self.gemini_api_key and len(self.gemini_api_key.strip()) > 5)

    def has_openai_key(self) -> bool:
        """Check if an OpenAI API key is available."""
        return bool(self.openai_api_key and len(self.openai_api_key.strip()) > 5)


# Singleton instance
settings = Settings()
