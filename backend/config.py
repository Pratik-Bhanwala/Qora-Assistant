"""Configuration helpers for the self persona chatbot backend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"


class Settings(BaseSettings):
    """Runtime configuration for the chatbot."""

    ollama_model: str = Field(
        "glm-4.6:cloud",
        description="Default Ollama model to use for responses.",
    )
    ollama_host: str = Field(
        "http://localhost:11434",
        description="Base URL for the Ollama server (set to https://cloud.ollama.com for cloud).",
    )
    ollama_api_key: str | None = Field(
        default=None,
        description="API key for Ollama Cloud requests if required.",
    )
    persona_profile_path: Path = Field(
        BASE_DIR / "data" / "profile.yaml",
        description="Path to the YAML file containing persona details.",
    )
    classification_style: Literal["conservative", "balanced", "creative"] = Field(
        "balanced",
        description="How proactively the agent should use persona knowledge.",
    )

    class Config:
        env_prefix = "CHATBOT_"
        env_file = ENV_PATH if ENV_PATH.exists() else None
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once and cache them for reuse."""

    if ENV_PATH.exists():
        # Load environment variables if an .env file is present.
        load_dotenv(ENV_PATH)
    return Settings()


settings = get_settings()

