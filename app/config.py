from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
SUPPORTED_EMBEDDING_PROVIDERS = frozenset({"openai", "local", "fake"})
DEFAULT_EMBEDDING_PROVIDER = "local"
DEFAULT_LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_LLM_PROVIDERS = frozenset({"openai", "ollama", "mock"})
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:1.5b"


def load_environment(env_file: str | Path = ENV_FILE) -> bool:
    loaded = load_dotenv(dotenv_path=env_file, override=False)
    logger.info(
        "Environment loaded; embedding_provider=%s OPENAI_API_KEY configured=%s",
        get_embedding_provider(),
        bool(os.getenv("OPENAI_API_KEY", "").strip()),
    )
    logger.info(
        "LLM provider=%s model=%s",
        get_llm_provider(),
        get_ollama_model() if get_llm_provider() == "ollama" else "configured",
    )
    return loaded


def get_embedding_provider() -> str:
    provider = os.getenv(
        "EMBEDDING_PROVIDER",
        DEFAULT_EMBEDDING_PROVIDER,
    ).strip().lower()
    if provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_EMBEDDING_PROVIDERS))
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER '{provider}'. "
            f"Expected one of: {supported}."
        )
    return provider


def get_local_embedding_model() -> str:
    return (
        os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            DEFAULT_LOCAL_EMBEDDING_MODEL,
        ).strip()
        or DEFAULT_LOCAL_EMBEDDING_MODEL
    )


def get_llm_provider() -> str:
    configured = os.getenv("LLM_PROVIDER", "").strip().lower()
    provider = configured or (
        "openai" if os.getenv("OPENAI_API_KEY", "").strip() else "ollama"
    )
    if provider not in SUPPORTED_LLM_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_LLM_PROVIDERS))
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            f"Expected one of: {supported}."
        )
    return provider


def get_ollama_model() -> str:
    return (
        os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
        or DEFAULT_OLLAMA_MODEL
    )


def is_development() -> bool:
    return os.getenv("APP_ENV", "development").strip().lower() in {
        "development",
        "dev",
        "local",
        "test",
    }
