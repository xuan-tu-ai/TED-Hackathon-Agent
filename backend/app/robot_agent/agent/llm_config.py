from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


Provider = Literal["openai", "anthropic", "google"]


class LLMConfig:
    def __init__(
        self,
        provider: Provider | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
    ):
        self.provider = provider or os.getenv("LM_PROVIDER") or os.getenv("DSPY_LLM_PROVIDER", "openai")
        self.model = model or os.getenv("LM_MODEL") or os.getenv("DSPY_LLM_MODEL", "gpt-4o-mini")
        self.api_base = api_base or os.getenv("LM_API_BASE") or os.getenv("DSPY_LLM_API_BASE")
        self.api_key = api_key or os.getenv("LM_API_KEY") or self._get_api_key(self.provider)

    def _get_api_key(self, provider: Provider) -> str:
        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        return os.getenv(key_map.get(provider, "OPENAI_API_KEY"), "")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key != "your_openai_key_here")


def get_llm_config() -> LLMConfig:
    return LLMConfig()


def get_lm() -> "dspy.LM" | None:
    """Create a DSPy LM instance. Returns None if not configured."""
    import dspy
    config = get_llm_config()

    if not config.is_configured:
        print(f"[dspy] Warning: LLM {config.provider}/{config.model} not fully configured (no API key)")
        return None

    lm_kwargs = {"api_key": config.api_key}
    if config.api_base:
        lm_kwargs["api_base"] = config.api_base

    if config.provider == "openai":
        return dspy.LM(config.model, **lm_kwargs)
    elif config.provider == "anthropic":
        return dspy.LM(f"anthropic/{config.model}", **lm_kwargs)
    elif config.provider == "google":
        return dspy.LM(f"google/{config.model}", **lm_kwargs)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def configure_dspy() -> None:
    """Legacy function - now does nothing. Use dspy.context(lm=get_lm()) instead."""
    pass
