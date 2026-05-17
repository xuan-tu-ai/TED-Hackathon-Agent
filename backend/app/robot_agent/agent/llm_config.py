"""LLM provider configuration and DSPy LM factory.

Reads environment variables (``LM_PROVIDER``, ``LM_MODEL``, ``LM_API_KEY``,
``LM_API_BASE``) and constructs a :class:`dspy.LM` instance.

Usage::

    from .llm_config import get_lm, get_llm_config

    lm = get_lm()
    cfg = get_llm_config()
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


Provider = Literal["openai", "anthropic", "google"]


class LLMConfig:
    """Holds resolved LLM provider, model, key, and base URL.

    Values are resolved from constructor arguments first, then from
    environment variables (``LM_PROVIDER``, ``LM_MODEL``, ``LM_API_KEY``,
    ``LM_API_BASE``), with DSPy fallbacks (``DSPY_LLM_*``) as a final
    default.
    """

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
        """Resolve the API key for the given provider from environment."""
        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        return os.getenv(key_map.get(provider, "OPENAI_API_KEY"), "")

    @property
    def is_configured(self) -> bool:
        """``True`` when a valid API key is present."""
        return bool(self.api_key and self.api_key != "your_openai_key_here")


def get_llm_config() -> LLMConfig:
    """Build and return a default :class:`LLMConfig` from environment."""
    return LLMConfig()


def get_lm() -> "dspy.LM | None":
    """Construct and return a :class:`dspy.LM` instance based on configuration.

    Returns ``None`` and prints a warning when the API key is missing.
    Provider prefixes (``anthropic/``, ``google/``) are prepended
    automatically for non-OpenAI providers.
    """
    import dspy
    config = get_llm_config()

    if not config.is_configured:
        print(f"[dspy] WARNING: LLM {config.provider}/{config.model} not configured (no API key)")
        return None

    lm_kwargs = {
        "api_key": config.api_key,
        "cache": False,
        "max_retries": 0,
    }
    if config.api_base:
        lm_kwargs["api_base"] = config.api_base

    model_str = config.model
    if config.provider == "anthropic":
        model_str = f"anthropic/{config.model}"
    elif config.provider == "google":
        model_str = f"google/{config.model}"

    print(f"[dspy] LM config: provider={config.provider} model={model_str} api_base={config.api_base}")
    return dspy.LM(model_str, **lm_kwargs)
