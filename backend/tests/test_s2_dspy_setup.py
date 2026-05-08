from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_llm_config_defaults():
    from app.robot_agent.agent.llm_config import LLMConfig, get_llm_config

    config = get_llm_config()
    assert config.provider in {"openai", "anthropic", "google"}
    assert config.model is not None
    assert isinstance(config.is_configured, bool)


def test_llm_config_explicit():
    from app.robot_agent.agent.llm_config import LLMConfig

    config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="test-key")
    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"
    assert config.api_key == "test-key"
    assert config.is_configured is True


def test_llm_config_not_configured():
    from app.robot_agent.agent.llm_config import LLMConfig

    config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="your_openai_key_here")
    assert config.is_configured is False


def test_configure_dspy_no_error():
    from app.robot_agent.agent.llm_config import configure_dspy

    configure_dspy()