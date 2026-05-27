from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_llm_config_defaults():
    from app.robot_agent.agent.config import config

    llm_cfg = config.get_llm_config()
    assert llm_cfg.provider in {"openai", "anthropic", "google"}
    assert llm_cfg.model is not None
    assert isinstance(llm_cfg.is_configured, bool)


def test_llm_config_explicit():
    # The config singleton reads from env; verify the raw env-reading path works
    from app.robot_agent.agent.config import AppConfig

    cfg = AppConfig()
    assert cfg.lm_provider is not None
    assert cfg.lm_model is not None


def test_llm_config_not_configured():
    from app.robot_agent.agent.config import config

    # Set a known-bad key and re-check
    old_key = config.lm_api_key
    config.lm_api_key = "your_openai_key_here"
    assert config.is_lm_configured is False
    config.lm_api_key = old_key  # restore


def test_get_lm_returns_none_when_not_configured():
    from app.robot_agent.agent.config import config

    old_key = config.lm_api_key
    config.lm_api_key = ""
    lm = config.get_lm()
    assert lm is None
    config.lm_api_key = old_key  # restore