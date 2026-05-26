"""Tests for the Databricks SQLAlchemy session module.

Tests cover URL construction, env var validation, and config property checks.
No live Databricks connection is required — all tests use mocked environment.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.robot_agent.db.databricks_session import (
    REQUIRED_DATABRICKS_ENV_VARS,
    _get_databricks_env,
    build_databricks_url,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_env_all():
    """Set all 5 required Databricks env vars to known values."""
    env = {
        "DATABRICKS_HOST": "adb-1234567890123456.7.azuredatabricks.net",
        "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/abc123def456",
        "DATABRICKS_TOKEN": "dapi1234567890abcdef",
        "DATABRICKS_CATALOG": "demo",
        "DATABRICKS_SCHEMA": "robot_forensics",
    }
    with patch.dict(os.environ, env, clear=False):
        yield env


@pytest.fixture
def mock_env_missing():
    """Set only 4 of 5 env vars — missing DATABRICKS_SCHEMA."""
    env = {
        "DATABRICKS_HOST": "adb-1234567890123456.7.azuredatabricks.net",
        "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/abc123def456",
        "DATABRICKS_TOKEN": "dapi1234567890abcdef",
        "DATABRICKS_CATALOG": "demo",
        # DATABRICKS_SCHEMA deliberately omitted
    }
    with patch.dict(os.environ, env, clear=False):
        yield env


# ── URL construction tests ───────────────────────────────────────────────────


class TestBuildDatabricksUrl:
    """Verify the ``databricks://`` connection URL is built correctly."""

    def test_url_built_correctly(self):
        """Given individual components, URL should match expected format."""
        url = build_databricks_url(
            host="adb-1234567890123456.7.azuredatabricks.net",
            http_path="/sql/1.0/warehouses/abc123def456",
            token="dapi1234567890abcdef",
            catalog="demo",
            schema="robot_forensics",
        )
        expected = (
            "databricks://token:dapi1234567890abcdef"
            "@adb-1234567890123456.7.azuredatabricks.net"
            "?http_path=/sql/1.0/warehouses/abc123def456"
            "&catalog=demo"
            "&schema=robot_forensics"
        )
        assert url == expected

    def test_url_with_special_chars(self):
        """Verify URL handles special characters in token/path."""
        url = build_databricks_url(
            host="adb-999.azuredatabricks.net",
            http_path="/sql/1.0/warehouses/test-warehouse",
            token="dapi_token_with_underscores",
            catalog="test_catalog",
            schema="test_schema",
        )
        assert url.startswith("databricks://token:dapi_token_with_underscores@")
        assert "http_path=/sql/1.0/warehouses/test-warehouse" in url
        assert "catalog=test_catalog" in url
        assert "schema=test_schema" in url


# ── Environment variable validation tests ────────────────────────────────────


class TestGetDatabricksEnv:
    """Verify env var validation logic."""

    def test_all_vars_present(self, mock_env_all):
        """When all 5 env vars are set, return them as a dict."""
        result = _get_databricks_env()
        for key in REQUIRED_DATABRICKS_ENV_VARS:
            assert result[key] == mock_env_all[key]

    def test_missing_env_var_raises(self, mock_env_missing):
        """When one required var is missing, raise RuntimeError."""
        with pytest.raises(RuntimeError) as exc_info:
            _get_databricks_env()
        assert "DATABRICKS_SCHEMA" in str(exc_info.value)
        assert "Databricks connection is not configured" in str(exc_info.value)

    def test_empty_env_var_raises(self):
        """When a required var is set but empty, raise RuntimeError."""
        env = {
            "DATABRICKS_HOST": "adb-host",
            "DATABRICKS_HTTP_PATH": "/sql/path",
            "DATABRICKS_TOKEN": "dapi_token",
            "DATABRICKS_CATALOG": "",
            "DATABRICKS_SCHEMA": "robot_forensics",
        }
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(RuntimeError) as exc_info:
                _get_databricks_env()
            assert "DATABRICKS_CATALOG" in str(exc_info.value)


# ── Config property tests ────────────────────────────────────────────────────


class TestDatabricksConfigured:
    """Verify ``config.databricks_configured`` property."""

    def test_configured_true(self, mock_env_all):
        """All 5 vars set → databricks_configured is True."""
        # Re-import config to pick up patched env for a fresh instance check
        from app.robot_agent.agent.config import AppConfig

        cfg = AppConfig()
        # After patch.dict, the singleton may still have old values;
        # manually set the attributes to simulate the patched environment
        cfg.databricks_host = mock_env_all["DATABRICKS_HOST"]
        cfg.databricks_http_path = mock_env_all["DATABRICKS_HTTP_PATH"]
        cfg.databricks_token = mock_env_all["DATABRICKS_TOKEN"]
        cfg.databricks_catalog = mock_env_all["DATABRICKS_CATALOG"]
        cfg.databricks_schema = mock_env_all["DATABRICKS_SCHEMA"]
        assert cfg.databricks_configured is True

    def test_configured_false_missing_host(self):
        """Missing host → databricks_configured is False."""
        from app.robot_agent.agent.config import AppConfig

        cfg = AppConfig()
        cfg.databricks_host = None
        cfg.databricks_http_path = "/sql/path"
        cfg.databricks_token = "dapi_token"
        cfg.databricks_catalog = "demo"
        cfg.databricks_schema = "robot_forensics"
        assert cfg.databricks_configured is False

    def test_configured_false_empty_string(self):
        """Empty string for a var → databricks_configured is False."""
        from app.robot_agent.agent.config import AppConfig

        cfg = AppConfig()
        cfg.databricks_host = "adb-host"
        cfg.databricks_http_path = ""
        cfg.databricks_token = "dapi_token"
        cfg.databricks_catalog = "demo"
        cfg.databricks_schema = "robot_forensics"
        assert cfg.databricks_configured is False

    def test_configured_false_all_missing(self):
        """All vars empty → databricks_configured is False."""
        from app.robot_agent.agent.config import AppConfig

        cfg = AppConfig()
        cfg.databricks_host = None
        cfg.databricks_http_path = None
        cfg.databricks_token = None
        cfg.databricks_catalog = None
        cfg.databricks_schema = None
        assert cfg.databricks_configured is False
