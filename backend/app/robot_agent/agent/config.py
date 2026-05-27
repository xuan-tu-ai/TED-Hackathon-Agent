"""Central configuration singleton.

All environment variables, file-system paths, and LLM configuration
are resolved once through this module.  No other module should read
``os.getenv`` or compute ``Path(__file__).parents[N]`` directly.

Usage::

    from .config import config

    db_url = config.database_url
    lm = config.get_lm()
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import ClassVar, Literal

from dotenv import load_dotenv

load_dotenv()

Provider = Literal["openai", "anthropic", "google"]


class AppConfig:
    """Singleton holding all env-driven configuration.

    Access the shared instance via the module-level ``config`` name::

        from .config import config
        print(config.database_url)
    """

    _instance: ClassVar[AppConfig | None] = None

    def __new__(cls) -> AppConfig:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    REQUIRED_ENV_VARS: ClassVar[list[str]] = [
        "AGENT_KNOWLEDGE_DIR",
        "DATA_DIR",
        "DATABASE_URL",
    ]

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        # ── Validate required env vars first ──────────────────
        missing = [k for k in self.REQUIRED_ENV_VARS if k not in os.environ]
        if missing:
            raise OSError(
                f"Missing required environment variable(s): {', '.join(missing)}. "
                "Set them in a .env file or your shell before starting the application."
            )

        # ── Required paths (must be set in .env / environment) ─
        self.agent_knowledge_dir = Path(os.environ["AGENT_KNOWLEDGE_DIR"]).resolve()
        self.data_dir = Path(os.environ["DATA_DIR"]).resolve()
        self.database_url: str = os.environ["DATABASE_URL"]

        # ── Optional sub-path overrides (default under DATA_DIR) ─
        self.chroma_db_dir = self._resolve_or(
            "CHROMA_DB_PATH", self.data_dir / "chroma_db"
        )
        self.session_dir = self._resolve_or(
            "SESSION_DIR", self.data_dir / "sessions"
        )
        self.memory_dir = self._resolve_or(
            "MEMORY_DIR", self.data_dir / "memory"
        )
        self.generated_data_dir = self._resolve_or(
            "GENERATED_DATA_DIR", self.data_dir / "generated"
        )

        # ── The only kept default ─────────────────────────────
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        # ── Derived paths ─────────────────────────────────────
        self.agents_md_path: Path = self.agent_knowledge_dir / "AGENTS.md"
        self.skills_base: Path = self.agent_knowledge_dir / "skills"
        self.rules_dir: Path = self.agent_knowledge_dir / "rules"
        self.db_path: Path = self.data_dir / "demo.db"
        self.sql_dir: Path = self.data_dir / "sql"

        # ── Anomaly detection thresholds (from agent_knowledge/thresholds.json) ──
        self.l1_density_threshold: float
        self.l2_min_slope: float
        self._load_thresholds()

        # ── Databricks configuration (OT data tables) ─────────
        self.databricks_host: str | None = os.getenv("DATABRICKS_HOST")
        self.databricks_http_path: str | None = os.getenv("DATABRICKS_HTTP_PATH")
        self.databricks_token: str | None = os.getenv("DATABRICKS_TOKEN")
        self.databricks_catalog: str | None = os.getenv("DATABRICKS_CATALOG")
        self.databricks_schema: str | None = os.getenv("DATABRICKS_SCHEMA")
        self.databricks_vs_index_pm_cards: str | None = os.getenv("DATABRICKS_VS_INDEX_PM_CARDS")

        # ── LLM configuration (no computed path defaults) ─────
        self.lm_provider: str = (
            os.getenv("LM_PROVIDER") or os.getenv("DSPY_LLM_PROVIDER", "openai")
        )
        self.lm_model: str = (
            os.getenv("LM_MODEL") or os.getenv("DSPY_LLM_MODEL", "gpt-4o-mini")
        )
        self.lm_api_base: str | None = (
            os.getenv("LM_API_BASE") or os.getenv("DSPY_LLM_API_BASE")
        )
        self.lm_api_key: str = (
            os.getenv("LM_API_KEY") or self._resolve_api_key(self.lm_provider)
        )

    # ── Threshold loader ─────────────────────────────────────

    def _load_thresholds(self) -> None:
        """Load L1/L2 thresholds from agent_knowledge/thresholds.json."""
        path = self.agent_knowledge_dir / "thresholds.json"
        defaults: dict = {
            "l1_density_threshold": 30.0,
            "l2_min_slope": 0.005,
            "sensor_thresholds": {
                "wire_feed_speed": 0.13,
                "wire_temperature": 12.06,
                "welding_current": 0.56,
            },
        }
        if path.exists():
            with path.open() as f:
                data = json.load(f)
            defaults.update({k: v for k, v in data.items() if not k.startswith("_")})
        self.l1_density_threshold = float(defaults["l1_density_threshold"])
        self.l2_min_slope = float(defaults["l2_min_slope"])
        self.sensor_thresholds: dict[str, float] = defaults["sensor_thresholds"]

    def reload_thresholds(self) -> None:
        """Hot-reload thresholds from disk without restarting the app."""
        self._load_thresholds()

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _resolve_or(env_key: str, default: Path) -> Path:
        """Return the env var value as an absolute path, or *default*."""
        raw = os.getenv(env_key)
        return Path(raw).resolve() if raw else default.resolve()

    @staticmethod
    def _resolve_api_key(provider: str) -> str:
        key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        return os.getenv(key_map.get(provider, "OPENAI_API_KEY"), "")

    @property
    def is_lm_configured(self) -> bool:
        """``True`` when a valid API key is present."""
        return bool(self.lm_api_key and self.lm_api_key != "your_openai_key_here")

    @property
    def databricks_configured(self) -> bool:
        """``True`` when all 5 Databricks environment variables are non-empty."""
        return all([
            bool(self.databricks_host),
            bool(self.databricks_http_path),
            bool(self.databricks_token),
            bool(self.databricks_catalog),
            bool(self.databricks_schema),
        ])

    # ── Factory methods (moved from llm_config.py) ────────────

    def get_llm_config(self):
        """Return an :class:`LLMConfig`-like object from current settings."""
        from dataclasses import dataclass

        @dataclass
        class LLMConfig:
            """Resolved LLM provider, model, key, and base URL."""
            provider: str
            model: str
            api_key: str
            api_base: str | None

            @property
            def is_configured(self) -> bool:
                return bool(self.api_key and self.api_key != "your_openai_key_here")

        return LLMConfig(
            provider=self.lm_provider,
            model=self.lm_model,
            api_key=self.lm_api_key,
            api_base=self.lm_api_base,
        )

    def get_lm(self):
        """Construct and return a :class:`dspy.LM` instance.

        Returns ``None`` and prints a warning when the API key is missing.
        Provider prefixes (``anthropic/``, ``google/``) are prepended
        automatically for non-OpenAI providers.
        """
        import dspy

        if not self.is_lm_configured:
            print(
                f"[dspy] WARNING: LLM {self.lm_provider}/{self.lm_model} "
                "not configured (no API key)"
            )
            return None

        lm_kwargs = {
            "api_key": self.lm_api_key,
            "cache": False,
            "max_retries": 0,
        }
        if self.lm_api_base:
            lm_kwargs["api_base"] = self.lm_api_base

        model_str = self.lm_model
        if self.lm_provider == "anthropic":
            model_str = f"anthropic/{self.lm_model}"
        elif self.lm_provider == "google":
            model_str = f"google/{self.lm_model}"

        print(
            f"[dspy] LM config: provider={self.lm_provider} "
            f"model={model_str} api_base={self.lm_api_base}"
        )
        return dspy.LM(model_str, **lm_kwargs)


# Module-level singleton instance
config = AppConfig()
