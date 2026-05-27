"""Re-exports from :mod:`~.config` for backward compatibility.

All LLM configuration logic has moved to :class:`~.config.AppConfig`.
This module is kept only for existing ``from .llm_config import ...``
imports and will be removed in a future cleanup.
"""

from .config import config as _cfg


def get_llm_config():
    """Delegate to :meth:`AppConfig.get_llm_config`."""
    return _cfg.get_llm_config()


def get_lm():
    """Delegate to :meth:`AppConfig.get_lm`."""
    return _cfg.get_lm()
