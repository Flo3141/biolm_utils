"""Configuration access and helpers using singleton pattern."""

import warnings
from typing import Any

from .loader import load_config


class ConfigManager:
    """Singleton manager for lazy configuration loading.

    Deprecated: prefer threading Hydra's DictConfig through your call stack
    instead of using this singleton. This remains for backward compatibility
    while call sites are migrated.
    """

    _instance = None

    @classmethod
    def get_config(cls):
        """Get the configuration, loading it lazily if needed."""
        warnings.warn(
            "ConfigManager is deprecated; use the Hydra-provided config directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if cls._instance is None:
            cls._instance = load_config()
        return cls._instance

    @classmethod
    def get_data_source(cls):
        return getattr(cls.get_config(), "data_source", None)

    @classmethod
    def get_training(cls):
        return getattr(cls.get_config(), "training", None)

    @classmethod
    def get_debugging(cls):
        return getattr(cls.get_config(), "debugging", None)

    @classmethod
    def get_inference(cls):
        return getattr(cls.get_config(), "inference", None)

    @classmethod
    def t_get(cls, key: str, default=None) -> Any:
        """Get training config value."""
        training = cls.get_training()
        if training is not None:
            return getattr(training, key, default)
        return default

    @classmethod
    def d_get(cls, key: str, default=None) -> Any:
        """Get debugging config value."""
        debugging = cls.get_debugging()
        if debugging is not None:
            return getattr(debugging, key, default)
        return default

    @classmethod
    def i_get(cls, key: str, default=None) -> Any:
        """Get inference config value."""
        inference = cls.get_inference()
        if inference is not None:
            return getattr(inference, key, default)
        return default
