"""Compatibility shim for legacy ConfigManager access.

Prefer threading Hydra configs explicitly; this singleton remains only for
backward compatibility when tests or legacy entry points inject a config
programmatically.
"""

import warnings
from typing import Any


class ConfigManager:
    _instance: Any = None

    @classmethod
    def set_config(cls, config: Any):
        """Seed the singleton without triggering Hydra loads (for tests/legacy)."""

        cls._instance = config

    @classmethod
    def get_config(cls):
        warnings.warn(
            "ConfigManager is deprecated; use the Hydra-provided config directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if cls._instance is None:
            raise RuntimeError(
                "ConfigManager has no active config. Call initialize_runtime(config) "
                "from biolm.biolm or explicitly inject via ConfigManager.set_config() in tests."
            )
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
    def t_get(cls, key: str, default=None):
        training = cls.get_training()
        return getattr(training, key, default) if training is not None else default

    @classmethod
    def d_get(cls, key: str, default=None):
        debugging = cls.get_debugging()
        return getattr(debugging, key, default) if debugging is not None else default

    @classmethod
    def i_get(cls, key: str, default=None):
        inference = cls.get_inference()
        return getattr(inference, key, default) if inference is not None else default
