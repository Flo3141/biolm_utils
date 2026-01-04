"""Compatibility shim for legacy ConfigManager access.

Prefer threading Hydra configs explicitly; this singleton remains only for
backward compatibility.
"""

import warnings
from typing import Any

from .loader import load_config


class ConfigManager:
    _instance: Any = None

    @classmethod
    def get_config(cls):
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
