"""Validation wrapper for backward compatibility."""

from .structured_config import BioLMConfig


def validate_config(cfg: BioLMConfig) -> None:
    """Validate the configuration."""
    cfg.validate()
