"""Canonical base model API for BioLM plugins.

Plugin authors should inherit from :class:`BioLMModel`.
"""

from __future__ import annotations

from typing import Any, Dict

from transformers import PretrainedConfig, PreTrainedModel


class BioLMModel(PreTrainedModel):
    """Base model class for plugins.

    Provides common hooks and ensures HuggingFace compatibility.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

    @classmethod
    def get_config(cls, **kwargs) -> PretrainedConfig:
        """Return a config for this model. Override in subclasses."""
        raise NotImplementedError

    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Hook for batch preprocessing. Override if needed."""
        return batch


__all__ = ["BioLMModel"]
