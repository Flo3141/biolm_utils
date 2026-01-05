"""Backward-compatible re-exports.

Historically the framework exposed a base model class as `biolm.base_model.BaseModel`.
New code should import from `biolm.biolm_model` instead.
"""

from __future__ import annotations

import warnings

from .biolm_model import BaseModel


warnings.warn(
    "`biolm.base_model` is deprecated; import `BioLMModel`/`BaseModel` from `biolm.biolm_model` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BaseModel"]
