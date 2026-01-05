"""Backward-compatible re-exports.

Historically this module contained model base classes, despite the filename.
New code should import from `biolm.biolm_model` instead.
"""

from __future__ import annotations

import warnings

from .biolm_model import BaseModel

warnings.warn(
    "`biolm.base_dataset` is deprecated; import `BioLMModel`/`BaseModel` from `biolm.biolm_model` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BaseModel"]
