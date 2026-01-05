"""Backward-compatible re-exports.

Historically this module contained model base classes, despite the filename.
New code should import from `biolm.base_model` instead.
"""

from __future__ import annotations

import warnings

from .base_model import BaseModel


warnings.warn(
    "`biolm.base_dataset` is deprecated; import `BaseModel` from `biolm.base_model` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BaseModel"]
