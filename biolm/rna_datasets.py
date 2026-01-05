"""Legacy dataset module.

New plugin code should import the canonical dataset base from `biolm.biolm_dataset`.
`RNABaseDataset` remains available for backwards compatibility.
"""

from __future__ import annotations

import warnings

from .biolm_dataset import BioLMDataset, RNABaseDataset


warnings.warn(
    "`biolm.rna_datasets` is deprecated; import `BioLMDataset`/`RNABaseDataset` from `biolm.biolm_dataset` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BioLMDataset", "RNABaseDataset"]
