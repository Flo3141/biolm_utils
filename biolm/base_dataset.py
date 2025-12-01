"""Base classes for plugin models in biolm.

Plugins should subclass these for consistency and to leverage framework hooks.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from torch.utils.data import Dataset
from transformers import PretrainedConfig, PreTrainedModel


class BaseModel(PreTrainedModel):
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


class BaseDataset(Dataset):
    """Base dataset class for plugins.

    Provides common interface for data loading and inherits from PyTorch Dataset.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
