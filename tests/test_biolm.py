import os
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch

# Mock config before importing
with patch("biolm.config_access.ConfigManager.get_config") as mock_get_config:
    from biolm.structured_config import BioLMConfig, TrainingConfig

    mock_cfg = BioLMConfig(
        mode="tokenize",
        task="regression",
        training=TrainingConfig(resume=False),  # Add training config
    )
    mock_get_config.return_value = mock_cfg
    from biolm.biolm import set_seed


class TestBiolm:
    def test_set_seed(self):
        seed = 42
        set_seed(seed)
        val1 = random.random()
        val2 = np.random.random()
        val3 = torch.rand(1).item()

        set_seed(seed)
        assert random.random() == val1
        assert np.random.random() == val2
        assert torch.rand(1).item() == val3

        # Check environment
        assert os.environ["PYTHONHASHSEED"] == str(seed)
