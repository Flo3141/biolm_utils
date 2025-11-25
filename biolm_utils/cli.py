"""CLI entry point and argument parsing."""

import hydra
from omegaconf import DictConfig

from .loader import _process_hydra_config


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def parse_args(cfg: DictConfig):
    """Hydra CLI entrypoint — returns processed BioLMConfig."""
    return _process_hydra_config(cfg)
