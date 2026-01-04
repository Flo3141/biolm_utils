from .config_access import ConfigManager
from .structured_config import BioLMConfig as Config


def get_config():
    """Legacy accessor; prefer passing Hydra's config explicitly."""
    return ConfigManager.get_config()


def set_config(config: Config):
    """Legacy setter; prefer threading config explicitly."""
    ConfigManager.set_config(config)
