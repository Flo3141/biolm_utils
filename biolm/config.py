from .structured_config import BioLMConfig as Config
from .config_access import ConfigManager

def get_config():
    return ConfigManager.get_config()

def set_config(config: Config):
    ConfigManager._instance = config
