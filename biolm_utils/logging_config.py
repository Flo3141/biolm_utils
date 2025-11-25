"""Logging configuration."""

import logging
from datetime import datetime

from .config_access import ConfigManager
from .path_setup import PathsManager


def setup_logging():
    """Set up logging based on config."""
    args = ConfigManager.get_config()
    paths = PathsManager.get_paths()

    # Switch off the 'The used dataset had no length, returning gathered tensors. You should drop the remainder yourself.' warning if desired.
    # if args.silent:
    logging.getLogger("accelerate").setLevel(logging.WARNING)

    now = datetime.now().strftime("%Y-%m-%d_%H:%M")
    logfile = paths["LOGPATH"] / f"{now}.log"
    logfile.touch(exist_ok=True)

    if not ConfigManager.d_get("dev", False):
        handlers = [
            logging.FileHandler(logfile, mode="w"),
            logging.StreamHandler(),
        ]
    else:
        handlers = [
            logging.StreamHandler(),
        ]

    # Convert all handlers to logging.Handler if not already
    handlers = [
        h if isinstance(h, logging.Handler) else logging.StreamHandler()
        for h in handlers
    ]

    logging.basicConfig(
        format=f"%(asctime)s ({args.mode} {paths['OUTPUTPATH'].stem}) - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
    )
