"""Path setup and directory management."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .config_access import ConfigManager


def _get_default_output_path(args) -> Path:
    """Determine the default output path."""
    if args.outputpath is None:
        if args.data_source and getattr(args.data_source, "filepath", None):
            args.outputpath = Path(args.data_source.filepath).stem
        else:
            args.outputpath = "output"
    return Path(args.outputpath)


def _get_model_load_path(args, outputpath: Path) -> Optional[Path]:
    """Determine the model load path based on mode."""
    mode_load_map = {
        "fine-tune": outputpath / "pre-train",
        "interpret": outputpath / "fine-tune",
        "predict": outputpath / "fine-tune",
    }
    load_path = mode_load_map.get(args.mode)

    return load_path


def _adjust_paths_for_pretrained_model(
    args, modelloadpath: Optional[Path], tokenizerfile: Path
) -> tuple[Optional[Path], Path]:
    """Adjust paths if a pretrained model is specified."""
    if args.inference and getattr(args.inference, "pretrainedmodel", None):
        pretrained_path = Path(args.inference.pretrainedmodel)
        if args.mode != "pre-train":
            return pretrained_path, pretrained_path / "tokenize"
        else:
            return modelloadpath, pretrained_path / "tokenize"
    # If in predict/interpret mode and no pretrainedmodel is set, use final_model if it exists
    if args.mode in ["predict", "interpret"] and modelloadpath is not None:
        final_model_dir = modelloadpath.parent / "final_model"
        if final_model_dir.exists():
            return final_model_dir, final_model_dir / "tokenize"
    return modelloadpath, tokenizerfile


def _resolve_config(args=None):
    """Return provided args or a previously injected config."""

    if args is not None:
        return args

    if ConfigManager._instance is not None:
        return ConfigManager.get_config()

    raise RuntimeError(
        "PathsManager requires a config. Call initialize_runtime(config) first or provide args explicitly."
    )


def setup_paths(args=None) -> Dict[str, Optional[Path]]:
    """Set up and return all path constants based on config."""
    args = _resolve_config(args)

    outputpath = _get_default_output_path(args)
    outputpath.mkdir(parents=True, exist_ok=True)

    tokenizerfile = (
        outputpath / "tokenize"
    )  # Directory containing tokenizer files (HuggingFace format)
    modelloadpath = _get_model_load_path(args, outputpath)
    modelloadpath, tokenizerfile = _adjust_paths_for_pretrained_model(
        args, modelloadpath, tokenizerfile
    )

    modelsavepath = outputpath / args.mode

    # Create directories for modes that need them
    if args.mode not in ["tokenize", "predict", "interpret"]:
        modelsavepath.mkdir(parents=True, exist_ok=True)

    reportfile = modelsavepath / "test_predictions.csv"
    rankfile = modelsavepath / "rank_deltas.csv"
    logpath = modelsavepath / "logs"
    logpath.mkdir(parents=True, exist_ok=True)

    datasetfile = (
        None
        if args.mode == "tokenize"
        else outputpath / args.mode / f"{args.mode}_dataset.pkl"
    )

    # Set up logging
    now = datetime.now().strftime("%Y-%m-%d_%H:%M")
    logfile = logpath / f"{now}.log"
    logfile.touch(exist_ok=True)

    return {
        "OUTPUTPATH": outputpath,
        "TOKENIZERFILE": tokenizerfile,
        "MODELLOADPATH": modelloadpath,
        "MODELSAVEPATH": modelsavepath,
        "REPORTFILE": reportfile,
        "RANKFILE": rankfile,
        "LOGPATH": logpath,
        "DATASETFILE": datasetfile,
        "LOGFILE": logfile,
        "TBPATH": logpath,
    }


class PathsManager:
    """Singleton manager for lazy path setup."""

    _instance = None
    _config = None

    @classmethod
    def set_config(cls, config) -> None:
        """Seed the path manager with a specific config and reset the cache."""
        cls._config = config
        cls._instance = None

    @classmethod
    def get_paths(cls, config=None) -> Dict[str, Optional[Path]]:
        """Get paths dict, setting up lazily if needed."""
        if cls._instance is None:
            cls._instance = setup_paths(args=config or cls._config)
        return cls._instance
