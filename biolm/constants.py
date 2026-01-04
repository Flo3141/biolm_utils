"""Derived constants and trainer class selection."""

import logging
import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from transformers.trainer import Trainer

from .config_access import ConfigManager
from .metrics import compute_metrics_for_classification, compute_metrics_for_regression
from .params import get_detected_ngpus
from .path_setup import PathsManager
from .trainer import (
    RegressionTrainer,
    WeightedRegressionTrainer,
    WeightedSamplingTrainer,
)


def _resolve_config(args=None):
    """Return provided args or a previously injected config."""

    if args is not None:
        return args

    if ConfigManager._instance is not None:
        return ConfigManager.get_config()

    raise RuntimeError(
        "Constants require a config. Call initialize_runtime(config) first or pass args explicitly."
    )


def setup_constants(log_params: bool = True, args=None):
    """Set up and return derived constants.

    log_params controls whether the parameter header is emitted. Set False for
    early imports; the real run can log once after plugin load.
    """
    args = _resolve_config(args)
    paths = PathsManager.get_paths(config=args)

    training_cfg = getattr(args, "training", None)
    debugging_cfg = getattr(args, "debugging", None)

    # We scale the gradient with respect to the number of GPUs to keep an
    # effective batch size of `args.batchsize` x `args.gradacc`
    if getattr(debugging_cfg, "dev", False):
        gradacc = 1
        detected_gpus = 1
    else:
        detected_gpus = get_detected_ngpus(args)
    # training.gradacc is the configured gradient-accumulation multiplier
    gradacc = max(
        1,
        int(
            round(
                float(getattr(training_cfg, "gradacc", 1)) / max(1, int(detected_gpus))
            )
        ),
    )

    # Resolve plugin/model class early so we can print a meaningful `model` field
    from .plugin_config import PluginManager

    plugin_config = PluginManager.get_config()
    if args.mode == "pre-train":
        model_cls = getattr(plugin_config, "model_cls_for_pretraining", None)
    else:
        model_cls = getattr(plugin_config, "model_cls_for_finetuning", None)

    # Best-effort lazy plugin load if the class is still missing (Hydra sometimes
    # instantiates configs before plugins are registered).
    if model_cls is None and getattr(args, "plugin", None):
        try:
            import importlib.metadata

            eps = importlib.metadata.entry_points(group="biolm.plugins")
            for ep in eps:
                if ep.name == args.plugin:
                    ep.load()()
                    plugin_config = PluginManager.get_config()
                    if args.mode == "pre-train":
                        model_cls = getattr(
                            plugin_config, "model_cls_for_pretraining", None
                        )
                    else:
                        model_cls = getattr(
                            plugin_config, "model_cls_for_finetuning", None
                        )
                    break
        except Exception:
            pass

    if log_params:
        logging.info(f"{'=== Params ===':>32}")

        for key, value in sorted(vars(args).items()):
            if key == "model" and (
                not value or (isinstance(value, DictConfig) and len(value) == 0)
            ):
                if model_cls is not None:
                    display_value = f"plugin={args.plugin}, class={model_cls.__module__}.{model_cls.__name__}"
                elif getattr(args, "plugin", None):
                    display_value = f"plugin={args.plugin}"
                else:
                    display_value = "(set by plugin config)"
            elif isinstance(value, DictConfig):
                if len(value) == 0:
                    display_value = "{}"
                else:
                    display_value = (
                        OmegaConf.to_yaml(value, resolve=True)
                        .strip()
                        .replace("\n", ", ")
                    )
            else:
                display_value = str(value)

            logging.info(f"{key:>25} : {display_value}")

        if model_cls is not None:
            logging.info(
                f"{'model_class':>25} : {model_cls.__module__}.{model_cls.__name__}"
            )
        else:
            logging.info(f"{'model_class':>25} : (not provided by plugin)")

        model_load_path = paths.get("MODELLOADPATH")
        model_save_path = paths.get("MODELSAVEPATH")
        logging.info(
            f"{'model_load_path':>25} : {model_load_path if model_load_path is not None else '(none)'}"
        )
        # For predict/interpret we don't persist models; omit save path noise
        if args.mode not in ["predict", "interpret"]:
            logging.info(
                f"{'model_save_path':>25} : {model_save_path if model_save_path is not None else '(none)'}"
            )

    if training_cfg and getattr(training_cfg, "resume", False):
        checkpointpath = max(
            paths["MODELSAVEPATH"].glob("checkpoint*"), key=os.path.getmtime
        )
        logging.info(f"Pretrained model to resume from: {checkpointpath}")
    else:
        checkpointpath = None

    regressiontrainer_cls = (
        WeightedRegressionTrainer
        if getattr(training_cfg, "weightedregression", False)
        else RegressionTrainer
    )

    classificationtrainer_cls = WeightedSamplingTrainer

    mlmtrainer_cls = Trainer

    metric = (
        compute_metrics_for_classification
        if args.task == "classification"
        else compute_metrics_for_regression
    )

    return {
        "GRADACC": gradacc,
        "CHECKPOINTPATH": checkpointpath,
        "REGRESSIONTRAINER_CLS": regressiontrainer_cls,
        "CLASSIFICATIONTRAINER_CLS": classificationtrainer_cls,
        "MLMTRAINER_CLS": mlmtrainer_cls,
        "METRIC": metric,
    }


# Global constants dict, set up lazily
_constants = None


def reset_constants():
    """Clear the cached constants so they recompute for a new config."""
    global _constants
    _constants = None


def get_constants(log_params: bool = True, args=None):
    """Get constants dict, setting up lazily if needed.

    log_params controls whether to emit the parameter header when computing
    constants.
    """
    global _constants
    if _constants is None:
        _constants = setup_constants(log_params=log_params, args=args)
    return _constants
