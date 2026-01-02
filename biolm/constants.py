"""Derived constants and trainer class selection."""

import logging
import os
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf
from transformers.trainer import Trainer

from .config_access import ConfigManager
from .params import get_detected_ngpus
from .path_setup import PathsManager
from .train_utils import (
    compute_metrics_for_classification,
    compute_metrics_for_regression,
)
from .trainer import (
    RegressionTrainer,
    WeightedRegressionTrainer,
    WeightedSamplingTrainer,
)


def setup_constants(log_params: bool = True):
    """Set up and return derived constants.

    log_params controls whether the parameter header is emitted. Set False for
    early imports; the real run can log once after plugin load.
    """
    args = ConfigManager.get_config()
    paths = PathsManager.get_paths()

    # We scale the gradient with respect to the number of GPUs to keep an
    # effective batch size of `args.batchsize` x `args.gradacc`
    if ConfigManager.d_get("dev", False):
        gradacc = 1
    else:
        detected_gpus = get_detected_ngpus(args)
    # training.gradacc is the configured gradient-accumulation multiplier
    gradacc = max(
        1,
        int(
            round(float(ConfigManager.t_get("gradacc", 1)) / max(1, int(detected_gpus)))
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

    if args.training.resume:
        checkpointpath = max(
            paths["MODELSAVEPATH"].glob("checkpoint*"), key=os.path.getmtime
        )
        logging.info(f"Pretrained model to resume from: {checkpointpath}")
    else:
        checkpointpath = None

    regressiontrainer_cls = (
        WeightedRegressionTrainer
        if ConfigManager.get_training().weightedregression
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


def get_constants(log_params: bool = True):
    """Get constants dict, setting up lazily if needed.

    log_params controls whether to emit the parameter header when computing
    constants.
    """
    global _constants
    if _constants is None:
        _constants = setup_constants(log_params=log_params)
    return _constants
