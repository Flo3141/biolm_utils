"""Derived constants and trainer class selection."""

import logging
import os
from pathlib import Path
from typing import Optional

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


def setup_constants():
    """Set up and return derived constants."""
    args = ConfigManager.get_config()
    paths = PathsManager.get_paths()

    # We scale the gradient with respect to the number of GPUs to keep an
    # effective batch size of `args.batchsize` x `args.gradacc`
    if ConfigManager.d_get("dev", False):
        gradacc = 1
    else:
        detected_gpus = get_detected_ngpus(args)
    # training.gradacc is the configured gradient-accumulation multiplier
    gradacc = max(1, int(round(float(ConfigManager.t_get("gradacc", 1)) / max(
        1, int(detected_gpus)
    ))))  # Log the arguments.
    import logging

    logging.info(f"{'=== Params ===':>32}")
    for k, v in sorted(vars(args).items()):
        logging.info(f"{k:>25} : {str(v):<25}")

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


def get_constants():
    """Get constants dict, setting up lazily if needed."""
    global _constants
    if _constants is None:
        _constants = setup_constants()
    return _constants
