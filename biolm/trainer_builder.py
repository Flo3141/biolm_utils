"""Trainer construction utilities extracted from train_utils."""

import logging

import numpy as np
import torch
from sklearn.utils import class_weight
from transformers import EarlyStoppingCallback

logger = logging.getLogger(__name__)


def _is_mlflow_enabled(args):
    """Check if MLflow logging is enabled in the config."""
    try:
        mlflow_conf = (
            getattr(args, "settings", None).mlflow
            if getattr(args, "settings", None)
            else None
        )
        return mlflow_conf and mlflow_conf.get("enabled", False)
    except Exception:
        return False


def _instantiate_trainer(trainer_cls, tokenizer, **kwargs):
    """Prefer processing_class param when available to avoid tokenizer deprecation."""

    base_kwargs = dict(kwargs)
    if tokenizer is None:
        return trainer_cls(**base_kwargs)

    try:
        return trainer_cls(processing_class=tokenizer, **base_kwargs)
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise
        logger.debug(
            "Falling back to tokenizer arg for %s because processing_class is unsupported",
            trainer_cls.__name__,
        )
        return trainer_cls(tokenizer=tokenizer, **base_kwargs)


def get_trainer(
    args,
    trainer_cls,
    model,
    tokenizer,
    training_args,
    train_dataset,
    val_dataset,
    data_collator,
    compute_metrics,
    labels,
):
    # Suppress verbose logging from transformers and accelerate
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.INFO)
    logging.getLogger("transformers").propagate = True
    # Remove any existing handlers to ensure propagation to root logger with proper formatting
    for handler in logging.getLogger("transformers").handlers[:]:
        logging.getLogger("transformers").removeHandler(handler)
    # Also for transformers.trainer
    logging.getLogger("transformers.trainer").setLevel(logging.INFO)
    logging.getLogger("transformers.trainer").propagate = False
    for handler in logging.getLogger("transformers.trainer").handlers[:]:
        logging.getLogger("transformers.trainer").removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.getLogger("transformers.trainer").addHandler(stream_handler)

    # Disable tqdm progress bar to ensure logs are properly formatted
    training_args.disable_tqdm = True

    # Check MLflow availability
    _mlflow_callback_available = False
    try:
        from transformers.integrations import MLflowCallback

        _mlflow_callback_available = True
    except Exception:
        pass

    # Set Tensorboard logging_dir if available
    logging_dir = None
    if hasattr(args, "outputpath") and args.outputpath is not None:
        logging_dir = str(args.outputpath)
    elif (
        hasattr(args, "settings")
        and getattr(args.settings, "outputpath", None) is not None
    ):
        logging_dir = str(args.settings.outputpath)
    if logging_dir is None and hasattr(training_args, "output_dir"):
        logging_dir = training_args.output_dir

    if hasattr(training_args, "logging_dir"):
        training_args.logging_dir = logging_dir

    if args.mode == "pre-train":
        callbacks = []
        if _mlflow_callback_available and _is_mlflow_enabled(args):
            callbacks.append(MLflowCallback())

        trainer = _instantiate_trainer(
            trainer_cls,
            tokenizer,
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks if callbacks else None,
        )
    elif args.task == "regression":
        callbacks = []
        if not getattr(getattr(args, "debugging", None), "dev", False):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=getattr(
                        getattr(args, "training", None), "patience", 10
                    )
                )
            )
        if _mlflow_callback_available and _is_mlflow_enabled(args):
            callbacks.append(MLflowCallback())

        trainer = _instantiate_trainer(
            trainer_cls,
            tokenizer,
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks if callbacks else None,
            compute_metrics=compute_metrics,
        )
    elif args.task == "classification":
        class_weights = class_weight.compute_class_weight(
            "balanced", classes=np.unique(labels), y=np.array(labels)
        )
        callbacks = []
        if not getattr(getattr(args, "debugging", None), "dev", False):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=getattr(
                        getattr(args, "training", None), "patience", 10
                    )
                )
            )
        if _mlflow_callback_available and _is_mlflow_enabled(args):
            callbacks.append(MLflowCallback())

        trainer = _instantiate_trainer(
            trainer_cls,
            tokenizer,
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks if callbacks else None,
            compute_metrics=compute_metrics,
            weights=torch.tensor(class_weights).float(),
        )
    else:
        raise ValueError(f"Unsupported task: {args.task}")

    return trainer
