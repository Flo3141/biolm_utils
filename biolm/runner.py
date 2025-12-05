"""Provide the run-once worker used by CrossValidator.

This module contains a factory to create the run_once function used by the
CrossValidator. The returned function implements the core per-fold logic that
previously lived as a nested function inside biolm.main(). Keeping the same
signature keeps migration simple.
"""

import logging
from typing import Any, Callable, Optional

from transformers.data.data_collator import DefaultDataCollator

from .interpret import loo_scores
from .mlflow_integration import start_mlflow_run
from .params import get_detected_ngpus
from .train_tokenizer import tokenize
from .train_utils import (
    create_reports,
    get_model_and_config,
    get_tokenizer,
    get_trainer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def make_run_fn(args, config, tokenizer, tokenizer_for_trainer, full_dataset):
    """Return a function implementing one training/validation/test run.

    Signature matches legacy code for compatibility:
      run(train_dataset, val_dataset, test_dataset, model_load_path, model_save_path, report_file, rank_file)
    """

    def run(
        train_dataset,
        val_dataset,
        test_dataset,
        model_load_path,
        model_save_path,
        report_file,
        rank_file,
    ):
        # Fast-path for tokenize mode: it doesn't use models and should return early.
        if args.mode == "tokenize":
            return tokenize(args)

        # When MLflow is enabled in the settings, start a run for this fold.
        with start_mlflow_run(model_save_path, args, config) as _mlflow:

            # Check if pre-train mode is attempted but plugin doesn't support it
            if args.mode == "pre-train" and not config.pretraining_required:
                raise ValueError(
                    f"Plugin {args.plugin} does not support pre-training. "
                    f"Set pretraining_required=True in plugin config to enable pre-training."
                )

            model_cls_map = {
                "pre-train": config.model_cls_for_pretraining,
                "fine-tune": config.model_cls_for_finetuning,
                "predict": config.model_cls_for_finetuning,
                "interpret": config.model_cls_for_finetuning,
            }
            model_cls = model_cls_map.get(args.mode)
            if model_cls is None:
                raise ValueError(f"Unknown mode: '{args.mode}'.")

            if args.mode == "pre-train":
                data_collator = config.datacollator_cls_for_pretraining(
                    tokenizer=tokenizer
                )
            else:
                # Use configured finetuning data collator, fallback to DefaultDataCollator
                data_collator_cls = (
                    config.datacollator_cls_for_finetuning or DefaultDataCollator
                )
                if (
                    hasattr(data_collator_cls, "__call__")
                    and tokenizer is not None
                    and data_collator_cls != DefaultDataCollator
                ):
                    # If it's a class that needs tokenizer, instantiate it
                    data_collator = data_collator_cls(tokenizer=tokenizer)
                else:
                    # If it's already instantiated or doesn't need tokenizer
                    data_collator = (
                        data_collator_cls()
                        if callable(data_collator_cls)
                        else data_collator_cls
                    )

            # The training path
            if args.mode in ["pre-train", "fine-tune"]:
                results, model = _train(
                    args,
                    train_dataset,
                    val_dataset,
                    data_collator,
                    model_load_path,
                    model_save_path,
                    tokenizer,
                    tokenizer_for_trainer,
                    full_dataset,
                    model_cls,
                    config,
                )

                if args.mode == "fine-tune" and test_dataset:
                    results = _test(
                        args,
                        test_dataset,
                        data_collator,
                        model_save_path,
                        report_file,
                        rank_file,
                        tokenizer,
                        tokenizer_for_trainer,
                        full_dataset,
                        model_cls,
                        config,
                        model,
                    )

                # If MLflow is active and results is a mapping, log numeric metrics
                if _mlflow is not None and isinstance(results, dict):
                    try:
                        numeric = {
                            k: float(v)
                            for k, v in results.items()
                            if isinstance(v, (int, float))
                        }
                        if numeric:
                            _mlflow.log_metrics(numeric)
                    except Exception:
                        pass

                return results

            elif args.mode == "predict":
                results = _test(
                    args,
                    test_dataset,
                    data_collator,
                    model_load_path,
                    report_file,
                    rank_file,
                    tokenizer,
                    tokenizer_for_trainer,
                    full_dataset,
                    model_cls,
                    config,
                    None,
                )

                if _mlflow is not None and isinstance(results, dict):
                    try:
                        numeric = {
                            k: float(v)
                            for k, v in results.items()
                            if isinstance(v, (int, float))
                        }
                        if numeric:
                            _mlflow.log_metrics(numeric)
                    except Exception:
                        pass
                return results

            elif args.mode == "interpret":
                res = loo_scores(
                    args=args,
                    tokenizer=tokenizer,
                    model_cls=model_cls,
                    test_dataset=test_dataset,
                    model_load_path=model_load_path,
                    output_path=model_save_path,
                    remove_first_last=config.add_special_tokens,
                )
                # interpretation outputs are not typically numeric metrics — return directly
                return res

        # tokenize handled early as fast path

    return run


def _train(
    args,
    train_dataset,
    val_dataset,
    data_collator,
    model_load_path,
    model_save_path,
    tokenizer,
    tokenizer_for_trainer,
    full_dataset,
    model_cls,
    config,
):
    from . import biolm as legacy_biolm

    metric_value, model = legacy_biolm.train(
        train_dataset,
        val_dataset,
        data_collator,
        model_load_path,
        model_save_path,
        tokenizer,
        tokenizer_for_trainer,
        full_dataset,
        model_cls,
        config,
    )

    return {"metric": metric_value}, model


def _test(
    args,
    test_dataset,
    data_collator,
    model_load_path,
    report_file,
    rank_file,
    tokenizer,
    tokenizer_for_trainer,
    full_dataset,
    model_cls,
    config,
    model,
):
    from . import biolm as legacy_biolm

    metric_value = legacy_biolm.test(
        test_dataset,
        data_collator,
        model_load_path,
        report_file,
        rank_file,
        tokenizer,
        tokenizer_for_trainer,
        full_dataset,
        model_cls,
        config,
        model,
    )

    return {"metric": metric_value}


def _run_via_cli():
    """Compat shim so `python -m biolm.runner` behaves like the CLI entry point."""
    # Import lazily to avoid Hydra side effects at module import time.
    from .cli import parse_args

    _rewrite_legacy_model_overrides()

    parse_args()


def _rewrite_legacy_model_overrides():
    """Translate legacy `model=` CLI overrides to the new plugin + model schema."""
    import sys

    rewritten = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg.startswith("model="):
            _, value = arg.split("=", 1)
            rewritten.append(f"plugin={value}")
        elif arg.startswith("model."):
            rewritten.append(f"+{arg}")
        elif arg.startswith("+model."):
            rewritten.append(arg)
        else:
            rewritten.append(arg)
    sys.argv = rewritten


if __name__ == "__main__":
    _run_via_cli()
