import json
import logging
import pickle
import sys
import tempfile
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils import class_weight

warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.",
)
from transformers import EarlyStoppingCallback

from .metrics import (
    IdentityScaler,
    LogScaler,
    compute_metrics_for_classification,
    compute_metrics_for_regression,
)
from .structured_config import TokenizationConfig

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


def _apply_model_overrides(model_config, model_overrides):
    if not model_overrides:
        return

    if hasattr(model_overrides, "items"):
        items = model_overrides.items()
    else:
        items = model_overrides.__dict__.items()

    alias_map = {
        "num_layers": "n_layer",
        "num_heads": "n_head",
        "hidden_size": "d_model",
        "intermediate_size": "d_inner",
    }

    for key, value in items:
        if value is None:
            continue
        targets = []
        if key in alias_map:
            targets.append(alias_map[key])
        targets.append(key)
        for attr in targets:
            try:
                setattr(model_config, attr, value)
            except Exception:
                # Some configs guard attribute setting; fall back to dict-style storage
                try:
                    model_config.__dict__[attr] = value
                except Exception:
                    pass


def get_tokenizer(args, tokenizer_file, tokenizer_cls, pretraining_required):

    # Support structured config and (for a short migration window) legacy flat top-level attributes for `mode`.
    mode = getattr(args, "mode", None)

    # if args.pretrainedmodel or (args.mode == "fine-tune" and pretraining_required):
    if mode == "fine-tune" and pretraining_required:
        tokenizer_config_file = (
            tokenizer_file.parent / "pre-train" / "tokenizer_config.json"
        )
        tokenizer_file = tokenizer_file.parent / "pre-train" / "tokenizer.json"
        # else:
        #     tokenizer_config_file = tokenizer_file.parent / "tokenizer_config.json"
        with open(
            tokenizer_config_file,
            "r",
        ) as ff:
            tok_config = json.load(ff)
            trunc_side = tok_config["truncation_side"]
            model_max_len = tok_config["model_max_length"]
            cls_token = tok_config["cls_token"]
            unk_token = tok_config["unk_token"]
            mask_token = tok_config["mask_token"]
            pad_token = tok_config["pad_token"]
            sep_token = tok_config["sep_token"]
            eos_token = tok_config["eos_token"]
            bos_token = tok_config["bos_token"]
        logger.info(
            f"Loaded tokenizer config from {tokenizer_config_file} "
            f"and setting it to {model_max_len} model max length"
        )
        with open(tokenizer_file, "r") as f:
            tokenizer_json = json.load(f)
        # Remove the meta data left and right correctly
        # [1] and [2] refer to the position where the sequence is isolated by means of the `columnsep`
        col = getattr(getattr(args, "data_source", None), "columnsep", "\t")
        seqpos = getattr(getattr(args, "data_source", None), "seqpos", 1)
        tokenizer_json["pre_tokenizer"]["pretokenizers"][1]["pattern"][
            "Regex"
        ] = f"([^{col}]*{col}){{{int(seqpos) - 1}}}"
        tokenizer_json["pre_tokenizer"]["pretokenizers"][2]["pattern"][
            "Regex"
        ] = f"{col}.*"
        tokensep = getattr(getattr(args, "data_source", None), "tokensep", None)
        if tokensep is not None:
            # Last position (-1) is for stripping the quotation marks.
            # We need to include the new tokensep replacement before.
            num_elements = len(tokenizer_json["normalizer"]["normalizers"])
            if (
                num_elements > 1
            ):  # this means a previous replacement with args.tokensep exists
                tokenizer_json["normalizer"]["normalizers"][-2]["pattern"][
                    "String"
                ] = tokensep
            else:  # here, we have to create a new one
                encoding = getattr(
                    getattr(args, "tokenization", None), "encoding", "atomic"
                )
                if encoding == "bpe":
                    replacement = ""
                elif encoding == "atomic":
                    replacement = " "
                pattern = (
                    {
                        "type": "Replace",
                        "pattern": {"String": tokensep},
                        "content": replacement,
                    },
                )
                tokenizer_json["normalizer"]["normalizers"].insert(0, pattern)
        # unfortunately we need to temporarily save the tokenizer as
        # some instances of TokenizerFast are deprived of the ability to load serialized tokenizers
        with tempfile.NamedTemporaryFile("r+") as tmp:
            json.dump(tokenizer_json, tmp)
            tmp.seek(0)
            tokenizer = tokenizer_cls(
                tokenizer_file=tmp.name,
                mask_token=mask_token,
                cls_token=cls_token,
                unk_token=unk_token,
                pad_token=pad_token,
                sep_token=sep_token,
                bos_token=bos_token,
                eos_token=eos_token,
                model_max_length=model_max_len,
                truncation=True,
                truncation_side=trunc_side,
            )
    else:  # pre-training data is the same as the data for tokenizing
        blocksize = getattr(getattr(args, "training", None), "blocksize", None)
        logger.info(
            f"Loading tokenizer from {tokenizer_file} and setting it to {blocksize} model max length"
        )

        # Load from HuggingFace format directory
        tokenizer_dir = tokenizer_file
        if (
            not tokenizer_dir.exists()
            or not (tokenizer_dir / "tokenizer_config.json").exists()
        ):
            parent_dir = tokenizer_file.parent
            logger.info(
                f"Tokenizer config not found in {tokenizer_dir}; trying {parent_dir}"
            )
            tokenizer_dir = parent_dir
        tokenizer_kwargs = {
            "model_max_length": blocksize,
            "truncation": True,
            "truncation_side": (
                "left"
                if getattr(getattr(args, "tokenization", None), "lefttailing", False)
                else "right"
            ),
        }

        try:
            tokenizer = tokenizer_cls.from_pretrained(
                str(tokenizer_dir),
                **tokenizer_kwargs,
            )
        except (TypeError, OSError, FileNotFoundError) as exc:
            logger.warning(
                "Tokenizer %s couldn't be loaded via from_pretrained: %s",
                tokenizer_cls.__name__,
                exc,
            )
            # Some fast tokenizers (e.g. XLNetTokenizerFast) still attempt to load a slow tokenizer
            # when using `from_pretrained`, which fails for our lightweight HuggingFace-format dumps
            # that only contain `tokenizer.json`. Fall back to constructing the tokenizer directly
            # from that JSON artifact so legacy pipelines keep working in tests.
            tokenizer_json = tokenizer_dir / "tokenizer.json"
            if not tokenizer_json.exists():
                raise

            tokenizer_config_json = tokenizer_dir / "tokenizer_config.json"
            config_overrides = {}
            if tokenizer_config_json.exists():
                with open(tokenizer_config_json, "r") as cfg:
                    tok_config = json.load(cfg)
                for key in [
                    "cls_token",
                    "unk_token",
                    "mask_token",
                    "pad_token",
                    "sep_token",
                    "bos_token",
                    "eos_token",
                    "model_max_length",
                    "truncation_side",
                ]:
                    if key in tok_config:
                        config_overrides[key] = tok_config[key]

            # Respect fallback kwargs but let config settings win when provided.
            tokenizer = tokenizer_cls(
                tokenizer_file=str(tokenizer_json),
                **{**tokenizer_kwargs, **config_overrides},
            )
            logger.warning(
                "Loaded tokenizer directly from %s due to %s", tokenizer_json, exc
            )
    tokenizer.name_or_path = tokenizer_file
    return tokenizer


def get_dataset(args, tokenizer, add_special_tokens, dataset_file, dataset_cls):
    current_blocksize = getattr(getattr(args, "training", None), "blocksize", None)
    if dataset_file.exists():
        logger.info(f"Loading dataset from {dataset_file}")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
        logger.info(f"First sample length: {len(dataset[0]['input_ids'])}")
        if (
            current_blocksize is not None
            and len(dataset[0]["input_ids"]) != current_blocksize
        ):
            logger.warning(
                f"Dataset blocksize mismatch ({len(dataset[0]['input_ids'])} vs {current_blocksize}), recreating dataset"
            )
            dataset_file.unlink(missing_ok=True)
        else:
            tokenizer = dataset.tokenizer
            return dataset

    # Create new dataset
    dataset = dataset_cls(
        tokenizer=tokenizer,
        args=args,
        add_special_tokens=add_special_tokens,
    )
    if not getattr(getattr(args, "debugging", None), "dev", False):
        logger.info(f"Saving dataset to {dataset_file}")
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset, f)
        # Save metadata
        metadata_file = dataset_file.with_suffix(".metadata.json")
        metadata = {
            "scaling_method": dataset.scaling_method,
        }
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
    if getattr(getattr(args, "debugging", None), "getdata", False):
        sys.exit()
    return dataset


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
    import logging

    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.INFO)
    logging.getLogger("transformers").propagate = True
    # Remove any existing handlers to ensure propagation to root logger with proper formatting
    for handler in logging.getLogger("transformers").handlers[:]:
        logging.getLogger("transformers").removeHandler(handler)
    # Also for transformers.trainer
    logging.getLogger("transformers.trainer").setLevel(logging.INFO)
    logging.getLogger("transformers.trainer").propagate = (
        False  # Don't propagate to avoid duplication
    )
    for handler in logging.getLogger("transformers.trainer").handlers[:]:
        logging.getLogger("transformers.trainer").removeHandler(handler)
    # Add a stream handler with the same formatter
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
    # If not found, fallback to training_args.output_dir
    if logging_dir is None and hasattr(training_args, "output_dir"):
        logging_dir = training_args.output_dir

    # Patch training_args to set logging_dir for Tensorboard
    if hasattr(training_args, "logging_dir"):
        training_args.logging_dir = logging_dir

    if args.mode == "pre-train":
        callbacks = []
        if _mlflow_callback_available and _is_mlflow_enabled(args):
            callbacks.append(MLflowCallback())

        trainer = trainer_cls(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks if callbacks else None,
            # labels=None,
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

        trainer = trainer_cls(
            model=model,
            tokenizer=tokenizer,
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

        trainer = trainer_cls(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=callbacks if callbacks else None,
            compute_metrics=compute_metrics,
            weights=torch.tensor(class_weights).float(),
        )

    return trainer


def create_reports(test_dataset, test_results, scaler, report_path, rank_path):
    seqs = [test_dataset.dataset.seq_idx[i] for i in test_dataset.indices]

    # Get the results.
    preds = test_results.predictions.squeeze()

    # Transform predictions and gold labels to original space.
    preds = scaler.inverse_transform(np.array(preds).reshape(1, -1)).squeeze()

    # Save the sequence idx, predictions and true labels to a csv file.
    if hasattr(test_dataset.dataset, "labels"):
        labels = [test_dataset.dataset.labels[i] for i in test_dataset.indices]
        labels = scaler.inverse_transform(labels).squeeze()
        # Create a file with rank deltas.
        label_tups = list(enumerate(labels))
        label_seq_tups = [x + tuple([y]) for x, y in zip(label_tups, seqs)]
        label_seq_tups = sorted(label_seq_tups, key=lambda x: x[1])
        pred_tups = list(enumerate(preds))
        pred_tups = sorted(pred_tups, key=lambda x: x[1])
        label_ranks, sorted_labels, sorted_seqs = zip(*label_seq_tups)
        pred_ranks, sorted_preds = zip(*pred_tups)
        rank_deltas = [x - y for x, y in zip(label_ranks, pred_ranks)]
        rank_df = pd.DataFrame(
            list(
                zip(
                    sorted_seqs,
                    sorted_labels,
                    label_ranks,
                    sorted_preds,
                    pred_ranks,
                    rank_deltas,
                )
            ),
            columns=["seqs", "label", "label_rank", "pred", "pred_rank", "rank_delta"],
        )
        logger.info(f"Saving test rankings to {rank_path}.")
        rank_df.to_csv(rank_path, index=False)
        report_df = pd.DataFrame(
            list(zip(seqs, labels, preds)),
            columns=["sequence", "label", "prediction"],
        )
    else:
        report_df = pd.DataFrame(
            list(zip(seqs, preds)),
            columns=["sequence", "prediction"],
        )
    logger.info(f"Saving test predictions to {report_path}.")
    report_df.to_csv(report_path, index=False)


def get_model_and_config(
    args,
    model_cls,
    model_config_cls,
    tokenizer,
    dataset,
    nlabels,
    model_load_path,
    pretraining_required,
    scaler=None,
):
    if args.mode == "pre-train" or (
        args.mode == "fine-tune"
        and (
            not pretraining_required
            or getattr(getattr(args, "training", None), "fromscratch", False)
        )
    ):
        model_config = model_cls.get_config(
            args=args,
            config_cls=model_config_cls,
            tokenizer=tokenizer,
            dataset=dataset,
            nlabels=nlabels,
        )
        _apply_model_overrides(model_config, getattr(args, "model", None))
        if not getattr(getattr(args, "training", None), "resume", False):
            if args.mode == "pre-train":
                logger.info(f"Initializing new {model_cls} model for pre-training.")
            else:
                logger.info(f"Initializing new {model_cls} model for fine-tuning.")
        else:
            logger.info(
                f"Initializing new {model_cls} model for later loading of pre-trained parameters."
            )
        model = model_cls(config=model_config)
        if args.mode == "pre-train":
            model.resize_token_embeddings(len(tokenizer))
    else:
        try:
            with open(Path(model_load_path) / "trainer_state.json") as f:
                trainer_state = json.load(f)
            n_epochs = trainer_state["log_history"][-1]["epoch"]
        except:
            pass
        try:
            n_epochs = trainer_state["epoch"]
        except:
            n_epochs = "unknown"
        model_config = model_config_cls.from_pretrained(model_load_path)
        model_config.num_labels = int(nlabels)
        model = model_cls.from_pretrained(
            model_load_path,
            config=model_config,
        )
        logger.info(
            f"Loaded {model_cls} model with weights from {model_load_path} saved on "
            f"{datetime.fromtimestamp(model_load_path.stat().st_ctime)} with {n_epochs} epochs trained."
        )
        model.scaling_method = getattr(model.config, "scaling_method", None)
    if args.mode != "pre-train":
        if scaler is not None:
            model.scaler = scaler
        else:
            # For predict/interpret, scaler should be loaded from dataset
            pass
    return model
