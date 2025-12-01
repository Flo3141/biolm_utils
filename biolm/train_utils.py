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
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.utils import class_weight

warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.",
)
from transformers import EarlyStoppingCallback

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


class LogScaler:
    def fit_transform(self, data):
        return np.log(data)

    def inverse_transform(self, data):
        return np.exp(data)


# Not pretty but complies best with the rest of the code.
class IdentityScaler:
    def fit_transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


def compute_metrics_for_regression(dataset, savepath):
    def _compute_metrics(pred):
        logits, labels = pred
        logits = logits.squeeze().tolist()
        labels = labels.squeeze().tolist()
        mse = mean_squared_error(labels, logits)
        spearman_rho, _ = spearmanr(logits, labels)
        return {
            "mse": mse,
            "spearman rho": spearman_rho,
        }

    return _compute_metrics


def compute_metrics_for_classification(dataset, savepath):
    def _compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="macro"
        )
        acc = accuracy_score(labels, preds)
        # target_names = [dataset.LE.classes_[x] for x in names]
        target_names = dataset.LE.classes_.tolist()
        # used_labels = list(set(preds).union(set(labels)))
        used_labels = list(range(len(target_names)))
        report = classification_report(
            labels,
            preds,
            output_dict=True,
            target_names=target_names,
            labels=used_labels,
            zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(savepath / "classification_report.csv")
        logging.info(
            classification_report(
                labels,
                preds,
                target_names=target_names,
                labels=used_labels,
                zero_division=0,
            )
        )
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    return _compute_metrics


def get_tokenizer(args, tokenizer_file, tokenizer_cls, pretraining_required):

    # Support structured config and (for a short migration window) legacy flat top-level attributes for `mode`.
    mode = getattr(args, "mode", None)

    # if args.pretrainedmodel or (args.mode == "fine-tune" and pretraining_required):
    if mode == "fine-tune" and pretraining_required:
        tokenizer_config_file = tokenizer_file / "pre-train" / "tokenizer_config.json"
        tokenizer_file = tokenizer_file / "pre-train" / "tokenizer.json"
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
        tokenizer = tokenizer_cls.from_pretrained(
            str(tokenizer_file),
            model_max_length=blocksize,
            truncation=True,
            truncation_side=(
                "left"
                if getattr(getattr(args, "tokenization", None), "lefttailing", False)
                else "right"
            ),
        )
    tokenizer.name_or_path = tokenizer_file
    return tokenizer


def get_dataset(args, tokenizer, add_special_tokens, dataset_file, dataset_cls):
    metadata_file = dataset_file.with_suffix(".metadata.json")
    current_blocksize = getattr(getattr(args, "training", None), "blocksize", None)
    tokenization = args.tokenization or TokenizationConfig()
    current_vocabsize = getattr(tokenization, "vocabsize", None)
    current_encoding = getattr(tokenization, "encoding", None)
    current_minfreq = getattr(tokenization, "minfreq", None)
    current_filepath = getattr(getattr(args, "data_source", None), "filepath", None)

    recreate = False
    if dataset_file.exists():
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            changed_params = []
            if metadata.get("blocksize") != current_blocksize:
                changed_params.append(
                    f"blocksize: {metadata.get('blocksize')} -> {current_blocksize}"
                )
            if metadata.get("vocabsize") != current_vocabsize:
                changed_params.append(
                    f"vocabsize: {metadata.get('vocabsize')} -> {current_vocabsize}"
                )
            if metadata.get("encoding") != current_encoding:
                changed_params.append(
                    f"encoding: {metadata.get('encoding')} -> {current_encoding}"
                )
            if metadata.get("minfreq") != current_minfreq:
                changed_params.append(
                    f"minfreq: {metadata.get('minfreq')} -> {current_minfreq}"
                )
            if str(metadata.get("filepath")) != str(current_filepath):
                changed_params.append(
                    f"filepath: {metadata.get('filepath')} -> {current_filepath}"
                )
            if changed_params:
                logger.warning(
                    f"Dataset parameters changed, recreating dataset. Changed: {', '.join(changed_params)}"
                )
                recreate = True
        else:
            logger.warning(
                "No metadata found for dataset, recreating to ensure compatibility"
            )
            recreate = True

    # Log current dataset parameters
    logger.info(
        f"Dataset parameters: blocksize={current_blocksize}, vocabsize={current_vocabsize}, "
        f"encoding={current_encoding}, minfreq={current_minfreq}, filepath={current_filepath}"
    )

    if recreate:
        dataset_file.unlink(missing_ok=True)
        metadata_file.unlink(missing_ok=True)
    else:
        if not dataset_file.exists():
            logger.warning(f"Dataset file {dataset_file} disappeared, recreating")
            recreate = True
        else:
            logger.info(f"Loading dataset from {dataset_file}")
            with open(dataset_file, "rb") as f:
                dataset = pickle.load(f)
            logger.info(f"First sample length: {len(dataset[0]['input_ids'])}")
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
        metadata = {
            "blocksize": current_blocksize,
            "vocabsize": current_vocabsize,
            "encoding": current_encoding,
            "minfreq": current_minfreq,
            "filepath": str(current_filepath),
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
    if args.mode != "pre-train":
        if scaler is not None:
            model.scaler = scaler
        else:
            with open(Path(model_load_path) / "scaler.pkl", "rb") as scaler_file:
                model.scaler = pickle.load(scaler_file)
    return model
