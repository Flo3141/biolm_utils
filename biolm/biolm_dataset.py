"""Canonical base dataset API for BioLM plugins.

Plugin authors should inherit from :class:`BioLMDataset`.

This dataset implements common BioLM preprocessing:
- read and parse input lines
- normalize and pre-tokenize sequences
- optional specifier extraction
- tokenization + padding/truncation
- optional label/scaler handling for supervised modes
"""

from __future__ import annotations

import json
import logging
import pickle
import re
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence, TypeVar

import numpy as np
import pandas as pd
import transformers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

from .train_utils import IdentityScaler, LogScaler


T = TypeVar("T")


class BioLMDataset(Dataset):
    """Base dataset implementation for BioLM plugins."""

    def __init__(
        self,
        tokenizer: Any,
        args: Any,
        add_special_tokens: bool,
        scaler: Optional[Any] = None,
    ):
        self.tokenizer = tokenizer
        self.args = args

        # Resolve max length early and use it consistently.
        # This avoids accidentally padding to an absurd HuggingFace default (e.g. 1e30)
        # when `tokenizer.model_max_length` is not configured.
        self.max_len = self._resolve_max_len(default_if_absurd=12288)

        self.nspecs = 0
        self.specs = None
        self.OHE = None

        self.LE: Optional[LabelEncoder] = None
        if getattr(args, "task", None) == "classification":
            self.LE = LabelEncoder()

        # Load raw lines
        filepath = self._resolve_filepath(args)
        self.lines = self._read_lines(
            filepath, stripheader=self._ds_get("stripheader", False)
        )

        # Basic parsing config
        self.columnsep = self._ds_get("columnsep", "\t")
        self.idpos = self._ds_get("idpos", None)
        if self.idpos is None:
            raise ValueError("data_source.idpos must be set (1-indexed)")
        self.seq_idx = [
            line.split(self.columnsep)[self.idpos - 1].strip('"') for line in self.lines
        ]

        tokensep = self._ds_get("tokensep", None)
        self.encoding = self._tk_get("encoding", "atomic")
        self.join_str = "" if tokensep is None or self.encoding == "bpe" else tokensep

        # Normalize, pre-tokenize
        self.normalized_lines = self._maybe_parallel_map(
            tokenizer.backend_tokenizer.normalizer.normalize_str,
            self.lines,
            stage="normalize",
        )
        logging.info("Normalizing sequences finished.")

        self._maybe_build_specs()
        self.seqs = self._pretokenize_sequences(self.normalized_lines)

        # Tokenized strings for logging/stats are computed lazily.
        self._tokenized_seqs: Optional[List[List[str]]] = None

        # Encode + pad/truncate
        encodings = self.tokenizer(
            self.seqs,
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
        )["input_ids"]
        logging.info("Encoding sequences finished.")
        # `max_length` should already guarantee correct shapes; keep a small safety net.
        if any(len(e) != self.max_len for e in encodings):
            encodings = self._pad_truncate(encodings, self.max_len)
        self.examples = np.array([{"input_ids": ids} for ids in encodings])

        # Scaler handling
        self.scaler = scaler
        self.scaling_method = self._resolve_scaling_method()
        if self.scaler is None:
            self.scaler = self._make_scaler(self.scaling_method)

        self._maybe_attach_labels()
        self._maybe_attach_target_values()

    # -----------------
    # Config access
    # -----------------
    def _ds_get(self, key: str, default: Any = None) -> Any:
        data_source = getattr(self.args, "data_source", None)
        if data_source is not None and hasattr(data_source, key):
            return getattr(data_source, key)
        return getattr(self.args, key, default)

    def _tk_get(self, key: str, default: Any = None) -> Any:
        tokenization = getattr(self.args, "tokenization", None)
        if tokenization is not None and hasattr(tokenization, key):
            return getattr(tokenization, key)
        return getattr(self.args, key, default)

    def _resolve_filepath(self, args: Any) -> str:
        filepath = getattr(getattr(args, "data_source", None), "filepath", None)
        if not filepath:
            filepath = getattr(args, "filepath", None)
        if not filepath:
            raise ValueError("data_source.filepath (or filepath) must be set")
        return str(filepath)

    # -----------------
    # IO / preprocessing
    # -----------------
    def _read_lines(self, filepath: str, stripheader: bool) -> List[str]:
        with open(filepath, encoding="utf-8") as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        if stripheader and lines:
            lines = lines[1:]
        return lines

    def _pretokenize_sequences(self, normalized_lines: Sequence[str]) -> List[str]:
        pre_tokenized = self._maybe_parallel_map(
            self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str,
            normalized_lines,
            stage="pretokenize",
        )
        logging.info("Pre-tokenizing sequences finished.")
        return [
            self.join_str.join([y[0] for y in x]).replace("Ġ", "")
            for x in pre_tokenized
        ]

    def _tokenize_raw_strings(self, seqs: Sequence[str]) -> List[List[str]]:
        log_lvl = transformers.utils.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        raw_encodings = self.tokenizer(
            seqs, add_special_tokens=False, truncation=False
        )["input_ids"]
        transformers.logging.set_verbosity(log_lvl)

        tokenized = [self.tokenizer.convert_ids_to_tokens(x) for x in raw_encodings]
        tokenized = [[t.replace("Ġ", "") for t in toks] for toks in tokenized]
        tokenized = [[t for t in toks if t] for toks in tokenized]
        logging.info("Raw tokenizing sequences finished.")
        return tokenized

    @property
    def tokenized_seqs(self) -> List[List[str]]:
        """Tokenized sequences for logging/statistics.

        This is intentionally computed lazily because it can be expensive and is
        not needed for most training/prediction runs.
        """

        if self._tokenized_seqs is None:
            self._tokenized_seqs = self._tokenize_raw_strings(self.seqs)
        return self._tokenized_seqs

    # -----------------
    # Specifier support
    # -----------------
    def _maybe_build_specs(self) -> None:
        specifiersep = self._ds_get("specifiersep", None)
        if specifiersep is None:
            return

        spec_tokenizer = self._build_spec_tokenizer()
        spec_normalized = self._maybe_parallel_map(
            spec_tokenizer.backend_tokenizer.normalizer.normalize_str,
            self.lines,
            stage="spec_normalize",
        )
        spec_pre_tok = self._maybe_parallel_map(
            spec_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str,
            spec_normalized,
            stage="spec_pretokenize",
        )
        spec_pre = [x[0][0] for x in spec_pre_tok]
        logging.info("Spec normalizing/tokenizing sequences finished.")
    def _dataset_num_workers(self) -> int:
        """Determine the number of worker threads for dataset preprocessing.

        This is intentionally conservative and bounded. Order is preserved.

        Overrides:
            - `BIOLM_DATASET_NUM_WORKERS` env var (int)
            - `args.settings.data_pre_processing.num_workers` (int)
            - `args.settings.data_pre_processing["num_workers"]` (int)
        """

        env = os.getenv("BIOLM_DATASET_NUM_WORKERS")
        if env is not None:
            try:
                return max(0, int(env))
            except Exception:
                return 0

        settings = getattr(self.args, "settings", None)
        dp = getattr(settings, "data_pre_processing", None) if settings is not None else None
        if dp is not None:
            if hasattr(dp, "num_workers"):
                try:
                    return max(0, int(getattr(dp, "num_workers")))
                except Exception:
                    return 0
            if isinstance(dp, dict) and "num_workers" in dp:
                try:
                    return max(0, int(dp.get("num_workers")))
                except Exception:
                    return 0

        # Auto-enable only for larger datasets to avoid overhead.
        n = len(getattr(self, "lines", []) or [])
        if n < 2000:
            return 0
        return max(1, min(8, (os.cpu_count() or 1)))

    def _maybe_parallel_map(
        self,
        fn: Callable[[T], Any],
        items: Sequence[T],
        stage: str,
    ) -> List[Any]:
        """Apply `fn` over `items`, optionally in a thread pool.

        Uses `executor.map` to preserve input order.
        """

        workers = self._dataset_num_workers()
        if workers <= 1:
            return [fn(x) for x in items]
        logging.info("Dataset preprocessing (%s) using %s workers", stage, workers)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, items))

        specs = [
            [
                re.findall(rf"(?<={specifiersep})[^{specifiersep}]+", y)
                for y in x.split(" ")
            ]
            for x in spec_pre
        ]

        self.nspecs = len(max(max([x for x in y]) for y in specs))
        specs_arr = [
            np.array(
                [
                    np.pad(
                        list(map(float, y)),
                        (0, self.nspecs - len(y)),
                        constant_values=0.0,
                    )
                    for y in x[: self.max_len]
                ]
            )
            for x in specs
        ]

        self.specs = [
            np.pad(
                x,
                ((0, self.max_len - x.shape[0]), (0, 0)),
                constant_values=0,
            )
            for x in specs_arr
        ]

    def _build_spec_tokenizer(self) -> Any:
        # This mirrors the legacy behaviour (strip spec-related components).
        with open(self.tokenizer.name_or_path, "r") as f:
            tokenizer_json = json.load(f)
        tokenizer_json["normalizer"]["normalizers"].pop(-3)
        tokenizer_json["pre_tokenizer"]["pretokenizers"].pop(-1)

        with tempfile.NamedTemporaryFile("r+") as tmp:
            json.dump(tokenizer_json, tmp)
            tmp.seek(0)
            return self.tokenizer.__class__(
                tokenizer_file=tmp.name,
                mask_token="[MASK]",
                cls_token="[CLS]",
                unk_token="[UNK]",
                pad_token="[PAD]",
                sep_token="[SEP]",
                bos_token="[BOS]",
                eos_token="[EOS]",
                model_max_length=self.max_len,
                truncation=True,
                truncation_side=(
                    "left" if self._tk_get("lefttailing", False) else "right"
                ),
            )

    # -----------------
    # Tokenization + padding
    # -----------------
    def _resolve_max_len(self, default_if_absurd: Optional[int]) -> int:
        max_len = getattr(self.tokenizer, "model_max_length", None)
        if max_len is None or (isinstance(max_len, int) and max_len > 1000000):
            max_len = getattr(
                getattr(self.args, "training", None),
                "blocksize",
                getattr(self.args, "blocksize", None),
            )
        if max_len is None or (isinstance(max_len, int) and max_len > 1000000):
            if default_if_absurd is None:
                raise ValueError(
                    "tokenizer.model_max_length is unset/invalid; set tokenizer.model_max_length or training.blocksize"
                )
            logging.warning(
                "Forcing max_len=%s due to invalid tokenizer.model_max_length.",
                default_if_absurd,
            )
            max_len = default_if_absurd
        try:
            return int(max_len)
        except Exception as e:
            raise ValueError("tokenizer.model_max_length must be an integer") from e

    def _pad_truncate(
        self, encodings: Sequence[Sequence[int]], max_len: int
    ) -> List[List[int]]:
        pad_id = int(self.tokenizer.pad_token_id)
        padded: List[List[int]] = []
        for e in encodings:
            arr = np.asarray(e, dtype=np.int64)
            cur_len = int(arr.shape[0])
            need = max_len - cur_len
            if need > 0:
                new = np.full((max_len,), int(pad_id), dtype=np.int64)
                new[:cur_len] = arr
                arr = new
            elif need < 0:
                arr = arr[:max_len]
            padded.append(arr.tolist())
        return padded

    # -----------------
    # Labels/scaling
    # -----------------
    def _resolve_scaling_method(self) -> str:
        scaling = getattr(
            getattr(self.args, "training", None),
            "scaling",
            getattr(self.args, "scaling", None),
        )
        return scaling or "identity"

    def _make_scaler(self, scaling_method: str) -> Any:
        if scaling_method == "minmax":
            return MinMaxScaler()
        if scaling_method == "standard":
            return StandardScaler()
        if scaling_method == "log":
            return LogScaler()
        return IdentityScaler()

    def _maybe_attach_labels(self) -> None:
        mode = getattr(self.args, "mode", None)
        labelpos = self._ds_get("labelpos", getattr(self.args, "labelpos", None))
        weightpos = getattr(self.args, "weightpos", None)
        if mode not in ["fine-tune", "predict", "interpret"] or labelpos is None:
            return

        if getattr(self.args, "task", None) == "regression":
            labels = [
                float(line.split(self.columnsep)[labelpos - 1].strip('"'))
                for line in self.lines
            ]

            if weightpos is not None:
                qualities = [
                    line.split(",")[weightpos].strip('"') for line in self.lines
                ]
                qual_dict = {"STRONG": 1.0, "GOOD": 0.75, "WEAK": 0.5, "POOR": 0.25}
                self.qualities = [qual_dict[x] for x in qualities]

            y = np.array(labels).reshape(-1, 1).astype(float)
            # Fit scaler during fine-tune; reuse during predict/interpret when provided.
            if mode == "fine-tune" or self.scaler is None:
                self.labels = self.scaler.fit_transform(y)
            else:
                self.labels = self.scaler.transform(y)

        elif getattr(self.args, "task", None) == "classification":
            if self.LE is None:
                self.LE = LabelEncoder()
            labels = [
                line.split(self.columnsep)[labelpos - 1].strip('"')
                for line in self.lines
            ]
            self.labels = self.LE.fit_transform(labels)
        else:
            return

        if weightpos is None:
            for l, e in zip(self.labels, self.examples):
                e.update({"labels": l})
        elif getattr(self.args, "data", None) == "protein":
            for l, e, q in zip(self.labels, self.examples, self.qualities):
                e.update({"labels": l})
                e.update({"qualities": q})

    def _maybe_attach_target_values(self) -> None:
        target_values = self._ds_get("target_values", None)
        if target_values is None:
            self.target_values = None
            return
        if self.scaler is None:
            self.target_values = target_values
            return
        if getattr(self.args, "mode", None) == "fine-tune":
            self.target_values = self.scaler.fit_transform(target_values)
        else:
            self.target_values = self.scaler.transform(target_values)

    def __len__(self):
        return len(self.examples)

    def log_raw_data(self):
        raw_data_df = pd.DataFrame()
        raw_data_df["seq"] = self.tokenized_seqs
        raw_data_df["lengths"] = raw_data_df["seq"].apply(lambda x: len(x))

        logging.info("Dataset raw statistics:")
        logging.info(raw_data_df.describe(include="all"))

    def log_data(self):
        data_df = pd.DataFrame()
        data_df["seq"] = [
            self.tokenizer.convert_ids_to_tokens(x["input_ids"]) for x in self.examples
        ]
        data_df["lengths"] = data_df["seq"].apply(lambda x: len(x))
        if getattr(self.args, "mode", None) in ["fine-tune", "predict", "interpret"]:
            data_df["labels"] = self.labels
        logging.info("Dataset statistics after truncation and adding special tokens:")
        logging.info(data_df.describe(include="all"))

    # Removed tokenize_kmers and all 3mer/5mer support (no longer used)

    def __getitem__(self, index: int):
        raise NotImplementedError

    def save(self, filepath):
        """Save the dataset along with the scaler."""
        data = {
            "lines": self.lines,
            "scaler": self.scaler,  # Save the scaler
            "scaling_method": self.scaling_method,  # Save scaling method
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath, tokenizer, args, add_special_tokens):
        """Load the dataset along with the scaler."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        dataset = cls(
            tokenizer=tokenizer,
            args=args,
            add_special_tokens=add_special_tokens,
            scaler=data.get("scaler"),  # Load the scaler
        )
        dataset.lines = data["lines"]
        dataset.scaling_method = data.get(
            "scaling_method", "identity"
        )  # Load scaling method
        return dataset


# Deprecated alias kept for existing code; prefer BioLMDataset.
RNABaseDataset = BioLMDataset

__all__ = ["BioLMDataset", "RNABaseDataset"]
