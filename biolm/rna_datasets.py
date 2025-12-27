import json
import logging
import pickle
import re
import tempfile

import numpy as np
import pandas as pd
import transformers
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

from .train_utils import IdentityScaler, LogScaler


class RNABaseDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        args,
        add_special_tokens,
        scaler=None,  # Add scaler as an optional parameter
    ):
        self.tokenizer = tokenizer

        self.args = args
        self.scaler = scaler  # Store the scaler in the dataset
        self.scaling_method = getattr(
            args, "scaling", "identity"
        )  # Store scaling method
        # Prepare helpers and resolved attributes for structured config
        data_source = getattr(args, "data_source", None)
        tokenization = getattr(args, "tokenization", None)
        settings = getattr(args, "settings", None)

        def ds_get(key, default=None):
            if data_source is not None and hasattr(data_source, key):
                return getattr(data_source, key)
            return getattr(args, key, default)

        def tk_get(key, default=None):
            if tokenization is not None and hasattr(tokenization, key):
                return getattr(tokenization, key)
            return getattr(args, key, default)

        def settings_get(key, default=None):
            if settings is not None:
                dp = getattr(settings, "data_pre_processing", None)
                if isinstance(dp, dict) and key in dp:
                    return dp.get(key)
            return getattr(args, key, default)

        self.nspecs = 0
        self.specs = None
        self.OHE = None
        if getattr(args, "task", None) == "classification":
            self.LE = LabelEncoder()

        # Resolve filepath (prefer nested data_source.filepath)
        filepath = getattr(getattr(args, "data_source", None), "filepath", None)
        if not filepath:
            filepath = getattr(args, "filepath", None)

        with open(filepath, encoding="utf-8") as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
            if ds_get("stripheader", False):
                lines = lines[1:]

        # We'll save the original input data lines for later reference.
        self.lines = lines
        columnsep = ds_get("columnsep", "\t")
        idpos = ds_get("idpos", None)
        self.seq_idx = [x.split(columnsep)[idpos - 1].strip('"') for x in self.lines]

        tokensep = ds_get("tokensep", None)
        encoding = tk_get("encoding", "atomic")
        self.join_str = "" if tokensep is None or encoding == "bpe" else tokensep

        # Expose frequently-used config options on the instance for use in
        # helper methods (backwards compatible with legacy flat top-level attributes).
        self.encoding = encoding

        # Normalize and pre-trokenize to obtain the sequences.
        normalized_seqs = [
            tokenizer.backend_tokenizer.normalizer.normalize_str(x) for x in lines
        ]
        # Keep a copy of normalized lines for helper methods that expect them
        self.normalized_lines = normalized_seqs

        logging.info("Normalizing sequences finished.")

        specifiersep = ds_get("specifiersep", None)
        if specifiersep is not None:
            with open(tokenizer.name_or_path, "r") as f:
                tokenizer_json = json.load(f)
            tokenizer_json["normalizer"]["normalizers"].pop(-3)
            tokenizer_json["pre_tokenizer"]["pretokenizers"].pop(-1)
            with tempfile.NamedTemporaryFile("r+") as tmp:
                json.dump(tokenizer_json, tmp)
                tmp.seek(0)
                spec_tokenizer = tokenizer.__class__(
                    tokenizer_file=tmp.name,
                    mask_token="[MASK]",
                    cls_token="[CLS]",
                    unk_token="[UNK]",
                    pad_token="[PAD]",
                    sep_token="[SEP]",
                    bos_token="[BOS]",
                    eos_token="[EOS]",
                    model_max_length=getattr(
                        getattr(args, "training", None),
                        "blocksize",
                        getattr(args, "blocksize", None),
                    ),
                    truncation=True,
                    truncation_side=(
                        "left" if tk_get("lefttailing", False) else "right"
                    ),
                )
            spec_normalized_seqs = [
                spec_tokenizer.backend_tokenizer.normalizer.normalize_str(x)
                for x in lines
            ]
            spec_pre_tokenized_seqs = [
                spec_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(x)[0][0]
                for x in spec_normalized_seqs
            ]
            logging.info("Spec normalizing/tokenizing sequences finished.")
            self.specs = [
                [
                    re.findall(rf"(?<={specifiersep})[^{specifiersep}]+", y)
                    for y in x.split(" ")
                ]
                for x in spec_pre_tokenized_seqs
            ]
            self.nspecs = len(max(max([x for x in y]) for y in self.specs))
            self.specs = [
                np.array(
                    [
                        np.pad(
                            list(map(float, y)),
                            (0, self.nspecs - len(y)),
                            constant_values=0.0,
                        )
                        for y in x[: tokenizer.model_max_length]
                    ]
                )
                for x in self.specs
            ]
            self.specs = [
                np.pad(
                    x,
                    ((0, tokenizer.model_max_length - x.shape[0]), (0, 0)),
                    constant_values=0,
                )
                for x in self.specs
            ]

        pre_tokenized_seqs = [
            tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(x)
            for x in normalized_seqs
        ]
        logging.info("Pre-tokenizing sequences finished.")
        self.seqs = [
            self.join_str.join([y[0] for y in x]).replace("Ġ", "")
            for x in pre_tokenized_seqs
        ]

        # Set the log level to error to supress the warning that we will
        # actually tokenize sequences which are longer than the model's max sequence length.
        log_lvl = transformers.utils.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        # Evaluate the length of the tokenized unmanipulated/untruncated data.
        if self.encoding in ["3mer", "5mer"]:
            self.seqs = self.tokenize_kmers(self.seqs, args)
            raw_encodings = self.tokenizer(
                self.seqs,
                add_special_tokens=False,
                truncation=False,
                is_split_into_words=True,
            )["input_ids"]
        else:
            raw_encodings = self.tokenizer(
                self.seqs, add_special_tokens=False, truncation=False
            )["input_ids"]
        logging.info("Raw tokenizing sequences finished.")
        # restore log lvl
        transformers.logging.set_verbosity(log_lvl)
        self.tokenized_seqs = [
            self.tokenizer.convert_ids_to_tokens(x) for x in raw_encodings
        ]
        logging.info("Re-builiding tokenized sequences finished.")
        self.tokenized_seqs = [
            list(map(lambda x: x.replace("Ġ", ""), y)) for y in self.tokenized_seqs
        ]
        self.tokenized_seqs = [[x for x in y if x != ""] for y in self.tokenized_seqs]

        encodings = self.tokenizer(
            self.seqs,
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding="max_length",
            is_split_into_words=self.encoding in ["3mer", "5mer"],
        )["input_ids"]
        logging.info("Encoding sequences finished.")

        # Use the tokenizer's `model_max_length` for padding/truncation.
        # Prefer the tokenizer attribute; if absent, fall back to the
        # training.blocksize in the config (still safe and minimal).
        # The plugin is responsible for setting this invariant (Saluki sets
        # it to 12288). If neither is set, raise a clear error.
        max_len = getattr(self.tokenizer, "model_max_length", None)
        if max_len is None:
            max_len = getattr(
                getattr(self.args, "training", None),
                "blocksize",
                getattr(self.args, "blocksize", None),
            )
            if max_len is None:
                raise ValueError(
                    "tokenizer.model_max_length is not set and args.training.blocksize is not set. Plugin must set a blocksize (e.g. Saluki requires 12288)."
                )
            else:
                logging.warning(
                    "tokenizer.model_max_length unset — using args.training.blocksize=%s",
                    max_len,
                )

        pad_id = int(self.tokenizer.pad_token_id)
        padded_encodings = []
        # Ensure max_len is a native Python int to avoid numpy pad type issues
        try:
            max_len = int(max_len)
        except Exception:
            raise ValueError("tokenizer.model_max_length must be an integer")

        for e in encodings:
            # convert to numpy array for safe padding/truncation
            arr = np.asarray(e, dtype=np.int64)
            cur_len = int(arr.shape[0])
            need = max_len - cur_len
            if need > 0:
                # Create a new array filled with pad_id and copy existing values.
                new = np.full((max_len,), int(pad_id), dtype=np.int64)
                new[:cur_len] = arr
                arr = new
            elif need < 0:
                arr = arr[:max_len]
            padded_encodings.append(arr.tolist())

        self.examples = np.array([{"input_ids": e} for e in padded_encodings])

        # TODO: Make this a model attribute
        # Set up the scaler
        scaling = getattr(
            getattr(args, "training", None), "scaling", getattr(args, "scaling", None)
        )
        if scaling == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling == "standard":
            self.scaler = StandardScaler()
        elif scaling == "log":
            self.scaler = LogScaler()
        else:
            # Not so pretty, but is currently the fastest adaptation for no scaling
            self.scaler = IdentityScaler()

        # get the labels and seq idx for each task.
        mode = getattr(args, "mode", None)
        labelpos = ds_get("labelpos", getattr(args, "labelpos", None))
        weightpos = getattr(args, "weightpos", None)
        if mode in ["fine-tune", "predict", "interpret"] and labelpos is not None:
            if args.task == "regression":
                labels = [
                    float(x.split(columnsep)[labelpos - 1].strip('"'))
                    for x in self.lines
                ]
                if weightpos is not None:
                    qualities = [x.split(",")[weightpos].strip('"') for x in self.lines]
                    qual_dict = {"STRONG": 1.0, "GOOD": 0.75, "WEAK": 0.5, "POOR": 0.25}
                    self.qualities = [qual_dict[x] for x in qualities]

                self.labels = self.scaler.fit_transform(
                    np.array(labels).reshape(-1, 1).astype(float)
                )
            elif getattr(args, "task", None) == "classification":
                labels = [
                    x.split(columnsep)[labelpos - 1].strip('"') for x in self.lines
                ]
                self.labels = self.LE.fit_transform(labels)

            # update self.examples with labels (and quality weights).
            if weightpos is None:
                for l, e in zip(self.labels, self.examples):
                    e.update({"labels": l})
            elif getattr(args, "data", None) == "protein":
                for l, e, q in zip(self.labels, self.examples, self.qualities):
                    e.update({"labels": l})
                    e.update({"qualities": q})

        # Apply scaling to target values if scaler is provided
        target_values = ds_get("target_values", None)
        if target_values is not None and self.scaler is not None:
            self.target_values = self.scaler.fit_transform(target_values)
        else:
            self.target_values = (
                target_values  # Use raw values if no scaler is provided
            )

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

    @staticmethod
    def tokenize_kmers(lines, args):
        """
        This method is also called when training tokenizers with `learn_tokenizer.py`,
        so we make it static.
        """
        split_lines = list()
        # Support both the structured BioLMConfig and legacy flat top-level attributes
        tokenization = getattr(args, "tokenization", None)
        data_source = getattr(args, "data_source", None)
        settings = getattr(args, "settings", None)

        def tk_get(key, default=None):
            if tokenization is not None and hasattr(tokenization, key):
                return getattr(tokenization, key)
            return getattr(args, key, default)

        def ds_get(key, default=None):
            if data_source is not None and hasattr(data_source, key):
                return getattr(data_source, key)
            return getattr(args, key, default)

        def settings_get(key, default=None):
            if settings is not None:
                dp = getattr(settings, "data_pre_processing", None)
                if isinstance(dp, dict) and key in dp:
                    return dp.get(key)
            return getattr(args, key, default)

        if tk_get("encoding", getattr(args, "encoding", None)) == "3mer":
            pattern = "s|[^xs]{3}|[^xs]{2}x[^xs]|[^xs]x[^xs]{2}|x"
        else:
            pattern = "s|[^xs]{5}|[^xs]{4}x[^xs]|[^xs]x[^xs]{4}||[^xs]{2}x[^xs]{3}|[^xs]{3}x[^xs]{2}|x"
        for line in lines:
            from .tokenization_helpers import parse_atomic_replacements

            atomicreplacements = tk_get(
                "atomicreplacements", getattr(args, "atomicreplacements", None)
            )
            rep = parse_atomic_replacements(atomicreplacements)
            if rep is not None:
                for k, v in rep.items():
                    tokensep = ds_get("tokensep", getattr(args, "tokensep", None))
                    if tokensep is not None:
                        line = line.replace(
                            f"{tokensep}{k}{tokensep}",
                            f"{tokensep}{v}{tokensep}",
                        )
                        line = line.replace(f"\n{k}{tokensep}", f"\n{v}{tokensep}")
                        line = line.replace(f"{tokensep}{k}\n", f"{tokensep}{v}\n")
                    else:
                        line = line.replace(k, v)
            centertoken = settings_get(
                "centertoken", getattr(args, "centertoken", None)
            )
            cds_end_pos = [i for i, x in enumerate(line) if x == centertoken]
            if not cds_end_pos:
                split_lines.append(re.findall(pattern, line))
                continue
            else:
                cds_end_pos = cds_end_pos[0]
                front = line[:cds_end_pos]
                back = line[cds_end_pos + 1 :]
                split_front = re.findall(pattern, front[::-1])[::-1]
                split_front = [x[::-1] for x in split_front]
                split_back = re.findall(pattern, back)
                split_line = split_front + ["s"] + split_back
                split_lines.append(split_line)
        return split_lines

    def __getitem__(example):
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
