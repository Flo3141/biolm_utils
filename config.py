from __future__ import annotations

from dataclasses import dataclass

from transformers import DefaultDataCollator, PretrainedConfig, PreTrainedModel
from transformers.image_processing_utils import ImageProcessingMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from biolm_utils.rna_datasets import RNABaseDataset


@dataclass
class Config:
    MODEL_CLS: PreTrainedModel
    TOKENIZER_CLS: PreTrainedTokenizerBase
    DATASET_CLS: RNABaseDataset
    LEARNINGRATE: float
    MAX_GRAD_NORM: float
    WEIGHT_DECAY: float
    SPECIAL_TOKENIZER_FOR_TRAINER_CLS: ImageProcessingMixin
    DATACOLLATOR_CLS_FOR_PRETRAINING: DefaultDataCollator
    DATACOLLATOR_CLS_FOR_FINETUNING: DefaultDataCollator
    ADD_SPECIAL_TOKENS: bool
    CONFIG_CLS: PretrainedConfig
    PRETRAINING_REQUIRED: bool


_config: Config | None = None


def get_config():
    global _config

    if _config is None:
        raise Exception("Config not initialized")
    return _config


def set_config(config: Config):
    global _config
    _config = config
