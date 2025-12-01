from transformers import (
    DataCollatorForPermutationLanguageModeling,
    DataCollatorWithPadding,
    XLNetConfig,
    XLNetTokenizerFast,
)

from biolm_utils.config import Config, set_config
from xlnet_dataset import RNALanguageDataset
from xlnet_models import RNA_XLNetForSequenceClassification, RNA_XLNetLMHeadModel

params = [
    RNA_XLNetLMHeadModel,  # 0
    RNA_XLNetForSequenceClassification,  # 1
    XLNetTokenizerFast,  # 2
    1e-5,  # 4
    1.0,  # 5
    0.0,  # 6
    None,  # 7
    DataCollatorForPermutationLanguageModeling,  # 8
    DataCollatorWithPadding,  # 9
    True,  # 10
    XLNetConfig,  # 11
    True,  # 12
    RNALanguageDataset,  # 13
]

config = Config(*params)
set_config(config)

from biolm_utils.biolm import main

main()
