"""Plugin configuration."""

from transformers import DefaultDataCollator, PreTrainedTokenizerFast

from biolm.plugin_config import PluginConfig

from .dataset import MyDataset
from .models import MyModel


def get_mymodel_config():
    """Factory function called by framework at runtime."""
    config = PluginConfig(
        model_cls_for_pretraining=None,
        model_cls_for_finetuning=MyModel,
        dataset_cls=MyDataset,
        tokenizer_cls=PreTrainedTokenizerFast,
        datacollator_cls_for_pretraining=None,
        datacollator_cls_for_finetuning=DefaultDataCollator,
        add_special_tokens=False,
        pretraining_required=False,
        learning_rate=0.001,
        max_grad_norm=0.4,
        weight_decay=0.001,
        config_cls=None,
        special_tokenizer_for_trainer_cls=None,
    )
    return config
