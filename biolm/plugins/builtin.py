"""Built-in plugin loaders used when entry points are unavailable."""

from __future__ import annotations

from typing import Callable, Dict

from ..plugin_config import PluginConfig, PluginManager


def register_saluki_plugin() -> bool:
    from transformers import BertConfig, DefaultDataCollator, PreTrainedTokenizerFast

    from .saluki.rna_cnn_dataset import RNACNNDataset
    from .saluki.rna_cnn_models import HFSaluki

    config = PluginConfig(
        model_cls_for_pretraining=None,
        model_cls_for_finetuning=HFSaluki,
        dataset_cls=RNACNNDataset,
        tokenizer_cls=PreTrainedTokenizerFast,
        datacollator_cls_for_pretraining=None,
        datacollator_cls_for_finetuning=DefaultDataCollator,
        add_special_tokens=False,
        config_cls=BertConfig,
        pretraining_required=False,
        learning_rate=1e-3,
        max_grad_norm=0.4,
        weight_decay=0.001,
        special_tokenizer_for_trainer_cls=None,
    )
    PluginManager.set_config(config)
    return True


def register_xlnet_plugin() -> bool:
    from transformers import (
        DataCollatorForPermutationLanguageModeling,
        DataCollatorWithPadding,
        XLNetConfig,
        XLNetTokenizerFast,
    )

    from .xlnet.xlnet_dataset import RNALanguageDataset
    from .xlnet.xlnet_models import (
        RNA_XLNetForSequenceClassification,
        RNA_XLNetLMHeadModel,
    )

    config = PluginConfig(
        model_cls_for_pretraining=RNA_XLNetLMHeadModel,
        model_cls_for_finetuning=RNA_XLNetForSequenceClassification,
        dataset_cls=RNALanguageDataset,
        tokenizer_cls=XLNetTokenizerFast,
        datacollator_cls_for_pretraining=DataCollatorForPermutationLanguageModeling,
        datacollator_cls_for_finetuning=DataCollatorWithPadding,
        add_special_tokens=True,
        config_cls=XLNetConfig,
        pretraining_required=True,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        weight_decay=0.0,
        special_tokenizer_for_trainer_cls=None,
    )
    PluginManager.set_config(config)
    return True


_BUILTIN_LOADERS: Dict[str, Callable[[], bool]] = {
    "saluki": register_saluki_plugin,
    "xlnet": register_xlnet_plugin,
}


def load_builtin_plugin(name: str) -> bool:
    """Attempt to load one of the built-in plugins by name."""
    if not name:
        return False
    loader = _BUILTIN_LOADERS.get(name.lower())
    if loader is None:
        return False
    return loader()


__all__ = [
    "load_builtin_plugin",
    "register_saluki_plugin",
    "register_xlnet_plugin",
]
