"""Example plugin template for educational use.

This demonstrates the minimal structure expected by a plugin that integrates
with biolm: provide a factory function returning a `PluginConfig` object and
expose it via the `biolm.plugins` entry-point group.

Use this as a scaffold for students to create custom plugins that plug into
the main biolm orchestration (tokenize / fine-tune / predict / interpret).
"""

from transformers import BertConfig, DefaultDataCollator, PreTrainedTokenizerFast

from biolm.plugin_config import PluginConfig, PluginManager


def get_example_plugin_config():
    """Factory used by the `biolm.plugins` entry point."""
    # Minimal example values — replace with real model/dataset classes.
    cfg = PluginConfig(
        model_cls_for_pretraining=None,
        model_cls_for_finetuning=None,
        dataset_cls=None,
        tokenizer_cls=PreTrainedTokenizerFast,
        datacollator_cls_for_pretraining=None,
        datacollator_cls_for_finetuning=DefaultDataCollator,
        add_special_tokens=False,
        config_cls=BertConfig,
        pretraining_required=False,
        learning_rate=1e-4,
        max_grad_norm=1.0,
        weight_decay=0.0,
        special_tokenizer_for_trainer_cls=None,
    )

    # Optional but explicit: store active plugin config in manager.
    PluginManager.set_config(cfg)
    return cfg
