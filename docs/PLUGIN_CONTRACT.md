# BioLM Plugin Contract

This document summarizes what a BioLM plugin must expose so the framework can load it reliably.

## Entry point
- Register an entry point in the `biolm.plugins` group. The entry point name is the plugin identifier users set in configs (e.g., `rna_saluki_cnn`).
- The entry point must resolve to a **factory function** that returns either:
  - a plain `dict` matching `biolm.config.Config` fields, or
  - an object with attributes matching those fields (e.g., a `PluginConfig` dataclass).

## Required fields
- `model_cls_for_finetuning`: torch `nn.Module` used for fine-tune/predict/interpret.
- `dataset_cls`: PyTorch `Dataset` that consumes `data_source` settings (idpos/seqpos/labelpos, delimiter) and emits items compatible with the model.
- `config_cls`: model configuration class (dataclass/config object) used by `get_model_and_config`.

## Optional/feature-specific fields
- `model_cls_for_pretraining`: torch `nn.Module` for pre-train (set and flag `pretraining_required=True` if mandatory).
- `pretraining_required`: `True` when pre-train must run before fine-tune.
- `tokenizer_cls`: custom tokenizer (defaults to HF `PreTrainedTokenizerFast`).
- `datacollator_cls_for_pretraining`: data collator for pre-train batches.
- `datacollator_cls_for_finetuning`: data collator for fine-tune/predict/interpret (defaults to HF `DefaultDataCollator`).
- `special_tokenizer_for_trainer_cls`: optional alternate tokenizer for the trainer only.
- `add_special_tokens`: whether to inject special tokens into the tokenizer.
- Training defaults such as `learning_rate`, `max_grad_norm`, `weight_decay` (used as fallbacks if not overridden in Hydra configs).

## Dataset expectations
- Dataset must honor the positions and delimiter from `data_source.*` (e.g., `filepath`, `columnsep`, `idpos`, `seqpos`, `labelpos`).
- Should surface `scaler` if normalization is used; interpret mode may read it.

## Registration flow (typical)
1. Define a factory `def build_plugin_config(): return PluginConfig(...)` with the fields above.
2. Expose it via `setup.cfg`/`pyproject.toml` entry point: `[project.entry-points."biolm.plugins"] my_plugin = mypkg.plugin:build_plugin_config`.
3. Optional: call `PluginManager.set_config(...)` in code paths if you use the programmatic API.

## Testing checklist for plugin authors
- `poetry run biolm list-plugins` shows your plugin name.
- `poetry run biolm tokenize/pre-train/fine-tune/predict/interpret --config-path ... plugin=<your_name>` runs end-to-end on a small sample.
- `test_predictions.csv` is produced under `${outputpath}/predict`; `loo_scores_<handletokens>.csv` under `${outputpath}/interpret`.
- Pre-train is enforced when `pretraining_required=True`.
