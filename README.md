# bioml_utils — utilities for bioinformatic language models

## Overview

A compact toolkit for tokenizing, pre-training and fine-tuning language models on biological sequences (RNA/protein). It also supports interpretation with leave-one-out (LOO) scores.

**Plugin Architecture**: The framework uses a configuration-first plugin system for extensibility. Plugins are included as git submodules in the `plugins/` directory and managed via Poetry with relative paths. Plugins provide config schemas (dicts) that specify custom models, datasets, and tokenizers.

Base classes (`BaseModel`, `BaseDataset`) ensure consistency and provide hooks for customization.

## Quick Start: One Command Setup

The fastest way to get started with BioLM and all plugins:

### Prerequisites
- Python 3.10+
- Poetry (install from [python-poetry.org](https://python-poetry.org/))
- Git

### Installation (Two Commands!)

```bash
# 1. Clone the framework with all plugins
git clone --recurse-submodules https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils

# 2. Run the setup script
./setup.sh
```

Done! Everything is configured and ready to use.

### Run Your First Experiment

```bash
# Start with a minimal example
poetry run biolm fine-tune --config-path ./exampleconfigs/minimal

# Or try Saluki-specific (RNA analysis)
poetry run biolm fine-tune --config-path ./exampleconfigs/saluki-rna-finetuning

# Or try XLNet-specific (Protein analysis)
poetry run biolm fine-tune --config-path ./exampleconfigs/xlnet-protein-finetuning
```

For help: `poetry run biolm --help`

## Configuration & Documentation

**New to BioLM?** Start here:

- **[README_CONFIGS.md](./README_CONFIGS.md)** — Complete configuration guide with parameter reference
- **[PLUGIN_TEMPLATE/](./PLUGIN_TEMPLATE/)** — Copy-paste template for your experiments
- **[exampleconfigs/](./exampleconfigs/)** — Working examples (minimal, Saluki-specific, XLNet-specific)

Key things to know:

- Each plugin has fixed blocksize: Saluki=12288, XLNet=512 (don't override!)
- Saluki requires comma-separated nucleotides (e.g., `a,t,g,c,a,g,t,c`)
- XLNet works with raw sequences (e.g., `MKVLWAALLVT...` or `atgcgatc...`)
- Column positions in data files are **1-indexed** (start from 1, not 0!)

### Available Plugins

Plugins are located in the `plugins/` directory:

- **Saluki**: RNA sequence analysis with CNN-based models (`./plugins/saluki`)
- **XLNet**: Protein/RNA sequence analysis with Transformer models (`./plugins/xlnet`)

## Manual Setup (Advanced Users / Development)

If you prefer manual control or are developing the framework/plugins:

```bash
# Clone with submodules (if not using --recurse-submodules initially)
git submodule update --init --recursive

# Install dependencies and plugins via Poetry
poetry install

# Or add MLflow support
poetry install --with mlflow

# Run tests
poetry run pytest -q
```

### For Plugin Developers

To modify a plugin while the framework is running:

```bash
# After making changes to a plugin (e.g., ./plugins/saluki)
# reinstall the plugin in develop mode
poetry install

# Your changes are immediately available
poetry run biolm fine-tune --config-path ...
```

## Using MLflow for Experiment Tracking

The framework integrates with MLflow for automatic logging of experiments, metrics, and models. To enable MLflow:

1. Install with MLflow support as shown above.
2. In your plugin config, set `settings.mlflow.enabled: true`.
3. Optionally configure `tracking_uri` and `experiment_name` (auto-set based on output path and mode if not specified).

During training, metrics and parameters are logged automatically. To view experiments:

```bash
# Start the MLflow UI
poetry run mlflow ui --backend-store-uri /path/to/your/output/mode/mlruns

# Access at http://127.0.0.1:5000 in your browser
```

For remote servers, use port forwarding: `ssh -N -f -L localhost:5000:localhost:5000 user@server`.

## Plugins

The framework supports plugins for specific models and datasets. Available plugins:

- **Saluki**: CNN-based model for RNA sequence analysis ([rna_saluki_cnn](https://github.com/dieterich-lab/rna_saluki_cnn))
- **XLNet**: Transformer model for protein/RNA sequences ([rna_protein_xlnet](https://github.com/dieterich-lab/rna_protein_xlnet))

Use the installer script to add plugins automatically.

## Pipfile → Poetry

This project migrated from Pipenv to Poetry. To migrate older environments:

```bash
poetry init --no-interaction
poetry lock
```

## Layout

Top-level package: `biolm_utils/` — main modules include:
- `biolm.py`     : CLI entrypoint for tokenize / pre-train / fine-tune / interpret / predict
- `config.py`    : Legacy Config dataclass and compatibility helpers
- `plugin_config.py`: **NEW** Modern PluginConfig system with comprehensive documentation
- `plugin_manager.py`: **NEW** PluginManager singleton for config management
- `base_model.py`: Base classes for plugin models
- `base_dataset.py`: Base classes for plugin datasets
- `plugin_registry.py`: Registry for plugin discovery and application
- `plugin_loader.py`: Automatic plugin discovery via entry-points and directories
- `cross_validation.py` : New CrossValidator orchestration (replaces decorator-based CV)
- `params.py` / `entry.py` : CLI parsing and runtime wiring
- `train_tokenizer.py`, `trainer.py`, `interpret.py`, `loo_utils.py` : core functionality

See the `biolm_utils/` package for full details.

## Documentation

The framework includes comprehensive documentation in the `DOCS/` directory:

- **`framework_and_saluki_plugin_guide.md`**: High-level overview of concepts, plugin architecture, and the Saluki example plugin
- **`internal_data_flow_guide.md`**: Detailed technical guide covering CLI parsing, configuration loading, plugin discovery, and the complete execution pipeline

These guides are essential reading for:
- **Plugin developers**: Understanding how to create and integrate plugins
- **Framework contributors**: Technical details for extending or modifying the system
- **Advanced users**: Deep dive into internal workings for debugging and optimization

## Plugin Development

Plugins extend the framework with custom models/datasets using the modern `PluginConfig` system. Create a separate repo with a clean 3-file structure:

### Plugin Structure (Recommended)

```
my_plugin/
├── dataset.py      # Dataset implementation (inherit from BaseDataset)
├── models.py       # Model implementation (inherit from BaseModel)  
└── config.py       # Plugin configuration factory
```

### PluginConfig System

The new `PluginConfig` dataclass provides comprehensive documentation and type safety:

```python
from biolm_utils.plugin_config import PluginConfig, PluginManager

def get_my_plugin_config():
    """Factory function that creates and returns the plugin configuration."""
    config = PluginConfig(
        # Model classes - set your model classes here
        model_cls_for_pretraining=None,  # For self-supervised pretraining
        model_cls_for_finetuning=MyModel,  # Your main model class
        
        # Dataset class - your dataset implementation
        dataset_cls=MyDataset,
        
        # Tokenizer - usually PreTrainedTokenizerFast
        tokenizer_cls=PreTrainedTokenizerFast,
        
        # Data collators - customize for your data preprocessing
        datacollator_cls_for_pretraining=None,  # For pretraining tasks
        datacollator_cls_for_finetuning=DefaultDataCollator,  # For supervised tasks
        
        # Training settings
        learning_rate=1e-4,
        max_grad_norm=1.0,
        weight_decay=0.0,
        
        # And many more options with full documentation...
    )
    
    PluginManager.set_config(config)
    return config
```

Each field in `PluginConfig` includes detailed docstrings explaining:
- What the field does
- When to use it vs leave as default
- Examples for different model types (RNN-CNN, transformers, etc.)

### Entry Points

Add to your `pyproject.toml`:
```toml
[project.entry-points."biolm_utils.plugins"]
myplugin = "my_plugin.config:get_my_plugin_config"
```

For local development, plugins are automatically discovered from the `plugins/` directory.

## Using Plugins

Once a plugin is installed (e.g., `pip install -e /path/to/plugin`), it's automatically discovered via entry-points.

### Activating Plugins

To activate a plugin in your code:

```python
from biolm_utils.plugin_registry import apply_plugin
apply_plugin('saluki')  # Activates Saluki's PluginConfig
```

Or use the new PluginManager directly:

```python
from my_plugin.config import get_my_plugin_config
config = get_my_plugin_config()  # Automatically sets as active config
```

### PluginConfig vs Legacy Config

The framework now uses `PluginConfig` (dataclass with comprehensive docs) instead of plain dicts. Benefits:

- **Type Safety**: Full type hints and validation
- **Documentation**: Every field has detailed explanations  
- **IDE Support**: Auto-completion and inline help
- **Consistency**: Standardized plugin interface

Legacy dict-based plugins still work via automatic conversion.

## Quick workflow — using plugins (short)

The typical developer workflow is:

1. Clone & install the framework (biolm_utils)
1) Install the framework locally using Poetry (recommended):

```bash
# create the project's venv and install dependencies
cd /path/to/biolm_utils
poetry install

# you can run the CLI using the Poetry environment
poetry run python biolm.py fine-tune
```
2) In another shell/terminal clone a plugin and install it into the same Poetry environment (editable for development):

```bash
# inside the plugin repo (assuming the same environment / venv created by Poetry)
cd /path/to/rna_saluki_cnn
poetry install
poetry run python -m pip install -e ./saluki_plugin
```

Detailed steps are below but the above is the minimal flow for getting started.


## Output layout

Experiments default to the `outputpath` in `params.py`. Typical layout:

```
my_experiment/
  tokenizer.json
  pre-train/
  fine-tune/<fold-id>/pytorch_model.bin
```

### Modes & examples

Main CLI: `biolm.py` (modes: `tokenize`, `pre-train`, `fine-tune`, `interpret`, `predict`).

Examples:

```bash
# tokenize
python biolm.py tokenize --configfile config.yaml

# fine-tune with plugin
python biolm.py fine-tune --configfile config.yaml --plugin saluki
```

#### Notes

- `splitpos=None` → 90/10 train/val (no test). If you provide split ids, the code will run cross-validation over splits.
- `specifiersep` (one-hot only) allows per-token float channels (e.g. `A#2.5`).
- `vocabsize`: The maximal size of the vocabulary at the end of the tokenization process.
- `minfreq`: The minimum frequency that a token should appear in the training file before it is recorded as vocabulary item.
- `atomicreplacements`: This is a dictionary with tokens that should be treated as atomic tokens during the byte pair encoding process. You have to specify both: The initial token and the character that it is to be mapped to. 
- `encoding`: The encoding to apply: character-wise (`atomic`) or BPE (`bpe`).
- `maxtokenlength`: The BPE tokenizer can come up with pretty long tokens. This number caps the length at a maximal length.
- `lefttailing`: If true, sequences are cropped from the left (keeps right-side context).

### Pre-training (language models only) and fine-tuning a model 

For pre-training an language model via Masked Language Modelling you will use the `pre-train` mode. For fine-tuning a model, the `fine-tune` mode is required. In your `config.yaml` you need to at least specify the parameters under `training`:

```yaml
training:
  general:
    batchsize: 8
    gradacc: 4
    nepochs: 10
    patience: 3
    resume: False # for resuming training
  fine-tuning:
    fromscratch: False # if we want to fine-tune without a pre-trained model (language models only)
    scaling: log # [log, minmax, standard]
    weightedregression: False
```

We also have to clarify data pre-processing and environment options:

```bash
data pre-processing:
  centertoken: False # either False or a token/character on which the sequence will be centered
environment:
  detected_ngpus: (auto-detected)  # Auto-detected; powers of two only (1,2,4,...)

BREAKING CHANGE: explicit GPU counts removed
------------------------------------------------
Note: The legacy `ngpus` option in `settings.environment` and `debugging.ngpus` has been removed. GPU counts are now auto-detected and exposed at `debugging.detected_ngpus` in the final `BioLMConfig` returned by `load_config()`.
 - Do not set `settings.environment.ngpus` or `debugging.ngpus` in your config YAMLs; they raise a ValueError.
 - Programmatic access: use `from biolm_utils.params import get_detected_ngpus` and call `get_detected_ngpus(args)`.
 - Example: `detected = get_detected_ngpus(args)`.
```

The `data processing` attributes refer to specific pre-processing options that are in detail explained by the command line help.

### Programmatic orchestration (train/dev/test runs with cross-validation)

If you want to orchestrate runs from other Python code (for example, to integrate
the library into a higher-level workflow or test harness) prefer the explicit
helpers introduced in the refactor: `make_run_fn`, `CrossValidator` and
`Paths`. These are easier to unit-test and avoid mutating global state.

Example (high-level):

```py
from biolm_utils.config import get_config
from biolm_utils.params import load_config
from biolm_utils.train_tokenizer import tokenize
from biolm_utils.train_utils import get_tokenizer, get_dataset
from biolm_utils.runner import make_run_fn
from biolm_utils.cross_validation import CrossValidator
from biolm_utils.paths import Paths

# Load your config / args (same objects used by the CLI)
config = get_config()
# load_config returns a BioLMConfig dataclass instance
args = load_config()

# Prepare tokenizer / datasets as usual
tokenizer = get_tokenizer(args, /* TOKENIZERFILE */, config.TOKENIZER_CLS, config.PRETRAINING_REQUIRED)
tokenizer_for_trainer = tokenizer
full_dataset = get_dataset(args, tokenizer, config.ADD_SPECIAL_TOKENS, /* DATASETFILE */, config.DATASET_CLS)

# Build the per-run callable (identical signature as legacy nested `run`):
run_once = make_run_fn(args, config, tokenizer, tokenizer_for_trainer, full_dataset)

# Create immutable per-run paths (these values come from biolm_utils.entry in the CLI)
base_paths = Paths(
  model_load_path=/* MODELLOADPATH */,
  model_save_path=/* MODELSAVEPATH */,
  output_path=/* OUTPUTPATH */,
  report_file=/* REPORTFILE */,
  rank_file=/* RANKFILE */,
)

# Instantiate CrossValidator and run the selected mode: fine-tune, predict, interpret, pre-train
cv = CrossValidator(params=args, dataset=full_dataset, run_once_fn=run_once, base_paths=base_paths)
result = cv.execute()

# `result` contains per-mode semantics (list of fold results for cross-validation, or a single value for predict)
```

## Configuration loader — programmatic usage and CLI behaviour

We simplified the configuration loader to be clearer and easier to test. Key
points you should be aware of:

- load_config now returns a structured `BioLMConfig` dataclass (no more implicit
  flattened argparse.Namespace). Use `cfg.data_source.filepath`,
  `cfg.training.batchsize`, `cfg.debugging.detected_ngpus`, etc.
- When calling programmatically prefer the explicit API: pass Hydra-style
  overrides as a list of strings (`key=value`). We purposely stopped auto-parsing
  sys.argv for programmatic calls — that behaviour was fragile and confusing.

Examples:

Programmatic:

```py
from biolm_utils.params import load_config

# Explicit list of overrides: 'key=value' strings
cfg = load_config(["mode=tokenize"])
print(cfg.mode)  # -> 'tokenize'
```

Via CLI (Hydra):

```bash
# Use Hydra-style overrides from the shell; Hydra CLI still works as before
python biolm.py mode=tokenize
```

Notes:

- Old behaviour where `load_config()` attempted to parse `sys.argv` and
  convert `--flag value` style arguments to hierarchical keys (e.g. `--filepath`
  -> `data_source.filepath`) has been removed. If you relied on that behaviour,
  update invocations to call `load_config` with explicit overrides or call the
  CLI directly (Hydra handles CLI args).
- Config validation and runtime GPU autodetection now live on the
  `BioLMConfig` dataclass via `cfg.validate()` and `cfg.autodetect_gpus()` and
  are run automatically when using `load_config`.

Notes & migration
- `run_once` keeps the original signature used by the old decorator: run(train, val, test, model_load, model_save, report, rank)
- The old `@parametrized_decorator` wrapper is still available for backward compatibility but is deprecated — prefer the `CrossValidator` + `make_run_fn` flow above.

### Cross-validation behaviour and pitfalls

Cross-validation configuration can be a little subtle — here are the rules and gotchas so you get deterministic, predictable behavior.

- `data_source.crossvalidation` accepts three kinds of values:
  - `null` / `0` / `False` (default) — no cross-validation. The code will either use `splitpos` + `devsplits` (deterministic splits) when provided, or a single random split when `splitratio` is specified.
  - `true` — *use predefined splits*. This requires `splitpos` to be set and `devsplits` (a list of split ids — and optionally `testsplits`) to be provided in your config or dataset file. This runs one pass per entry in `devsplits` (and `testsplits` if set) deterministically.
  - integer >= 2 — *random k-fold cross-validation* (k-fold). This performs k independent shuffled runs and requires `splitratio` (e.g., `[80,10,10]` or `[80,20]`) to determine train/val/(test) percentages. Note: `crossvalidation=1` is not allowed because it is ambiguous.

Pitfalls to avoid:
- `crossvalidation=true` without `splitpos` is ambiguous and will now raise an error — either provide `splitpos` (and `devsplits`) or set `crossvalidation` to a positive integer >= 2 and a `splitratio`.
- `crossvalidation` as an integer while `splitpos` is present is conflicting — numeric crossvalidation implies random splits and therefore conflicts with predefined split positions; prefer `crossvalidation=true` for predefined splits.
- `splitpos` set without `devsplits` is invalid — you must provide `devsplits` (and optionally `testsplits`) to define which splits are used for validation/testing.

Example YAML snippets:

1) Predefined splits (one deterministic CV run per entry of devsplits):

```yaml
data_source:
  splitpos: 3
  devsplits: [[1], [2]]  # list-of-lists: each tuple defines dev/test groupings
  testsplits: [[3], [4]] # optional
  crossvalidation: true
```

2) Random 5-fold cross-validation with 80/10/10 train/val/test:

```yaml
data_source:
  crossvalidation: 5
  splitratio: [80, 10, 10]
```

3) No CV (single run): deterministic with splits or a single random split

```yaml
data_source:
  crossvalidation: 0
  splitpos: 1
  devsplits: [2]
```

The library also validates these combinations early — invalid or ambiguous settings will raise a helpful error explaining the expected fix.

Automatic migration helper

To help migrate older configs that may use ambiguous forms, we've added a small helper in `biolm_utils.cfg_migration`:

- `analyze_crossvalidation(params)` — returns human-readable notes about ambiguous or problematic settings.
- `migrate_crossvalidation(params, auto_apply=False)` — returns a copy of `params` and recommended fixes; with `auto_apply=True` it will apply safe conversions (e.g. `0 -> False`, `True + splitratio -> convert to default k-fold`).

Usage example:

```py
from biolm_utils.cfg_migration import analyze_crossvalidation, migrate_crossvalidation

# analyze
notes = analyze_crossvalidation(args)
for n in notes:
  print("TODO:", n)

# apply safe migrations
new_args, applied_notes = migrate_crossvalidation(args, auto_apply=True)
```


Under `environment`, you can decide if you want to train on GPU or CPU and on how many GPUs you want to train. GPU count is auto-detected and restricted to powers-of-two values (1, 2, 4, 8...).

### Extract LOO-scores for a model

To calculate importance scores for indidvidual input tokens, we can use the mode `interpret`. The script will then run over the test splits and extracts leave-one-out (LOO) scores. The LOO scores are estimated by leaving a certain token blank (or delete comepletely, see options below), run the model with this "defective" sequence and compare the results to the prediction of the model for the original sequence. Positive scores denote, that leaving the input out leads to higher prediction, v.v. negative score means, leaving the input out leads to lower predictions. 

```yaml
looscores:
  handletokens: remove # remove, mask, replace
  replacementdict: None # dict of atomic tokens that should be replaced against each other if `--handletokens` is set to `replace`."
```

The scripts will then extract LOO scores for all splits of the fine-tuning data and saves them as `.csv` under the corresponding fine-tuning path as `loo_scores_{handle_tokens}.csv`.

### Inference:

Inference means sending a fine-tuned model on unseen data and let it make predictions. For this, run the main script with in the `predict` mode. The configfile mirrors only a fraction of the attributes compared to the complete pipeline.

### Resuming a model

There are two use cases to resume a model using the `--resume` argument:
1) `--resume` (without parameters) triggers the huggingface internal `resume_from_checkpoint` option which will only _continue_
a training that has been interrupted. For example, a planned training that was to run for 50 epochs and was interrupted  at epoch
23 can be resumed from the best checkpoint to be run from epoch 23 to planned epoch 50.
2) `--resume X` will trigger further pre-training a model from its best checkpoint for additional `X` epochs.


## Customization

This framwework on it's own does not provide full functionality. It is meant to be employed with plugins that implement the following classes and methods:
- A custom model class that inherits from 🤗 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/model#transformers.PreTrainedModel) and provides a static `getconfig()` method.
- A custom dataset class that inherits from [RNABaseDataset](./biolm_utils/rna_datasets.py) and provides the `__getitem__()` method.
- A main script that imports the `run()` method from [biolm.py](./biolm_utils/biolm.py) and defines a custom `Config` object from [config.py](./biolm_utils/config.py) via `setconfig()`.

## License

## Unified Installation: Framework + Plugin

To install both the biolm_utils framework and the Saluki plugin in a single Poetry environment:

```bash
# 1. Install framework dependencies
cd /prj/RNA_NLP/biolm_utils
poetry install
poetry add mlflow

# 2. Add Saluki plugin as a local dependency (Poetry way)
poetry add /absolute/path/to/rna_saluki_cnn

# 3. Run your code inside the Poetry environment
poetry run python your_entrypoint.py --config your_config.yaml
```

- No manual venv activation needed—`poetry run` ensures the correct environment is used.
- The plugin must have a valid `pyproject.toml` and all code inside the `saluki_plugin/` directory.


## Troubleshooting

- If you encounter build errors, ensure the plugin's `pyproject.toml` includes:

  ```toml
  [tool.poetry]
  name = "rna-saluki-cnn"
  version = "0.1.0"
  packages = [{ include = "saluki_plugin" }]
  ...
  ```

- Remove any `package-mode = false` lines.
- Only one `pyproject.toml` per package.


## Example Usage

```bash
poetry run python src/entry.py --config exampleconfigs/predict_interpret.yaml
```
