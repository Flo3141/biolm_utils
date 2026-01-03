> **Note:** The `biolm-2.0` branch contains the latest, actively developed version of BioLM with major improvements and a new plugin architecture. The `main` branch is legacy. For the newest features and code, please [switch to the `biolm-2.0` branch](https://github.com/dieterich-lab/biolm_utils/tree/biolm-2.0).
> **Note:** The `biolm-2.0` branch contains the latest, actively developed version of BioLM with major improvements and a new plugin architecture. The `main` branch is legacy. For the newest features and code, please [switch to the `biolm-2.0` branch](https://github.com/dieterich-lab/biolm_utils/tree/biolm-2.0).

# BioLM 2.0 Framework

A modular PyTorch framework for training language models on biological sequences (RNA/protein). Features a **plugin architecture** where model implementations are separate packages developed and versioned independently.

---

## Table of Contents

- [Installation](#installation)
- [Adding Plugins](#adding-plugins)
- [Data Format](#data-format)
- [Modes Overview](#modes-overview)
- [Usage](#usage)
- [Available Plugins](#available-plugins)
- [Configuration Management](#configuration-management)
- [Output Directory Structure](#output-directory-structure)
- [MLflow Tracking](#mlflow-tracking)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)

---

## 🚀 Installation

**Requirements:**
- Python 3.10+
- Poetry ([install guide](https://python-poetry.org/docs/#installation))

**Framework Installation (no plugins yet):**
```bash
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
git checkout biolm-2.0
./install.sh
```

`install.sh` installs only the BioLM framework. Plugins are installed separately (see below).

**Adding Plugins:**

- **Standard (user) install — clones into `./plugins/`**
  ```bash
  poetry run biolm install-plugin <path-or-git-url>
  poetry run biolm list-plugins
  ```
  `install-plugin` will clone the plugin repo into `./plugins/<name>` inside this project and install it in editable mode. Use this flow if you just need to run the plugin without editing its source elsewhere.

- **Developer install — track your upstream repo live**
  ```bash
  poetry add --editable /absolute/path/to/rna_saluki_cnn
  poetry add --editable /absolute/path/to/rna_protein_xlnet
  ```
  This wires the Poetry environment directly to your existing plugin repos (no copies under `./plugins`). Edits you make in those repos are immediately picked up when running BioLM.

If you previously used `install-plugin` and no longer want the cloned copies, you can safely remove the `./plugins` directory; the CLI will recreate it on demand for future user installs.

---

## 📊 Data Format

Input files must specify the delimiter using the `data_source.columnsep` configuration. By default, the delimiter is set to tab (`\t`). Example:

```
ID          Label    Sequence
seq_001     1.5      a,t,g,c,a,g,t,c,...
seq_002     2.3      a,t,g,c,a,g,t,c,...
```

---

## ⚡ Modes Overview

| Mode         | Description                                                                 | Typical Use/Plugin         |
|--------------|-----------------------------------------------------------------------------|---------------------------|
| tokenize     | Build vocabulary/tokenizer from data.                                       | All models                |
| pre-train    | (Optional) Pre-train language model on unlabeled data.                      | Required for LMs (XLNet)  |
| fine-tune    | Train model on labeled data for your task.                                  | All models                |
| predict      | Run inference/prediction on new data.                                       | All models                |
| interpret    | Feature importance/interpretation (e.g., saliency, attention, etc.).        | All models                |

**Notes:**
- Language models (e.g., XLNet) require pre-training before fine-tuning.
- CNN-based models (e.g., Saluki) do **not** require pre-training.

### Mode Quickstart

Below are the canonical commands, vital configuration knobs, and outputs for each mode. Reference paths assume you keep experiment-specific overrides under `./my_experiment` and set `outputpath` inside that config.

**Tokenize**

```bash
poetry run biolm tokenize --config-path ./my_experiment
```

- Key config values: `data_source.filepath`, `tokenization.encoding`, `tokenization.vocabsize`.
- Output: tokenizer artifacts in `${outputpath}/tokenize` (e.g., merges.txt, vocab.json).

**Pre-train**

```bash
poetry run biolm pre-train --config-path ./my_experiment
```

- Requires a plugin whose config sets `task: pre-train` (see `mode/pre-train.yaml`).
- Important options: `training.nepochs`, `training.batchsize`, `training.scaling`, `settings.mlflow.enabled`.
- Output: pretrained weights and logs in `${outputpath}/pre-train`.

**Fine-tune**

```bash
poetry run biolm fine-tune --config-path ./my_experiment
```

- Make sure `plugin` points to the installed model package and `task` matches the plugin expectation (classification/regression).
- Main toggles: `data_source.splitratio`, `training.nepochs`, `training.patience`, `training.gradacc`.
- Output: fine-tuned checkpoints, metrics, and MLflow logs in `${outputpath}/fine-tune`.

**Predict**

```bash
poetry run biolm predict --config-path ./my_experiment inference.pretrainedmodel=/path/to/model.ckpt
```

- Ensure `inference.pretrainedmodel` is set to the checkpoint produced by fine-tuning or pre-training.
- Optional overrides: `inference.looscores.handletokens` (defaults to `mask` here), `debugging.dev` for quick dry-runs.
- Output: `${outputpath}/predict/test_predictions.csv` (IDs plus plugin-specific scores/probabilities) and logs in `${outputpath}/predict/logs/`.

**Interpret**

```bash
poetry run biolm interpret --config-path ./my_experiment inference.pretrainedmodel=/path/to/model.ckpt
```

- Core options under `inference.looscores`:
  - `handletokens`: `mask` (default) or `remove` to control occlusion behaviour.
  - `replacementdict`: dictionary limiting replacements per token; leave `null` for full masking.
  - `replacespecifier`: boolean to include sequence specifier fields in replacements.
- Other useful flags: `debugging.dev` to restrict the number of samples, `training.batchsize` for occlusion batching.
- Output: `${outputpath}/interpret/loo_scores_<handletokens>.csv` and `.pkl` plus logs in `${outputpath}/interpret/logs/`.

---

## 🛠️ Usage

Run any mode with:
```bash
poetry run biolm {tokenize | pre-train | fine-tune | predict | interpret} --config-path ./my_experiment
```

## 🧭 Execution Flow (at a glance)

1. CLI parses args and Hydra composes configs.
2. `plugin_registry` resolves the plugin entry point; plugin config classes are loaded.
3. Data is loaded/prepared (tokenizer built or loaded); datasets are cached under `${outputpath}/{mode}`.
4. Mode dispatcher (`runner`) calls the appropriate trainer/evaluator.
5. Artifacts and logs are written to `${outputpath}/{mode}`; MLflow (if enabled) logs params/metrics/artifacts to `${outputpath}/mlruns`.

---

## 🔌 Available Plugins

| Plugin | Model | Sequences | Pre-training | Use Case |
|--------|-------|-----------|--------------|----------|
| [rna_protein_xlnet](https://github.com/dieterich-lab/rna_protein_xlnet) | XLNet | RNA/Protein | Yes | General sequence analysis and prediction |
| [rna_saluki_cnn](https://github.com/dieterich-lab/rna_saluki_cnn) | CNN | RNA | No | m6A modification site prediction |

---

## ⚙️ Configuration Management

BioLM uses Hydra for flexible configuration. Compose configs from multiple files and override values at runtime:

```bash
poetry run biolm fine-tune --config-path ./my_experiment training.nepochs=50
```

### Default Configuration Structure

BioLM provides a default configuration structure:

```plaintext
biolm/conf
├── config.yaml          # Base configuration
├── mode
│   ├── tokenize.yaml    # Tokenization-specific settings
│   ├── pre-train.yaml   # Pre-training-specific settings
│   ├── fine-tune.yaml   # Fine-tuning-specific settings
│   ├── predict.yaml     # Prediction-specific settings
│   └── interpret.yaml   # Interpretation-specific settings
```

### Important Configuration Settings (suggested order)

- **`plugin`**, **`task`**, **`outputpath`**: Select the installed plugin, set `classification` or `regression`, and choose where artifacts are written.
- **`data_source.filepath`**, **`data_source.columnsep`**, **`data_source.splitratio`**: Point to the data file, delimiter (default `\t`), and splits.
- **`training.nepochs`**, **`training.batchsize`**, **`training.blocksize`**: Core training knobs; `training.batchsize` is also used by interpret.
- **`inference.pretrainedmodel`**: Checkpoint path required for `predict` and `interpret`.
- **`inference.looscores.*`**: `handletokens` (`mask`/`remove`), `replacementdict` (limit substitutions), `replacespecifier` (include sequence specifier fields).
- **`mlflow.enabled`**, **`mlflow.tracking_uri`**: Toggle tracking and set the MLflow artifact store (default `${outputpath}/mlruns`).

**Hardware note:** XLNet-style LMs are GPU-oriented; Saluki CNN can run on CPU but is faster on GPU. The framework will pick GPU if available (`gpu.py`) and fall back to CPU.

### Example Configuration File

```yaml
# config.yaml
plugin: <plugin_name>                       # Replace with the plugin identifier you installed
outputpath: /path/to/results               # Output directory
task: regression                            # regression or classification

data_source:
  filepath: /path/to/data.txt              # Tab-separated data
  idpos: 1                                  # ID column (1-indexed)
  seqpos: 3                                 # Sequence column
  labelpos: 2                               # Label column
  splitratio: [70, 15, 15]                 # Train/val/test split

training:
  nepochs: 100                              # Number of epochs
  batchsize: 8                              # Batch size
  blocksize: 512                            # Max sequence length
```

### Hydra Composition

Hydra enables you to compose configurations by merging the base configuration with mode-specific configurations. For example:

**Base Configuration (`config.yaml`):**

```yaml
plugin: <plugin_name>
outputpath: /path/to/results
task: regression
```

**Mode Configuration (`mode/fine-tune.yaml`):**

```yaml
training:
  nepochs: 100
  batchsize: 8
  blocksize: 512
```

When running the `fine-tune` mode, Hydra automatically merges these configurations. You can also specify additional overrides at runtime.

### Creating New Compositions

To create a new composition, you can define additional configuration files. For example, if you want to create a custom training setup:

**Custom Configuration (`custom_training.yaml`):**

```yaml
training:
  nepochs: 50
  batchsize: 16
  learning_rate: 0.001
```

Run with the custom configuration:

```bash
poetry run biolm fine-tune --config-name custom_training --config-path ./my_experiment
```

---

## 📂 Output Directory Structure

The framework organizes outputs in the following structure:

```
output/
├── tokenize/          # Tokenizer files (HuggingFace format)
├── pre-train/         # Pre-trained models
├── fine-tune/         # Fine-tuned models and logs
├── predict/           # Prediction outputs
├── interpret/         # Interpretation results
└── mlruns/            # MLflow logs and artifacts
```

- **Tokenizer**: Located in `output/tokenize/`.
- **Trained Models**: Found in `output/pre-train/` or `output/fine-tune/`.
- **Training Results**: Logs and metrics are in `output/fine-tune/logs/`.
- **Prediction Outputs**: `output/predict/test_predictions.csv` plus logs in `output/predict/logs/`.
- **Interpretation Outputs**: `output/interpret/loo_scores_<handletokens>.csv` and `.pkl` plus logs in `output/interpret/logs/`.
- **MLflow Logs**: Stored in `output/mlruns/` (params/metrics/artifacts per run).

### Artifact contents (what to expect)

- **Checkpoints**: Saved under `${outputpath}/pre-train` and `${outputpath}/fine-tune` (plugin-specific filenames). Reuse them by pointing `inference.pretrainedmodel` (for predict/interpret) or `model_load_path` (for continued training).
- **`test_predictions.csv`**: Typically includes sample identifiers plus plugin-specific scores/probabilities; labels may appear if available. Schemas can differ by plugin—consult the plugin README for exact columns.
- **`loo_scores_<handletokens>.csv` / `.pkl`**: Per-position leave-one-out scores; includes sequence IDs, positions, tokens, and plugin-specific score deltas. The `<handletokens>` suffix reflects the occlusion strategy (`mask`/`remove`).
- **MLflow run folders**: Contain `params`, `metrics`, and `artifacts` (including checkpoints and logs). MLflow UI can browse and download these directly.

---

## 📈 MLflow Tracking

BioLM integrates with MLflow for experiment tracking. To enable MLflow:

1. Set `mlflow.enabled: true` in the configuration.
2. Access the MLflow UI:
   ```bash
   poetry run mlflow ui --backend-store-uri output/mlruns
   ```
3. Download artifacts (e.g., models, logs) directly from the UI.

---

## 🧪 Testing

Run tests with:
```bash
poetry run pytest tests/
```

For specific suites:
```bash
poetry run pytest tests/integration/      # Plugin system tests
poetry run pytest tests/test_*.py         # Unit tests
```

With coverage:
```bash
poetry run pytest --cov=biolm --cov-report=html
```

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request

**Plugin development:** See Plugin Development Guide below

---

## 📝 Citation

```bibtex
@software{biolm2024,
  title = {BioLM 2.0: A Modular Framework for Biological Language Models},
  author = {Dieterich Lab},
  year = {2024},
  url = {https://github.com/dieterich-lab/biolm_utils}
}
```
