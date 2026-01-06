> **Note:** The `biolm-2.0` branch contains the latest, actively developed version of BioLM with major improvements and a new plugin architecture. The `main` branch is legacy. For the newest features and code, please [switch to the `biolm-2.0` branch](https://github.com/dieterich-lab/biolm_utils/tree/biolm-2.0).

# BioLM 2.0 Framework

A modular PyTorch framework for training language models on biological sequences (RNA/protein). Features a **plugin architecture** where model implementations are separate packages developed and versioned independently.

---

## Table of Contents

- [Installation](#installation)
- [Adding Plugins](#adding-plugins)
  - Notes on plugin discovery: plugins expose their factories through the `biolm.plugins` entry-point group, so installing via `install-plugin`, `develop-plugin`, or an editable pip install lets BioLM pick them up automatically.
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

  **Notes on plugin discovery:** Plugins register themselves via the `biolm.plugins` entry-point group in their `pyproject.toml` (e.g., `xlnet = "xlnet_plugin.config:get_xlnet_config"`, `saluki = "saluki_plugin.config:get_saluki_config"`). As long as the package is installed in the same Poetry environment as BioLM (`install-plugin`, `develop-plugin`, or `pip install -e <path-or-git-url>`), the framework automatically discovers the entry point—no additional registration is needed.

- **Developer install — keep framework metadata clean**

-  ```bash
-  # inside the biolm_utils repo
-  poetry install --no-interaction --with dev
-  # install your local plugin(s) into the env without touching pyproject
-  poetry run biolm develop-plugin /path/to/your/plugin
-  ```

-  This keeps `pyproject.toml` unchanged while wiring editable installs through the CLI. Edits in your plugin repos are picked up immediately. To remove, run `poetry run pip uninstall <plugin-name>`.

If you previously used `install-plugin` and no longer want the cloned copies, you can safely remove the `./plugins` directory; the CLI will recreate it on demand for future user installs.

---

## 📊 Data Format

Input files must specify the delimiter using the `data_source.columnsep` configuration. By default, the delimiter is set to tab (`\t`). Example (tab-separated columns, raw sequence text):

```tsv
ID	Label	Sequence
seq_001	1.5	AUGCUAGCUAGC
seq_002	2.3	AUGGCUAUGGCU
```

---

## ⚡ Modes Overview

| Mode         | Description                                                                 | Typical Use/Plugin         |
|--------------|-----------------------------------------------------------------------------|---------------------------|
| tokenize     | Build vocabulary/tokenizer from data.                                       | All models                |
| pre-train    | (Optional) Pre-train language model on unlabeled data.                      | Required for LMs          |
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

Always pass an explicit `--config-path`/`--config-name`; runtime initialization no longer relies on implicit defaults.

## 🧭 Execution Flow (at a glance)

1. CLI parses args and Hydra composes configs.
2. `plugin_registry` resolves the plugin entry point; plugin config classes are loaded.
3. Data is loaded/prepared (tokenizer built or loaded); datasets are cached under `${outputpath}/{mode}`.
4. Mode dispatcher (`runner`) calls the appropriate trainer/evaluator.
5. Artifacts and logs are written to `${outputpath}/{mode}`; MLflow (if enabled) logs params/metrics/artifacts to `${outputpath}/mlruns`.

## ⚡ Quickstart Examples

- **Saluki**

  ```bash
  # minimal config (saluki_quick.yaml)
  plugin: rna_saluki_cnn
  outputpath: /tmp/biolm_saluki_quick
  task: classification
  data_source:
    filepath: /prj/RNA_NLP/biolm_utils/examples/data/quickstart_sequences.tsv
    columnsep: "\t"
    idpos: 1
    seqpos: 3
    labelpos: 2
    splitratio: [70, 15, 15]
  training:
    nepochs: 3
    batchsize: 4
  ```

  Run: `poetry run biolm fine-tune --config-path . --config-name saluki_quick`

- **XLNet**

  ```bash
  # minimal config (xlnet_quick.yaml)
  plugin: rna_protein_xlnet
  outputpath: /tmp/biolm_xlnet_quick
  task: classification
  data_source:
    filepath: /prj/RNA_NLP/biolm_utils/examples/data/quickstart_sequences.tsv
    columnsep: "\t"
    idpos: 1
    seqpos: 2
    labelpos: 3
    splitratio: [70, 15, 15]
  training:
    nepochs: 1
    batchsize: 2
    blocksize: 256
  ```

  Run sequence:
  1) `poetry run biolm tokenize --config-path . --config-name xlnet_quick`
  2) `poetry run biolm pre-train --config-path . --config-name xlnet_quick`
  3) `poetry run biolm fine-tune --config-path . --config-name xlnet_quick`
  4) `poetry run biolm predict --config-path . --config-name xlnet_quick`

---

## 🔌 Available Plugins

| Plugin | Model | Sequences | Pre-training | Use Case |
|--------|-------|-----------|--------------|----------|
| [rna_protein_xlnet](https://github.com/dieterich-lab/rna_protein_xlnet) | XLNet | RNA/DNA/Protein | Yes | General sequence modeling (pre-train + downstream tasks) |
| [rna_saluki_cnn](https://github.com/dieterich-lab/rna_saluki_cnn) | CNN | RNA/DNA/Protein | No | Sequence classification/regression without pre-train |

---

## ⚙️ Configuration Management

BioLM uses Hydra for flexible configuration. Compose configs from multiple files and override values at runtime:

```bash
poetry run biolm fine-tune --config-path ./my_experiment training.nepochs=50
```

### Important Configuration Settings (suggested order)

- **`plugin`**, **`task`**, **`outputpath`**: Select the installed plugin, set `classification` or `regression`, and choose where artifacts are written.
- **`data_source.filepath`**, **`data_source.columnsep`**, **`data_source.splitratio`**: Point to the data file, delimiter (default `\t`), and splits.
- **`training.nepochs`**, **`training.batchsize`**, **`training.blocksize`**: Core training knobs; `training.batchsize` is also used by interpret.
- **`inference.pretrainedmodel`**: Checkpoint path required for `predict` and `interpret`.
- **`inference.looscores.*`**: `handletokens` (`mask`/`remove`), `replacementdict` (limit substitutions), `replacespecifier` (include sequence specifier fields).
- **`mlflow.enabled`**, **`mlflow.tracking_uri`**: Toggle tracking and set the MLflow artifact store (default `${outputpath}/mlruns`).

**Sample data for quickstarts:** Both quickstarts use the bundled `examples/data/quickstart_sequences.tsv` (tab-separated, columns: id, label, sequence; 100 rows) so they run out-of-the-box and can split cleanly without the plugin repos checked out.

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
```

Run with the custom configuration:

```bash
poetry run biolm fine-tune --config-name custom_training --config-path ./my_experiment
```

---

## 📂 Output Directory Structure

The framework organizes outputs in the following structure:

```plaintext
output/
├── tokenize/
│   ├── vocab.json           # Tokenizer vocabulary
│   └── merges.txt           # Merge rules (if BPE)
├── pre-train/
│   ├── checkpoint-XX/       # Checkpoint folders
│   ├── model.safetensors    # Final model weights
│   ├── config.json          # Model configuration
│   └── logs/                # Training logs
├── fine-tune/
│   ├── checkpoint-XX/       # Checkpoint folders
│   ├── model.safetensors    # Fine-tuned model weights
│   ├── all_results.json     # Aggregated metrics
│   ├── test_predictions.csv # Predictions on test set
│   └── logs/                # Training logs
├── predict/
│   ├── test_predictions.csv # Model predictions
│   └── logs/                # Execution logs
├── interpret/
│   ├── loo_scores.csv       # Feature importance scores
│   └── logs/                # Execution logs
└── mlruns/                  # MLflow tracking data
```

### Artifact contents (what to expect)

- **Checkpoints**: Saved under `${outputpath}/pre-train` and `${outputpath}/fine-tune` (plugin-specific filenames, e.g., `model.safetensors`). Reuse them by pointing `inference.pretrainedmodel` (for predict/interpret) or `model_load_path` (for continued training).
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

Tracking is scoped to each run’s `outputpath` (default `${outputpath}/mlruns`) rather than a global store; set `mlflow.tracking_uri` if you want a shared backend.

---

## 📜 Plugin Contract (for plugin authors)

See [docs/PLUGIN_CONTRACT.md](docs/PLUGIN_CONTRACT.md) for the required entry point, factory return shape, and dataset/model/tokenizer expectations.

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
  author = {Philipp Wiesenbach},
  year = {2024},
  url = {https://github.com/dieterich-lab/biolm_utils}
}
```
