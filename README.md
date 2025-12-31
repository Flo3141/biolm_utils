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

**Framework Installation:**
```bash
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
git checkout biolm-2.0
./install.sh
```

**Adding Plugins:**
Install plugins using the CLI:
```bash
poetry run biolm install-plugin <path-or-git-url>
poetry run biolm list-plugins
```

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

---

## 🛠️ Usage

Run any mode with:
```bash
poetry run biolm {tokenize | pre-train | fine-tune | predict | interpret} --config-path ./my_experiment
```

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

### Important Configuration Settings

- **`data_source.columnsep`**: Specifies the delimiter for input files (default: `\t`).
- **`data_source.splitratio`**: Defines the train/validation/test split ratios.
- **`training.nepochs`**: Number of training epochs (default: 100).
- **`training.batchsize`**: Batch size for training (default: 8).
- **`mlflow.enabled`**: Enables MLflow tracking (default: `true`).
- **`mlflow.tracking_uri`**: Directory for MLflow logs (default: `${outputpath}/mlruns`).

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
- **Prediction Outputs**: Stored in `output/predict/`.

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
