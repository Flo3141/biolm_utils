# BioLM 2.0 Framework

A modular PyTorch framework for training language models on biological sequences (RNA/protein). Features a **plugin architecture** where model implementations are separate packages developed and versioned independently.

## Table of contents

- Quick Start
- Installation
- Plugins (installation)
- Verify Installation
- Run Your First Training
- Data Format
- Usage (pipeline)
- Configuration Management (Hydra)
- Plugin Discovery
- Testing
- Contributing
- Full Documentation (merged)


## 🎯 Quick Start

### Prerequisites

- Python 3.10+
- Poetry ([install guide](https://python-poetry.org/docs/#installation))

### Installation

1. **Install Framework**

  ```bash
  git clone https://github.com/dieterich-lab/biolm_utils.git
  cd biolm_utils
  git checkout biolm-2.0
  ./install.sh
  ```

1. **Plugins (External packages)**

  The framework is intentionally "bare": models and model-specific code live in separate plugin packages. Plugins are not bundled with the framework and must be installed separately by users who need them.

  The framework is intentionally "bare": model implementations are distributed as separate plugin packages and must be installed when you need them.

  Canonical plugin installation (required)

  Use only the framework CLI helper to install plugins. This wrapper standardizes installation, verifies the package exposes the required entry point, and runs a lightweight verification after installation:

  ```bash
  poetry run biolm install-plugin <path-or-git-url>
  ```

  The helper will:

- Install the plugin package into your current environment (uses Poetry when available, falls back to pip).
- Verify the package exposes a `biolm.plugins` entry point and that the entry-point target is importable.
- Run a small verification step; on failure it will report the error and will not leave a partially-installed plugin.

  After installing, confirm the plugin is visible with:

  ```bash
  poetry run biolm list-plugins
  ```

### Verify Installation

You can list discovered/registered plugins with:

```bash
poetry run biolm list-plugins
```

### Run Your First Training

```bash
# Copy a minimal config template to a working directory
cp -r biolm/examples/plugin_template my_experiment

# Edit my_experiment/config.yaml to point at your data and output path
#   - set `outputpath: /path/to/results`
#   - set `data_source.filepath: /path/to/data.txt`

# Run fine-tuning
poetry run biolm fine-tune --config-path ./my_experiment
```

**📖 See Installation Guide below**

## 📊 Data Format

Your input file should be **tab-separated**:

```
ID          Label    Sequence
seq_001     1.5      a,t,g,c,a,g,t,c,...
seq_002     2.3      a,t,g,c,a,g,t,c,...
```

**Important:** Column positions are **1-indexed** (1, 2, 3...).

## 🚀 Usage

### Training Pipeline

```bash
# Step 1: Tokenize (builds vocabulary)
poetry run biolm tokenize --config-path ./my_experiment

# Step 2: Pre-train (optional; model-dependent)
poetry run biolm pre-train --config-path ./my_experiment

# Step 3: Fine-tune on your task
poetry run biolm fine-tune --config-path ./my_experiment

# Step 4: Make predictions
poetry run biolm predict --config-path ./my_experiment

# Step 5: Interpret (feature importance)
poetry run biolm interpret --config-path ./my_experiment
```

### Available Plugins

| Plugin | Model | Sequences | Pre-training | Use Case |
|--------|-------|-----------|--------------|----------|
Plugins are external packages; the framework itself does not bundle any model implementations. Install and register plugins separately (see above).

## ⚙️ Configuration Management with Hydra

BioLM uses Hydra for flexible and powerful configuration management. Hydra allows you to compose configurations from multiple files and override specific values at runtime.

### Default Configuration Structure

BioLM provides a default configuration structure that includes:

- **Base Configurations**: General settings shared across all modes.
- **Mode-Specific Configurations**: Settings tailored for specific modes like `tokenize`, `pre-train`, `fine-tune`, etc.

The default configuration files are located in the `biolm/conf` directory:

```plaintext
biolm/conf
├── config.yaml          # Base configuration
├── mode
│   ├── tokenize.yaml    # Tokenization-specific settings
│   ├── pre-train.yaml   # Pre-training-specific settings
│   ├── fine-tune.yaml   # Fine-tuning-specific settings
│   ├── predict.yaml     # Prediction-specific settings
│   └── interpret.yaml   # Interpretation-specific settings (LOO, saliency)
```

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

### Minimal Overrides

For quick experiments, you can override specific values directly from the command line without modifying configuration files:

```bash
poetry run biolm fine-tune training.nepochs=50 training.batchsize=16
```

This approach is useful for testing different hyperparameters without creating new configuration files.

### Example: Fine-Tuning with Custom Overrides

Suppose you want to fine-tune a model with:

- 20 epochs
- Batch size of 32
- Learning rate of 0.0005

You can achieve this with minimal overrides:

```bash
poetry run biolm fine-tune training.nepochs=20 training.batchsize=32 training.learning_rate=0.0005
```

### Tips for Using Hydra

1. **Understand the Default Configurations**: Start by exploring the default `config.yaml` and mode-specific configurations in `biolm/conf`.
2. **Use Composition for Reusability**: Create reusable configuration files for common setups.
3. **Leverage Minimal Overrides for Flexibility**: Use command-line overrides for quick experiments.

**📖 See Configuration Reference below**
## 🏗️ Architecture

```
biolm_utils/              # Framework (this repo)
├── biolm/               # Core framework
│   ├── plugins/         # Plugin loader and discovery
│   └── examples/        # Config templates (framework-provided)
└── tests/               # Tests

# Plugins live in separate repositories and register themselves via Python entry points.
```

### Plugin Discovery (how it works)

Plugins register via **Python entry points**. When a plugin package is installed, it exposes an entry point that points to a factory or configuration function. At runtime the framework queries installed entry points in the `biolm.plugins` group and imports the plugin-provided configuration.

Example entry point (plugin package):

```toml
[project.entry-points."biolm.plugins"]
mymodel = "my_plugin.config:get_config"
```

The framework discovers and loads these entry points automatically—this keeps the core framework independent of any specific plugin implementation.

## 🧪 Testing

```bash
# Run all tests (61 tests)
poetry run pytest tests/

# Specific suites
poetry run pytest tests/integration/      # Plugin system tests
poetry run pytest tests/test_*.py         # Unit tests

# With coverage
poetry run pytest --cov=biolm --cov-report=html
```

**📖 Detailed test documentation:** See Testing Guide below  
**📖 CI/CD workflows explained:** See CI/CD Guide below

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request

**Plugin development:** See Plugin Development Guide below

## 📝 Citation

```bibtex
@software{biolm2024,
  title = {BioLM 2.0: A Modular Framework for Biological Language Models},
  author = {Dieterich Lab},
  year = {2024},
  url = {https://github.com/dieterich-lab/biolm_utils}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🆘 Support

- **Issues:** [GitHub Issues](https://github.com/dieterich-lab/biolm_utils/issues)
- **Discussions:** [GitHub Discussions](https://github.com/dieterich-lab/biolm_utils/discussions)

## 🎯 What's New in 2.0

- ✅ **Plugin Architecture** - Models as separate packages
- ✅ **Entry Point Discovery** - Automatic plugin loading  
- ✅ **Independent Versioning** - Plugins released separately
- ✅ **Clean Codebase** - Framework contains only core logic
- ✅ **Comprehensive Testing** - 61 framework tests, plugin tests in plugin repos
- ✅ **Modern Tooling** - Poetry, GitHub Actions, CI/CD
- ✅ **Educational Documentation** - Complete guides for testing, CI/CD, plugins

---

## Full Documentation (merged from docs/)

The following sections contain the complete documentation previously stored in `docs/`. They are provided here to keep the repository self-contained. If you maintain copies elsewhere, keep them in sync.

---

<!-- INSTALLATION.md -->

### Installation Guide

Complete installation instructions for BioLM 2.0 framework and plugins.

Prerequisites

- Python 3.10+
- Poetry
- Git

Quick Installation (Recommended)

Install the framework (recommended method):

```bash
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
./install.sh
```

The `install.sh` script installs framework dependencies and runs optional verifications. Plugin installation is intentionally performed separately using the framework CLI helper (see below).

Manual Installation

Option 1: Framework Only

```bash
cd biolm_utils
poetry install
```

Plugin installation (manual methods deprecated)

Installing plugins via manual `poetry add` or by running `poetry install` inside plugin repositories is no longer recommended. Use the CLI helper described above to ensure consistent installation and verification.

Installation Script Details

The `install.sh` script supports several options (see script for exact flags in repo):

```bash
./install.sh [OPTIONS]

Options:
  --skip-tests            Skip post-installation tests
  --help                  Show help message
```

Verification

Check Framework Installation

```bash
cd biolm_utils
poetry run python -c "import biolm; print('BioLM import OK')"
```

Check Plugin Discovery

```bash
poetry run python -c "
import importlib.metadata
print('Registered plugins:')
eps = importlib.metadata.entry_points(group='biolm.plugins')
for ep in eps:
    print(f'  • {ep.name}')
"
```

Run Tests

```bash
# Framework tests only
poetry run pytest tests/ --ignore=tests/integration --ignore=tests/end_to_end

# With plugins (integration + end-to-end)
poetry run pytest tests/
```

---

<!-- CONFIGURATION.md -->

### Configuration Reference

Complete guide to configuring BioLM experiments. (This section contains essential parameters, examples, and CLI override usage.)

Essential Parameters

Core Settings (example)

```yaml
# config.yaml
plugin: <plugin_name>
task: regression
outputpath: /absolute/path/to/results

data_source:
  filepath: /path/to/data.txt
  columnsep: "\t"
  idpos: 1
  seqpos: 3
  labelpos: 2
  splitratio: [70,15,15]

training:
  nepochs: 100
  batchsize: 8
  patience: 10
```

Data format and plugin-specific notes are important — validate `idpos`, `seqpos`, `labelpos` (1-indexed).

Command-Line Overrides

```bash
poetry run biolm fine-tune --config-path ./my_experiment training.nepochs=50
poetry run biolm fine-tune --config-path ./my_experiment +training.blocksize=512
```

Interpret Mode

`mode/interpret.yaml` provides defaults for interpretation (LOO, saliency). Typical fields include `interpret.method`, `interpret.n_perturbations` and reporting options. Compose `interpret` with your base config and override minimal fields as needed.

---

<!-- PLUGIN_DEVELOPMENT.md -->

### Plugin Development Guide (summary)

BioLM uses a plugin architecture where model implementations are separate packages that register via Python entry points.

Quick checklist to create a plugin:

1. Create a package with `pyproject.toml` and register an entry point in `[project.entry-points."biolm.plugins"]`.
2. Provide a factory function that returns a `PluginConfig` (see `biolm/plugin_config.py`).
3. Implement `model`, `dataset`, and optional tokenizer/collator classes.
4. Write tests verifying plugin loads and returns a valid config.

Entry point example:

```toml
[project.entry-points."biolm.plugins"]
plugin_name = "module.path:function_name"
```

The framework loads these entry points at runtime—no hard-coded plugin imports.

Best practices: pin `biolm` version, provide default plugin configs, document required plugin parameters (blocksize, tokenization), and add tests.

---

<!-- TESTING.md -->

### Testing Guide (summary)

Run tests with `poetry run pytest tests/`.

Structure: unit tests, integration tests (plugin discovery), and optional end-to-end tests.

Quick commands:

```bash
poetry run pytest tests/              # run full suite
poetry run pytest tests/integration/  # plugin integration tests
poetry run pytest tests/end_to_end/   # end-to-end (may require plugins)
```

---

<!-- CI_CD.md -->

### CI/CD Guide (summary)

Workflows live in `.github/workflows/` and cover unit tests, full matrix tests, plugin compatibility, and lockfile checks. The `plugin-compat.yml` workflow demonstrates checking plugin compatibility by checking out plugins alongside the framework.

Key principle: keep CI fast for PRs (`unit-tests-fast`) and run heavier runs on merges or as scheduled jobs (`unit-tests-full`).

---

<!-- PUBLISHING.md -->

### Publishing Guide (summary)

Use Poetry to build and publish. Typical flow:

```bash
poetry build
poetry publish -r testpypi
poetry publish
```

Tag releases and maintain a changelog.

---

<!-- tests/end_to_end/README.md (merged) -->

### End-to-End Testing Notes

End-to-end tests verify plugin integration and full pipeline behavior. They live under `tests/end_to_end/` and include tests that validate plugin loading, tokenization, pre-training (when supported), and small full-pipeline runs used for CI smoke tests.

Run example:

```bash
poetry run pytest tests/end_to_end/test_xlnet_saluki.py -v
```

---

If you want these sections split out again later for editing, I can recreate the `docs/` directory from this README.
