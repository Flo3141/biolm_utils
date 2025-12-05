# BioLM Utils

A toolkit for tokenizing, pre-training, and fine-tuning language models on biological sequences (RNA/protein). Supports interpretation with leave-one-out (LOO) scores.

## Quick Start

### Prerequisites
- Python 3.10+
- Poetry ([python-poetry.org](https://python-poetry.org/))

### Installation

```bash
# Clone the framework
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils

# Install dependencies
poetry install
```

Built-in plugins (Saluki and XLNet) are loaded automatically. No separate plugin installation required.

### Run Your First Experiment

```bash
# Start with minimal example
poetry run biolm fine-tune --config-path ./exampleconfigs/minimal

# Or copy template for your data
cp -r PLUGIN_TEMPLATE my_experiment
# Edit my_experiment/config.yaml, then:
poetry run biolm fine-tune --config-path ./my_experiment
```

## Documentation

- **[QUICKSTART.md](./QUICKSTART.md)** 
- **[README_CONFIGS.md](./README_CONFIGS.md)** 
- **[PLUGIN_TEMPLATE/](./PLUGIN_TEMPLATE/)**

