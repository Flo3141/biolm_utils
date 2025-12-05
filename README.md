# BioLM 2.0

A modular framework for training and interpreting language models on biological sequences (RNA/protein). Features a **plugin architecture** where models are separate packages that can be developed, versioned, and released independently.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Poetry ([python-poetry.org](https://python-poetry.org/))

### Installation

#### Option 1: Using the install script (recommended)

```bash
# Clone repositories side by side
git clone https://github.com/dieterich-lab/biolm_utils.git
git clone https://github.com/dieterich-lab/rna_saluki_cnn.git
git clone https://github.com/dieterich-lab/rna_protein_xlnet.git

# Run installation script
cd biolm_utils
./install.sh --with-plugins
```

#### Option 2: Manual installation

```bash
# Install framework
cd biolm_utils
poetry install

# Install plugins (optional)
cd ../rna_saluki_cnn && poetry install
cd ../rna_protein_xlnet && poetry install

# Link plugins to framework
cd ../biolm_utils
poetry install --with plugins
```

### Verify Installation

```bash
poetry run python -c "
import importlib.metadata
eps = importlib.metadata.entry_points(group='biolm.plugins')
print('Available plugins:')
for ep in eps:
    print(f'  - {ep.name}')
"
```

### Run Your First Experiment

```bash
# Use Saluki model for RNA regression
poetry run biolm fine-tune \
    --model saluki \
    --task regression \
    --data-source.train data/train.tsv \
    --data-source.dev data/dev.tsv
```

## 🧩 Plugin Architecture

BioLM 2.0 uses a **plugin system** where model implementations are separate packages:

```
biolm_utils/              # Core framework
  ├── biolm/              # Training/evaluation engine
  ├── tests/              # Framework tests
  └── install.sh          # Installation script

rna_saluki_cnn/           # Saluki plugin (separate repo)
  └── saluki_plugin/      
      ├── models.py       # CNN-based model
      ├── dataset.py      # One-hot encoding dataset
      └── config.py       # Plugin registration

rna_protein_xlnet/        # XLNet plugin (separate repo)
  └── xlnet_plugin/
      ├── models.py       # XLNet architecture
      ├── dataset.py      # Tokenized dataset
      └── config.py       # Plugin registration
```

### Available Plugins

| Plugin | Model | Task | Repository |
|--------|-------|------|------------|
| **saluki** | CNN with GRU | RNA stability prediction | [rna_saluki_cnn](https://github.com/dieterich-lab/rna_saluki_cnn) |
| **xlnet** | Transformer (XLNet) | RNA/protein pretraining & fine-tuning | [rna_protein_xlnet](https://github.com/dieterich-lab/rna_protein_xlnet) |

### Creating Your Own Plugin

See [PLUGIN_DEVELOPMENT.md](./PLUGIN_DEVELOPMENT.md) for detailed guide.

Quick overview:
1. Create package with `pyproject.toml`
2. Register entry point: `[project.entry-points."biolm.plugins"]`
3. Implement model, dataset, and config
4. Install and use: `poetry install && biolm fine-tune --model myplugin`

Template available at: [plugin-template/](./plugin-template/)

## 📖 Documentation

- **[PLUGIN_DEVELOPMENT.md](./PLUGIN_DEVELOPMENT.md)** - Create custom plugins
- **[PUBLISHING.md](./PUBLISHING.md)** - Release packages to PyPI
- **[QUICKSTART.md](./QUICKSTART.md)** - Tutorial and examples
- **[README_CONFIGS.md](./README_CONFIGS.md)** - Configuration reference

## 🔧 Development

### Running Tests

```bash
# Framework tests only
poetry run pytest tests/ --ignore=tests/integration --ignore=tests/end_to_end

# With plugins (integration tests)
poetry install --with plugins
poetry run pytest tests/integration/test_plugin_discovery.py
```

### Using Makefiles

Each plugin has a Makefile for common tasks:

```bash
# In plugin directory
make help           # Show available targets
make install        # Install plugin
make test           # Run plugin tests
make verify-plugin  # Check entry point registration
make bootstrap      # Full setup (framework + plugin)
```

## 🏗️ Architecture

### Entry Point System

Plugins register themselves via Python entry points:

```toml
# In plugin's pyproject.toml
[project.entry-points."biolm.plugins"]
mymodel = "my_plugin.config:get_mymodel_config"
```

Framework discovers plugins at runtime:
```python
import importlib.metadata
plugins = importlib.metadata.entry_points(group='biolm.plugins')
```

No hard-coded dependencies between framework and plugins!

### Version Compatibility

| Framework | Saluki | XLNet | Notes |
|-----------|--------|-------|-------|
| 0.0.3     | 0.1.0  | 0.1.0 | Current (biolm-2.0 branch) |
| 0.1.0     | TBD    | TBD   | First stable release |

Plugins specify compatible framework versions in their `pyproject.toml`:
```toml
dependencies = [
    "biolm>=0.0.3,<0.1.0",  # Semantic versioning
]
```

## 🤝 Contributing

### Framework Development

1. Fork and clone `biolm_utils`
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes and test: `poetry run pytest`
4. Submit PR to `biolm-2.0` branch

### Plugin Development

Plugins are independent repositories. See existing plugins for examples:
- [rna_saluki_cnn](https://github.com/dieterich-lab/rna_saluki_cnn)
- [rna_protein_xlnet](https://github.com/dieterich-lab/rna_protein_xlnet)

## 📜 License

MIT License - see [LICENSE](./LICENSE) file

## 🙏 Acknowledgments

Developed by the Dieterich Lab for bioinformatics research.

## 📞 Support

- Issues: [GitHub Issues](https://github.com/dieterich-lab/biolm_utils/issues)
- Discussions: [GitHub Discussions](https://github.com/dieterich-lab/biolm_utils/discussions)

