# BioLM 2.0 Framework

A modular PyTorch framework for training language models on biological sequences (RNA/protein). Features a **plugin architecture** where model implementations are separate packages developed and versioned independently.

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

2. **Install Plugins (Optional)**
   You can now install plugins directly using the `biolm` CLI:

   **For Saluki (RNA):**
   ```bash
   biolm install-plugin https://github.com/dieterich-lab/rna_saluki_cnn.git
   ```

   **For XLNet (Protein):**
   ```bash
   biolm install-plugin https://github.com/dieterich-lab/rna_protein_xlnet.git
   ```

### Verify Installation

You can list installed plugins with:
```bash
biolm list-plugins
```

### Run Your First Training

```bash
# Copy config template
cp -r biolm/examples/plugin_template my_experiment

# Edit my_experiment/config.yaml:
#   - outputpath: /path/to/results
#   - data_source.filepath: /path/to/data.txt
#   - plugin: saluki (RNA) or xlnet (protein)

# Run fine-tuning
poetry run biolm fine-tune --config-path ./my_experiment
```

**📖 For detailed instructions, see [docs/INSTALLATION.md](docs/INSTALLATION.md)**

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

# Step 2: Pre-train (XLNet only)
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
| **saluki** | Saluki CNN | RNA | ❌ No | RNA regulatory prediction (12K tokens) |
| **xlnet** | XLNet | Protein | ✅ Required | Protein function (512 tokens) |

## ⚙️ Configuration Basics

```yaml
# config.yaml
plugin: saluki                              # Model: saluki or xlnet
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

**📖 For complete reference, see [docs/CONFIGURATION.md](docs/CONFIGURATION.md)**

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[Installation Guide](docs/INSTALLATION.md)** | Detailed setup, troubleshooting |
| **[Configuration Reference](docs/CONFIGURATION.md)** | All parameters explained |
| **[Plugin Development](docs/PLUGIN_DEVELOPMENT.md)** | Create custom plugins |
| **[Publishing Guide](docs/PUBLISHING.md)** | Release to PyPI |

## 🏗️ Architecture

```
biolm_utils/              # Framework (this repo)
├── biolm/               # Core framework
│   ├── plugins/         # Plugin loader
│   └── examples/        # Config templates
└── tests/               # Tests (67 passing)

rna_saluki_cnn/          # Saluki plugin (separate repo)
├── saluki_plugin/       # Plugin code
└── pyproject.toml       # Entry point registration

rna_protein_xlnet/       # XLNet plugin (separate repo)
├── xlnet_plugin/        # Plugin code
└── pyproject.toml       # Entry point registration
```

### Plugin Discovery

Plugins register via **Python entry points**:

```toml
[project.entry-points."biolm.plugins"]
mymodel = "my_plugin.config:get_config"
```

Framework discovers them automatically at runtime—no hard dependencies!

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

**📖 Detailed test documentation:** [docs/TESTING.md](docs/TESTING.md)  
**📖 CI/CD workflows explained:** [docs/CI_CD.md](docs/CI_CD.md)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request

**Plugin development:** See [docs/PLUGIN_DEVELOPMENT.md](docs/PLUGIN_DEVELOPMENT.md)

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
