# Installation Guide

Complete installation instructions for BioLM 2.0 framework and plugins.

## Prerequisites

### Required
- **Python 3.10+** 
- **Poetry** ([installation guide](https://python-poetry.org/docs/#installation))
- **Git**

### Check Prerequisites

```bash
python3 --version    # Should be 3.10 or higher
poetry --version     # Should be 1.0+
git --version
```

## Quick Installation (Recommended)

Use the automated installation script for framework + plugins:

```bash
# 1. Clone repositories side by side
git clone https://github.com/dieterich-lab/biolm_utils.git
git clone https://github.com/dieterich-lab/rna_saluki_cnn.git
git clone https://github.com/dieterich-lab/rna_protein_xlnet.git

# 2. Run installation script
cd biolm_utils
./install.sh --with-plugins
```

The script will:
- ✅ Install framework dependencies
- ✅ Discover and install plugins from neighboring directories
- ✅ Configure entry points
- ✅ Run tests to verify installation

## Manual Installation

### Option 1: Framework Only

If you only need the framework without plugins:

```bash
cd biolm_utils
poetry install
```

### Option 2: Framework + Specific Plugin

```bash
# Install framework
cd biolm_utils
poetry install

# Install one plugin
cd ../rna_saluki_cnn
poetry install

# Verify plugin registration
cd ../biolm_utils
poetry run python -c "
import importlib.metadata
eps = importlib.metadata.entry_points(group='biolm.plugins')
for ep in eps:
    print(f'{ep.name}: {ep.value}')
"
```

### Option 3: Development Setup

For plugin development with editable installs:

```bash
# Install framework in development mode
cd biolm_utils
poetry install

# Install plugins in development mode
cd ../rna_saluki_cnn
poetry install

cd ../rna_protein_xlnet
poetry install
```

## Installation Script Details

The `install.sh` script supports several options:

```bash
./install.sh [OPTIONS]

Options:
  --with-plugins          Install framework + all discovered plugins
  --saluki-path PATH      Custom path to Saluki plugin
  --xlnet-path PATH       Custom path to XLNet plugin
  --skip-tests            Skip post-installation tests
  --help                  Show help message
```

### Examples

```bash
# Install only framework
./install.sh

# Install with plugins from default locations
./install.sh --with-plugins

# Install with custom plugin paths
./install.sh --with-plugins \
  --saluki-path /home/user/my_saluki \
  --xlnet-path /home/user/my_xlnet

# Quick install without running tests
./install.sh --with-plugins --skip-tests
```

## Verification

### Check Framework Installation

```bash
cd biolm_utils
poetry run python -c "import biolm; print(f'BioLM version: {biolm.__version__}')"
```

### Check Plugin Discovery

```bash
poetry run python -c "
import importlib.metadata

print('Registered plugins:')
eps = importlib.metadata.entry_points(group='biolm.plugins')
for ep in eps:
    print(f'  • {ep.name}')
    try:
        config = ep.load()()
        print(f'    ✅ Loaded successfully')
    except Exception as e:
        print(f'    ❌ Failed: {e}')
"
```

### Run Tests

```bash
# Framework tests only
poetry run pytest tests/ --ignore=tests/integration --ignore=tests/end_to_end

# With plugins (integration + end-to-end)
poetry run pytest tests/
```

Expected: **67 tests passing**

## Troubleshooting

### Poetry Not Found

**Problem:** `poetry: command not found`

**Solution:** Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Add to PATH (add to `~/.bashrc` or `~/.zshrc`):
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Python Version Mismatch

**Problem:** `Python 3.10+ is required`

**Solution:** Use pyenv or conda to install Python 3.10+:

```bash
# Using pyenv
pyenv install 3.10.12
pyenv local 3.10.12

# Using conda
conda create -n biolm python=3.10
conda activate biolm
```

### Plugin Not Discovered

**Problem:** Plugin doesn't appear in entry points list

**Solution:** 

1. Check plugin's `pyproject.toml` has correct entry point:
```toml
[project.entry-points."biolm.plugins"]
mymodel = "my_plugin.config:get_mymodel_config"
```

2. Reinstall plugin:
```bash
cd plugin_directory
poetry install
```

3. Verify from framework directory:
```bash
cd biolm_utils
poetry run python -c "
import importlib.metadata
eps = importlib.metadata.entry_points(group='biolm.plugins')
print([ep.name for ep in eps])
"
```

### Module Import Errors

**Problem:** `ModuleNotFoundError: No module named 'biolm'`

**Solution:** 

1. Ensure you're in the virtual environment:
```bash
cd biolm_utils
poetry shell
```

2. Or use `poetry run`:
```bash
poetry run python script.py
poetry run biolm fine-tune ...
```

### Dependency Conflicts

**Problem:** Poetry reports dependency conflicts

**Solution:**

1. Clear lock file and reinstall:
```bash
rm poetry.lock
poetry install
```

2. If issue persists, check plugin dependencies match framework requirements:
```toml
# In plugin's pyproject.toml
[tool.poetry.dependencies]
python = ">=3.10,<4.0"
biolm = {path = "../biolm_utils", develop = true}
torch = "^2.0.0"  # Should match framework's torch version
```

## Uninstallation

```bash
# Remove virtual environments
cd biolm_utils && rm -rf .venv
cd ../rna_saluki_cnn && rm -rf .venv
cd ../rna_protein_xlnet && rm -rf .venv

# Remove installed packages (optional)
rm -rf ~/.cache/pypoetry
```

## Next Steps

- **Configure your experiment:** See [CONFIGURATION.md](CONFIGURATION.md)
- **Run first training:** Follow quick start in main [README](../README.md)
- **Develop a plugin:** See [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md)
