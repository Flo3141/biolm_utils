# Plugin Template

Minimal working plugin template for BioLM 2.0.

## Quick Start

1. Copy this directory to create your plugin:
   ```bash
   cp -r plugin-template/ ../my-rna-model/
   ```

2. Rename files and classes:
   - Replace `my_plugin` with your plugin name
   - Replace `MyModel` with your model name
   - Update entry point in `pyproject.toml`

3. Install and test:
   ```bash
   cd my-rna-model
   poetry install
   poetry run python -c "import importlib.metadata; print(list(importlib.metadata.entry_points(group='biolm.plugins')))"
   ```

See `PLUGIN_DEVELOPMENT.md` for detailed guide.
