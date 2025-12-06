# Plugin Development Guide

This guide explains how to create custom plugins for the BioLM 2.0 framework.

## Overview

BioLM 2.0 uses a **plugin architecture** where model implementations are separate packages that register themselves via Python entry points. The framework discovers and loads plugins dynamically at runtime.

## Quick Start: Creating a New Plugin

### 1. Project Structure

```
my_rna_model/
├── pyproject.toml           # Package metadata + entry point registration
├── README.md
├── my_plugin/
│   ├── __init__.py
│   ├── config.py            # Plugin registration function
│   ├── models.py            # Model implementation
│   └── dataset.py           # Dataset implementation
└── tests/
    └── test_plugin.py
```

### 2. Minimal `pyproject.toml`

```toml
[project]
name = "my-rna-model"
version = "0.1.0"
description = "My custom RNA model plugin for BioLM"
requires-python = ">=3.10"
dependencies = [
    "biolm>=0.0.3",  # Framework dependency
    "torch>=2.0",
    "transformers>=4.0",
]

# ⭐ CRITICAL: Entry point registration
[project.entry-points."biolm.plugins"]
mymodel = "my_plugin.config:get_mymodel_config"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "my-rna-model"
version = "0.1.0"
description = "My custom RNA model plugin"

[tool.poetry.dependencies]
python = ">=3.10,<4.0"
biolm = {path = "../biolm_utils", develop = true}  # Dev: local path
# biolm = "^0.0.3"  # Production: published version
```

### 3. Plugin Registration (`my_plugin/config.py`)

```python
"""Plugin configuration that gets called by the framework."""

from biolm.plugin_config import PluginConfig
from transformers import DefaultDataCollator, PreTrainedTokenizerFast

from .models import MyModel
from .dataset import MyDataset


def get_mymodel_config():
    """
    Factory function called by framework at runtime.
    
    The framework calls this when user specifies model='mymodel' in their config.
    """
    config = PluginConfig(
        # Model classes
        model_cls_for_pretraining=None,        # If you support pretraining
        model_cls_for_finetuning=MyModel,      # Your main model class
        
        # Data handling
        dataset_cls=MyDataset,
        tokenizer_cls=PreTrainedTokenizerFast,
        datacollator_cls_for_pretraining=None,
        datacollator_cls_for_finetuning=DefaultDataCollator,
        
        # Training settings
        add_special_tokens=False,
        pretraining_required=False,
        learning_rate=0.001,
        max_grad_norm=0.4,
        weight_decay=0.001,
        
        # Advanced (optional)
        config_cls=None,
        special_tokenizer_for_trainer_cls=None,
    )
    
    return config
```

### 4. Model Implementation (`my_plugin/models.py`)

```python
"""Your model implementation."""

import torch.nn as nn
from biolm.base_dataset import BaseModel  # Or use transformers.PreTrainedModel


class MyModel(BaseModel):
    """Your custom RNA model."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Access config attributes
        input_size = config.input_size
        num_labels = config.num_labels
        
        # Build your model
        self.encoder = nn.Linear(input_size, 128)
        self.classifier = nn.Linear(128, num_labels)
    
    def forward(self, input_ids, **kwargs):
        """Forward pass - must return dict with 'logits' key."""
        x = self.encoder(input_ids)
        logits = self.classifier(x)
        return {"logits": logits}
    
    @staticmethod
    def get_config(args, config_cls, tokenizer, dataset, nlabels):
        """
        Create model config from training arguments.
        
        Called by framework during setup to prepare model configuration.
        """
        from transformers import PretrainedConfig
        
        config = PretrainedConfig(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Add custom attributes
        config.input_size = dataset.sequence_length
        config.num_labels = nlabels
        
        return config
```

### 5. Dataset Implementation (`my_plugin/dataset.py`)

```python
"""Your dataset implementation."""

import torch
from biolm.rna_datasets import RNABaseDataset


class MyDataset(RNABaseDataset):
    """Custom dataset for your model."""
    
    def __init__(self, **args):
        super().__init__(**args)
        # Add custom initialization
        self.sequence_length = 512
    
    def __getitem__(self, i):
        """Get a single example."""
        example = self.examples[i].copy()
        
        # Convert to tensor
        example["input_ids"] = torch.tensor(
            example["input_ids"], 
            dtype=torch.long
        )
        
        return example
```

## Entry Point System Explained

### How It Works

1. **Registration**: When your package is installed, Poetry/pip registers the entry point
2. **Discovery**: Framework queries `importlib.metadata.entry_points(group='biolm.plugins')`
3. **Loading**: Framework calls your `get_mymodel_config()` function
4. **Usage**: Model classes become available for training/evaluation

### Entry Point Format

```toml
[project.entry-points."biolm.plugins"]
plugin_name = "module.path:function_name"
```

- `plugin_name`: What users specify in config (`model: mymodel`)
- `module.path`: Python import path
- `function_name`: Function that returns `PluginConfig`

### Verification

Test that your plugin is registered:

```bash
poetry install  # Install your plugin
poetry run python -c "
import importlib.metadata
eps = importlib.metadata.entry_points(group='biolm.plugins')
for ep in eps:
    print(f'{ep.name}: {ep.value}')
"
```

Should show:
```
mymodel: my_plugin.config:get_mymodel_config
```

## Installation & Development

### Development Setup

```bash
# Clone repos side by side
cd ~/projects
git clone https://github.com/dieterich-lab/biolm_utils.git
git clone https://github.com/yourorg/my_rna_model.git

# Install framework
cd biolm_utils
poetry install

# Install your plugin
cd ../my_rna_model
poetry install

# Link plugin to framework (for development)
cd ../biolm_utils
# Update pyproject.toml to add your plugin path
poetry install --with plugins
```

### Testing Your Plugin

```python
# tests/test_plugin.py
def test_plugin_loads():
    from biolm.plugin_config import PluginManager
    config = PluginManager.load_plugin('mymodel')
    assert config is not None
    assert config.model_cls_for_finetuning is not None
```

### Using Your Plugin

```yaml
# config.yaml
mode: fine-tune
model: mymodel  # Your plugin name!
task: regression

data_source:
  train: data/train.tsv
  dev: data/dev.tsv
```

Run with:
```bash
biolm fine-tune --config-path config.yaml
```

## Best Practices

### ✅ DO

- Use semantic versioning
- Pin framework version: `biolm = "^0.0.3"`
- Write tests for your plugin
- Document model requirements (blocksize, encoding, etc.)
- Validate inputs in `__init__` and `get_config`
- Return dict with `"logits"` key from `forward()`

### ❌ DON'T

- Hard-code paths or system-specific settings
- Depend on framework internals (use public APIs only)
- Forget to register entry point
- Mix training logic into model class
- Break backward compatibility without version bump

## Advanced Topics

### Custom Tokenizers

```python
config = PluginConfig(
    tokenizer_cls=MyCustomTokenizer,
    special_tokenizer_for_trainer_cls=MyTokenizerWrapper,
    ...
)
```

### Custom Data Collators

```python
from transformers import DataCollatorWithPadding

config = PluginConfig(
    datacollator_cls_for_finetuning=DataCollatorWithPadding,
    ...
)
```

### Model-Specific Config Classes

```python
from transformers import PretrainedConfig

class MyModelConfig(PretrainedConfig):
    model_type = "mymodel"
    
    def __init__(self, hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

config = PluginConfig(
    config_cls=MyModelConfig,
    ...
)
```

## Troubleshooting

### Plugin Not Found

```python
# Check if registered
import importlib.metadata
eps = list(importlib.metadata.entry_points(group='biolm.plugins'))
print([ep.name for ep in eps])
```

### Import Errors

- Ensure `biolm` is in dependencies
- Check that module paths in entry point are correct
- Verify package is installed: `pip list | grep my-rna-model`

### Config Not Loading

- Verify `get_mymodel_config()` returns `PluginConfig` instance
- Check for exceptions in config function
- Test loading manually: `PluginManager.load_plugin('mymodel')`

## Example Plugins

See existing plugins for reference:
- **Saluki**: `/home/pwiesenbach/rna_saluki_cnn/`
- **XLNet**: `/home/pwiesenbach/rna_protein_xlnet/`

## Publishing Your Plugin

See `PUBLISHING.md` for details on releasing to PyPI.

## Support

- Framework issues: https://github.com/dieterich-lab/biolm_utils/issues
- Plugin template: `/prj/RNA_NLP/biolm_utils/plugin-template/`
