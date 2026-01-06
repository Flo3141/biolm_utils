# Plugin Contract

To add a model to BioLM, you need to expose a **configuration factory** via a Python entry point.

## 1. The Code (Example)

Here is a complete example of a plugin configuration (e.g., `my_plugin/config.py`).

```python
from biolm.plugin_config import PluginConfig, PluginManager
from transformers import PreTrainedTokenizerFast, DefaultDataCollator
from my_plugin.dataset import MyDataset
from my_plugin.models import MyModel

def get_plugin_config():
    """Factory function returning the plugin configuration."""
    
    config = PluginConfig(
        # -- Required --
        model_cls_for_finetuning=MyModel,
        dataset_cls=MyDataset,
        
        # -- Optional / Defaults --
        tokenizer_cls=PreTrainedTokenizerFast,
        datacollator_cls_for_finetuning=DefaultDataCollator,
        
        # -- Training Hyperparameters (Fixed per plugin) --
        learning_rate=1e-4,
        max_grad_norm=1.0,
        weight_decay=0.01,
        
        # -- Pre-training (Optional) --
        pretraining_required=False,  # Set True if model needs pre-training
        model_cls_for_pretraining=None,
    )
    
    # Register as active config
    PluginManager.set_config(config)
    return config
```

### Key Fields

| Field | Description |
| :--- | :--- |
| `model_cls_for_finetuning` | Your PyTorch `nn.Module` for downstream tasks. |
| `dataset_cls` | Your `torch.utils.data.Dataset` class. Must handle `data_source` config (filepath, delimiter, etc.). |
| `learning_rate` | **Fixed** learning rate. Users cannot override this via Hydra to ensure reproducibility. |
| `pretraining_required` | If `True`, users must run `pre-train` before `fine-tune`. |

## 2. The Entry Point

Register your factory function in `pyproject.toml` (or `setup.cfg`) under the `biolm.plugins` group.

```toml
[project.entry-points."biolm.plugins"]
# The name 'my_cool_model' is what users will use in config: plugin=my_cool_model
my_cool_model = "my_plugin.config:get_plugin_config"
```

## 3. Dataset Implementation

Your dataset class receives the global config. It must respect the `data_source` settings.

```python
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split: str):
        # cfg.data_source.filepath -> Path to data
        # cfg.data_source.columnsep -> Delimiter (e.g. "\t")
        # cfg.data_source.idpos/seqpos/labelpos -> Column indices
        pass
```

## 4. Verification

1. **Install**: `poetry run biolm develop-plugin /path/to/your/plugin` (runs `pip install -e` on the checkout)
    - alternatively, `pip install -e .` works as well when already inside the repo
2. **List**: `biolm list-plugins` (should show `my_cool_model`)
3. **Run**: `biolm fine-tune plugin=my_cool_model ...`
