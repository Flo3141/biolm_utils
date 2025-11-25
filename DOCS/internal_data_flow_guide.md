# BioLM Utils — Internal Data Flow Guide

This document provides a detailed technical overview of the internal data flow within the biolm_utils framework. It covers how CLI arguments are parsed, configurations are loaded and validated, plugins are discovered and loaded, and how the execution pipeline works from start to finish.

Audience: framework maintainers, contributors, and advanced users who need to understand the internal workings for debugging, extending, or modifying the system.

---

## 1 — Entry Points and CLI Parsing

### 1.1 Main Entry Point (`biolm.py`)

The framework's execution begins in `biolm.py::main()`. This function serves as the primary CLI entry point and orchestrates the entire pipeline.

**Key responsibilities:**
- Parse command-line arguments using Hydra
- Load and validate configuration
- Discover and apply plugins
- Route execution to appropriate mode handlers (tokenize, pre-train, fine-tune, predict, interpret)

### 1.2 Hydra Configuration System

The framework uses Hydra for configuration management, which provides:

- Hierarchical configuration files (`conf/*.yaml`)
- Command-line overrides (`key=value` syntax)
- Configuration composition and inheritance
- Automatic config validation

**Configuration loading flow:**
```
CLI args → Hydra config parsing → BioLMConfig dataclass → Validation → Plugin application
```

**Example CLI invocation:**
```bash
python biolm.py mode=fine-tune data_source.filepath=data.csv training.nepochs=50
```

This gets parsed into a `BioLMConfig` object with nested attributes.

---

## 2 — Configuration Loading and Validation

### 2.1 `params.py::load_config()`

The `load_config()` function in `params.py` is responsible for:

1. **Parsing CLI arguments** using Hydra's configuration system
2. **Merging defaults** from `conf/*.yaml` files
3. **Applying overrides** from command line
4. **Creating BioLMConfig** dataclass instance
5. **Running validation** and GPU autodetection

**Key steps:**
```python
def load_config(overrides=None):
    # 1. Initialize Hydra with config path
    with initialize(config_path="conf"):
        # 2. Compose configuration with overrides
        cfg = compose(config_name="config", overrides=overrides)
    
    # 3. Convert to BioLMConfig dataclass
    args = BioLMConfig(**cfg)
    
    # 4. Validate configuration
    args.validate()
    
    # 5. Auto-detect GPUs
    args.autodetect_gpus()
    
    return args
```

### 2.2 BioLMConfig Structure

The `BioLMConfig` dataclass (`structured_config.py`) contains:

- **mode**: Execution mode (`tokenize`, `pre-train`, `fine-tune`, `predict`, `interpret`)
- **task**: Task type (`classification`, `regression`)
- **data_source**: Dataset configuration (filepath, cross-validation settings)
- **training**: Training hyperparameters (batch size, epochs, patience)
- **debugging**: Debug settings (GPU detection, verbosity)
- **settings**: Framework settings (MLflow, output paths)

### 2.3 Configuration Validation

Configuration validation happens in `BioLMConfig.validate()`:

- **Cross-validation consistency**: Ensures `crossvalidation`, `splitpos`, and `devsplits` are compatible
- **GPU settings**: Validates GPU count is a power of 2
- **Path validation**: Checks output directories exist or can be created
- **Plugin compatibility**: Validates plugin settings match the selected mode

---

## 3 — Plugin Discovery and Loading

### 3.1 Plugin Registry System

The framework supports dynamic plugin loading through:

1. **Entry points** (preferred for packaged plugins)
2. **Local plugins directory** (for development)
3. **Explicit registration** (programmatic)

### 3.2 Entry Point Discovery (`plugin_loader.py`)

**Entry point loading flow:**
```python
def discover_entrypoint_plugins():
    # 1. Use importlib.metadata to find entry points
    eps = entry_points(group="biolm_utils.plugins")
    
    # 2. Load each entry point
    for ep in eps:
        factory = ep.load()
        register_plugin(ep.name, factory)
```

**Example entry point (in pyproject.toml):**
```toml
[project.entry-points."biolm_utils.plugins"]
saluki = "saluki_plugin.config:get_saluki_config"
```

### 3.3 Plugin Application (`plugin_registry.py`)

When a plugin is applied:

```python
def apply_plugin(name: str):
    factory = get_plugin_factory(name)
    config_obj = factory()  # Call plugin factory
    
    # Handle both dict and PluginConfig instances
    if isinstance(config_obj, dict):
        config = Config(**config_obj)
    else:
        config = Config(**config_obj.__dict__)
    
    set_config(config)  # Store in global state
```

### 3.4 PluginConfig vs Legacy Config

The framework supports two plugin configuration systems:

**Legacy Config (dict-based):**
- Plugins return plain dictionaries
- Limited type safety
- Minimal documentation

**Modern PluginConfig (dataclass-based):**
- Plugins return `PluginConfig` instances
- Full type hints and validation
- Comprehensive field documentation
- Better IDE support

**Migration path:**
```python
# Old way
def get_plugin_config():
    return {
        "model_cls_for_finetuning": MyModel,
        "dataset_cls": MyDataset,
        # ... many fields without documentation
    }

# New way
def get_plugin_config():
    from biolm_utils.plugin_config import PluginConfig, PluginManager
    
    config = PluginConfig(
        model_cls_for_finetuning=MyModel,  # Well-documented field
        dataset_cls=MyDataset,
        # ... all fields have docstrings
    )
    
    PluginManager.set_config(config)
    return config
```

---

## 4 — Execution Pipeline

### 4.1 Mode Routing

After configuration loading, execution routes based on `args.mode`:

```python
def main():
    args = load_config()
    
    if args.mode == "tokenize":
        tokenize(args)
    elif args.mode in ["pre-train", "fine-tune"]:
        run_training_pipeline(args)
    elif args.mode == "predict":
        run_prediction_pipeline(args)
    elif args.mode == "interpret":
        run_interpretation_pipeline(args)
```

### 4.2 Training Pipeline Flow

For training modes (`pre-train`, `fine-tune`):

1. **Plugin Configuration**: `PluginManager.get_config()` retrieves active plugin
2. **Tokenizer Setup**: `get_tokenizer()` creates tokenizer instance
3. **Dataset Loading**: `get_dataset()` loads and preprocesses data
4. **Cross-Validation Setup**: `CrossValidator` orchestrates k-fold execution
5. **Per-Fold Execution**: `make_run_fn()` creates fold-specific runners

**Detailed flow:**
```python
def run_training_pipeline(args):
    # 1. Get plugin configuration
    config = PluginManager.get_config()
    
    # 2. Setup tokenizer
    tokenizer = get_tokenizer(args, tokenizer_path, config.tokenizer_cls, config.pretraining_required)
    
    # 3. Load dataset
    dataset = get_dataset(args, tokenizer, config.add_special_tokens, data_path, config.dataset_cls)
    
    # 4. Create run function factory
    run_fn = make_run_fn(args, config, tokenizer, tokenizer_for_trainer, dataset)
    
    # 5. Setup cross-validation
    cv = CrossValidator(params=args, dataset=dataset, run_once_fn=run_fn, base_paths=paths)
    
    # 6. Execute cross-validation
    results = cv.execute()
    
    return results
```

### 4.3 Runner Factory (`make_run_fn`)

The `make_run_fn` creates a function that executes a single training run:

```python
def make_run_fn(args, config, tokenizer, tokenizer_for_trainer, full_dataset):
    def run_single_fold(train_indices, val_indices, test_indices, model_load_path, model_save_path, report_file, rank_file):
        # 1. Split dataset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices) if test_indices else None
        
        # 2. Create data collator
        data_collator = select_data_collator(args.mode, config)
        
        # 3. Get model and config
        model, model_config = get_model_and_config(args, config, tokenizer)
        
        # 4. Setup trainer
        trainer = get_trainer(args, config, model, model_config, training_args, train_dataset, val_dataset, data_collator, compute_metrics, tokenizer_for_trainer)
        
        # 5. Execute training
        results = trainer.train()
        
        # 6. Optional testing
        if test_dataset:
            test_results = trainer.predict(test_dataset, metric_key_prefix="test")
            results.update(test_results.metrics)
        
        return results
    
    return run_single_fold
```

---

## 5 — Data Processing Pipeline

### 5.1 Tokenization Flow

**Tokenization mode execution:**
```python
def tokenize(args):
    # 1. Load tokenizer configuration
    config = PluginManager.get_config()
    
    # 2. Create tokenizer instance
    tokenizer = config.tokenizer_cls(
        vocab_size=args.tokenizer.vocabsize,
        max_len=args.tokenizer.maxtokenlength,
        **tokenizer_kwargs
    )
    
    # 3. Train tokenizer on data
    tokenizer.train_from_iterator(
        get_training_corpus(args.data_source.filepath),
        vocab_size=args.tokenizer.vocabsize,
        min_frequency=args.tokenizer.minfreq
    )
    
    # 4. Save tokenizer
    tokenizer.save_pretrained(args.paths.tokenizer)
```

### 5.2 Dataset Loading and Preprocessing

**Dataset creation flow:**
```python
def get_dataset(args, tokenizer, add_special_tokens, data_path, dataset_cls):
    # 1. Instantiate dataset class
    dataset = dataset_cls(
        data_path=data_path,
        tokenizer=tokenizer,
        add_special_tokens=add_special_tokens,
        **dataset_kwargs
    )
    
    # 2. Apply preprocessing
    if hasattr(dataset, 'preprocess'):
        dataset.preprocess()
    
    # 3. Apply centering/tokenization
    if args.data_preprocessing.centertoken:
        dataset.center_sequences(args.data_preprocessing.centertoken)
    
    return dataset
```

### 5.3 Data Collator Selection

**Data collator selection logic:**
```python
def select_data_collator(mode, config):
    if mode == "pre-train":
        # Use pretraining collator if specified, otherwise fallback
        collator_cls = config.datacollator_cls_for_pretraining or DefaultDataCollator
    else:
        # Use finetuning collator, fallback to default
        collator_cls = config.datacollator_cls_for_finetuning or DefaultDataCollator
    
    # Instantiate collator
    if hasattr(collator_cls, '__call__') and tokenizer is not None:
        return collator_cls(tokenizer=tokenizer)
    else:
        return collator_cls() if callable(collator_cls) else collator_cls
```

---

## 6 — Training Orchestration

### 6.1 CrossValidator Implementation

The `CrossValidator` manages k-fold cross-validation:

```python
class CrossValidator:
    def __init__(self, params, dataset, run_once_fn, base_paths):
        self.params = params
        self.dataset = dataset
        self.run_once_fn = run_once_fn
        self.base_paths = base_paths
    
    def execute(self):
        results = []
        
        for fold_idx in range(self._get_num_folds()):
            # 1. Generate fold indices
            train_idx, val_idx, test_idx = self._get_fold_indices(fold_idx)
            
            # 2. Create fold-specific paths
            fold_paths = self._create_fold_paths(fold_idx)
            
            # 3. Execute single fold
            fold_result = self.run_once_fn(
                train_idx, val_idx, test_idx,
                fold_paths.model_load_path,
                fold_paths.model_save_path,
                fold_paths.report_file,
                fold_paths.rank_file
            )
            
            results.append(fold_result)
        
        return self._aggregate_results(results)
```

### 6.2 Trainer Setup and Execution

**Trainer creation flow:**
```python
def get_trainer(args, config, model, model_config, training_args, train_dataset, val_dataset, data_collator, compute_metrics, tokenizer_for_trainer):
    # 1. Create transformers TrainingArguments
    training_args = TrainingArguments(
        output_dir=model_save_path,
        num_train_epochs=args.training.nepochs,
        per_device_train_batch_size=args.training.batchsize,
        per_device_eval_batch_size=args.training.batchsize,
        # ... other args from config
    )
    
    # 2. Create trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer_for_trainer,
        # ... callbacks and other config
    )
    
    return trainer
```

---

## 7 — Result Collection and Output

### 7.1 Metrics Aggregation

**Cross-validation result aggregation:**
```python
def _aggregate_results(self, fold_results):
    aggregated = {}
    
    for metric_name in fold_results[0].keys():
        if metric_name.endswith('_loss') or metric_name.endswith('_acc'):
            # Average numeric metrics across folds
            values = [fold[metric_name] for fold in fold_results]
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values
            }
        else:
            # Keep non-numeric results as lists
            aggregated[metric_name] = [fold[metric_name] for fold in fold_results]
    
    return aggregated
```

### 7.2 Output Path Management

**Path generation for experiments:**
```python
def _create_fold_paths(self, fold_idx):
    base_path = Path(self.base_paths.model_save_path)
    
    if self._is_cross_validation():
        fold_path = base_path / f"fold_{fold_idx}"
    else:
        fold_path = base_path
    
    return Paths(
        model_load_path=self.base_paths.model_load_path,
        model_save_path=str(fold_path),
        output_path=str(base_path.parent),
        report_file=str(fold_path / "report.json"),
        rank_file=str(fold_path / "rankings.csv")
    )
```

### 7.3 MLflow Integration (Optional)

When MLflow is enabled, results are automatically logged:

```python
def _log_to_mlflow(results, args, config):
    if not args.settings.mlflow.enabled:
        return
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            'mode': args.mode,
            'task': args.task,
            'learning_rate': config.learning_rate,
            # ... other config fields
        })
        
        # Log metrics
        for metric_name, metric_data in results.items():
            if isinstance(metric_data, dict) and 'mean' in metric_data:
                mlflow.log_metric(f"{metric_name}_mean", metric_data['mean'])
                mlflow.log_metric(f"{metric_name}_std", metric_data['std'])
            else:
                mlflow.log_metric(metric_name, metric_data)
```

---

## 8 — Error Handling and Validation

### 8.1 Configuration Validation

**Runtime validation checks:**
- Dataset file existence
- Tokenizer compatibility
- Model configuration validity
- GPU memory requirements
- Cross-validation parameter consistency

### 8.2 Error Propagation

**Error handling strategy:**
```python
try:
    # Execute pipeline step
    result = execute_step(args, config)
except ConfigurationError as e:
    logger.error(f"Configuration error: {e}")
    raise
except TrainingError as e:
    logger.error(f"Training failed: {e}")
    # Attempt cleanup
    cleanup_partial_results()
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Log full traceback for debugging
    logger.debug("Full traceback:", exc_info=True)
    raise
```

### 8.3 Logging and Debugging

**Logging levels and outputs:**
- **INFO**: Progress updates, key metrics
- **DEBUG**: Detailed internal state, API calls
- **WARNING**: Non-critical issues, deprecated usage
- **ERROR**: Failures that prevent execution

**Debug output includes:**
- Configuration dumps
- Dataset statistics
- Memory usage
- GPU utilization
- Timing information

---

## 9 — Extension Points

### 9.1 Custom Plugins

**Plugin extension points:**
- Model classes (pretraining/finetuning)
- Dataset implementations
- Tokenizer customizations
- Data collators
- Metrics computation
- Training callbacks

### 9.2 Framework Extensions

**Framework extension points:**
- Custom cross-validation strategies
- Alternative trainers (non-HuggingFace)
- Custom data loading pipelines
- Specialized tokenization schemes
- Result aggregation strategies

### 9.3 Configuration Extensions

**Configuration extension patterns:**
```python
@dataclass
class CustomConfig(BioLMConfig):
    """Extended configuration with custom fields."""
    custom_field: str = "default_value"
    
    def validate(self):
        super().validate()
        # Custom validation logic
        if self.custom_field not in VALID_OPTIONS:
            raise ValueError(f"Invalid custom_field: {self.custom_field}")
```

---

## 10 — Performance Considerations

### 10.1 Memory Management

**Memory optimization strategies:**
- Gradient accumulation for large batches
- Mixed precision training (FP16)
- Dataset streaming for large corpora
- GPU memory monitoring and cleanup

### 10.2 Parallelization

**Parallel execution patterns:**
- Multi-GPU training (DataParallel/DistributedDataParallel)
- Cross-validation fold parallelization
- Dataset preprocessing parallelization
- I/O operation overlapping

### 10.3 Caching and Optimization

**Performance optimizations:**
- Tokenizer caching
- Dataset preprocessing caching
- Model checkpoint resumption
- Result caching for expensive computations

---

## 3 — Experiment Tracking and Visualization

### 3.1 Tensorboard Integration

BioLM Utils integrates Tensorboard for real-time experiment tracking and visualization during training. Tensorboard logs are automatically written by the HuggingFace Trainer to the directory specified by the `outputpath` configuration (or fallback to the training output directory).

**How it works:**
- The Trainer writes metrics, losses, and other training statistics to Tensorboard log files.
- The log directory is set via `outputpath` in your config, ensuring logs are organized per experiment.
- No manual setup is required; logging is enabled by default when training starts.

**Launching Tensorboard:**
```bash
tensorboard --logdir <your_outputpath>
```
Replace `<your_outputpath>` with the path used in your experiment config.

**What you get:**
- Interactive plots of training/validation loss, metrics, and more
- Step-by-step progress tracking
- Easy comparison between runs

**Note:**
Tensorboard complements MLflow integration. MLflow handles experiment metadata, parameters, and artifacts, while Tensorboard provides rich visualizations of training dynamics.

---
