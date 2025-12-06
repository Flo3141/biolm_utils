# Configuration Reference

Complete guide to configuring BioLM experiments.

## Quick Start

```bash
# Copy template
cp -r biolm/examples/plugin_template my_experiment

# Edit my_experiment/config.yaml
# Minimal required changes:
#   - outputpath: /path/to/results
#   - data_source.filepath: /path/to/data.txt
#   - plugin: saluki or xlnet

# Run
poetry run biolm fine-tune --config-path ./my_experiment
```

## Configuration Structure

```
my_experiment/
├── config.yaml          # Main configuration
└── mode/                # Mode-specific overrides (optional)
    ├── tokenize.yaml
    ├── pre-train.yaml
    ├── fine-tune.yaml
    ├── predict.yaml
    └── interpret.yaml
```

## Essential Parameters

### Core Settings

```yaml
# config.yaml
plugin: saluki                              # Model: saluki (RNA) or xlnet (protein)
task: regression                            # regression or classification
outputpath: /absolute/path/to/results      # Output directory

# Data configuration
data_source:
  filepath: /path/to/data.txt              # Tab-separated input file
  columnsep: "\t"                           # Column separator
  idpos: 1                                  # ID column (1-indexed)
  seqpos: 3                                 # Sequence column (1-indexed)  
  labelpos: 2                               # Label column (1-indexed)
  splitratio: [70, 15, 15]                 # Train/validation/test split (%)

# Training
training:
  nepochs: 100                              # Number of epochs
  batchsize: 8                              # Batch size
  patience: 10                              # Early stopping patience
  seed: 42                                  # Random seed

# Model (optional - plugin defaults usually sufficient)
model:
  num_layers: 2                             # Model depth
  hidden_size: 128                          # Hidden dimension
```

## Data Format

Your input file must be **tab-separated**:

```
ID          Label    Sequence
seq_001     1.5      a,t,g,c,a,g,t,c,...
seq_002     2.3      a,t,g,c,a,g,t,c,...
```

**Important Notes:**
- Column positions are **1-indexed** (1, 2, 3... not 0, 1, 2)
- Saluki requires comma-separated nucleotides: `a,t,g,c,...`
- XLNet uses raw sequences: `MKVLWAALLVT...`

## Plugin-Specific Requirements

### Saluki (RNA)

```yaml
plugin: saluki

tokenization:
  encoding: atomic                          # Required: one-hot encoding

training:
  blocksize: 12288                          # Required: fixed sequence length
  
data_source:
  seqpos: 3                                 # Comma-separated nucleotides
  tokensep: ","                             # Required for Saluki
```

**Pre-training:** ❌ Not supported

### XLNet (Protein)

```yaml
plugin: xlnet

tokenization:
  encoding: bpe                             # Byte-pair encoding
  vocabsize: 20000                          # Vocabulary size

training:
  blocksize: 512                            # Standard transformer length
  
data_source:
  seqpos: 3                                 # Raw amino acid sequences
```

**Pre-training:** ✅ Required before fine-tuning

## Complete Parameter Reference

### data_source

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filepath` | string | **required** | Path to input data file |
| `stripheader` | bool | `false` | Remove first line of file |
| `columnsep` | string | `"\t"` | Column separator |
| `tokensep` | string | `","` | Token separator (Saluki) |
| `idpos` | int | `1` | ID column position (1-indexed) |
| `seqpos` | int | `3` | Sequence column position |
| `labelpos` | int | `2` | Label column position |
| `splitratio` | list | `[70,15,15]` | Train/val/test split percentages |
| `crossvalidation` | bool | `false` | Enable cross-validation |
| `splitpos` | int | `null` | Column defining pre-defined splits |
| `devsplits` | list | `null` | Validation split IDs |
| `testsplits` | list | `null` | Test split IDs |

### training

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nepochs` | int | `100` | Number of training epochs |
| `batchsize` | int | `8` | Per-device batch size |
| `gradacc` | int | `4` | Gradient accumulation steps |
| `blocksize` | int | plugin default | Max sequence length (with padding) |
| `patience` | int | `10` | Early stopping patience |
| `seed` | int | `42` | Random seed |
| `learning_rate` | float | plugin default | Learning rate |
| `max_grad_norm` | float | plugin default | Gradient clipping |
| `weight_decay` | float | plugin default | Weight decay |
| `scaling` | string | `"log"` | Target scaling: `log` or `standard` |
| `resume` | bool | `false` | Resume from checkpoint |

### tokenization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoding` | string | `"bpe"` | Encoding: `bpe` or `atomic` |
| `vocabsize` | int | `20000` | Vocabulary size (BPE only) |
| `minfreq` | int | `2` | Minimum token frequency |
| `maxtokenlength` | int | `10` | Maximum token length |
| `samplesize` | int | `null` | Downsample data if needed |
| `lefttailing` | bool | `false` | Truncate from left vs right |

### model

Model parameters depend on the plugin. Common ones:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_layers` | int | varies | Number of layers |
| `hidden_size` | int | varies | Hidden dimension size |
| `num_heads` | int | varies | Attention heads (transformers) |
| `intermediate_size` | int | varies | FFN intermediate size |
| `dropout` | float | varies | Dropout rate |

### debugging

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `accelerator` | string | `"auto"` | Device: `auto`, `cpu`, `gpu` |
| `silent` | bool | `false` | Suppress verbose output |
| `dev` | bool | `false` | Use tiny subset for testing |
| `forcenewdata` | bool | `false` | Rebuild dataset cache |

### settings.mlflow

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | `false` | Enable MLflow tracking |
| `tracking_uri` | string | `null` | MLflow server URI |
| `experiment_name` | string | `null` | Experiment name |

## Command-Line Overrides

Override any parameter from command line:

```bash
# Override single parameter
poetry run biolm fine-tune --config-path ./my_experiment \
  training.nepochs=50

# Override multiple parameters
poetry run biolm fine-tune --config-path ./my_experiment \
  training.nepochs=50 \
  training.batchsize=16 \
  data_source.splitratio=[80,10,10]

# Add parameter not in config (use +)
poetry run biolm fine-tune --config-path ./my_experiment \
  +training.blocksize=512
```

## Training Pipeline

### 1. Tokenization

Build vocabulary from your sequences:

```bash
poetry run biolm tokenize --config-path ./my_experiment
```

Creates: `{outputpath}/tokenizer.json`

### 2. Pre-training (XLNet only)

Pre-train on unlabeled sequences:

```bash
poetry run biolm pre-train --config-path ./my_experiment
```

Creates: `{outputpath}/pre-train/`

### 3. Fine-tuning

Train on your labeled task:

```bash
poetry run biolm fine-tune --config-path ./my_experiment
```

Creates:
- `{outputpath}/fine-tune/checkpoint-*/`
- `{outputpath}/fine-tune/test_results.json`

### 4. Prediction

Make predictions on new data:

```bash
poetry run biolm predict --config-path ./my_experiment
```

### 5. Interpretation

Compute feature importance:

```bash
poetry run biolm interpret --config-path ./my_experiment
```

## Example Configurations

### Minimal (Saluki)

```yaml
# config.yaml
plugin: saluki
task: regression
outputpath: /home/user/experiments/rna_exp1

data_source:
  filepath: data/rna_sequences.txt
  idpos: 1
  seqpos: 3
  labelpos: 2

training:
  nepochs: 50
  batchsize: 8
```

### Full (XLNet with Pre-training)

```yaml
# config.yaml
plugin: xlnet
task: classification
outputpath: /home/user/experiments/protein_exp1

data_source:
  filepath: data/proteins.txt
  idpos: 1
  seqpos: 2
  labelpos: 3
  splitratio: [70, 15, 15]

tokenization:
  encoding: bpe
  vocabsize: 10000
  minfreq: 3

training:
  nepochs: 100
  batchsize: 16
  blocksize: 512
  patience: 20
  learning_rate: 0.0001

model:
  num_layers: 4
  hidden_size: 256
  num_heads: 8

debugging:
  accelerator: gpu

settings:
  mlflow:
    enabled: true
    experiment_name: protein_classification
```

## Troubleshooting

### "blocksize must be 12288 for Saluki"

**Problem:** You're overriding blocksize for Saluki

**Solution:** Remove `training.blocksize` from config or use default

### "Could not convert string to float"

**Problem:** Wrong column positions (labelpos pointing to sequence)

**Solution:** Check your data format and verify `idpos`, `seqpos`, `labelpos` are correct (1-indexed!)

### "Tokenizer not found"

**Problem:** Haven't run tokenization step

**Solution:** Run `poetry run biolm tokenize --config-path ./my_experiment` first

### "Pre-training required for XLNet"

**Problem:** Trying to fine-tune XLNet without pre-training

**Solution:** Run `poetry run biolm pre-train --config-path ./my_experiment` first

## Next Steps

- **Run first training:** Follow quick start above
- **Develop plugin:** See [PLUGIN_DEVELOPMENT.md](PLUGIN_DEVELOPMENT.md)
- **Publish to PyPI:** See [PUBLISHING.md](PUBLISHING.md)
