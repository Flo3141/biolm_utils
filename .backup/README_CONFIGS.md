# BioLM Configuration Guide

## Quick Start: Use an Example

**New to BioLM?** Start with one of the minimal examples:

```bash
# Minimal example (bare minimum to run)
poetry run biolm fine-tune --config-path ./exampleconfigs/minimal

# Real-world example: Saluki fine-tuning on RNA
poetry run biolm fine-tune --config-path ./exampleconfigs/saluki-rna-finetuning

# Real-world example: XLNet fine-tuning on proteins
poetry run biolm fine-tune --config-path ./exampleconfigs/xlnet-protein-finetuning
```

Then copy `PLUGIN_TEMPLATE/` and modify it for your data!

---

## Configuration Structure

All configs follow this Hydra structure:

```
your_experiment/
├── config.yaml                # Main config (framework + plugin settings)
└── mode/                      # Mode-specific overrides
    ├── tokenize.yaml          # Settings for tokenization
    ├── fine-tune.yaml         # Settings for fine-tuning
    ├── pre-train.yaml         # Settings for pre-training
    ├── predict.yaml           # Settings for prediction
    └── interpret.yaml         # Settings for interpretation
```

**Example:**
```bash
# Uses config.yaml + mode/fine-tune.yaml
poetry run biolm fine-tune --config-path ./exampleconfigs/minimal
```

---

## Framework Parameters (All Configs)

### Core Settings

| Parameter | Type | Required | Default | Notes |
|-----------|------|----------|---------|-------|
| `mode` | string | ✅ Yes | — | `tokenize` / `pre-train` / `fine-tune` / `predict` / `interpret` |
| `task` | string | ✅ Yes | — | `regression` or `classification` (plugin-dependent) |
| `plugin` | string | ✅ Yes | — | Plugin name: `saluki` or `xlnet` |
| `outputpath` | path | ✅ Yes | — | Where to save models/results. Use absolute path for clarity. |

**Example:**
```yaml
mode: fine-tune
task: regression
plugin: saluki
outputpath: /home/user/experiments/my_rna_experiment
```

---

### Data Source Configuration

Used for **tokenization, pre-training, and fine-tuning**. Describe your data file structure:

```yaml
data_source:
  filepath: "path/to/your/data.txt"           # Can be relative or absolute
  stripheader: false                           # Remove first line?
  columnsep: "\t"                              # Column separator: "\t", ",", "|", etc.
  tokensep: ","                                # Token separator (if sequences are pre-tokenized)
  specifiersep: null                           # Usually null
  idpos: 1                                     # Column with ID (1-indexed)
  seqpos: 3                                    # Column with sequence (1-indexed)
  labelpos: 2                                  # Column with label (for fine-tuning; 1-indexed)
  
  # Cross-validation or data splitting
  crossvalidation: false                       # Use cross-validation?
  splitratio: [80, 20]                        # [train%, val%, test%] - must sum to 100
  splitpos: null                               # Column defining splits (if pre-split in data)
  devsplits: null                              # Which splits for validation (if using splitpos)
  testsplits: null                             # Which splits for testing (if using splitpos)
```

**Data File Format:**
```
ID    Label    Sequence
seq1  1.5      a,t,g,c,g,a,t,c,...    (for Saluki: comma-separated nucleotides)
seq2  2.3      a,t,g,c,g,a,t,c,...    OR
seq1  1.5      atgcgatc...             (for XLNet: raw sequences)
```

**Column Positions (1-indexed!):**
```
ID    Label    Sequence       Other
1     2        3              4      ← use idpos=1, labelpos=2, seqpos=3
```

---

### Tokenization Configuration

Used for creating your tokenizer (**before fine-tuning**):

```yaml
tokenization:
  encoding: bpe                           # Byte-pair encoding (standard)
  samplesize: null                        # Downsample if data too large (e.g., 50000)
  vocabsize: 20000                        # Vocabulary size
  minfreq: 2                              # Min frequency for token inclusion
  maxtokenlength: 10                      # Max token length
  lefttailing: false                      # Cut from left (true) or right (false)?
  atomicreplacements: null                # Advanced: predefined atomic tokens
```

**Plugin-Specific Override:**
- **Saluki**: `encoding` must be `atomic` (one-hot encoding); `vocabsize` is ignored
- **XLNet**: `encoding` can be `bpe` (default) for flexible tokenization

---

### Training Configuration

Used for **pre-training and fine-tuning**. Place in `config.yaml` or override in `mode/fine-tune.yaml`:

```yaml
training:
  seed: 42                                # Random seed for reproducibility
  batchsize: 8                            # Per-device batch size
  gradacc: 4                              # Gradient accumulation steps
  blocksize: null                         # Sequence length - AUTO-SET by plugin!
  nepochs: 100                            # Number of epochs
  patience: 10                            # Early stopping patience
  resume: false                           # Resume from checkpoint?
  scaling: log                            # Target scaling: "log" or "standard"
  weightedregression: false               # Weight loss by target value?
```

⚠️ **IMPORTANT: Blocksize is Enforced by Plugin**
- **Saluki**: blocksize = 12288 (ENFORCED - do not override)
- **XLNet**: blocksize = 512 (ENFORCED - do not override)

Don't set blocksize in your config; the plugin will enforce it automatically.

---

### Inference & Interpretation Configuration

```yaml
inference:
  looscores: {}                           # Leave-one-out scores (for interpret mode)
  pretrainedmodel: null                   # Path to pre-trained model

debugging:
  silent: false                           # Suppress debug output?
  dev: false                              # Development mode (mini dataset)?
  getdata: false                          # Just load data, don't train?
  forcenewdata: false                     # Rebuild dataset cache?
  accelerator: auto                       # "auto", "cpu", "gpu", "tpu"
  detected_ngpus: null                    # Auto-detected; don't set manually
```

---

### MLflow & Settings

```yaml
settings:
  mlflow:
    enabled: true                         # Enable experiment tracking?
    tracking_uri: null                    # Auto-set if null
    experiment_name: null                 # Auto-set if null
  
  data_pre_processing: {}                 # Plugin-specific preprocessing
  environment: {}                         # Environment variables
```

---

## Plugin-Specific Constraints

### Saluki (RNA Sequence Analysis)

**Blocksize:** 12288 (fixed, cannot change)

**Data Format:**
```
ID        Label    Sequence (comma-separated nucleotides)
seq_001   1.5      a,t,g,c,a,g,t,c,g,a,g,c,a,t,g,c,...
seq_002   2.3      a,t,g,c,a,g,t,c,g,a,g,c,a,t,g,c,...
```

**Tokenization:**
```yaml
tokenization:
  encoding: atomic              # Must be "atomic" (one-hot)
  # vocabsize, minfreq ignored for Saluki
```

**Task Support:** `regression` and `classification`

**Key Features:**
- Nucleotide-level one-hot encoding (4 dimensions: a,t,g,c)
- CNN-based architecture
- Works with long sequences (blocksize 12288)

---

### XLNet (Protein & RNA Sequence Analysis)

**Blocksize:** 512 (fixed, cannot change)

**Data Format (flexible):**
```
ID        Label    Sequence (any token format)
prot_001  1.5      MKVLWAALLVTFLAGCAKAKQ...    (protein: amino acids)
or
seq_001   2.3      atgcgatcgatcgatcgatc...     (RNA: nucleotides)
```

**Tokenization:**
```yaml
tokenization:
  encoding: bpe                 # Byte-pair encoding (flexible)
  vocabsize: 20000              # Customize as needed
  minfreq: 2
```

**Task Support:** `regression` and `classification`

**Key Features:**
- Transformer-based (XLNet)
- Flexible tokenization (BPE)
- Supports both proteins and RNA
- Better for transfer learning

---

## Common Patterns

### Pattern 1: Simple Fine-Tuning on Your Data

```yaml
# config.yaml
mode: fine-tune
task: regression
plugin: saluki
outputpath: /absolute/path/to/experiments/my_experiment

data_source:
  filepath: /path/to/my_data.txt
  stripheader: false
  columnsep: "\t"
  tokensep: ","
  idpos: 1
  seqpos: 3
  labelpos: 2
  splitratio: [80, 20]

training:
  seed: 42
  batchsize: 8
  nepochs: 50
  patience: 5
```

```yaml
# mode/fine-tune.yaml
training:
  batchsize: 16                 # Override batch size for fine-tuning
  nepochs: 100

settings:
  mlflow:
    enabled: true
```

### Pattern 2: Cross-Validation

```yaml
data_source:
  filepath: /path/to/my_data.txt
  splitpos: 4                   # Column 4 defines splits
  crossvalidation: true
  # Specify different splits per fold in devsplits
  devsplits: [[1], [2], [3], [4]]
  testsplits: [[5]]
```

### Pattern 3: Different Data for Tokenization vs Fine-tuning

```yaml
# Use different files
data_source:  # For fine-tuning
  filepath: /path/to/fine_tune_data.txt
  
tokenization:
  # Saluki uses atomic encoding; XLNet uses BPE
  # Can provide separate tokenization data via framework
```

---

## Output Structure

After running `poetry run biolm fine-tune --config-path ./exampleconfigs/minimal`, you'll get:

```
outputpath/
├── fine-tune/
│   ├── checkpoint-1/          # Intermediate checkpoints
│   ├── checkpoint-2/
│   ├── pytorch_model.bin       # Best model weights
│   ├── config.json             # Model config
│   ├── training_args.bin       # Training arguments
│   └── fine-tune_dataset.pkl   # Cached dataset
├── fine-tune.log              # Experiment log
└── mlruns/                    # MLflow tracking (if enabled)
    └── 0/
        └── <run_id>/
            ├── metrics/
            ├── params/
            └── artifacts/
```

---

## Troubleshooting

### Error: "blocksize must be X"
→ Don't set `blocksize` in config! The plugin enforces it automatically.

### Error: "Column position X out of range"
→ Check your data file with `head -3 your_data.txt`
→ Remember: positions are **1-indexed**, not 0-indexed!

### Error: "Data file has X rows but Y expected"
→ Check `stripheader` setting
→ Verify separator is correct with `head your_data.txt | cat -A` (shows whitespace)

### Model not improving
→ Check `scaling` setting (use "log" for data with wide range)
→ Increase `nepochs` or adjust `batchsize`
→ Check data quality: verify labels and sequences are correct

### Output files not created
→ Use absolute path for `outputpath`
→ Ensure directory path is writable

---

## Next Steps

1. **Copy a template:** `cp -r PLUGIN_TEMPLATE my_experiment`
2. **Edit config:** Open `my_experiment/config.yaml` and update paths
3. **Run:** `poetry run biolm fine-tune --config-path ./my_experiment`
4. **Monitor:** Check `outputpath/fine-tune.log` for progress

**Questions?** See individual plugin READMEs or check example configs in `./exampleconfigs/`
