# Quick Configuration Reference

## 30-Second Setup

```bash
# 1. Copy the template
cp -r PLUGIN_TEMPLATE my_experiment

# 2. Edit these 3 things in my_experiment/config.yaml:
#    - outputpath: /absolute/path/to/save/results
#    - data_source.filepath: /path/to/your/data.txt
#    - plugin: saluki (for RNA) or xlnet (for protein)

# 3. Run!
poetry run biolm fine-tune --config-path ./my_experiment
```

## Data Format

Your data file should be **tab-separated**:

```
ID         Label    Sequence
seq_001    1.5      a,t,g,c,a,g,t,c,...   (Saluki format)
seq_002    2.3      a,t,g,c,a,g,t,c,...
```

OR for XLNet:
```
protein_id  binding_affinity  MKVLWAALLVT...
prot_001    1.5              MKVLWAALLVT...
prot_002    2.3              MKIFVSYDTSA...
```

**Remember:** Column positions are **1-indexed** (1, 2, 3... not 0, 1, 2...)

## Essential Parameters

| Parameter | What It Does | Example |
|-----------|--------------|---------|
| `plugin` | Which model: `saluki` (RNA) or `xlnet` (protein) | `plugin: saluki` |
| `outputpath` | Where to save results | `/home/user/experiments/exp1` |
| `data_source.filepath` | Your data file path | `/data/my_rna_sequences.txt` |
| `task` | `regression` or `classification` | `task: regression` |
| `training.batchsize` | Batch size (adjust for GPU memory) | `batchsize: 8` |
| `training.nepochs` | How long to train | `nepochs: 100` |

## Plugin Constraints

**Saluki (RNA):**
- blocksize: **12288** (fixed, don't override)
- Data format: comma-separated nucleotides (`a,t,g,c,a,g,t,c`)
- Encoding: atomic (one-hot)

**XLNet (Protein/RNA):**
- blocksize: **512** (fixed, don't override)
- Data format: raw sequences (`MKVLWAALLVT...` or `atgcgatc...`)
- Encoding: flexible BPE

## Common Commands

```bash
# Run fine-tuning with your config
poetry run biolm fine-tune --config-path ./my_experiment

# Test on a small dataset (dev mode)
# Edit my_experiment/config.yaml: set debugging.dev: true
poetry run biolm fine-tune --config-path ./my_experiment

# View results
tail -f /outputpath/fine-tune.log
ls /outputpath/fine-tune/
```

## Example Configs

**Start with these:**

```bash
# Minimal example (copy this and modify)
poetry run biolm fine-tune --config-path ./exampleconfigs/minimal

# Real Saluki example
poetry run biolm fine-tune --config-path ./exampleconfigs/saluki-rna-finetuning

# Real XLNet example
poetry run biolm fine-tune --config-path ./exampleconfigs/xlnet-protein-finetuning
```

## For Full Details

See **[README_CONFIGS.md](README_CONFIGS.md)** for:
- Complete parameter reference
- Advanced patterns (cross-validation, preprocessing)
- Troubleshooting
- Configuration structure

See **[PLUGIN_TEMPLATE/README.md](PLUGIN_TEMPLATE/README.md)** for step-by-step guidance.
