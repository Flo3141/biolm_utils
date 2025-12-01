# README for PLUGIN_TEMPLATE
# ==============================================================

## How to Use This Template

### Step 1: Copy the Template
```bash
cp -r PLUGIN_TEMPLATE my_first_experiment
```

### Step 2: Edit the Configuration
```bash
vi my_first_experiment/config.yaml
```

**Make these 3 changes:**
1. **outputpath**: Change to where you want results saved
2. **data_source.filepath**: Point to your data file
3. **plugin**: Choose `saluki` (RNA) or `xlnet` (protein)

### Step 3: Prepare Your Data

Your data file should be tab-separated (or your chosen `columnsep`):
```
ID         Label    Sequence
seq_001    1.5      a,t,g,c,a,g,t,c,...      (Saluki: comma-separated)
seq_002    2.3      MKVLWAALLVT...           (XLNet: raw sequence)
```

**Important:** Column positions are **1-indexed**!
- Column 1, 2, 3 means: idpos=1, labelpos=2, seqpos=3

### Step 4: Run!
```bash
poetry run biolm fine-tune --config-path ./my_first_experiment
```

### Step 5: Monitor & Collect Results
```bash
# Watch the log file
tail -f /outputpath/fine-tune.log

# Results will be in:
# /outputpath/fine-tune/
#   ├── pytorch_model.bin       (trained model)
#   ├── config.json             (model config)
#   ├── training_results.json   (metrics)
#   └── fine-tune_dataset.pkl   (cached dataset)
```

## Common Issues

### "blocksize must be X"
→ Don't set blocksize in config! It's auto-enforced.

### "Column position X out of range"
→ Check your data file structure:
```bash
head -3 /path/to/data.txt
# Count columns (1-indexed!): 1st, 2nd, 3rd...
```

### "No such file or directory"
→ Use absolute path for `outputpath`:
```yaml
outputpath: /home/user/experiments/my_exp    # ✅ Good
outputpath: ~/experiments/my_exp              # ❌ Avoid
outputpath: ./outputs/my_exp                  # ❌ Avoid
```

## For More Details

See `README_CONFIGS.md` in the parent directory for:
- Complete parameter reference
- Plugin-specific constraints
- Advanced patterns (cross-validation, preprocessing, etc.)
- Troubleshooting guide

## Need Help?

Check existing examples in `exampleconfigs/`:
```
exampleconfigs/
├── minimal/                    (simplest example)
├── saluki-rna-finetuning/     (RNA-specific)
└── xlnet-protein-finetuning/  (protein-specific)
```
