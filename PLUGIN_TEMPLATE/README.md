# Plugin Template

This directory contains a copy-paste template for creating BioLM experiments.

## Usage

```bash
# Copy the template
cp -r PLUGIN_TEMPLATE my_experiment

# Edit config.yaml (change outputpath, data_source.filepath, plugin)
vi my_experiment/config.yaml

# Run
poetry run biolm fine-tune --config-path ./my_experiment
```

## What to Edit

In `config.yaml`, change these 3 settings:
- `outputpath`: Absolute path for results
- `data_source.filepath`: Path to your data file
- `plugin`: `saluki` (RNA) or `xlnet` (protein)

## Data Format

Tab-separated file with columns: ID, Label, Sequence
- Column positions are 1-indexed
- Saluki: comma-separated nucleotides (a,t,g,c,...)
- XLNet: raw sequences (MKVLWAALLVT... or atgcgatc...)

## For Help

See parent directory:
- `QUICKSTART.md` - Getting started guide
- `README_CONFIGS.md` - Full configuration reference
