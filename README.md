> **Note:** The `biolm-2.0` branch contains the latest, actively developed version of BioLM with major improvements and a new plugin architecture. The `main` branch is legacy. For the newest features and code, please [switch to the `biolm-2.0` branch](https://github.com/dieterich-lab/biolm_utils/tree/biolm-2.0).

# `biolm_utils`: A Framework for Bioinformatical Language Models

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](README.md)

This project implements pre-training and fine-tuning of neural models for regressing half-lives of RNA and protein sequences. In addition, it supports the extraction of leave-one-out (LOO) scores for fine-tuned models to analyze importance scores of individual inputs.

## Key Features

- **Tokenization of RNA/Protein sequences** via:
  - Byte Pair Encoding (BPE)
  - Atomic one-hot encoding
- **Pre-train** a language model via Masked Language Modelling (MLM)
- **Fine-tune** any model for regressing half-lives
- **Calculate leave-one-out (LOO) scores** for your fine-tuned model to assess token importance

## Quick Start

Get up and running with `biolm_utils` in just a few steps:

```bash
# Clone the repository
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils

# Create and activate a virtual environment
python3 -m venv biolm
source biolm/bin/activate  # On Windows: biolm\Scripts\activate

# Install dependencies
pip install -e .

# Run a basic command to verify installation
python biolm_utils/biolm.py -h
```

## Installation

First, clone the repository and navigate into it. We recommend creating a dedicated environment using [Python venv](https://docs.python.org/3/library/venv.html) for the project. Then, install the project via the [pyproject.toml](./pyproject.toml) file.

### Standard Installation

```bash
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
python3 -m venv biolm
source biolm/bin/activate  # On Windows: biolm\Scripts\activate
pip install -e .
```

### Alternative Installation with Pipenv

```bash
git clone https://github.com/dieterich-lab/biolm_utils.git
cd biolm_utils
pip install pipenv
pipenv install
```

## File Structure

The project is organized as follows:

```bash
biolm_utils/
├── biolm_utils/
│   ├── biolm.py                # Main script for tokenizing, training, testing, predicting, and LOO scores
│   ├── config.py               # Config class that needs to be initialized by plugins
│   ├── cross_validation.py    # Wrapper that manages fine-tuning on different splits
│   ├── entry.py                # Main entry point after params.py, fixing paths and global variables
│   ├── __init__.py
│   ├── interpret.py            # Script controlling the LOO score calculation
│   ├── loo_utils.py            # Custom evaluator to extract LOO scores for regression tasks
│   ├── params.py               # Argument parser
│   ├── rna_datasets.py         # Dataset class handling tokenized and vectorized sequences
│   ├── trainer.py              # Custom trainer classes for fine-tuning models on regression tasks
│   ├── train_tokenizer.py     # Script controlling the tokenization process
│   └── train_utils.py          # Helper functions (e.g., loading models/tokenizers, creating reports)
├── pyproject.toml              # Project configuration and dependencies
└── README.md                   # This file
```

## Pathing

The software saves all experiment data in the `outputpath` specified in [params.py](biolm_utils/params.py) (or falls back to the file path stem of the input file given in `filepath` if not provided). This directory will be created if it doesn't exist. The software saves the dataset (tokenized samples from the given filepath), the tokenizer, and the models.

For example, considering we use cross-validation via splits and after having pre-trained (language models only) and fine-tuned a model, the directory will look as follows:

```bash
my_experiment/
├── fine-tune/
│   ├── 0/
│   │   └── pytorch_model.bin
│   ├── 1/
│   │   └── pytorch_model.bin
│   ├── 2/
│   │   └── pytorch_model.bin
│   └── dataset.json
├── pre-train/
│   ├── dataset.json
│   └── pytorch_model.bin
└── tokenizer.json
```

## Usage

### General

The main script is [biolm.py](biolm_utils/biolm.py). It contains a `run()` function that can be imported into your custom project. It accesses the given parameters from [`params.py`](biolm_utils/params.py) and additionally from a custom `Config` object located in [config.py](biolm_utils/config.py) that can be set via `set_config()`.

To get a verbose explanation of all the possible parameters, run:

```bash
python biolm_utils/biolm.py -h
```

All options besides the training `mode` are optional and are mostly populated with sensible default parameters. The `mode` can be one of the following:

- `tokenize` 
- `pre-train`
- `fine-tune`
- `interpret`
- `predict`

As an example, you can run training with command-line parameters:

```bash
python biolm_utils/biolm.py pre-train --filepath "xxx" --outputpath "xxx" --...
```

Or start tokenization with a config file:

```bash
python biolm_utils/biolm.py tokenize --configfile config.yaml
```

The parameters in the config file will then be parsed by the argument parser in [params.py](biolm_utils/params.py) to rule out any conflicts. Parameters parsed from the command line have priority over those from the config file.

### Configuring the Data

We designed options to give varying data sources for either tokenization/pre-training (we expect that the data for training the tokenizer will be the same as for the pre-training process) and for the fine-tuning step. You also have to let the scripts know where exactly to find information about labels, sequences, and splits in your data file. The two corresponding sections in the config file are listed below. Attributes should be self-explanatory by their comments or explained by the command-line parser. 

```yaml
# Description of the datasource used for:
# - Training the tokenizer
# - Pre-training (for LM)
tokenizing and pre-training data source:
  filepath: "tokenizing_and_pre-training_data_file.txt"
  stripheader: False              # If the custom data file has a header that needs to be stripped
  columnsep: "\t"                 # Could be ",", "|", "\t", etc.
  tokensep: ","
  specifiersep: None
  idpos: 1                        # Position of the identifier column
  seqpos: 1                       # Position of the sequence column
  pretrainedmodel: None           # If the tokenizer for pre-training differs from the chosen data

# Description of the fine-tuning source
fine-tuning data source:
  filepath: "fine-tuning_data_file.txt"
  stripheader: False              # If the custom data file has a header that needs to be stripped
  columnsep: "\t"                 # Could be ",", "|", "\t", etc.
  tokensep: ","
  specifiersep: None
  idpos: 1                        # Position of the identifier column
  seqpos: 1                       # Position of the sequence column
  labelpos: 1                     # Position of the label column
  weightpos: None                 # Position of the column containing quality labels
  splitpos: 1                     # Position of the split identifier for cross-validation
  pretrainedmodel: None           # If the pre-trained model differs from the chosen data
```

An example prototypical dataset file would look like this (without header)

```csv
0	ENST00000488147	ENSG00000227232	653635	WASH7P	unprocessed_pseudogene	0.204213162843933	3.39423360819142	0.121582579281952	0.374739086478062	a,t,g,g,g,a,g,c,c,g,t,g,t,g,c,a,c,g,t,c,g,g,g,a,g,c,t,c,g,g,a,g,t,g,a,g,c,gej,c,a,c,c,a,t,g,a,c,t,c,c,t,g,t,g,a,g,g,a,t,g,c,a,g,c,a,c,t,c,c,c,t,g,g,c,a,g,g,t,c,a,g,a,c,c,t,a,t,g,c,c,g,t,g,c,c,c,t,t,c,a,t,c,c,a,g,c,c,a,g,a,c,c,t,g,c,g,g,c,g,a,g,a,g,g,a,g,g,c,c,g,t,c,c,a,g,c,a,g,a,t,g,g,c,g,g,a,t,g,c,c,c,t,g,c,a,g,t,a,c,c,t,g,c,a,g,a,a,g,g,t,c,t,c,t,g,g,a,g,a,c,a,t,c,t,t,c,a,g,c,a,g,gej,t,a,g,a,g,c,a,g,a,g,c,c,g,g,a,g,c,c,a,g,g,t,g,c,a,g,g,c,c,a,t,t,g,g,a,g,a,g,a,a,g,g,t,c,t,c,c,t,t,g,g,c,c,c,a,g,g,c,c,a,a,g,a,t,t,g,a,g,a,a,g,a,t,c,a,a,g,g,g,c,a,g,c,a,a,g,a,a,g,g,c,c,a,t,c,a,a,g,gej,t,g,t,t,c,t,c,c,a,g,t,g,c,c,a,a,g,t,a,c,c,c,t,g,c,t,c,c,a,g,g,g,c,g,c,c,t,g,c,a,g,g,a,a,t,a,t,g,g,c,t,c,c,a,t,c,t,t,c,a,c,g,g,g,c,g,c,c,c,a,g,g,a,c,c,c,t,g,g,c,c,t,g,c,a,g,a,g,a,c,g,c,c,c,c,c,g,c,c,a,c,a,g,g,a,t,c,c,a,g,a,g,c,a,a,g,c,a,c,c,g,c,c,c,c,c,t,g,g,a,c,g,a,g,c,g,g,g,c,c,c,t,g,c,a,g,gej,a,g,a,a,g,c,t,g,a,a,g,g,a,c,t,t,t,c,c,t,g,t,g,t,g,c,g,t,g,a,g,c,a,c,c,a,a,g,c,c,g,g,a,g,c,c,c,g,a,g,g,a,c,g,a,t,g,c,a,g,a,a,g,a,g,g,g,a,c,t,t,g,g,g,g,g,t,c,t,t,c,c,c,a,g,c,a,a,c,a,t,c,a,g,c,t,c,t,g,t,c,a,g,c,t,c,c,t,t,g,c,t,g,c,t,c,t,t,c,a,a,c,a,c,c,a,c,c,g,a,g,a,a,c,c,t,gej,t,a,g,a,a,g,a,a,g,t,a,t,g,t,c,t,t,c,c,t,g,g,a,c,c,c,c,c,t,g,g,c,t,g,g,t,g,c,t,g,t,a,a,c,a,a,a,g,a,c,c,c,a,t,g,t,g,a,t,g,c,t,g,g,g,g,g,c,a,g,a,g,a,c,a,g,a,g,g,a,g,a,a,g,c,t,g,t,t,t,g,a,t,g,c,c,c,c,c,t,t,g,t,c,c,a,t,c,a,g,c,a,a,g,a,g,a,g,a,g,c,a,g,c,t,g,g,a,a,c,a,g,c,a,g,gej,t,c,c,c,a,g,a,g,a,a,c,t,a,c,t,t,c,t,a,t,g,t,g,c,c,a,g,a,c,c,t,g,g,g,c,c,a,g,g,t,g,c,c,t,g,a,g,a,t,t,g,a,t,g,t,t,c,c,a,t,c,c,t,a,c,c,t,g,c,c,t,g,a,c,c,t,g,c,c,c,g,g,c,a,t,t,g,c,c,a,a,c,g,a,c,c,t,c,a,t,g,t,a,c,a,t,t,g,c,c,g,a,c,c,t,g,g,g,c,c,c,c,g,g,c,a,t,t,g,c,c,c,c,c,t,c,t,g,c,c,c,c,t,g,g,c,a,c,c,a,t,t,c,c,a,g,a,a,c,t,g,c,c,c,a,c,c,t,t,c,c,a,c,a,c,t,g,a,g,g,t,a,g,c,c,g,a,g,c,c,t,c,t,c,a,a,g,aej,c,c,t,a,c,a,a,g,a,t,g,g,g,g,t,a,c,t,a,a,c,a,c,c,a,c,c,c,c,c,a,c,c,g,c,c,c,c,c,a,c,c,a,c,c,a,c,c,c,c,c,a,g,c,t,c,c,t,g,a,g,g,t,g,c,t,g,g,c,c,a,g,t,g,c,a,c,c,c,c,c,a,c,t,c,c,c,a,c,c,c,t,c,a,a,c,c,g,c,g,g,c,c,c,c,t,g,t,a,g,g,c,c,a,a,g,g,c,g,c,c,a,g,g,c,a,g,g,a,c,g,a,c,a,g,c,a,g,c,a,g,c,a,g,c,g,c,g,t,c,t,c,c,t,t,c,a,g,tej,c,c,a,g,g,g,a,g,c,t,c,c,c,a,g,g,g,a,a,g,t,g,g,t,t,g,a,c,c,c,c,t,c,c,g,g,t,g,g,c,t,g,g,c,c,a,c,t,c,t,g,c,t,a,g,a,g,t,c,c,a,t,c,c,g,c,c,a,a,g,c,t,g,g,g,g,g,c,a,t,c,g,g,c,a,a,g,g,c,c,a,a,g,c,t,g,c,g,c,a,g,c,a,t,g,a,a,g,g,a,g,c,g,a,a,a,g,c,t,g,g,a,g,a,a,g,c,a,g,c,a,g,c,a,g,a,a,g,g,a,g,c,a,g,g,a,g,c,a,a,g,tej,g,a,g,a,g,c,c,a,c,g,a,g,c,c,a,a,g,g,t,g,g,g,c,a,c,t,t,g,a,t,g,t,c,gej,c,t,c,c,a,t,g,g,g,g,g,g,a,c,g,g,c,t,c,c,a,c,c,c,a,g,c,c,t,g,c,g,c,c,a,c,t,g,t,g,t,t,c,t,t,a,a,g,a,g,g,c,t,t,c,c,a,g,a,g,a,a,a,a,c,g,g,c,a,c,a,c,c,a,a,t,c,a,a,t,a,a,a,g,a,a,c,t,g,a,g,c,a,g,a,a,a
```

There are certain specifics regarding the following entries:

- **`splitpos`**: If it is set to `None`, fine-tuning is carried out on a 90/10 train/val split with no subsequent testing. If a split position is given, we expect at least three different splits on which we do cross-validation by:
  - Setting each split as a dedicated test set
  - Setting the following split as a dedicated validation set
  - Training on the rest of the splits

- **`specifiersep`** (one-hot encoding only): If you want to decorate your atomic tokens with float numbers, you can do so by denoting a separator after which you append the float number(s) to the atomic token. For example, you could specify `specifiersep: #` for generating your samples as: `a#2.5, c, A, g#5.7, ...` or even with multiple modifiers like `a#2.5#0.2, c, A, g#5.7, ...`. The decorating float numbers are then appended to new "channels" of the one-hot encoding. Regarding the last sample from above, this would result in a one-hot encoding of (assuming a vocabulary of `[a, c, g, t, A, C, G, T]`):

```
a | 1  | 0 | 0 | 0 |
c | 0  | 1 | 0 | 0 |
g | 0  | 0 | 0 | 1 |
t | 0  | 0 | 0 | 0 |
A | 0  | 0 | 1 | 0 |
C | 0  | 0 | 0 | 0 |
G | 0  | 0 | 0 | 0 |
T | 0  | 0 | 0 | 0 |
  |2.5 | 0 | 0 |5.7|
  |0.2 | 0 | 0 |5.7|
```



### Training a Tokenizer

To train a tokenizer, you'll be using the `tokenize` mode. The `encoding` parameter in the config file offers different encoding options. Under the section `tokenization`, you'll find options to further customize the encoding process. 

```yaml
tokenization:
  samplesize: None                      # If your data is too big to learn a tokenizer, you can downsample it
  vocabsize: 20_000
  minfreq: 2
  atomicreplacements: None              # Dictionary of replacements, e.g., {"a": "A", "bcd": "xyz"}
  encoding: atomic                      # [bpe, atomic]
  bpe:
    maxtokenlength: 10
  lefttailing: True
```

Where:

- **`samplesize`**: Offers the option to downsample your data. If your file has, for example, 10M lines, training a BPE tokenizer on all these might become very costly or computationally infeasible. You can instead give a sample size of `250_000` to make the process much faster.
- **`vocabsize`**: The maximal size of the vocabulary at the end of the tokenization process.
- **`minfreq`**: The minimum frequency that a token should appear in the training file before it is recorded as a vocabulary item.
- **`atomicreplacements`**: This is a dictionary with tokens that should be treated as atomic tokens during the byte pair encoding process. You have to specify both the initial token and the character that it is to be mapped to.
- **`encoding`**: The actual encoding to be applied. Either character-wise (`atomic`) or using a word piece tokenizer for byte pair encoding (`bpe`).
- **`maxtokenlength`**: The BPE tokenizer can come up with pretty long tokens. This number caps the length at a maximal length.
- **`lefttailing`**: If true, the sequences will be cut from the left (beginning from the right end).

### Pre-training (Language Models Only) and Fine-tuning a Model

For pre-training a language model via Masked Language Modelling, you will use the `pre-train` mode. For fine-tuning a model, the `fine-tune` mode is required. In your `config.yaml`, you need to at least specify the parameters under `training`:

```yaml
training:
  general:
    batchsize: 8
    gradacc: 4
    blocksize: 512
    nepochs: 10
    patience: 3
    resume: False                       # For resuming training
  fine-tuning:
    fromscratch: False                  # If we want to fine-tune without a pre-trained model (language models only)
    scaling: log                        # [log, minmax, standard]
    weightedregression: False
```

The attributes under `training: general` should be mostly self-explanatory: `blocksize` refers to the sequence length and might lead to errors when chosen bigger than `512` (for XLNET). For Saluki, we were able to set this maximum sequence length to `12288`. Sequences will then be truncated by the tokenizer or will be tokenized, re-centered, and cropped when using the option `cdscentered` (see below).

We also have to clarify data pre-processing and environment options:

```yaml
data pre-processing:
  centertoken: False                    # Either False or a token/character on which the sequence will be centered
environment:
  ngpus: 1                              # [1, 2, 4]
```

The `data pre-processing` attributes refer to specific pre-processing options that are in detail explained by the command-line help.

Under `environment`, you can decide if you want to train on GPU or CPU and on how many GPUs you want to train. We allow training on 1, 2, or 4 GPUs as this even number will be offset against the `gradacc` (gradient accumulation) option to preserve a fixed effective batch size.

### Extract LOO Scores for a Model

To calculate importance scores for individual input tokens, we can use the mode `interpret`. The script will then run over the test splits and extract leave-one-out (LOO) scores. The LOO scores are estimated by leaving a certain token blank (or deleting it completely, see options below), running the model with this "defective" sequence, and comparing the results to the prediction of the model for the original sequence. Positive scores denote that leaving the input out leads to higher prediction; conversely, negative scores mean leaving the input out leads to lower predictions. 

```yaml
looscores:
  handletokens: remove                  # [remove, mask, replace]
  replacementdict: None                 # Dict of atomic tokens that should be replaced against each other if handletokens is set to 'replace'
```

The scripts will then extract LOO scores for all splits of the fine-tuning data and save them as `.csv` under the corresponding fine-tuning path as `loo_scores_{handle_tokens}.csv`.

### Inference

Inference means sending a fine-tuned model on unseen data and letting it make predictions. For this, run the main script in the `predict` mode. The config file mirrors only a fraction of the attributes compared to the complete pipeline.

### Resuming a Model

There are two use cases to resume a model using the `--resume` argument:

1. **`--resume`** (without parameters): Triggers the Hugging Face internal `resume_from_checkpoint` option which will only _continue_ a training that has been interrupted. For example, a planned training that was to run for 50 epochs and was interrupted at epoch 23 can be resumed from the best checkpoint to continue from epoch 23 to the planned epoch 50.
2. **`--resume X`**: Will trigger further pre-training a model from its best checkpoint for additional `X` epochs.


## Customization

This framework on its own does not provide full functionality. It is meant to be employed with plugins that implement the following classes and methods:

- A custom model class that inherits from 🤗 [PreTrainedModel](https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/model#transformers.PreTrainedModel) and provides a static `getconfig()` method
- A custom dataset class that inherits from [RNABaseDataset](./biolm_utils/rna_datasets.py) and provides the `__getitem__()` method
- A main script that imports the `run()` method from [biolm.py](./biolm_utils/biolm.py) and defines a custom `Config` object from [config.py](./biolm_utils/config.py) via `setconfig()`

## Roadmap

We have several exciting features and improvements planned for future releases:

- [ ] Enhanced support for additional sequence types (DNA methylation patterns, protein modifications)
- [ ] Integration with popular bioinformatics frameworks (Biopython, BioConductor)
- [ ] Improved model interpretability tools and visualization dashboards
- [ ] Pre-built models and tokenizers for common use cases
- [ ] Support for distributed training on multiple nodes
- [ ] Comprehensive benchmarking suite for model comparison
- [ ] Extended documentation with Jupyter notebook tutorials
- [ ] REST API for model inference

Community contributions are welcome! Please see our [contribution guidelines](CONTRIBUTING.md) for more information.

## Contributing

We welcome contributions from the community! Whether it's bug reports, feature requests, documentation improvements, or code contributions, your help is appreciated.

### How to Contribute

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes and commit them (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{biolm_utils,
  title={biolm_utils: A Framework for Bioinformatical Language Models},
  author={Dieterich Lab},
  year={2024},
  url={https://github.com/dieterich-lab/biolm_utils}
}
```

## Support

For questions, issues, or discussions:

- Open an [issue](https://github.com/dieterich-lab/biolm_utils/issues) on GitHub
- Check the [documentation](README.md)
- Visit the [biolm-2.0 branch](https://github.com/dieterich-lab/biolm_utils/tree/biolm-2.0) for the latest development version

## Acknowledgments

This project builds upon the excellent work of the Hugging Face Transformers library and the broader open-source bioinformatics community.
