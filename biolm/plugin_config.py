"""Simple plugin configuration system.

This replaces the complex plugin registry system with a simple, extensible
configuration class that plugins can subclass.

Example usage:
    from biolm.plugin_config import PluginConfig, PluginManager
    from my_models import MyModel
    from my_dataset import MyDataset

    config = PluginConfig(
        model_cls_for_finetuning=MyModel,
        dataset_cls=MyDataset,
        learning_rate=1e-3,
    )
    PluginManager.set_config(config)
"""

from dataclasses import dataclass
from typing import Any, Optional, Type

# Optional imports - plugins can override these
try:
    from transformers import PreTrainedTokenizerFast
    from transformers.data.data_collator import DefaultDataCollator
except ImportError:
    PreTrainedTokenizerFast = None
    DefaultDataCollator = None


@dataclass
class PluginConfig:
    """Configuration for a plugin.

    Plugins can subclass this to provide their specific model, dataset,
    tokenizer, and data collator classes.

    When developing a new model (e.g., RNN-CNN), you need to specify:
    - Which model class to use for pretraining vs finetuning
    - Which data collator to use for preparing batches during training
    - Whether special tokens should be added to the tokenizer
    - Training hyperparameters specific to your model

    Data Collators:
    - Data collators prepare batches of data for model training by handling
      padding, masking, and other preprocessing steps.
    - Pretraining data collators often handle self-supervised tasks like
      masked language modeling, where inputs need special masking.
    - Finetuning data collators typically handle supervised tasks with
      standard padding and no special masking.
    - For RNA sequence models, pretraining might use masked sequence modeling
      while finetuning uses standard sequence classification/regression collation.
    """

    # Model classes - override these in subclasses
    model_cls_for_pretraining: Optional[Type] = None
    """Model class used during pretraining phase.
    
    Pretraining typically involves self-supervised learning tasks like
    masked language modeling on unlabeled data. Set to None if your model
    doesn't require pretraining (e.g., RNN-CNN models trained from scratch).
    
    Example: For BERT-like models, this would be the base model class.
    For RNN-CNN models, this is often None since they finetune from random init.
    """

    model_cls_for_finetuning: Optional[Type] = None
    """Model class used during finetuning and prediction phases.
    
    Finetuning involves supervised learning on labeled data. This is the
    main model class for most downstream tasks.
    
    Example: For sequence classification, this would be your RNN-CNN model
    with a classification head. For regression tasks, it would have a
    regression head.
    """

    # Dataset class
    dataset_cls: Optional[Type] = None
    """Dataset class for loading and preprocessing your data.
    
    This should be a PyTorch Dataset subclass that handles your specific
    data format (e.g., RNA sequences, gene expression data).
    
    Example: RNADataset for loading FASTA files, or ExpressionDataset
    for loading gene expression matrices.
    """

    # Tokenizer class
    tokenizer_cls: Optional[Type] = PreTrainedTokenizerFast
    """Tokenizer class for converting raw sequences to model inputs.
    
    Defaults to HuggingFace's PreTrainedTokenizerFast. Override if you
    need custom tokenization logic for your data type.
    
    Example: For DNA/RNA sequences, you might need a custom tokenizer
    that handles k-mer tokenization or special nucleotide encodings.
    """

    # Data collator classes
    datacollator_cls_for_pretraining: Optional[Type] = None
    """Data collator for pretraining batches.
    
    Used only during pretraining to prepare batches with special preprocessing
    like masked language modeling. Should be a callable that takes a tokenizer
    and returns a data collator instance.
    
    Set to None if pretraining is not needed (common for RNN-CNN models).
    
    Example: For masked sequence modeling, this might randomly mask nucleotides
    in RNA sequences. For generative pretraining, it might handle next-token
    prediction masking.
    """

    datacollator_cls_for_finetuning: Optional[Type] = DefaultDataCollator
    """Data collator for finetuning, prediction, and interpretation batches.
    
    Used for supervised learning tasks. Handles standard batch preparation
    like padding sequences to the same length.
    
    Defaults to HuggingFace's DefaultDataCollator for basic padding.
    Override for custom collation logic (e.g., special padding for RNNs).
    
    Example: For sequence classification, this ensures all sequences in a
    batch have the same length. For regression tasks, it might handle
    variable-length inputs with appropriate padding.
    """

    # Special tokenizer for trainer (optional)
    special_tokenizer_for_trainer_cls: Optional[Type] = None
    """Special tokenizer class used only by the trainer.
    
    Some models need different tokenization during training vs inference.
    Set this if your training process requires special tokenization logic
    that's different from the main tokenizer.
    
    Most models can leave this as None.
    """

    # Config class for model configuration
    config_cls: Optional[Type] = None
    """Model configuration class.
    
    Should be a dataclass or config class that defines model hyperparameters
    like hidden dimensions, number of layers, attention heads, etc.
    
    Example: BertConfig, or your custom RNNConfig with layers/ hidden_size fields.
    """

    # Training hyperparameters
    learning_rate: float = 1e-4
    """Learning rate for training.
    
    Typical values: 1e-5 to 1e-3. Lower values (1e-5) for fine-tuning
    pretrained models, higher values (1e-3) for training from scratch.
    """

    max_grad_norm: float = 1.0
    """Maximum gradient norm for gradient clipping.
    
    Prevents exploding gradients during training. Typical values: 1.0-5.0.
    Set to 0.0 to disable gradient clipping.
    """

    weight_decay: float = 0.0
    """Weight decay (L2 regularization) coefficient.
    
    Helps prevent overfitting. Typical values: 0.0-0.01.
    Set to 0.0 to disable weight decay.
    """

    # Other options
    add_special_tokens: bool = False
    """Whether to add special tokens to the tokenizer.
    
    Set to True if your model needs special tokens like [CLS], [SEP], [MASK]
    for sequence modeling tasks. Most RNN-CNN models don't need this.
    """

    pretraining_required: bool = False
    """Whether this model requires pretraining before finetuning.
    
    Set to True if your model_cls_for_pretraining is not None and pretraining
    is mandatory before finetuning. Set to False for models that can be
    trained end-to-end or fine-tuned from random initialization.
    """


class PluginManager:
    """Singleton manager for plugin configuration."""

    _instance = None

    @classmethod
    def get_config(cls) -> PluginConfig:
        """Get the current plugin configuration."""
        if cls._instance is None:
            cls._instance = PluginConfig()
        return cls._instance

    @classmethod
    def set_config(cls, config: PluginConfig) -> None:
        """Set the plugin configuration."""
        cls._instance = config
