"""Model implementation."""

import torch.nn as nn

from biolm.base_dataset import BaseModel


class MyModel(BaseModel):
    """Minimal model example."""

    def __init__(self, config):
        super().__init__(config)
        self.encoder = nn.Linear(config.input_size, 128)
        self.classifier = nn.Linear(128, config.num_labels)

    def forward(self, input_ids, **kwargs):
        x = self.encoder(input_ids)
        logits = self.classifier(x)
        return {"logits": logits}

    @staticmethod
    def get_config(args, config_cls, tokenizer, dataset, nlabels):
        from transformers import PretrainedConfig

        config = PretrainedConfig(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
        config.input_size = 512  # Customize based on your data
        config.num_labels = nlabels
        return config
