import torch

from biolm_utils.rna_datasets import RNABaseDataset


class RNALanguageDataset(RNABaseDataset):

    def __getitem__(self, i):
        example = self.examples[i].copy()
        example["input_ids"] = torch.tensor(example["input_ids"], dtype=torch.long)
        return example
