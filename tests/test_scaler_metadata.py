import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from biolm.biolm_dataset import BioLMDataset
from biolm.train_utils import IdentityScaler, LogScaler, get_model_and_config


class DummyArgs:
    def __init__(self, mode="fine-tune"):
        self.mode = mode
        self.training = SimpleNamespace()


class DummyModel:
    @staticmethod
    def get_config(args, config_cls, tokenizer, dataset, nlabels):
        return SimpleNamespace()

    def __init__(self, config=None):
        self.config = config or SimpleNamespace()


def test_dataset_save_includes_scaler_and_scaling_method(tmp_path):
    ds = object.__new__(BioLMDataset)
    # provide minimal attributes used by save()
    ds.lines = ["A\tSAMPLE\t1"]
    ds.scaler = IdentityScaler()
    ds.scaling_method = "identity"

    out = tmp_path / "ds.pkl"
    ds.save(out)

    with open(out, "rb") as f:
        data = pickle.load(f)

    assert "lines" in data
    assert "scaler" in data
    assert "scaling_method" in data
    assert data["scaling_method"] == "identity"


def test_get_model_and_config_attaches_scaler():
    args = DummyArgs(mode="fine-tune")
    scaler = LogScaler()

    model = get_model_and_config(
        args=args,
        model_cls=DummyModel,
        model_config_cls=SimpleNamespace,
        tokenizer=None,
        dataset=None,
        nlabels=1,
        model_load_path=Path("."),
        pretraining_required=False,
        scaler=scaler,
    )

    assert hasattr(model, "scaler")
    assert model.scaler is scaler


if __name__ == "__main__":
    pytest.main([__file__])
