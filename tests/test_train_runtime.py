from pathlib import Path
from types import SimpleNamespace

import pytest

from biolm import biolm


def test_train_uses_existing_runtime(monkeypatch):
    called = []

    def fake_ensure_runtime(config=None, log_params=False):
        called.append(config)

    def fake_get_trainer_class(mode, task):
        return object

    def fake_get_num_labels(mode, task, dataset):
        return 1

    class DummyException(Exception):
        pass

    def fake_get_model_and_config(*args, **kwargs):
        raise DummyException()

    monkeypatch.setattr(biolm, "ensure_runtime", fake_ensure_runtime)
    monkeypatch.setattr(biolm, "_get_trainer_class", fake_get_trainer_class)
    monkeypatch.setattr(biolm, "_get_num_labels", fake_get_num_labels)
    monkeypatch.setattr(biolm, "get_model_and_config", fake_get_model_and_config)

    original_args = biolm.args
    biolm.args = SimpleNamespace(
        mode="pre-train",
        task="regression",
        training=SimpleNamespace(resume=False, batchsize=1, gradacc=1, nepochs=1),
        debugging=SimpleNamespace(dev=False),
        settings=SimpleNamespace(mlflow={"enabled": False}),
    )

    class DummyTrainDataset:
        dataset = SimpleNamespace(scaler=None)

    with pytest.raises(DummyException):
        biolm.train(
            train_dataset=DummyTrainDataset(),
            val_dataset=None,
            data_collator=None,
            model_load_path=Path("./model"),
            model_save_path=Path("./model"),
            tokenizer=SimpleNamespace(),
            tokenizer_for_trainer=SimpleNamespace(),
            full_dataset=SimpleNamespace(),
            model_cls=SimpleNamespace(),
            config=SimpleNamespace(
                config_cls=SimpleNamespace(),
                pretraining_required=False,
                datacollator_cls_for_pretraining=SimpleNamespace(),
                datacollator_cls_for_finetuning=SimpleNamespace(),
            ),
        )

    biolm.args = original_args
    assert called == [None]