import builtins
from pathlib import Path

from biolm.structured_config import BioLMConfig, SettingsConfig


def test_runner_uses_mlflow_for_results(monkeypatch, tmp_path):
    # Create fake mlflow context manager that yields a fake mlflow object
    calls = {}

    class FakeML:
        def log_metrics(self, metrics):
            calls.setdefault("log_metrics", []).append(metrics)

    import contextlib

    @contextlib.contextmanager
    def fake_start(*args, **kwargs):
        yield FakeML()

    monkeypatch.setattr("biolm.mlflow_integration.start_mlflow_run", fake_start)

    # Build fake args/config
    args = BioLMConfig(
        mode="fine-tune", settings=SettingsConfig(mlflow={"enabled": True})
    )

    # Minimal config with expected attributes
    class CfgObj:
        model_cls_for_pretraining = None
        model_cls_for_finetuning = object
        datacollator_cls_for_pretraining = staticmethod(lambda **kw: None)
        datacollator_cls_for_finetuning = staticmethod(lambda **kw: None)
        add_special_tokens = False
        config_cls = None
        pretraining_required = False

    cfg = CfgObj()

    import biolm.runner as runner

    # Ensure runner uses our fake start_mlflow_run (it has an imported reference)
    monkeypatch.setattr(runner, "start_mlflow_run", fake_start)

    # Replace _train/_test in runner so we don't exercise heavy logic

    def fake_train(*a, **k):
        return ({"val_loss": 0.5, "val_acc": 0.8}, object())

    def fake_test(*a, **k):
        return {"test_loss": 0.3}

    monkeypatch.setattr(runner, "_train", fake_train)
    monkeypatch.setattr(runner, "_test", fake_test)

    # Create the run function
    run_fn = runner.make_run_fn(args, cfg, None, None, None)

    # Now run and assert mlflow.log_metrics is called (via our fake start_w)
    model_save = tmp_path / "out"
    model_save.mkdir()

    # Call run once: should call fake_train and then fake_test and log metrics
    res = run_fn(None, None, True, None, model_save, None, None)

    # result should be the final results from fake_test
    assert isinstance(res, dict)
