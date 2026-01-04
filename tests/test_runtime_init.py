import pytest

from biolm import biolm
from biolm.config_access import ConfigManager
from biolm.structured_config import BioLMConfig, DebuggingConfig, TrainingConfig


def _reset_state(monkeypatch):
    monkeypatch.setattr(biolm, "args", None, raising=False)
    monkeypatch.setattr(biolm, "constants", None, raising=False)
    monkeypatch.setattr(biolm, "paths", None, raising=False)
    ConfigManager._instance = None


def _make_config():
    return BioLMConfig(
        mode="tokenize",
        task="regression",
        debugging=DebuggingConfig(dev=False, silent=True),
        training=TrainingConfig(resume=False),
    )


def test_ensure_runtime_requires_config(monkeypatch):
    _reset_state(monkeypatch)
    called = []
    monkeypatch.setattr(
        biolm, "initialize_runtime", lambda *a, **k: called.append(True)
    )

    with pytest.raises(RuntimeError):
        biolm.ensure_runtime()

    assert called == []


def test_ensure_runtime_initializes_with_config(monkeypatch):
    _reset_state(monkeypatch)
    cfg = _make_config()
    calls = {}

    def fake_init(config, log_params=False):
        calls["config"] = config
        calls["log_params"] = log_params
        monkeypatch.setattr(biolm, "args", config, raising=False)

    monkeypatch.setattr(biolm, "initialize_runtime", fake_init)

    biolm.ensure_runtime(cfg, log_params=True)

    assert calls["config"] is cfg
    assert calls["log_params"] is True


def test_ensure_runtime_skips_when_same_config(monkeypatch):
    _reset_state(monkeypatch)
    cfg = _make_config()
    monkeypatch.setattr(biolm, "args", cfg, raising=False)
    ConfigManager.set_config(cfg)

    called = []
    monkeypatch.setattr(
        biolm, "initialize_runtime", lambda *a, **k: called.append(True)
    )

    biolm.ensure_runtime(cfg)

    assert called == []


def test_ensure_runtime_uses_preset_config(monkeypatch):
    _reset_state(monkeypatch)
    cfg = _make_config()
    ConfigManager.set_config(cfg)

    called = []

    def fake_init(config, log_params=False):
        called.append(config)
        monkeypatch.setattr(biolm, "args", config, raising=False)

    monkeypatch.setattr(biolm, "initialize_runtime", fake_init)

    biolm.ensure_runtime()

    assert called == [cfg]
