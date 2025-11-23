import sys
from types import SimpleNamespace

from biolm_utils import plugin_loader
from biolm_utils.plugin_registry import get_plugin_factory, unregister_plugin


def test_discover_entrypoint_plugins_registers(monkeypatch):
    # Ensure test isolation
    if get_plugin_factory("dummy_test_plugin") is not None:
        unregister_plugin("dummy_test_plugin")

    # Create a dummy entry point object with .name and .load()
    class DummyEP:
        def __init__(self, name, factory):
            self.name = name
            self._factory = factory

        def load(self):
            return self._factory

    def fake_entry_points():
        return SimpleNamespace(
            select=lambda group=None: [DummyEP("dummy_test_plugin", lambda: "cfg")]
        )

    monkeypatch.setattr(plugin_loader, "entry_points", fake_entry_points)

    registered = plugin_loader.discover_entrypoint_plugins()
    assert "dummy_test_plugin" in registered
    assert get_plugin_factory("dummy_test_plugin") is not None

    # cleanup
    unregister_plugin("dummy_test_plugin")


def test_discover_plugins_from_dir(tmp_path, monkeypatch):
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()

    # Write a small plugin module with a factory
    plugin_file = plugins_dir / "examplemod.py"
    plugin_file.write_text("def get_examplemod_config():\n    return 'cfg'\n")

    # Add plugins_dir to sys.path so import_module can find it
    monkeypatch.syspath_prepend(str(plugins_dir))

    registered = plugin_loader.discover_plugins_from_dir(str(plugins_dir))
    assert "examplemod" in registered
    # ensure the plugin factory is accessible via registry
    assert get_plugin_factory("examplemod") is not None

    # cleanup
    unregister_plugin("examplemod")
