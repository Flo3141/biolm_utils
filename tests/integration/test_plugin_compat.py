import os

from biolm_utils.plugin_loader import discover_entrypoint_plugins
from biolm_utils.plugin_registry import get_plugin_factory


def test_installed_plugins_are_discoverable():
    """Integration test: assert any plugins listed by PLUGINS_LIST are discoverable.

    The CI job installs a list of plugin packages and sets PLUGINS_LIST to their
    plugin names (comma-separated). The test runner verifies that discover
    registers the installed plugin factories.
    """
    plugins_env = os.environ.get("PLUGINS_LIST")
    if not plugins_env:
        # Test skipped when no external plugins configured (local dev runs)
        return

    expected = [p.strip() for p in plugins_env.split(",") if p.strip()]
    assert expected, "PLUGINS_LIST provided but empty"

    registered = discover_entrypoint_plugins()

    missing = [p for p in expected if p not in registered]
    # Also check registry factory is present
    not_registered = [p for p in expected if get_plugin_factory(p) is None]

    assert not missing, f"Plugins advertised but not discovered: {missing}"
    assert (
        not not_registered
    ), f"Plugins discovered but not registered as factories: {not_registered}"
