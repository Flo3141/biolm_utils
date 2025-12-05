"""Integration tests for plugin discovery and loading.

These tests verify that plugins are correctly registered via entry points
and can be loaded by the framework at runtime.
"""

import importlib.metadata

import pytest


def test_plugin_entry_points_exist():
    """Verify that plugin entry points are registered in the biolm.plugins group."""
    entry_points = list(importlib.metadata.entry_points(group="biolm.plugins"))

    assert (
        len(entry_points) > 0
    ), "No plugins found in 'biolm.plugins' entry point group"

    plugin_names = {ep.name for ep in entry_points}
    print(f"\nFound {len(entry_points)} plugin(s): {plugin_names}")

    # Store for other tests
    return entry_points


def test_saluki_plugin_registered():
    """Verify that the Saluki plugin is registered."""
    entry_points = {
        ep.name: ep for ep in importlib.metadata.entry_points(group="biolm.plugins")
    }

    assert "saluki" in entry_points, "Saluki plugin not found in entry points"

    saluki_ep = entry_points["saluki"]
    assert (
        "saluki_plugin" in saluki_ep.value
    ), f"Saluki should come from saluki_plugin package, got: {saluki_ep.value}"


def test_xlnet_plugin_registered():
    """Verify that the XLNet plugin is registered."""
    entry_points = {
        ep.name: ep for ep in importlib.metadata.entry_points(group="biolm.plugins")
    }

    assert "xlnet" in entry_points, "XLNet plugin not found in entry points"

    xlnet_ep = entry_points["xlnet"]
    assert (
        "xlnet_plugin" in xlnet_ep.value
    ), f"XLNet should come from xlnet_plugin package, got: {xlnet_ep.value}"


def test_plugin_loading_saluki():
    """Test that the Saluki plugin can be loaded via entry point."""
    import importlib.metadata

    from biolm.plugin_config import PluginManager

    # Load plugin via entry point (simulates what cli.py does)
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    saluki_ep = next((ep for ep in eps if ep.name == "saluki"), None)
    assert saluki_ep is not None, "Saluki entry point not found"

    # Load and execute the plugin function
    plugin_func = saluki_ep.load()
    plugin_func()

    # Get the config that was set by the plugin
    config = PluginManager.get_config()
    assert config is not None, "Saluki plugin config is None"
    assert (
        config.model_cls_for_finetuning is not None
    ), "Saluki should have a fine-tuning model class"
    assert config.dataset_cls is not None, "Saluki should have a dataset class"
    assert config.pretraining_required is False, "Saluki doesn't require pretraining"


def test_plugin_loading_xlnet():
    """Test that the XLNet plugin can be loaded via entry point."""
    import importlib.metadata

    from biolm.plugin_config import PluginManager

    # Load plugin via entry point (simulates what cli.py does)
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    xlnet_ep = next((ep for ep in eps if ep.name == "xlnet"), None)
    assert xlnet_ep is not None, "XLNet entry point not found"

    # Load and execute the plugin function
    plugin_func = xlnet_ep.load()
    plugin_func()

    # Get the config that was set by the plugin
    config = PluginManager.get_config()
    assert config is not None, "XLNet plugin config is None"
    assert (
        config.model_cls_for_pretraining is not None
    ), "XLNet should have a pretraining model class"
    assert (
        config.model_cls_for_finetuning is not None
    ), "XLNet should have a fine-tuning model class"
    assert config.dataset_cls is not None, "XLNet should have a dataset class"
    assert config.pretraining_required is True, "XLNet requires pretraining"


def test_no_builtin_plugins():
    """Verify that no builtin plugins exist (all should be external)."""
    entry_points = {
        ep.name: ep.value
        for ep in importlib.metadata.entry_points(group="biolm.plugins")
    }

    for name, value in entry_points.items():
        assert (
            "biolm.plugins.builtin" not in value
        ), f"Plugin '{name}' still uses builtin path: {value}. All plugins should be external packages."


def test_plugin_discovery_without_framework_import():
    """Test that plugins can be discovered without importing heavy framework modules."""
    # This should be fast - no torch/transformers imports needed
    import importlib.metadata

    eps = list(importlib.metadata.entry_points(group="biolm.plugins"))

    assert (
        len(eps) >= 2
    ), f"Expected at least 2 plugins (Saluki, XLNet), found {len(eps)}"

    # Verify we can inspect without loading
    for ep in eps:
        assert hasattr(ep, "name"), "Entry point should have 'name' attribute"
        assert hasattr(ep, "value"), "Entry point should have 'value' attribute"
        assert (
            ":" in ep.value
        ), f"Entry point value should be 'module:callable' format, got: {ep.value}"


@pytest.mark.parametrize("plugin_name", ["saluki", "xlnet"])
def test_plugin_config_attributes(plugin_name):
    """Test that plugin configs have all required attributes."""
    import importlib.metadata

    from biolm.plugin_config import PluginManager

    # Load plugin via entry point
    eps = importlib.metadata.entry_points(group="biolm.plugins")
    ep = next((e for e in eps if e.name == plugin_name), None)
    assert ep is not None, f"{plugin_name} entry point not found"
    plugin_func = ep.load()
    plugin_func()
    config = PluginManager.get_config()

    # Required attributes
    required_attrs = [
        "model_cls_for_finetuning",
        "dataset_cls",
        "tokenizer_cls",
        "datacollator_cls_for_finetuning",
        "pretraining_required",
        "learning_rate",
        "max_grad_norm",
        "weight_decay",
    ]

    for attr in required_attrs:
        assert hasattr(
            config, attr
        ), f"Plugin '{plugin_name}' missing required attribute: {attr}"


def test_plugin_models_are_callable():
    """Verify that plugin model classes can be instantiated."""
    import importlib.metadata
    from unittest.mock import MagicMock

    from biolm.plugin_config import PluginManager

    for plugin_name in ["saluki", "xlnet"]:
        # Load plugin via entry point
        eps = importlib.metadata.entry_points(group="biolm.plugins")
        ep = next((e for e in eps if e.name == plugin_name), None)
        assert ep is not None, f"{plugin_name} entry point not found"
        plugin_func = ep.load()
        plugin_func()
        config = PluginManager.get_config()

        # Check that model classes are callable
        model_cls = config.model_cls_for_finetuning
        assert callable(
            model_cls
        ), f"{plugin_name} fine-tuning model class should be callable"

        # Verify it's a class (has __init__)
        assert hasattr(
            model_cls, "__init__"
        ), f"{plugin_name} model should be a class with __init__"


if __name__ == "__main__":
    # Allow running standalone for quick verification
    print("Testing plugin discovery...")
    test_plugin_entry_points_exist()
    test_saluki_plugin_registered()
    test_xlnet_plugin_registered()
    test_no_builtin_plugins()
    print("\n✅ All plugin discovery tests passed!")
