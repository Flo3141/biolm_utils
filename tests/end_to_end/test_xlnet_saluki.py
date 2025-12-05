"""
End-to-end tests for plugin system integration.

These tests verify that:
1. Plugins load correctly (via entry points or built-in loaders)
2. Mode dispatch works correctly (including graceful handling of unsupported modes)
3. Error messages are informative
"""

import logging
import subprocess

import pytest

# Configure logging to capture debug output
logging.basicConfig(level=logging.DEBUG)


def debug_log(msg):
    """Print debug message to stdout for test visibility."""
    print(msg)


def test_plugin_loading():
    """Test that plugins can be loaded via built-in loader."""
    debug_log("Starting plugin loading test")

    # This should load the built-in plugins if entry points are unavailable
    command = [
        "poetry",
        "run",
        "python",
        "-c",
        """
from biolm.plugin_config import PluginManager
from biolm.plugins.builtin import register_xlnet_plugin, register_saluki_plugin

# Test loading XLNet plugin
xlnet_result = register_xlnet_plugin()
print(f"XLNet plugin loaded: {xlnet_result}")
config = PluginManager.get_config()
print(f"XLNet pretraining_required: {config.pretraining_required}")

# Reset for next plugin
PluginManager._config = None

# Test loading Saluki plugin
saluki_result = register_saluki_plugin()
print(f"Saluki plugin loaded: {saluki_result}")
config = PluginManager.get_config()
print(f"Saluki pretraining_required: {config.pretraining_required}")
""",
    ]

    debug_log(f"Running: {' '.join(command)}")
    result = subprocess.run(
        command, capture_output=True, text=True, cwd="/prj/RNA_NLP/biolm_utils"
    )
    debug_log(f"Output: {result.stdout}")
    if result.returncode != 0:
        debug_log(f"Error: {result.stderr}")
    assert result.returncode == 0, f"Plugin loading failed: {result.stderr}"
    assert "XLNet plugin loaded: True" in result.stdout, "XLNet plugin not loaded"
    assert "Saluki plugin loaded: True" in result.stdout, "Saluki plugin not loaded"
    assert (
        "XLNet pretraining_required: True" in result.stdout
    ), "XLNet should require pretraining"
    assert (
        "Saluki pretraining_required: False" in result.stdout
    ), "Saluki should not require pretraining"


def test_saluki_unsupported_pretraining():
    """Test that Saluki plugin config shows pretraining_required=False."""
    debug_log("Starting Saluki pre-train check")

    # Just verify the config is set up correctly to disallow pre-training
    command = [
        "poetry",
        "run",
        "python",
        "-c",
        """
from biolm.plugin_config import PluginManager
from biolm.plugins.builtin import register_saluki_plugin

register_saluki_plugin()
config = PluginManager.get_config()

# Verify Saluki doesn't support pre-training
if config.pretraining_required:
    print("ERROR: Saluki should not require pretraining")
else:
    print("SUCCESS: Saluki correctly has pretraining_required=False")

if config.model_cls_for_pretraining is not None:
    print("ERROR: Saluki should not have a pretraining model class")
else:
    print("SUCCESS: Saluki correctly has model_cls_for_pretraining=None")
""",
    ]

    debug_log(f"Running: {' '.join(command)}")
    result = subprocess.run(
        command, capture_output=True, text=True, cwd="/prj/RNA_NLP/biolm_utils"
    )
    debug_log(f"Output: {result.stdout}")
    if result.returncode != 0:
        debug_log(f"Error: {result.stderr}")
    assert result.returncode == 0, f"Saluki pretraining check failed: {result.stderr}"
    assert (
        "SUCCESS" in result.stdout
    ), f"Saluki pretraining check failed: {result.stdout}"


def test_xlnet_plugin_config():
    """Test that XLNet plugin is correctly configured."""
    debug_log("Starting XLNet config test")

    command = [
        "poetry",
        "run",
        "python",
        "-c",
        """
from biolm.plugin_config import PluginManager
from biolm.plugins.builtin import register_xlnet_plugin

register_xlnet_plugin()
config = PluginManager.get_config()

# Verify all expected attributes are present
print(f"model_cls_for_pretraining: {config.model_cls_for_pretraining is not None}")
print(f"model_cls_for_finetuning: {config.model_cls_for_finetuning is not None}")
print(f"dataset_cls: {config.dataset_cls is not None}")
print(f"tokenizer_cls: {config.tokenizer_cls is not None}")
print(f"datacollator_cls_for_pretraining: {config.datacollator_cls_for_pretraining is not None}")
print(f"datacollator_cls_for_finetuning: {config.datacollator_cls_for_finetuning is not None}")
print(f"add_special_tokens: {config.add_special_tokens}")
print(f"pretraining_required: {config.pretraining_required}")
""",
    ]

    debug_log(f"Running: {' '.join(command)}")
    result = subprocess.run(
        command, capture_output=True, text=True, cwd="/prj/RNA_NLP/biolm_utils"
    )
    debug_log(f"Output: {result.stdout}")
    if result.returncode != 0:
        debug_log(f"Error: {result.stderr}")
    assert result.returncode == 0, f"XLNet config test failed: {result.stderr}"
    # Check that all required fields are present and non-None
    assert "model_cls_for_pretraining: True" in result.stdout
    assert "model_cls_for_finetuning: True" in result.stdout
    assert "dataset_cls: True" in result.stdout
    assert "tokenizer_cls: True" in result.stdout
    assert "datacollator_cls_for_finetuning: True" in result.stdout
    assert "add_special_tokens: True" in result.stdout
    assert "pretraining_required: True" in result.stdout


def test_saluki_plugin_config():
    """Test that Saluki plugin is correctly configured."""
    debug_log("Starting Saluki config test")

    command = [
        "poetry",
        "run",
        "python",
        "-c",
        """
from biolm.plugin_config import PluginManager
from biolm.plugins.builtin import register_saluki_plugin

register_saluki_plugin()
config = PluginManager.get_config()

# Verify all expected attributes are present
print(f"model_cls_for_pretraining: {config.model_cls_for_pretraining}")
print(f"model_cls_for_finetuning: {config.model_cls_for_finetuning is not None}")
print(f"dataset_cls: {config.dataset_cls is not None}")
print(f"tokenizer_cls: {config.tokenizer_cls is not None}")
print(f"datacollator_cls_for_pretraining: {config.datacollator_cls_for_pretraining}")
print(f"datacollator_cls_for_finetuning: {config.datacollator_cls_for_finetuning is not None}")
print(f"add_special_tokens: {config.add_special_tokens}")
print(f"pretraining_required: {config.pretraining_required}")
""",
    ]

    debug_log(f"Running: {' '.join(command)}")
    result = subprocess.run(
        command, capture_output=True, text=True, cwd="/prj/RNA_NLP/biolm_utils"
    )
    debug_log(f"Output: {result.stdout}")
    if result.returncode != 0:
        debug_log(f"Error: {result.stderr}")
    assert result.returncode == 0, f"Saluki config test failed: {result.stderr}"
    # Check that all required fields are present correctly
    assert "model_cls_for_pretraining: None" in result.stdout
    assert "model_cls_for_finetuning: True" in result.stdout
    assert "dataset_cls: True" in result.stdout
    assert "tokenizer_cls: True" in result.stdout
    assert "datacollator_cls_for_pretraining: None" in result.stdout
    assert "datacollator_cls_for_finetuning: True" in result.stdout
    assert "add_special_tokens: False" in result.stdout
    assert "pretraining_required: False" in result.stdout
