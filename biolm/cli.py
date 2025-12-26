"""CLI entry point and argument parsing."""

import warnings

import hydra
from omegaconf import DictConfig, OmegaConf

from .loader import _process_hydra_config
from .logging_config import setup_logging

warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.",
)


@hydra.main(
    config_path="/prj/RNA_NLP/biolm_utils/biolm/conf",
    config_name="config",
    version_base="1.1",
)
def _hydra_main(cfg: DictConfig):
    """Hydra CLI entrypoint — returns processed BioLMConfig."""
    # Set up logging early
    setup_logging()

    # Allow legacy overrides that address keys not present in the base config
    OmegaConf.set_struct(cfg, False)

    processed_config = _process_hydra_config(cfg)

    # Set the config for the framework to use
    from .config_access import ConfigManager

    ConfigManager._instance = processed_config

    # Set MLflow defaults if enabled
    if (
        processed_config.settings
        and hasattr(processed_config.settings, "mlflow")
        and processed_config.settings.mlflow
        and processed_config.settings.mlflow.get("enabled", False)
    ):
        if processed_config.settings.mlflow.get("tracking_uri") is None:
            processed_config.settings.mlflow["tracking_uri"] = (
                f"{processed_config.outputpath}/{processed_config.mode}/mlruns"
            )
        if processed_config.settings.mlflow.get("experiment_name") is None:
            processed_config.settings.mlflow["experiment_name"] = (
                f"{processed_config.mode}_{processed_config.outputpath.split('/')[-1]}"
            )

    # Reset cached module-level state to use the new config
    from . import biolm as biolm_module
    from . import constants as constants_module
    from .path_setup import PathsManager

    PathsManager._instance = None
    constants_module._constants = None
    biolm_module.args = processed_config
    biolm_module.constants = constants_module.get_constants()
    biolm_module.paths = PathsManager.get_paths()

    # Load plugin if specified
    if processed_config.plugin:
        plugin_loaded = False
        available_plugins = []
        try:
            # Try to load plugin via entry points
            import importlib.metadata

            eps = importlib.metadata.entry_points(group="biolm.plugins")
            available_plugins = [ep.name for ep in eps]
            for ep in eps:
                if ep.name == processed_config.plugin:
                    plugin_func = ep.load()
                    plugin_func()
                    print(
                        f"Plugin {processed_config.plugin} loaded successfully via entry point."
                    )
                    plugin_loaded = True
                    break
        except Exception as e:
            print(f"Warning: Could not load plugin {processed_config.plugin}: {e}")
            import traceback

            traceback.print_exc()

        if not plugin_loaded:
            print(
                "Warning: Plugin {name} could not be loaded. Available entry-point plugins: {available}.".format(
                    name=processed_config.plugin,
                    available=available_plugins,
                )
            )

    # Run the main function
    from .biolm import main

    main()


def parse_args():
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "plugin":
            from .plugin_manager import handle_plugin_command

            handle_plugin_command(sys.argv[2:])
            return
        elif sys.argv[1] == "install-plugin":
            from .plugin_manager import handle_plugin_command

            # Map 'install-plugin <url>' to 'plugin install <url>'
            handle_plugin_command(["install"] + sys.argv[2:])
            return
        elif sys.argv[1] == "list-plugins":
            from .plugin_manager import handle_plugin_command

            # Map 'list-plugins' to 'plugin list'
            handle_plugin_command(["list"] + sys.argv[2:])
            return
        elif sys.argv[1] == "remove-plugin":
            from .plugin_manager import handle_plugin_command

            # Map 'remove-plugin <name>' to 'plugin remove <name>'
            handle_plugin_command(["remove"] + sys.argv[2:])
            return

    _hydra_main()


if __name__ == "__main__":
    parse_args()
