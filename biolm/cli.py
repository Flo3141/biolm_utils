"""CLI entry point and argument parsing."""

import warnings

import hydra
from omegaconf import DictConfig

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
def parse_args(cfg: DictConfig):
    """Hydra CLI entrypoint — returns processed BioLMConfig."""
    # Set up logging early
    setup_logging()

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

    # Reset paths to use the new config
    from .path_setup import PathsManager

    PathsManager._instance = None

    # Load plugin if specified
    if processed_config.plugin:
        try:
            # Try to load plugin via entry points
            import importlib.metadata

            eps = importlib.metadata.entry_points(group="biolm.plugins")
            plugin_names = [ep.name for ep in eps]
            if processed_config.plugin in plugin_names:
                for ep in eps:
                    if ep.name == processed_config.plugin:
                        plugin_func = ep.load()
                        plugin_func()
                        print(f"Plugin {processed_config.plugin} loaded successfully.")
                        break
            else:
                print(
                    f"Warning: Plugin {processed_config.plugin} not found in entry points. Available: {plugin_names}"
                )
        except Exception as e:
            print(f"Warning: Could not load plugin {processed_config.plugin}: {e}")
            import traceback

            traceback.print_exc()

    # Run the main function
    from .biolm import main

    main()


if __name__ == "__main__":
    parse_args()
