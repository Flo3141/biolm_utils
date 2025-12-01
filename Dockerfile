# BioLM Docker Setup

This directory contains Docker configurations for running the BioLM framework with plugins.

## Quick Start

1. **Build the Docker Image**:
   ```bash
   docker build -t biolm:latest .
   ```

2. **Run an Experiment**:
   ```bash
   docker run --rm -v /path/to/your/configs:/configs -v /path/to/output:/output biolm:latest biolm fine-tune --config-path /configs/your_experiment
   ```

   - Replace `/path/to/your/configs` with the path to your plugin configs (e.g., `/home/pwiesenbach/rna_saluki_cnn/_exampleconfigs`).
   - Replace `/path/to/output` with where you want experiment outputs saved.

## Files

- `Dockerfile`: Defines the base image with framework and plugins installed.
- `docker-compose.yml` (optional): For advanced setups with multiple services.
- `requirements.txt`: Pinned dependencies for reproducibility.

## Customization

- To add a new plugin, edit the `Dockerfile` to include it (e.g., `RUN pip install git+https://github.com/your/plugin.git`).
- For GPU support, use `--gpus all` in the `docker run` command (requires NVIDIA Docker).

## Notes

- This setup assumes plugins are installed via pip. For local plugins, mount them as volumes.
- Ensure Docker has access to your data paths.