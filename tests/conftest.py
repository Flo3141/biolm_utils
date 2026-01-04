import os
import shutil

import pytest


@pytest.fixture(scope="session", autouse=True)
def mlflow_isolated_tracking(tmp_path_factory):
    """Force MLflow to use a session-scoped temp tracking dir and clean it up.

    This prevents test runs from leaving an `mlruns` folder in the repository root
    while still allowing tests to override `tracking_uri` explicitly when needed.
    """

    tmpdir = tmp_path_factory.mktemp("mlflow_tracking")
    prev = os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_URI"] = f"file://{tmpdir}"
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
    os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)

    yield

    # Restore environment
    if prev is not None:
        os.environ["MLFLOW_TRACKING_URI"] = prev
    else:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
    os.environ.pop("MLFLOW_EXPERIMENT_NAME", None)

    # Clean up temp and any stray repo-level mlruns
    shutil.rmtree(tmpdir, ignore_errors=True)
    shutil.rmtree("mlruns", ignore_errors=True)
