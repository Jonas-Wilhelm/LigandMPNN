"""Resolve default model weight paths relative to the package installation directory."""

import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PARAMS_DIR = os.path.join(_PACKAGE_DIR, "model_params")


def get_model_params_dir() -> str:
    """Return the path to the model_params directory inside the package."""
    return _MODEL_PARAMS_DIR


def default_checkpoint(filename: str) -> str:
    """Return the absolute path to a checkpoint file in model_params/."""
    return os.path.join(_MODEL_PARAMS_DIR, filename)


def validate_checkpoint(path: str) -> None:
    """Check that a checkpoint file exists; exit with a helpful message if not."""
    if not os.path.isfile(path):
        msg = f"Checkpoint not found: {path}"
        if path.startswith(_MODEL_PARAMS_DIR):
            msg += "\nRun 'ligandmpnn-fetch-weights --all' to download model weights."
        raise SystemExit(msg)
