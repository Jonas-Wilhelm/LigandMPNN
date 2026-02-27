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
