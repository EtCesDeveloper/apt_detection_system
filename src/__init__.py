"""
APT detection package — Multitask CNN+LSTM for CICIDS2017.

This package exposes the high-level building blocks of the project so that
external scripts (or notebooks) can do:

    from src import build_multitask_model, load_cicids2017, build_dataset

instead of having to dig into submodules.

Public API
----------
- config                  : global paths and hyperparameters (module)
- load_cicids2017         : load and clean the raw CSV files
- build_dataset           : turn the cleaned DataFrame into 3D sequences
- build_multitask_model   : build and compile the CNN+LSTM multitask model
"""

from __future__ import annotations
import logging

# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------
__version__ = "0.1.0"
__author__ = "APT Detection Project"
__all__ = [
    "config",
    "load_cicids2017",
    "build_dataset",
    "build_multitask_model",
]

# ---------------------------------------------------------------------------
# Logging configuration (run once at import time)
# ---------------------------------------------------------------------------
# Using a NullHandler is the recommended pattern for libraries: it prevents
# "No handlers could be found" warnings while letting the application that
# uses the package configure its own logging output.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Eager re-export of the lightweight `config` module.
# Heavier modules (model, data_loader, preprocessing) are loaded lazily
# below to avoid importing TensorFlow until it's actually needed.
# ---------------------------------------------------------------------------
from . import env_setup  # noqa: E402  must come before any TF import
from . import config     # noqa: E402


def __getattr__(name: str):
    """
    PEP 562 lazy attribute loader.

    TensorFlow / pandas are heavy to import (several seconds). By loading
    model and data symbols only when actually requested, `import src`
    stays fast for tools that only need `src.config` (e.g. unit tests,
    CLI --help, or quick path lookups).
    """
    if name == "load_cicids2017":
        from .data_loader import load_cicids2017
        return load_cicids2017
    if name == "build_dataset":
        from .preprocessing import build_dataset
        return build_dataset
    if name == "build_multitask_model":
        from .model import build_multitask_model
        return build_multitask_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Make tab-completion show the public API even with lazy loading."""
    # Only expose the documented public API + dunder metadata, hiding
    # internal imports (logging, annotations, etc.) from autocompletion.
    return sorted(set(__all__) | {"__version__", "__author__"})
