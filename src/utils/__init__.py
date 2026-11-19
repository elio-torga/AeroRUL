"""Cross-cutting helpers: configuration, logging, reproducibility and paths."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def load_config(path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """Load the YAML configuration file.

    Parameters
    ----------
    path:
        Optional path to a YAML file. Defaults to ``<project_root>/config.yaml``.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(relative: str) -> Path:
    """Resolve a config path relative to the project root and create it."""
    path = Path(relative)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a module logger writing to stderr with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def add_file_handler(logger: logging.Logger, log_file: Path) -> None:
    """Additionally persist a logger's records to ``log_file``."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(handler)


def set_seed(seed: int) -> None:
    """Seed every random number generator used by the project."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
