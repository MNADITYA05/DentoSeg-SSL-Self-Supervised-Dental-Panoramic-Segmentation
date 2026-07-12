"""
utils/config.py
────────────────
YAML configuration loader with graceful fallback.

If PyYAML is available the config is loaded from a .yaml file.
If not, the raw dict is returned as-is (useful when configs are
constructed in-notebook without writing to disk).
"""

import pathlib
from typing import Any, Dict, Union

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


def load_config(path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """
    Load a YAML config file into a nested dict.

    Parameters
    ----------
    path : str or Path
        Path to a .yaml config file.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    ImportError
        If PyYAML is not installed.
    FileNotFoundError
        If the config file does not exist.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if not _YAML_AVAILABLE:
        raise ImportError(
            "PyYAML is required to load config files. "
            "Install it with: pip install pyyaml"
        )

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def flat_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten a nested config dict one level deep into dot-separated keys.
    Useful for logging hyperparameters to CSV/W&B.

    Example
    -------
    {'training': {'lr': 1e-4}} → {'training.lr': 1e-4}
    """
    flat = {}
    for section, values in config.items():
        if isinstance(values, dict):
            for key, val in values.items():
                flat[f"{section}.{key}"] = val
        else:
            flat[section] = values
    return flat
