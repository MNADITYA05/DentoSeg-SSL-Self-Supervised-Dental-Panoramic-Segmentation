"""
utils/env.py
─────────────
Runtime environment detection and default path resolution.

Supported environments
──────────────────────
  kaggle  — /kaggle/input dataset mount, /kaggle/working output
  colab   — Google Colab (/content/*)
  local   — local filesystem (relative paths from project root)

Use detect_environment() at the top of scripts/notebooks and pass the
result to get_default_paths() to obtain sensible defaults without
hardcoding any paths in source code.
"""

import os
from typing import Dict


def detect_environment() -> str:
    """
    Detect the current compute environment.

    Returns
    -------
    str
        One of: 'kaggle', 'colab', 'local'.
    """
    if os.path.exists("/kaggle"):
        return "kaggle"
    if "COLAB_GPU" in os.environ or "COLAB_BACKEND_URL" in os.environ:
        return "colab"
    return "local"


def get_default_paths(env: str = None) -> Dict[str, str]:
    """
    Return sensible default data and output paths for the given environment.

    Parameters
    ----------
    env : str or None
        One of 'kaggle', 'colab', 'local'.  Auto-detected if None.

    Returns
    -------
    dict with keys: 'images', 'masks', 'output'
    """
    if env is None:
        env = detect_environment()

    if env == "kaggle":
        base = (
            "/kaggle/input/childrens-dental-panoramic-radiographs-dataset"
            "/Dental_dataset/Adult tooth segmentation dataset"
            "/Panoramic radiography database"
        )
        return {
            "images": f"{base}/images",
            "masks":  f"{base}/mask",
            "output": "/kaggle/working/outputs",
        }

    if env == "colab":
        return {
            "images": "/content/data/images",
            "masks":  "/content/data/masks",
            "output": "/content/outputs",
        }

    # local
    return {
        "images": "data/images",
        "masks":  "data/masks",
        "output": "outputs",
    }
