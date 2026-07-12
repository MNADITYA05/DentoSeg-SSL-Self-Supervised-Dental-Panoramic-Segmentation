"""
utils/seed.py
──────────────
Global seed initialisation for reproducible experiments.

Sets seeds for Python's random module, NumPy, and TensorFlow.
Call set_global_seed(seed) once at the top of every script/notebook
before any model or data operations.
"""

import os
import random

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int = 42) -> None:
    """
    Set all relevant random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value (default 42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Global seed set to {seed}.")
