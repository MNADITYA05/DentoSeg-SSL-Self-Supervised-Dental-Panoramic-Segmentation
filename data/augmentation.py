"""
data/augmentation.py
─────────────────────
Augmentation layers suited to dental panoramic X-rays.

Key design choices
──────────────────
• RandomContrast is implemented as a proper tf.keras.layers.Layer (not a
  Lambda) so that models containing it can be saved and reloaded cleanly.
• Augmentation parameters are configurable via keyword arguments so they
  can be driven from YAML configs without code changes.
"""

from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers


# ──────────────────────────────────────────────────────────────────────
#  Custom serialisable contrast layer
# ──────────────────────────────────────────────────────────────────────

class RandomContrast(tf.keras.layers.Layer):
    """
    Randomly adjusts the contrast of a grayscale image during training.

    Applies: output = (input − mean) * factor + mean
    where factor ~ Uniform(lower, upper).

    Unlike tf.keras.layers.Lambda, this layer is fully serialisable.

    Parameters
    ----------
    lower : float
        Minimum contrast scale factor (< 1.0 reduces contrast).
    upper : float
        Maximum contrast scale factor (> 1.0 increases contrast).
    """

    def __init__(self, lower: float = 0.85, upper: float = 1.15, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lower = lower
        self.upper = upper

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        if training:
            factor = tf.random.uniform(shape=[], minval=self.lower, maxval=self.upper)
            mean = tf.reduce_mean(x)
            x = (x - mean) * factor + mean
            x = tf.clip_by_value(x, 0.0, 1.0)
        return x

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"lower": self.lower, "upper": self.upper})
        return config


# ──────────────────────────────────────────────────────────────────────
#  Augmentation pipeline factory
# ──────────────────────────────────────────────────────────────────────

def get_augmentation_pipeline(
    horizontal_flip: bool = True,
    rotation_factor: float = 0.1,
    gaussian_noise_stddev: float = 0.05,
    contrast_lower: float = 0.85,
    contrast_upper: float = 1.15,
) -> tf.keras.Sequential:
    """
    Build a Keras Sequential augmentation pipeline for contrastive pretraining.

    The pipeline is designed for dental X-ray specifics:
    • Horizontal flip — valid for panoramic views (symmetric dentition).
    • Small rotation — accounts for slight head tilt variation.
    • Gaussian noise — simulates X-ray detector noise.
    • Random contrast — simulates exposure variation between machines.

    Parameters
    ----------
    horizontal_flip : bool
        Include random horizontal flip.
    rotation_factor : float
        Max rotation as a fraction of 2π (e.g. 0.1 → ±36°).
    gaussian_noise_stddev : float
        Standard deviation of additive Gaussian noise.
    contrast_lower : float
        Lower bound for contrast scale factor.
    contrast_upper : float
        Upper bound for contrast scale factor.

    Returns
    -------
    tf.keras.Sequential
        A stateless augmentation pipeline.
    """
    aug_layers = []

    if horizontal_flip:
        aug_layers.append(layers.RandomFlip("horizontal"))

    if rotation_factor > 0:
        aug_layers.append(layers.RandomRotation(rotation_factor))

    if gaussian_noise_stddev > 0:
        aug_layers.append(layers.GaussianNoise(gaussian_noise_stddev))

    aug_layers.append(
        RandomContrast(lower=contrast_lower, upper=contrast_upper)
    )

    return tf.keras.Sequential(aug_layers, name="augmentation_pipeline")
