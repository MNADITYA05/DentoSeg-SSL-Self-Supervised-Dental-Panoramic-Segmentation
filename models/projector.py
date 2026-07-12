"""
models/projector.py
────────────────────
Projection head used during contrastive pretraining only.

Maps the encoder bottleneck (spatial feature map) → a fixed-size embedding
vector via GlobalAveragePooling + two-layer MLP.

This head is discarded after pretraining; only the encoder weights are
transferred to the segmentation model.
"""

import tensorflow as tf
from tensorflow.keras import layers


def create_projection_head(
    encoder_output: tf.Tensor,
    projection_dim: int = 128,
    hidden_dim: int = 512,
) -> tf.Tensor:
    """
    Attach a projection MLP to the encoder bottleneck tensor.

    Parameters
    ----------
    encoder_output : tf.Tensor
        Bottleneck feature map from the encoder: shape (B, h, w, C).
    projection_dim : int
        Dimensionality of the final projection vector (SimCLR z-space).
    hidden_dim : int
        Width of the intermediate dense layer.

    Returns
    -------
    tf.Tensor
        L2-normalised projection vector of shape (B, projection_dim).
    """
    x = layers.GlobalAveragePooling2D(name="proj_gap")(encoder_output)
    x = layers.Dense(hidden_dim, name="proj_dense1")(x)
    x = layers.BatchNormalization(name="proj_bn")(x)
    x = layers.Activation("relu", name="proj_relu")(x)
    x = layers.Dense(projection_dim, name="proj_dense2")(x)
    return x
