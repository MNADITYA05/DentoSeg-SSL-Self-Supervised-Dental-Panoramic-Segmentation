"""
models/segmentation.py
───────────────────────
Assembles the two end-to-end models:

  1. create_contrastive_model  — encoder + two augmented views + projection
                                 heads; used during SSL pretraining.
  2. create_segmentation_model — pretrained encoder + U-Net decoder;
                                 used during supervised fine-tuning.
"""

import tensorflow as tf
from tensorflow.keras import layers

from .encoder import create_encoder
from .decoder import create_decoder
from .projector import create_projection_head
from data.augmentation import get_augmentation_pipeline


def create_contrastive_model(
    encoder: tf.keras.Model,
    input_shape: tuple = (224, 224, 1),
    projection_dim: int = 128,
    augmentation_cfg: dict = None,
) -> tf.keras.Model:
    """
    Build the SimCLR-style contrastive model.

    A single input image is augmented two independent times to produce two
    views.  Both views are encoded and projected.  The model outputs the pair
    of projection vectors (z1, z2) so the NT-Xent loss can be computed
    outside the model (enabling @tf.function optimisation of the loss).

    Parameters
    ----------
    encoder : tf.keras.Model
        The shared encoder (from create_encoder).
    input_shape : tuple
        (H, W, C) matching the encoder's input.
    projection_dim : int
        Projection head output dimension.
    augmentation_cfg : dict or None
        Keyword arguments forwarded to get_augmentation_pipeline().
        If None, defaults are used.

    Returns
    -------
    tf.keras.Model
        Inputs: (B, H, W, C) image batch.
        Outputs: [(B, projection_dim), (B, projection_dim)] — (z1, z2).
    """
    if augmentation_cfg is None:
        augmentation_cfg = {}

    augmentation = get_augmentation_pipeline(**augmentation_cfg)

    inputs = layers.Input(shape=input_shape, name="contrastive_input")

    # Two independently-augmented views of the same image
    view1 = augmentation(inputs, training=True)
    view2 = augmentation(inputs, training=True)

    # Shared encoder — bottleneck only (skip connections not needed here)
    bottleneck1, *_ = encoder(view1, training=True)
    bottleneck2, *_ = encoder(view2, training=True)

    # Separate projection heads (weights are NOT shared between the two heads)
    z1 = create_projection_head(bottleneck1, projection_dim=projection_dim, name="proj1")
    z2 = create_projection_head(bottleneck2, projection_dim=projection_dim, name="proj2")

    model = tf.keras.Model(inputs=inputs, outputs=[z1, z2], name="contrastive_model")
    return model


def create_segmentation_model(
    encoder: tf.keras.Model,
    input_shape: tuple = (224, 224, 1),
    base_filters: int = 64,
) -> tf.keras.Model:
    """
    Assemble the full U-Net segmentation model using a pretrained encoder.

    Parameters
    ----------
    encoder : tf.keras.Model
        Pretrained encoder (weights transferred from contrastive pretraining).
    input_shape : tuple
        (H, W, C) matching the encoder's input.
    base_filters : int
        Must match the base_filters used when the encoder was created.

    Returns
    -------
    tf.keras.Model
        Inputs: (B, H, W, C).
        Outputs: (B, H, W, 1) sigmoid segmentation mask.
    """
    inputs = layers.Input(shape=input_shape, name="seg_input")

    encoder_outputs = encoder(inputs)                    # [bottleneck, skip1..4]
    segmentation_mask = create_decoder(encoder_outputs, base_filters=base_filters)

    model = tf.keras.Model(inputs=inputs, outputs=segmentation_mask,
                           name="segmentation_model")
    return model
