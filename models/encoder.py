"""
models/encoder.py
──────────────────
U-Net encoder backbone (5-block CNN).

Returns the bottleneck feature map AND four skip-connection tensors so the
same encoder can be reused for both contrastive pretraining and segmentation
fine-tuning without duplication.

Block channel progression: base → 2×base → 4×base → 8×base → 16×base
Default (base=64):          64  →  128   →  256   →   512  →  1024
"""

import tensorflow as tf
from tensorflow.keras import layers


def _conv_block(x: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Two Conv2D → BN → ReLU layers."""
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def create_encoder(
    input_shape: tuple = (224, 224, 1),
    base_filters: int = 64,
) -> tf.keras.Model:
    """
    Build the U-Net encoder.

    Parameters
    ----------
    input_shape : tuple
        (H, W, C) of the input images. Default is (224, 224, 1) for grayscale.
    base_filters : int
        Number of filters in the first block; doubles each subsequent block.

    Returns
    -------
    tf.keras.Model
        Outputs: [bottleneck, skip1, skip2, skip3, skip4]
        where skip1 is the highest-resolution feature map (½ of input).
    """
    inputs = layers.Input(shape=input_shape, name="encoder_input")
    f = base_filters  # shorthand

    # Block 1 — full resolution
    skip1 = _conv_block(inputs, f, name="enc_block1")
    x = layers.MaxPooling2D(name="enc_pool1")(skip1)          # H/2

    # Block 2
    skip2 = _conv_block(x, f * 2, name="enc_block2")
    x = layers.MaxPooling2D(name="enc_pool2")(skip2)          # H/4

    # Block 3
    skip3 = _conv_block(x, f * 4, name="enc_block3")
    x = layers.MaxPooling2D(name="enc_pool3")(skip3)          # H/8

    # Block 4
    skip4 = _conv_block(x, f * 8, name="enc_block4")
    x = layers.MaxPooling2D(name="enc_pool4")(skip4)          # H/16

    # Block 5 — bottleneck
    bottleneck = _conv_block(x, f * 16, name="enc_bottleneck")  # H/32

    encoder = tf.keras.Model(
        inputs=inputs,
        outputs=[bottleneck, skip1, skip2, skip3, skip4],
        name="encoder",
    )
    return encoder
