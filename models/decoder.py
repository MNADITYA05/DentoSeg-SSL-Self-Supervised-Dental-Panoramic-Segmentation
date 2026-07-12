"""
models/decoder.py
──────────────────
U-Net decoder: takes the encoder's [bottleneck, skip1..skip4] outputs and
upsamples back to the original spatial resolution, producing a 1-channel
sigmoid mask.
"""

import tensorflow as tf
from tensorflow.keras import layers


def _up_block(x: tf.Tensor, skip: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    """Transposed conv upsample → concat skip → two conv blocks."""
    x = layers.Conv2DTranspose(filters, 3, strides=2, padding="same",
                               name=f"{name}_upsample")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def create_decoder(encoder_outputs: list, base_filters: int = 64) -> tf.Tensor:
    """
    Build the U-Net decoder as a tensor graph attached to encoder outputs.

    Parameters
    ----------
    encoder_outputs : list
        [bottleneck, skip1, skip2, skip3, skip4] from create_encoder().
    base_filters : int
        Must match the base_filters used in the encoder.

    Returns
    -------
    tf.Tensor
        Output segmentation mask of shape (B, H, W, 1) with sigmoid activation.
    """
    bottleneck, skip1, skip2, skip3, skip4 = encoder_outputs
    f = base_filters

    x = _up_block(bottleneck, skip4, f * 8, name="dec_block1")
    x = _up_block(x,          skip3, f * 4, name="dec_block2")
    x = _up_block(x,          skip2, f * 2, name="dec_block3")
    x = _up_block(x,          skip1, f,     name="dec_block4")

    output = layers.Conv2D(1, 1, padding="same", activation="sigmoid",
                           name="segmentation_output")(x)
    return output
