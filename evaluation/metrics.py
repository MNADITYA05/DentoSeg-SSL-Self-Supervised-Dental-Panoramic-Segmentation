"""
evaluation/metrics.py
──────────────────────
Keras-compatible metric functions for binary segmentation.

Both functions follow the Keras metric API (y_true, y_pred) and can be
passed directly to model.compile(metrics=[...]).
"""

import tensorflow as tf


def iou_metric(y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """
    Mean Intersection-over-Union for binary segmentation.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth mask, values in {0, 1}.
    y_pred : tf.Tensor
        Predicted probability map, values in [0, 1].
    threshold : float
        Decision threshold to binarise predictions.

    Returns
    -------
    tf.Tensor
        Scalar mean IoU over the batch.
    """
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # Sum over spatial dims and channel; keep batch dim
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection

    iou = tf.reduce_mean((intersection + 1e-7) / (union + 1e-7))
    return iou


def dice_coefficient(
    y_true: tf.Tensor, y_pred: tf.Tensor, threshold: float = 0.5, smooth: float = 1.0
) -> tf.Tensor:
    """
    Dice coefficient metric (higher is better, in [0, 1]).

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth binary mask.
    y_pred : tf.Tensor
        Predicted probability map.
    threshold : float
        Decision threshold.
    smooth : float
        Laplace smoothing constant.

    Returns
    -------
    tf.Tensor
        Scalar Dice coefficient.
    """
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    return (2.0 * intersection + smooth) / (union + smooth)
