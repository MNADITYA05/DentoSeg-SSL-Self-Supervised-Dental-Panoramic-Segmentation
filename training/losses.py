"""
training/losses.py
───────────────────
Loss functions for both training stages.

NT-Xent (Normalised Temperature-scaled Cross Entropy)
──────────────────────────────────────────────────────
The correct SimCLR formulation:
  • Concatenate z1 and z2 into a 2B × D matrix.
  • Build a 2B × 2B cosine similarity matrix, scaled by temperature.
  • Mask out self-similarities (diagonal).
  • For each row i, the positive pair is row (i + B) mod 2B.
  • Loss = mean cross-entropy over all 2B rows.

The original code used tf.linalg.set_diag with k=batch_size which is
invalid — set_diag only supports the main diagonal.  This version uses
tf.nn.sparse_softmax_cross_entropy_with_logits with explicit label indices,
which is both correct and efficient.

Dice Loss
─────────
Standard soft-Dice loss for binary segmentation.
"""

import tensorflow as tf


# ──────────────────────────────────────────────────────────────────────
#  NT-Xent loss  (SimCLR, corrected)
# ──────────────────────────────────────────────────────────────────────

def nt_xent_loss(z1: tf.Tensor, z2: tf.Tensor, temperature: float = 0.1) -> tf.Tensor:
    """
    Normalised Temperature-scaled Cross Entropy loss.

    Parameters
    ----------
    z1, z2 : tf.Tensor
        Projection vectors of shape (B, D).  Need not be pre-normalised.
    temperature : float
        Softmax temperature τ.  Smaller values sharpen the distribution.

    Returns
    -------
    tf.Tensor
        Scalar loss value.
    """
    # L2-normalise both sets of projections
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    batch_size = tf.shape(z1)[0]

    # Concatenate to 2B × D, then compute 2B × 2B cosine similarity
    z = tf.concat([z1, z2], axis=0)                        # (2B, D)
    sim = tf.matmul(z, z, transpose_b=True) / temperature  # (2B, 2B)

    # Remove self-similarity (diagonal) by adding a large negative value
    self_mask = tf.eye(2 * batch_size) * -1e9
    sim = sim + self_mask

    # Positive pair for row i (i < B)  → row i + B
    # Positive pair for row i (i >= B) → row i - B
    labels = tf.concat(
        [tf.range(batch_size, 2 * batch_size),
         tf.range(batch_size)],
        axis=0,
    )  # shape (2B,)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=sim)
    )
    return loss


# ──────────────────────────────────────────────────────────────────────
#  Dice loss
# ──────────────────────────────────────────────────────────────────────

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1.0) -> tf.Tensor:
    """
    Soft Dice loss for binary segmentation.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground-truth binary mask, values in {0, 1}.
    y_pred : tf.Tensor
        Predicted probability map, values in [0, 1].
    smooth : float
        Laplace smoothing constant to prevent division by zero.

    Returns
    -------
    tf.Tensor
        Scalar loss value in [0, 1].  Lower is better.
    """
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice_coeff
