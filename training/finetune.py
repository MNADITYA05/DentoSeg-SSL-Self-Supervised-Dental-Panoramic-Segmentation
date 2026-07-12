"""
training/finetune.py
─────────────────────
Supervised fine-tuning of the segmentation model.

Key design decisions vs. the original code
───────────────────────────────────────────
• Encoder freeze schedule: the encoder is frozen for the first
  `freeze_encoder_epochs` epochs so the randomly-initialised decoder can
  warm up without corrupting pretrained representations; then the encoder
  is unfrozen for joint fine-tuning.
• Validation split uses train_test_split (shuffled) rather than a blind
  slice of the first 20% of the training array.
• Model saved in .keras format (not deprecated .h5).
• Returns the Keras History object so callers can plot or inspect it.
"""

import pathlib
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from evaluation.metrics import iou_metric, dice_coefficient
from .losses import dice_loss


def train_segmentation(
    model: tf.keras.Model,
    encoder: tf.keras.Model,
    train_images: np.ndarray,
    train_masks: np.ndarray,
    *,
    val_ratio: float = 0.2,
    batch_size: int = 16,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    freeze_encoder_epochs: int = 10,
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
    best_model_path: str = "outputs/finetune/best_model.keras",
    final_model_path: str = "outputs/finetune/dental_segmentation_model.keras",
    log_csv_path: Optional[str] = None,
    seed: int = 42,
) -> tf.keras.callbacks.History:
    """
    Fine-tune the segmentation model on labelled data.

    Parameters
    ----------
    model : tf.keras.Model
        Full segmentation model (encoder + decoder).
    encoder : tf.keras.Model
        The shared encoder inside `model`.  Used for the freeze schedule.
    train_images : np.ndarray
        Shape (N, H, W, 1).
    train_masks : np.ndarray
        Shape (N, H, W, 1), binary float32.
    val_ratio : float
        Fraction of training data used for validation.
    batch_size : int
    epochs : int
    learning_rate : float
    freeze_encoder_epochs : int
        Encoder is frozen for this many epochs; then unfrozen.
        Set to 0 to skip freezing entirely.
    early_stopping_patience : int
    reduce_lr_patience : int
    reduce_lr_factor : float
    best_model_path : str
        Where to save the best checkpoint during training.
    final_model_path : str
        Where to save the fully trained model after training completes.
    log_csv_path : str or None
        If given, a CSV of per-epoch metrics is written here.
    seed : int
        Random seed for val split.

    Returns
    -------
    tf.keras.callbacks.History
    """
    # ── Validation split (shuffled) ────────────────────────────────────
    tr_img, val_img, tr_msk, val_msk = train_test_split(
        train_images, train_masks, test_size=val_ratio,
        random_state=seed, shuffle=True,
    )
    print(f"Fine-tune split → train: {len(tr_img)}  val: {len(val_img)}")

    # ── Ensure output directories exist ────────────────────────────────
    for p in [best_model_path, final_model_path]:
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)

    # ── Compile helper ─────────────────────────────────────────────────
    def _compile(lr: float) -> None:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=dice_loss,
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                iou_metric,
                dice_coefficient,
            ],
        )

    # ── Common callbacks ───────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coefficient",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coefficient",
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_dice_coefficient",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]
    if log_csv_path:
        pathlib.Path(log_csv_path).parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(tf.keras.callbacks.CSVLogger(log_csv_path))

    # ── Phase 1: frozen encoder warm-up ───────────────────────────────
    history = None
    if freeze_encoder_epochs > 0:
        print(f"\n[Phase 1] Encoder FROZEN — training decoder for "
              f"{freeze_encoder_epochs} epochs …")
        encoder.trainable = False
        _compile(learning_rate)
        history = model.fit(
            tr_img, tr_msk,
            validation_data=(val_img, val_msk),
            batch_size=batch_size,
            epochs=freeze_encoder_epochs,
            callbacks=callbacks,
            verbose=1,
        )

    # ── Phase 2: joint fine-tuning ─────────────────────────────────────
    remaining = epochs - (freeze_encoder_epochs if freeze_encoder_epochs > 0 else 0)
    if remaining > 0:
        print(f"\n[Phase 2] Encoder UNFROZEN — joint fine-tuning for "
              f"{remaining} epochs …")
        encoder.trainable = True
        _compile(learning_rate * 0.1)        # lower LR to protect pretrained weights
        initial_epoch = freeze_encoder_epochs if history else 0
        history = model.fit(
            tr_img, tr_msk,
            validation_data=(val_img, val_msk),
            batch_size=batch_size,
            epochs=epochs,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
            verbose=1,
        )

    # ── Save final model ───────────────────────────────────────────────
    model.save(final_model_path)
    print(f"\nFinal model saved → {final_model_path}")

    return history
