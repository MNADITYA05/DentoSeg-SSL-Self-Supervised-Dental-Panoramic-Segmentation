"""
training/pretrain.py
─────────────────────
Contrastive pre-training loop for the SSL stage.

ContrastiveTrainer wraps:
  • a compiled @tf.function train step (fast GPU/TPU execution)
  • per-epoch progress bars via tqdm
  • CSV loss logging
  • encoder checkpoint saving at the end of pretraining
"""

import csv
import pathlib
from typing import Optional

import tensorflow as tf
from tqdm import tqdm

from .losses import nt_xent_loss


class ContrastiveTrainer:
    """
    Trains the contrastive model using NT-Xent loss.

    Parameters
    ----------
    model : tf.keras.Model
        The contrastive model (outputs [z1, z2]).
    encoder : tf.keras.Model
        The shared encoder inside `model`; saved separately after training.
    temperature : float
        NT-Xent temperature hyperparameter.
    learning_rate : float
        Adam learning rate.
    log_path : str or None
        If provided, per-epoch losses are appended to this CSV file.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        encoder: tf.keras.Model,
        temperature: float = 0.1,
        learning_rate: float = 3e-4,
        log_path: Optional[str] = None,
    ) -> None:
        self.model = model
        self.encoder = encoder
        self.temperature = temperature
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.log_path = pathlib.Path(log_path) if log_path else None

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "avg_loss"])

    # ------------------------------------------------------------------
    # Single-step (compiled for speed)
    # ------------------------------------------------------------------

    @tf.function
    def _train_step(self, images: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            z1, z2 = self.model(images, training=True)
            loss = nt_xent_loss(z1, z2, self.temperature)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: tf.data.Dataset,
        epochs: int,
        encoder_checkpoint: Optional[str] = None,
    ) -> None:
        """
        Run the contrastive pretraining loop.

        Parameters
        ----------
        dataset : tf.data.Dataset
            Batched dataset of images (masks are not used here).
        epochs : int
            Number of pretraining epochs.
        encoder_checkpoint : str or None
            If provided, save the encoder weights to this path after training.
        """
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            num_batches = 0

            pbar = tqdm(dataset, desc=f"Pretrain {epoch}/{epochs}", unit="batch")
            for batch in pbar:
                loss = self._train_step(batch)
                epoch_loss += float(loss)
                num_batches += 1
                pbar.set_postfix({"loss": f"{float(loss):.4f}"})

            avg_loss = epoch_loss / max(num_batches, 1)
            print(f"  Epoch {epoch}/{epochs} — avg loss: {avg_loss:.4f}")

            if self.log_path:
                with open(self.log_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, f"{avg_loss:.6f}"])

        # Save encoder weights for transfer to segmentation model
        if encoder_checkpoint:
            path = pathlib.Path(encoder_checkpoint)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.encoder.save(str(path))
            print(f"\nEncoder saved → {path}")
