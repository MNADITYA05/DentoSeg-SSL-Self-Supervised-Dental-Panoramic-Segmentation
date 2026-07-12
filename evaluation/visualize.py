"""
evaluation/visualize.py
────────────────────────
Visualisation utilities:
  • make_gradcam_heatmap  — Grad-CAM for segmentation models
  • overlay_gradcam       — blend heatmap onto original image
  • visualize_results     — grid of (image | ground truth | prediction | Grad-CAM)
  • plot_training_history — loss / metric curves across epochs
"""

import pathlib
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# ──────────────────────────────────────────────────────────────────────
#  Grad-CAM
# ──────────────────────────────────────────────────────────────────────

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: Optional[str] = None,
) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap for a binary segmentation model.

    Parameters
    ----------
    img_array : np.ndarray
        Single image with a batch dimension: shape (1, H, W, C).
    model : tf.keras.Model
        Trained segmentation model.
    last_conv_layer_name : str or None
        Name of the conv layer to hook.  Auto-detected if None.

    Returns
    -------
    np.ndarray
        Normalised heatmap in [0, 1], shape (h', w').
    """
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if "conv" in layer.name and "transpose" not in layer.name:
                last_conv_layer_name = layer.name
                break
        if last_conv_layer_name is None:
            raise ValueError("No convolutional layer found in model.")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_tensor)
        # Use mean prediction across the spatial output as the scalar target
        target = tf.reduce_mean(preds[..., 0])

    grads = tape.gradient(target, conv_output)              # (1, h, w, C)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)

    conv_output_np = conv_output.numpy()[0]                 # (h, w, C)
    pooled_grads_np = pooled_grads.numpy()

    # Weight channels by gradient importance
    for i in range(pooled_grads_np.shape[-1]):
        conv_output_np[:, :, i] *= pooled_grads_np[i]

    heatmap = np.mean(conv_output_np, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-10)
    heatmap = 1.0 - heatmap   # invert: high activation → bright region

    return heatmap


def overlay_gradcam(
    img: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on a grayscale image.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image, shape (H, W) or (H, W, 1), values in [0, 1] or [0, 255].
    heatmap : np.ndarray
        Heatmap in [0, 1], any spatial size (will be resized).
    alpha : float
        Heatmap opacity.
    colormap : int
        OpenCV colormap constant.

    Returns
    -------
    np.ndarray
        RGB overlay image, dtype uint8.
    """
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    img = np.squeeze(img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)

    overlay = colored * alpha + img * (1.0 - alpha)
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────
#  Result grid
# ──────────────────────────────────────────────────────────────────────

def visualize_results(
    model: tf.keras.Model,
    images: np.ndarray,
    masks: np.ndarray,
    num_samples: int = 5,
    save_path: Optional[str] = None,
    seed: int = 42,
) -> None:
    """
    Plot a grid of (Original | Ground Truth | Prediction | Grad-CAM) rows.

    Parameters
    ----------
    model : tf.keras.Model
    images : np.ndarray   shape (N, H, W, 1)
    masks  : np.ndarray   shape (N, H, W, 1)
    num_samples : int
    save_path : str or None
        If provided, figure is saved to this path.
    seed : int
        For reproducible sample selection.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(images), size=min(num_samples, len(images)), replace=False)

    pred_masks = model.predict(images[indices], verbose=0)
    pred_binary = (pred_masks > 0.5).astype(np.float32)

    fig, axes = plt.subplots(len(indices), 4, figsize=(20, 5 * len(indices)))
    if len(indices) == 1:
        axes = [axes]

    col_titles = ["Original", "Ground Truth", "Prediction", "Grad-CAM"]
    for col, title in enumerate(col_titles):
        axes[0][col].set_title(title, fontsize=14, fontweight="bold")

    for row, idx in enumerate(indices):
        orig = np.squeeze(images[idx])

        # Original
        axes[row][0].imshow(orig, cmap="gray")
        axes[row][0].axis("off")

        # Ground truth overlay
        axes[row][1].imshow(orig, cmap="gray")
        axes[row][1].imshow(np.squeeze(masks[idx]), alpha=0.5, cmap="Reds")
        axes[row][1].axis("off")

        # Prediction overlay
        axes[row][2].imshow(orig, cmap="gray")
        axes[row][2].imshow(np.squeeze(pred_binary[row]), alpha=0.5, cmap="Greens")
        axes[row][2].axis("off")

        # Grad-CAM
        try:
            heatmap = make_gradcam_heatmap(images[idx:idx+1], model)
            overlay = overlay_gradcam(orig, heatmap)
            axes[row][3].imshow(overlay)
        except Exception as exc:
            axes[row][3].text(0.5, 0.5, f"Grad-CAM\nunavailable\n{exc}",
                              ha="center", va="center", transform=axes[row][3].transAxes)
        axes[row][3].axis("off")

    plt.tight_layout()

    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Results saved → {save_path}")

    plt.show()


# ──────────────────────────────────────────────────────────────────────
#  Training history
# ──────────────────────────────────────────────────────────────────────

def plot_training_history(
    history: tf.keras.callbacks.History,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot loss and metric curves for both train and validation sets.

    Parameters
    ----------
    history : tf.keras.callbacks.History
    save_path : str or None
    """
    metrics = ["loss", "accuracy", "iou_metric", "dice_coefficient"]
    titles  = ["Loss", "Accuracy", "IoU", "Dice Coefficient"]

    available = [(m, t) for m, t in zip(metrics, titles) if m in history.history]
    n = len(available)
    if n == 0:
        print("No metrics found in history.")
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (metric, title) in zip(axes, available):
        ax.plot(history.history[metric], label="Train")
        val_key = f"val_{metric}"
        if val_key in history.history:
            ax.plot(history.history[val_key], label="Validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"History plot saved → {save_path}")

    plt.show()
