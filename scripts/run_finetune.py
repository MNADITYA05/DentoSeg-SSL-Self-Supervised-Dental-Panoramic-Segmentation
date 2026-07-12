"""
scripts/run_finetune.py
────────────────────────
CLI entry point for supervised segmentation fine-tuning.

Usage (local):
    python scripts/run_finetune.py --config configs/finetune.yaml

Usage (override paths):
    python scripts/run_finetune.py \
        --config configs/finetune.yaml \
        --images /path/to/images \
        --masks  /path/to/masks \
        --encoder outputs/pretrain/encoder_pretrained.keras

The script loads the pretrained encoder, assembles the U-Net, trains it,
and saves evaluation plots and the final model to the output directory.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tensorflow as tf

from data.augmentation import RandomContrast  # needed for custom object registration
from data.dataset import DentalDataset
from evaluation.metrics import iou_metric, dice_coefficient
from evaluation.visualize import visualize_results, plot_training_history
from models.encoder import create_encoder
from models.segmentation import create_segmentation_model
from training.finetune import train_segmentation
from training.losses import dice_loss
from utils.config import load_config
from utils.env import detect_environment, get_default_paths
from utils.seed import set_global_seed
from utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DentoSeg-SSL Segmentation Fine-tuning")
    parser.add_argument("--config", default="configs/finetune.yaml",
                        help="Path to finetune YAML config")
    parser.add_argument("--images", default=None)
    parser.add_argument("--masks", default=None)
    parser.add_argument("--encoder", default=None,
                        help="Override encoder checkpoint path from config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    log = get_logger("finetune")

    seed = cfg["training"].get("seed", 42)
    set_global_seed(seed)

    # ── Paths ─────────────────────────────────────────────────────────
    env = detect_environment()
    defaults = get_default_paths(env)
    log.info(f"Environment: {env}")

    images_path   = args.images  or cfg["data"].get("images_path") or defaults["images"]
    masks_path    = args.masks   or cfg["data"].get("masks_path")  or defaults["masks"]
    encoder_ckpt  = args.encoder or cfg["model"].get("encoder_checkpoint")

    log.info(f"Images:  {images_path}")
    log.info(f"Masks:   {masks_path}")
    log.info(f"Encoder: {encoder_ckpt}")

    # ── Data ──────────────────────────────────────────────────────────
    image_size = cfg["data"].get("image_size", 224)
    dataset = DentalDataset(images_path, masks_path, image_size=image_size)
    (train_images, train_masks), (test_images, test_masks) = dataset.prepare_data(
        test_ratio=cfg["data"].get("test_ratio", 0.2),
        seed=seed,
    )

    # ── Encoder ───────────────────────────────────────────────────────
    input_shape = (image_size, image_size, cfg["data"].get("channels", 1))

    if encoder_ckpt and pathlib.Path(encoder_ckpt).exists():
        log.info(f"Loading pretrained encoder from {encoder_ckpt} …")
        encoder = tf.keras.models.load_model(
            encoder_ckpt,
            custom_objects={"RandomContrast": RandomContrast},
        )
    else:
        log.warning("No pretrained encoder found — initialising from scratch.")
        encoder = create_encoder(input_shape=input_shape)

    # ── Segmentation model ────────────────────────────────────────────
    seg_model = create_segmentation_model(encoder=encoder, input_shape=input_shape)
    log.info("Segmentation model summary:")
    seg_model.summary(print_fn=log.info)

    # ── Fine-tune ─────────────────────────────────────────────────────
    out = cfg["output"]
    history = train_segmentation(
        model=seg_model,
        encoder=encoder,
        train_images=train_images,
        train_masks=train_masks,
        val_ratio=cfg["data"].get("val_ratio", 0.2),
        batch_size=cfg["training"]["batch_size"],
        epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        freeze_encoder_epochs=cfg["model"].get("freeze_encoder_epochs", 10),
        early_stopping_patience=cfg["training"].get("early_stopping_patience", 10),
        reduce_lr_patience=cfg["training"].get("reduce_lr_patience", 5),
        reduce_lr_factor=cfg["training"].get("reduce_lr_factor", 0.5),
        best_model_path=out.get("best_model", "outputs/finetune/best_model.keras"),
        final_model_path=out.get("final_model", "outputs/finetune/dental_segmentation_model.keras"),
        log_csv_path=out.get("log_csv"),
        seed=seed,
    )

    # ── Evaluate ──────────────────────────────────────────────────────
    log.info("Evaluating on test set …")
    results = seg_model.evaluate(test_images, test_masks, verbose=1)
    metric_names = ["Loss", "Accuracy", "Precision", "Recall", "IoU", "Dice"]
    for name, val in zip(metric_names, results):
        log.info(f"Test {name}: {val:.4f}")

    # ── Visualise ─────────────────────────────────────────────────────
    if history:
        plot_training_history(
            history,
            save_path=out.get("history_plot", "outputs/finetune/training_history.png"),
        )

    visualize_results(
        seg_model, test_images, test_masks,
        num_samples=5,
        save_path=out.get("results_plot", "outputs/finetune/segmentation_results.png"),
        seed=seed,
    )

    log.info("Fine-tuning complete.")


if __name__ == "__main__":
    main()
