"""
scripts/run_pretrain.py
────────────────────────
CLI entry point for contrastive SSL pretraining.

Usage (local):
    python scripts/run_pretrain.py --config configs/pretrain.yaml

Usage (override paths at runtime):
    python scripts/run_pretrain.py \
        --config configs/pretrain.yaml \
        --images /path/to/images \
        --masks  /path/to/masks

The encoder weights are saved to the path specified in the config
(output.checkpoint) and can be loaded by run_finetune.py.
"""

import argparse
import pathlib
import sys

# Ensure project root is on the path when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import tensorflow as tf

from data.dataset import DentalDataset
from models.encoder import create_encoder
from models.segmentation import create_contrastive_model
from training.pretrain import ContrastiveTrainer
from utils.config import load_config
from utils.env import detect_environment, get_default_paths
from utils.seed import set_global_seed
from utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DentoSeg-SSL Contrastive Pretraining")
    parser.add_argument("--config", default="configs/pretrain.yaml",
                        help="Path to pretrain YAML config")
    parser.add_argument("--images", default=None,
                        help="Override images path from config")
    parser.add_argument("--masks", default=None,
                        help="Override masks path from config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    log = get_logger("pretrain")

    # ── Seed ────────────────────────────────────────────────────────────
    seed = cfg["training"].get("seed", 42)
    set_global_seed(seed)

    # ── Paths ────────────────────────────────────────────────────────────
    env = detect_environment()
    defaults = get_default_paths(env)
    log.info(f"Environment: {env}")

    images_path = args.images or cfg["data"].get("images_path") or defaults["images"]
    masks_path  = args.masks  or cfg["data"].get("masks_path")  or defaults["masks"]
    log.info(f"Images: {images_path}")
    log.info(f"Masks:  {masks_path}")

    # ── Data ─────────────────────────────────────────────────────────────
    image_size = cfg["data"].get("image_size", 224)
    dataset = DentalDataset(images_path, masks_path, image_size=image_size)
    (train_images, _), _ = dataset.prepare_data(
        test_ratio=cfg["data"].get("test_ratio", 0.2),
        seed=seed,
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(len(train_images), seed=seed)
        .batch(cfg["training"]["batch_size"])
        .prefetch(tf.data.AUTOTUNE)
    )

    # ── Models ───────────────────────────────────────────────────────────
    input_shape = (image_size, image_size, cfg["data"].get("channels", 1))
    encoder = create_encoder(
        input_shape=input_shape,
        base_filters=cfg["model"].get("encoder_base_filters", 64),
    )
    contrastive_model = create_contrastive_model(
        encoder=encoder,
        input_shape=input_shape,
        projection_dim=cfg["model"].get("projection_dim", 128),
        augmentation_cfg=cfg.get("augmentation", {}),
    )

    log.info("Contrastive model summary:")
    contrastive_model.summary(print_fn=log.info)

    # ── Train ────────────────────────────────────────────────────────────
    trainer = ContrastiveTrainer(
        model=contrastive_model,
        encoder=encoder,
        temperature=cfg["training"].get("temperature", 0.1),
        learning_rate=cfg["training"].get("learning_rate", 3e-4),
        log_path=cfg["output"].get("log_csv"),
    )

    trainer.train(
        dataset=train_ds,
        epochs=cfg["training"]["epochs"],
        encoder_checkpoint=cfg["output"].get("checkpoint"),
    )

    log.info("Pretraining complete.")


if __name__ == "__main__":
    main()
