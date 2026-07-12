"""
data/dataset.py
───────────────
DentalDataset: loads, preprocesses, and splits panoramic X-ray images and masks.

Supported image extensions: .png, .jpg, .jpeg, .bmp, .tif, .tiff
Images are loaded as grayscale, resized to `image_size x image_size`,
and normalised to [0, 1]. Masks are binarised (>0 → 1).
"""

import pathlib
from typing import Tuple, List

import cv2
import natsort
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class DentalDataset:
    """
    Loads paired dental panoramic images and segmentation masks.

    Parameters
    ----------
    images_path : str
        Directory containing raw X-ray images.
    masks_path : str
        Directory containing corresponding binary masks.
    image_size : int
        Target spatial resolution (images are resized to image_size × image_size).
    """

    def __init__(self, images_path: str, masks_path: str, image_size: int = 224) -> None:
        self.images_path = pathlib.Path(images_path)
        self.masks_path = pathlib.Path(masks_path)
        self.image_size = image_size
        self._image_paths: List[pathlib.Path] = []
        self._mask_paths: List[pathlib.Path] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_paths(self) -> None:
        """Collect and naturally-sort all valid image/mask file paths."""
        self._image_paths = natsort.natsorted(
            [p for p in self.images_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
        )
        self._mask_paths = natsort.natsorted(
            [p for p in self.masks_path.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
        )
        if len(self._image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.images_path}")
        if len(self._mask_paths) == 0:
            raise FileNotFoundError(f"No masks found in {self.masks_path}")
        if len(self._image_paths) != len(self._mask_paths):
            raise ValueError(
                f"Image/mask count mismatch: "
                f"{len(self._image_paths)} images vs {len(self._mask_paths)} masks"
            )

    def _load_array(self, paths: List[pathlib.Path], is_mask: bool = False) -> np.ndarray:
        """Load a list of image paths into a float32 numpy array."""
        label = "masks" if is_mask else "images"
        arrays = []
        for p in tqdm(paths, desc=f"Loading {label}", unit="img"):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise IOError(f"Could not read file: {p}")
            img = cv2.resize(img, (self.image_size, self.image_size),
                             interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            if is_mask:
                img = (img > 0).astype(np.float32)
            arrays.append(img)
        return np.stack(arrays, axis=0)  # (N, H, W)

    @staticmethod
    def _add_channel(arr: np.ndarray) -> np.ndarray:
        """Add a trailing channel dimension if missing: (N,H,W) → (N,H,W,1)."""
        if arr.ndim == 3:
            return np.expand_dims(arr, axis=-1)
        return arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Load all data and return train / test splits.

        Parameters
        ----------
        test_ratio : float
            Fraction of data reserved for the test set.
        seed : int
            Random seed for reproducible splitting.

        Returns
        -------
        (train_images, train_masks), (test_images, test_masks)
            All arrays have shape (N, image_size, image_size, 1), dtype float32.
        """
        self._collect_paths()
        print(f"Found {len(self._image_paths)} images and {len(self._mask_paths)} masks.")

        images = self._add_channel(self._load_array(self._image_paths, is_mask=False))
        masks = self._add_channel(self._load_array(self._mask_paths, is_mask=True))

        train_images, test_images, train_masks, test_masks = train_test_split(
            images, masks, test_size=test_ratio, random_state=seed, shuffle=True
        )

        print(f"Split → train: {len(train_images)}  test: {len(test_images)}")
        return (train_images, train_masks), (test_images, test_masks)
