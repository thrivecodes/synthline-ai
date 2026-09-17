"""Module for loading seed images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from synthline_ai.config.defaults import SUPPORTED_EXTENSIONS
from synthline_ai.config.models import ImageInfo, QualityWarning


def load_seeds(
    seeds_dir: Path,
) -> tuple[list[ImageInfo], list[np.ndarray], list[QualityWarning]]:
    """
    Load seed images from a directory and compute their basic metadata.

    Args:
        seeds_dir: Directory containing seed images.

    Returns:
        A tuple of (images, arrays, warnings).

    Raises:
        FileNotFoundError: If the seeds directory does not exist.
        ValueError: If the seeds directory is empty.
    """
    if not seeds_dir.exists() or not seeds_dir.is_dir():
        raise FileNotFoundError(f"Seeds directory not found: {seeds_dir}")

    files = sorted(
        [f for f in seeds_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    )

    if not files:
        raise ValueError(f"No valid seed images found in {seeds_dir}")

    images: list[ImageInfo] = []
    arrays: list[np.ndarray] = []
    warnings: list[QualityWarning] = []

    for path in files:
        # Load image
        img = cv2.imread(str(path))
        if img is None:
            warnings.append(
                QualityWarning(
                    path=path,
                    code="corrupt",
                    message="Failed to read image file.",
                    severity="error",
                )
            )
            continue

        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) == 3 else 1

        # Mean brightness
        mean_brightness = float(np.mean(img))

        # Blur score (variance of Laplacian of grayscale version)
        if len(img.shape) == 3 and channels == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        images.append(
            ImageInfo(
                path=path,
                width=width,
                height=height,
                channels=channels,
                mean_brightness=mean_brightness,
                blur_score=blur_score,
            )
        )
        arrays.append(img)

    return images, arrays, warnings
