from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from synthline_ai.generation.base import GenerationResult


def create_contact_sheet(
    results: list[GenerationResult],
    output_path: Path,
    max_samples: int = 16,
    thumb_size: int = 256,
) -> Path:
    """Create a grid contact sheet showing generated samples.

    Layout:
    - Take up to max_samples results
    - Create 2 rows: top row is generated images, bottom row is mask overlays
    - Actually simplest: Create a grid where each position is thumb_size x thumb_size.
    - Row arrangement: for N samples with C columns:
      - Row 0: images[0..C-1]
      - Row 1: mask overlays[0..C-1]
      - Row 2: images[C..2C-1]
      - Row 3: mask overlays[C..2C-1]

    Saves as JPEG at output_path. Returns output_path.
    """
    if not results:
        # Just save an empty image
        cv2.imwrite(str(output_path), np.zeros((thumb_size, thumb_size, 3), dtype=np.uint8))
        return output_path

    n_samples = min(len(results), max_samples)
    cols = min(n_samples, 4)
    n_pairs = math.ceil(n_samples / cols)
    rows = n_pairs * 2

    sheet = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)

    for i in range(n_samples):
        res = results[i]

        # Resize image and mask
        img = cv2.resize(res.image, (thumb_size, thumb_size))
        mask = cv2.resize(res.mask, (thumb_size, thumb_size))

        # Create overlay
        overlay = img.copy()
        # Red overlay where mask > 0
        mask_bool = mask > 0
        overlay[mask_bool] = overlay[mask_bool] * 0.5 + np.array([0, 0, 255]) * 0.5

        # Determine position
        pair_row = i // cols
        col = i % cols

        img_row = pair_row * 2
        mask_row = img_row + 1

        sheet[
            img_row * thumb_size : (img_row + 1) * thumb_size,
            col * thumb_size : (col + 1) * thumb_size,
        ] = img
        sheet[
            mask_row * thumb_size : (mask_row + 1) * thumb_size,
            col * thumb_size : (col + 1) * thumb_size,
        ] = overlay

    cv2.imwrite(str(output_path), sheet)
    return output_path
