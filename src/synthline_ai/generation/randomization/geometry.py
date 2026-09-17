"""Micro-geometric transformations preserving mask-to-image alignment."""

from __future__ import annotations

import cv2
import numpy as np


def apply_geometry_variation(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.RandomState,
    max_rotation_deg: float = 3.0,
    max_shift_px: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply slight rotation and sub-pixel translation to image and defect mask.

    Both image and mask undergo identical affine transformations to guarantee
    exact label registration without drift.

    Args:
        image: Source image array (H, W, C) or (H, W) in uint8.
        mask: Defect binary mask (H, W) in uint8 (0 and 255).
        rng: Reproducible numpy RandomState.
        max_rotation_deg: Maximum rotation angle in degrees (+/-).
        max_shift_px: Maximum translation shift in pixels (+/-).

    Returns:
        (transformed_image, transformed_mask) tuple.
    """
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    angle = rng.uniform(-max_rotation_deg, max_rotation_deg)
    shift_x = rng.uniform(-max_shift_px, max_shift_px)
    shift_y = rng.uniform(-max_shift_px, max_shift_px)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    matrix[0, 2] += shift_x
    matrix[1, 2] += shift_y

    # Transform image (reflect border to prevent black boundary artifacts)
    transformed_image = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    # Transform mask (constant zero border so defects don't wrap from outside)
    transformed_mask = cv2.warpAffine(
        mask,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # Ensure mask remains strictly binary {0, 255}
    transformed_mask = np.where(transformed_mask > 127, np.uint8(255), np.uint8(0))

    return transformed_image, transformed_mask
