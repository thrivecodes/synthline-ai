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
    extra_masks: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply slight rotation and sub-pixel translation to image and defect mask.

    Both image and mask undergo identical affine transformations to guarantee
    exact label registration without drift.
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

    if extra_masks is not None:
        for idx, em in enumerate(extra_masks):
            warped = cv2.warpAffine(
                em,
                matrix,
                (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            extra_masks[idx][:] = np.where(warped > 127, np.uint8(255), np.uint8(0))

    return transformed_image, transformed_mask
