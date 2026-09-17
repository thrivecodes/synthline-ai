from __future__ import annotations

import cv2
import numpy as np


def mask_to_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (x, y, width, height) in COCO format from a binary mask.

    Uses cv2.boundingRect on non-zero pixels.
    Returns (0, 0, 0, 0) if mask is empty.
    """
    if np.count_nonzero(mask) == 0:
        return (0, 0, 0, 0)
    x, y, w, h = cv2.boundingRect(mask)
    return (int(x), int(y), int(w), int(h))


def mask_to_yolo_bbox(
    mask: np.ndarray,
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    """Convert binary mask to normalized YOLO bbox: (x_center, y_center, width, height).

    Coordinates are normalized to [0.0, 1.0].
    Returns (0.0, 0.0, 0.0, 0.0) if mask is empty or dimensions are invalid.
    """
    if np.count_nonzero(mask) == 0 or img_width <= 0 or img_height <= 0:
        return (0.0, 0.0, 0.0, 0.0)

    x, y, w, h = cv2.boundingRect(mask)
    x_center = (x + w / 2.0) / img_width
    y_center = (y + h / 2.0) / img_height
    norm_w = w / float(img_width)
    norm_h = h / float(img_height)

    return (
        float(np.clip(x_center, 0.0, 1.0)),
        float(np.clip(y_center, 0.0, 1.0)),
        float(np.clip(norm_w, 0.0, 1.0)),
        float(np.clip(norm_h, 0.0, 1.0)),
    )


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    """Return width * height."""
    _, _, w, h = bbox
    return int(w * h)
