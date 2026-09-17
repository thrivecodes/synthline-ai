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


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    """Return width * height."""
    _, _, w, h = bbox
    return int(w * h)
