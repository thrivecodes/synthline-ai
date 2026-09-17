from __future__ import annotations

import cv2
import numpy as np


def validate_mask(mask: np.ndarray) -> bool:
    """Check mask is uint8, single-channel (2D), contains only 0 and 255."""
    if mask.dtype != np.uint8:
        return False
    if mask.ndim != 2:
        return False
    unique_vals = np.unique(mask)
    return all(val in (0, 255) for val in unique_vals)



def mask_area(mask: np.ndarray) -> int:
    """Count non-zero pixels in the mask."""
    return int(np.count_nonzero(mask))


def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """Convert binary mask to polygon contours for COCO segmentation.

    Returns list of polygons, each polygon is a flat list of [x1,y1,x2,y2,...] coordinates.
    Uses cv2.findContours with RETR_EXTERNAL and CHAIN_APPROX_SIMPLE.
    Filters out contours with fewer than 3 points.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.shape[0] >= 3:
            # Flatten to [x1, y1, x2, y2, ...]
            poly = contour.flatten().tolist()
            polygons.append(poly)
    return polygons
