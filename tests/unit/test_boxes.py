from __future__ import annotations

import numpy as np

from synthline_ai.labeling.boxes import bbox_area, mask_to_bbox


def test_mask_to_bbox() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[10:20, 30:50] = 255
    bbox = mask_to_bbox(mask)
    assert bbox == (30, 10, 20, 10)


def test_mask_to_bbox_empty() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    bbox = mask_to_bbox(mask)
    assert bbox == (0, 0, 0, 0)


def test_bbox_area() -> None:
    assert bbox_area((10, 20, 30, 40)) == 1200
