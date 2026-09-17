from __future__ import annotations

import numpy as np

from synthline_ai.labeling.masks import mask_area, mask_to_polygon, validate_mask


def test_validate_valid_mask() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    assert validate_mask(mask) is True


def test_validate_invalid_values() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[0, 0] = 128
    assert validate_mask(mask) is False


def test_validate_wrong_dtype() -> None:
    mask = np.zeros((64, 64), dtype=np.float32)
    assert validate_mask(mask) is False


def test_mask_area() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[0:10, 0:10] = 255
    assert mask_area(mask) == 100


def test_mask_area_empty() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    assert mask_area(mask) == 0


def test_mask_to_polygon() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    polygons = mask_to_polygon(mask)
    assert len(polygons) == 1
    poly = polygons[0]
    assert len(poly) >= 6
    assert all(isinstance(x, (int, float)) for x in poly)
