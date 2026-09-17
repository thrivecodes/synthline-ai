"""Unit tests for ROI extraction and workpiece boundary constraint."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from synthline_ai.generation.roi import apply_roi_constraint, compute_roi_mask


def test_compute_roi_mask_dark_background() -> None:
    # Black background with a bright central workpiece disk
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(img, (60, 60), 40, (200, 200, 200), -1)

    roi = compute_roi_mask(img)
    assert roi.shape == (120, 120)
    assert roi.dtype == np.uint8
    # Center should be within ROI
    assert roi[60, 60] == 255
    # Corner should be outside ROI (background)
    assert roi[5, 5] == 0


def test_compute_roi_mask_bright_background() -> None:
    # White background with a dark metallic part in center
    img = np.full((120, 120, 3), 240, dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (90, 90), (60, 60, 60), -1)

    roi = compute_roi_mask(img)
    assert roi[60, 60] == 255
    assert roi[5, 5] == 0


def test_compute_roi_mask_uniform_texture() -> None:
    # Homogeneous surface, entire image is valid workpiece
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    roi = compute_roi_mask(img)
    assert np.all(roi == 255)


def test_apply_roi_constraint_restores_background() -> None:
    clean = np.full((100, 100, 3), 100, dtype=np.uint8)
    defective = clean.copy()
    # Apply a large defect covering the whole image
    defective[:, :] = 255
    mask = np.full((100, 100), 255, dtype=np.uint8)

    # ROI is only a small central block (rows 30:70, cols 30:70)
    roi = np.zeros((100, 100), dtype=np.uint8)
    roi[30:70, 30:70] = 255

    res_img, res_mask = apply_roi_constraint(defective, mask, clean, roi)

    # Inside ROI: defect retained
    assert np.all(res_img[35:65, 35:65] == 255)
    assert np.all(res_mask[35:65, 35:65] == 255)

    # Outside ROI: original clean pixels restored, mask zeroed out
    assert np.all(res_img[5, 5] == 100)
    assert res_mask[5, 5] == 0
    assert np.all(res_mask[0:20, :] == 0)


def test_pipeline_with_auto_roi(tmp_path: Path) -> None:
    from synthline_ai.config.models import DefectType, GenerationConfig, ImageInfo
    from synthline_ai.generation.pipeline import run_generation

    # Create an image with a central workpiece (disk) and black background
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    cv2.circle(img, (60, 60), 40, (200, 200, 200), -1)

    info = ImageInfo(
        path=tmp_path / "part.png",
        width=120,
        height=120,
        channels=3,
        mean_brightness=100.0,
        blur_score=500.0,
    )
    config = GenerationConfig(
        seeds_dir=tmp_path,
        defect_type=DefectType.SCRATCH,
        count=3,
        output_dir=tmp_path / "out",
        auto_roi=True,
    )
    results = run_generation(config, [img], [info])
    assert len(results) == 3
    for res in results:
        assert res.metadata.get("auto_roi") is True
        assert res.mask[5, 5] == 0
        assert np.array_equal(res.image[5, 5], img[5, 5])
