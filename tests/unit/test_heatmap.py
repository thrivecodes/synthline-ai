"""Unit tests for spatial defect heatmap generation and coverage metrics."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from synthline_ai.generation.base import GenerationResult
from synthline_ai.validation.heatmap import (
    compute_spatial_metrics,
    generate_defect_heatmap,
)


def test_compute_spatial_metrics_empty() -> None:
    acc = np.zeros((64, 64), dtype=np.float32)
    metrics = compute_spatial_metrics(acc)
    assert metrics["coverage_ratio"] == 0.0
    assert metrics["spatial_entropy"] == 0.0
    assert metrics["centroid_x"] == 0.5
    assert metrics["centroid_y"] == 0.5


def test_compute_spatial_metrics_clustered() -> None:
    acc = np.zeros((100, 100), dtype=np.float32)
    # Put all defects in top-left corner (0:20, 0:20)
    acc[0:20, 0:20] = 10.0

    metrics = compute_spatial_metrics(acc)
    assert metrics["coverage_ratio"] == 0.04
    # Centroid should be in top-left
    assert metrics["centroid_x"] < 0.2
    assert metrics["centroid_y"] < 0.2
    # Clustered defects have lower spatial entropy than uniform
    assert metrics["spatial_entropy"] < 0.5


def test_compute_spatial_metrics_uniform() -> None:
    acc = np.full((100, 100), 5.0, dtype=np.float32)
    metrics = compute_spatial_metrics(acc)
    assert metrics["coverage_ratio"] == 1.0
    # Centroid at center
    assert 0.45 <= metrics["centroid_x"] <= 0.55
    assert 0.45 <= metrics["centroid_y"] <= 0.55
    # Maximum spatial dispersion
    assert metrics["spatial_entropy"] > 0.95


def test_generate_defect_heatmap_output(tmp_path: Path) -> None:
    results = []
    for i in range(4):
        img = np.full((80, 80, 3), 120, dtype=np.uint8)
        mask = np.zeros((80, 80), dtype=np.uint8)
        # Shift defect position per sample
        x1, y1 = 10 + i * 15, 10 + i * 15
        mask[y1 : y1 + 15, x1 : x1 + 15] = 255

        res = GenerationResult(
            image=img,
            mask=mask,
            metadata={},
            source_seed="seed.png",
            defect_type="scratch",
            random_seed=i,
        )
        results.append(res)

    heatmap_path = tmp_path / "heatmap.png"
    out_path, metrics = generate_defect_heatmap(results, heatmap_path)

    assert out_path.exists()
    saved = cv2.imread(str(out_path))
    assert saved is not None
    assert saved.shape == (80, 80, 3)

    assert metrics["coverage_ratio"] > 0.0
    assert 0.0 <= metrics["spatial_entropy"] <= 1.0


def test_generate_defect_heatmap_empty_results(tmp_path: Path) -> None:
    heatmap_path = tmp_path / "empty_heat.png"
    out_path, metrics = generate_defect_heatmap([], heatmap_path)

    assert out_path.exists()
    assert metrics["coverage_ratio"] == 0.0
