"""Spatial defect density heatmap and coverage analysis."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from synthline_ai.generation.base import GenerationResult


def compute_spatial_metrics(
    accumulator: np.ndarray,
) -> dict[str, float]:
    """Compute quantitative spatial dispersion and centroid metrics.

    Args:
        accumulator: 2D float array containing defect pixel counts.

    Returns:
        Dict containing coverage_ratio, centroid_x, centroid_y, and spatial_entropy.
    """
    h, w = accumulator.shape[:2]
    total_pixels = h * w
    nonzero_count = int(np.count_nonzero(accumulator))
    coverage_ratio = float(nonzero_count / total_pixels) if total_pixels > 0 else 0.0

    if nonzero_count == 0:
        return {
            "coverage_ratio": 0.0,
            "centroid_x": 0.5,
            "centroid_y": 0.5,
            "spatial_entropy": 0.0,
        }

    # Normalized weighted centroid
    y_indices, x_indices = np.where(accumulator > 0)
    weights = accumulator[y_indices, x_indices]
    weight_sum = float(np.sum(weights))

    centroid_x = float(np.sum(x_indices * weights) / (weight_sum * w))
    centroid_y = float(np.sum(y_indices * weights) / (weight_sum * h))

    # Spatial entropy across an 8x8 cell grid
    grid_rows, grid_cols = 8, 8
    cell_h = max(1, h // grid_rows)
    cell_w = max(1, w // grid_cols)
    cell_sums: list[float] = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            sub = accumulator[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            cell_sums.append(float(np.sum(sub)))

    total_cell_weight = sum(cell_sums)
    if total_cell_weight > 0:
        probs = [cs / total_cell_weight for cs in cell_sums if cs > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(grid_rows * grid_cols)
        norm_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    else:
        norm_entropy = 0.0

    return {
        "coverage_ratio": round(coverage_ratio, 4),
        "centroid_x": round(centroid_x, 4),
        "centroid_y": round(centroid_y, 4),
        "spatial_entropy": round(norm_entropy, 4),
    }


def generate_defect_heatmap(
    results: list[GenerationResult],
    output_path: Path,
    colormap: int = cv2.COLORMAP_TURBO,
) -> tuple[Path, dict[str, float]]:
    """Accumulate defect masks and render a spatial density heatmap.

    Args:
        results: List of GenerationResult objects.
        output_path: Destination path for rendered heatmap (e.g. heatmap.png).
        colormap: OpenCV colormap enum (default COLORMAP_TURBO).

    Returns:
        Tuple of (output_path, metrics_dict).
    """
    if not results:
        # Fallback empty 256x256 image
        blank = np.zeros((256, 256, 3), dtype=np.uint8)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), blank)
        return output_path, {
            "coverage_ratio": 0.0,
            "centroid_x": 0.5,
            "centroid_y": 0.5,
            "spatial_entropy": 0.0,
        }

    # Determine canvas dimensions from first sample
    first_h, first_w = results[0].image.shape[:2]
    accumulator = np.zeros((first_h, first_w), dtype=np.float32)
    image_acc = np.zeros((first_h, first_w, 3), dtype=np.float32)

    for res in results:
        m = res.mask
        if m.shape[:2] != (first_h, first_w):
            m = cv2.resize(m, (first_w, first_h), interpolation=cv2.INTER_NEAREST)
        accumulator += (m > 0).astype(np.float32)

        img = res.image
        if img.shape[:2] != (first_h, first_w):
            img = cv2.resize(img, (first_w, first_h), interpolation=cv2.INTER_AREA)
        image_acc += img.astype(np.float32)

    avg_image = (image_acc / len(results)).astype(np.uint8)
    metrics = compute_spatial_metrics(accumulator)

    max_val = float(np.max(accumulator))
    if max_val > 0:
        norm = (accumulator / max_val * 255.0).astype(np.uint8)
        colorized = cv2.applyColorMap(norm, colormap)

        # Blend where defects occurred; keep base image elsewhere
        mask_any = accumulator > 0
        composite = avg_image.copy()
        blended = cv2.addWeighted(avg_image, 0.35, colorized, 0.65, 0)
        composite[mask_any] = blended[mask_any]
    else:
        composite = avg_image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), composite)

    return output_path, metrics
