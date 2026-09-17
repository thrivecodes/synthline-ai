"""Module for checking seed image quality."""

from __future__ import annotations

import statistics

import numpy as np

from synthline_ai.config.defaults import (
    BLUR_THRESHOLD,
    BRIGHTNESS_HIGH,
    BRIGHTNESS_LOW,
    MAX_DIMENSION,
    MIN_DIMENSION,
)
from synthline_ai.config.models import ImageInfo, QualityWarning, SeedSetReport
from synthline_ai.ingestion.deduplication import check_seed_duplicates


def check_seed_quality(images: list[ImageInfo], arrays: list[np.ndarray]) -> SeedSetReport:
    """
    Check the quality of seed images.

    Args:
        images: Metadata about the seed images.
        arrays: Raw image arrays (currently unused, but kept for future).

    Returns:
        A report summarizing the seed image quality.
    """
    warnings: list[QualityWarning] = []

    if not images:
        return SeedSetReport(
            images=[],
            warnings=[],
            valid_count=0,
            rejected_count=0,
        )

    median_width = statistics.median([img.width for img in images])
    median_height = statistics.median([img.height for img in images])

    for img in images:
        if img.mean_brightness < BRIGHTNESS_LOW:
            warnings.append(
                QualityWarning(
                    path=img.path,
                    code="too_dark",
                    message=f"Image is too dark (brightness: {img.mean_brightness:.2f}).",
                    severity="warning",
                )
            )

        if img.mean_brightness > BRIGHTNESS_HIGH:
            warnings.append(
                QualityWarning(
                    path=img.path,
                    code="too_bright",
                    message=f"Image is too bright (brightness: {img.mean_brightness:.2f}).",
                    severity="warning",
                )
            )

        if img.blur_score < BLUR_THRESHOLD:
            warnings.append(
                QualityWarning(
                    path=img.path,
                    code="blurry",
                    message=f"Image is blurry (blur score: {img.blur_score:.2f}).",
                    severity="warning",
                )
            )

        if img.width < MIN_DIMENSION or img.height < MIN_DIMENSION:
            warnings.append(
                QualityWarning(
                    path=img.path,
                    code="too_small",
                    message=f"Image is too small ({img.width}x{img.height}).",
                    severity="warning",
                )
            )

        if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
            warnings.append(
                QualityWarning(
                    path=img.path,
                    code="too_large",
                    message=f"Image is too large ({img.width}x{img.height}).",
                    severity="warning",
                )
            )

        width_diff = abs(img.width - median_width) / median_width
        height_diff = abs(img.height - median_height) / median_height

        if width_diff > 0.5 or height_diff > 0.5:
            warnings.append(
                QualityWarning(
                    path=img.path,
                    code="odd_dimensions",
                    message=(
                        f"Image dimensions ({img.width}x{img.height}) "
                        "differ significantly from median."
                    ),
                    severity="warning",
                )
            )

    # Check for identical and near-duplicate seeds
    if arrays and len(arrays) == len(images):
        warnings.extend(check_seed_duplicates(images, arrays))

    return SeedSetReport(
        images=images,
        warnings=warnings,
        valid_count=len(images),
        rejected_count=0,
    )
