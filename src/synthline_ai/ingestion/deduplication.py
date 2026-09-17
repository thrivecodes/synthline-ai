"""Perceptual hashing and duplicate detection for seed images."""

from __future__ import annotations

import cv2
import numpy as np

from synthline_ai.config.models import ImageInfo, QualityWarning


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """Compute 64-bit difference hash (dHash) for an image.

    Converts image to grayscale, resizes to (hash_size + 1, hash_size),
    compares horizontally adjacent pixels, and serializes as a 64-bit int.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Resize: width = hash_size + 1, height = hash_size
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)

    # Compute difference between adjacent horizontal pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # Convert boolean array to 64-bit unsigned integer
    bit_str = "".join("1" if b else "0" for b in diff.flatten())
    return int(bit_str, 2)


def hamming_distance(hash1: int, hash2: int) -> int:
    """Compute Hamming distance between two 64-bit integer hashes."""
    return (hash1 ^ hash2).bit_count()


def find_duplicates(
    arrays: list[np.ndarray],
    threshold: int = 4,
) -> list[tuple[int, int, int]]:
    """Find pairs of images that are identical or near-duplicates.

    Args:
        arrays: List of image numpy arrays.
        threshold: Maximum Hamming distance to consider a pair as duplicate
            (0 = identical visual structure, <=4 = near-duplicate).

    Returns:
        List of (idx_a, idx_b, distance) tuples.
    """
    hashes = [compute_dhash(img) for img in arrays]
    duplicates: list[tuple[int, int, int]] = []

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dist = hamming_distance(hashes[i], hashes[j])
            if dist <= threshold:
                duplicates.append((i, j, dist))

    return duplicates


def check_seed_duplicates(
    images: list[ImageInfo],
    arrays: list[np.ndarray],
    threshold: int = 4,
) -> list[QualityWarning]:
    """Inspect seed set for duplicates and return QualityWarning instances."""
    if len(arrays) < 2:
        return []

    duplicates = find_duplicates(arrays, threshold=threshold)
    warnings: list[QualityWarning] = []

    for idx_a, idx_b, dist in duplicates:
        info_a = images[idx_a]
        info_b = images[idx_b]
        dup_type = "Identical duplicate" if dist == 0 else f"Near-duplicate (distance={dist})"
        warnings.append(
            QualityWarning(
                path=info_b.path,
                code="near_duplicate" if dist > 0 else "duplicate",
                severity="warning",
                message=f"{dup_type} of '{info_a.path.name}' detected in seed set.",
            )
        )

    return warnings
