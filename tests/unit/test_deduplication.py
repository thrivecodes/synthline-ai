"""Unit tests for perceptual hashing and seed deduplication."""

from pathlib import Path

import numpy as np

from synthline_ai.config.models import ImageInfo
from synthline_ai.ingestion.deduplication import (
    check_seed_duplicates,
    compute_dhash,
    find_duplicates,
    hamming_distance,
)
from synthline_ai.ingestion.quality import check_seed_quality


def test_dhash_and_hamming_distance() -> None:
    img1 = np.full((64, 64, 3), 100, dtype=np.uint8)
    img1[:, :32] = 200  # Left bright, right dark

    img2 = img1.copy()

    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    assert h1 == h2
    assert hamming_distance(h1, h2) == 0

    # Inverted image
    img_inv = 255 - img1
    h_inv = compute_dhash(img_inv)
    assert hamming_distance(h1, h_inv) >= 8


def test_find_duplicates() -> None:
    base = np.random.RandomState(42).randint(50, 200, (64, 64, 3), dtype=np.uint8)
    exact = base.copy()
    diff = np.random.RandomState(99).randint(50, 200, (64, 64, 3), dtype=np.uint8)

    duplicates = find_duplicates([base, exact, diff], threshold=4)
    assert len(duplicates) == 1
    idx_a, idx_b, dist = duplicates[0]
    assert (idx_a, idx_b) == (0, 1)
    assert dist == 0


def test_check_seed_duplicates_warnings() -> None:
    img_a = np.full((64, 64, 3), 120, dtype=np.uint8)
    img_b = img_a.copy()

    infos = [
        ImageInfo(
            path=Path("sample_a.png"),
            width=64,
            height=64,
            channels=3,
            mean_brightness=120.0,
            blur_score=100.0,
        ),
        ImageInfo(
            path=Path("sample_b.png"),
            width=64,
            height=64,
            channels=3,
            mean_brightness=120.0,
            blur_score=100.0,
        ),
    ]

    warnings = check_seed_duplicates(infos, [img_a, img_b], threshold=2)
    assert len(warnings) == 1
    assert warnings[0].code == "duplicate"
    assert "sample_a.png" in warnings[0].message

    # Test integration with check_seed_quality
    report = check_seed_quality(infos, [img_a, img_b])
    assert any(w.code == "duplicate" for w in report.warnings)
