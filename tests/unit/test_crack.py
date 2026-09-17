"""Unit tests for procedural crack and fracture defect generator."""

from __future__ import annotations

import numpy as np

from synthline_ai.generation.procedural.crack import CrackGenerator


def test_crack_generates_valid_output() -> None:
    generator = CrackGenerator()
    assert generator.defect_type == "crack"

    img = np.full((128, 128, 3), 160, dtype=np.uint8)
    result = generator.generate(
        image=img,
        seed_name="seed_test.png",
        random_seed=42,
        severity=0.6,
        frequency=1.0,
    )

    assert result.image.shape == img.shape
    assert result.image.dtype == np.uint8
    assert result.mask.shape == (128, 128)
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)).issubset({0, 255})
    assert (result.mask > 0).any()
    assert result.defect_type == "crack"
    assert "total_branches" in result.metadata
    assert "total_length_pixels" in result.metadata


def test_crack_deterministic() -> None:
    generator = CrackGenerator()
    img = np.full((64, 64, 3), 140, dtype=np.uint8)

    res1 = generator.generate(img, "s.png", 100, severity=0.5)
    res2 = generator.generate(img, "s.png", 100, severity=0.5)

    np.testing.assert_array_equal(res1.image, res2.image)
    np.testing.assert_array_equal(res1.mask, res2.mask)


def test_crack_different_seeds_produce_different_outputs() -> None:
    generator = CrackGenerator()
    img = np.full((64, 64, 3), 140, dtype=np.uint8)

    res1 = generator.generate(img, "s.png", 101, severity=0.5)
    res2 = generator.generate(img, "s.png", 202, severity=0.5)

    assert not np.array_equal(res1.mask, res2.mask)


def test_crack_severity_range() -> None:
    generator = CrackGenerator()
    img = np.full((64, 64, 3), 150, dtype=np.uint8)

    low = generator.generate(img, "s.png", 42, severity=0.0)
    high = generator.generate(img, "s.png", 42, severity=1.0)

    assert (low.mask > 0).any()
    assert (high.mask > 0).any()
    assert np.sum(high.mask > 0) >= np.sum(low.mask > 0)
