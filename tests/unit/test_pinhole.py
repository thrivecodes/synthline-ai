"""Unit tests for procedural pinhole and pitting defect generator."""

from __future__ import annotations

import numpy as np

from synthline_ai.generation.procedural.pinhole import PinholeGenerator


def test_pinhole_generates_valid_output() -> None:
    generator = PinholeGenerator()
    assert generator.defect_type == "pinhole"

    img = np.full((128, 128, 3), 170, dtype=np.uint8)
    result = generator.generate(
        image=img,
        seed_name="seed_test.png",
        random_seed=42,
        severity=0.5,
        frequency=1.5,
    )

    assert result.image.shape == img.shape
    assert result.image.dtype == np.uint8
    assert result.mask.shape == (128, 128)
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask)).issubset({0, 255})
    assert (result.mask > 0).any()
    assert result.defect_type == "pinhole"
    assert "num_pinholes" in result.metadata
    assert "avg_radius_pixels" in result.metadata


def test_pinhole_deterministic() -> None:
    generator = PinholeGenerator()
    img = np.full((64, 64, 3), 130, dtype=np.uint8)

    res1 = generator.generate(img, "s.png", 55, severity=0.4)
    res2 = generator.generate(img, "s.png", 55, severity=0.4)

    np.testing.assert_array_equal(res1.image, res2.image)
    np.testing.assert_array_equal(res1.mask, res2.mask)


def test_pinhole_frequency_scaling() -> None:
    generator = PinholeGenerator()
    img = np.full((128, 128, 3), 130, dtype=np.uint8)

    low_freq = generator.generate(img, "s.png", 10, severity=0.3, frequency=0.5)
    high_freq = generator.generate(img, "s.png", 10, severity=0.3, frequency=3.0)

    assert low_freq.metadata["num_pinholes"] <= high_freq.metadata["num_pinholes"]
