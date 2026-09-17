"""Unit tests for procedural dent defect generator."""

from __future__ import annotations

import numpy as np

from synthline_ai.generation.procedural.dent import DentGenerator


def _make_canvas(h: int = 128, w: int = 128) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_dent_generates_valid_output() -> None:
    canvas = _make_canvas()
    gen = DentGenerator()
    res = gen.generate(canvas, "test.png", random_seed=42, severity=0.6)

    assert res.image.shape == (128, 128, 3)
    assert res.image.dtype == np.uint8
    assert res.mask.shape == (128, 128)
    assert res.mask.dtype == np.uint8
    assert set(np.unique(res.mask)).issubset({0, 255})
    assert np.any(res.mask == 255)
    assert res.defect_type == "dent"


def test_dent_deterministic() -> None:
    canvas = _make_canvas()
    gen = DentGenerator()
    res1 = gen.generate(canvas, "test.png", random_seed=99, severity=0.5)
    res2 = gen.generate(canvas, "test.png", random_seed=99, severity=0.5)

    np.testing.assert_array_equal(res1.image, res2.image)
    np.testing.assert_array_equal(res1.mask, res2.mask)


def test_dent_different_seeds() -> None:
    canvas = _make_canvas()
    gen = DentGenerator()
    res1 = gen.generate(canvas, "test.png", random_seed=1, severity=0.5)
    res2 = gen.generate(canvas, "test.png", random_seed=2, severity=0.5)

    assert not np.array_equal(res1.image, res2.image)


def test_dent_directional_shading() -> None:
    """Dent shading should create both highlighted and darkened regions relative to base."""
    canvas = np.full((128, 128, 3), 128, dtype=np.uint8)
    gen = DentGenerator()
    res = gen.generate(canvas, "test.png", random_seed=123, severity=0.8)

    # In the mask area, some pixels should be brighter than 128 and some darker than 128
    defect_pixels = res.image[res.mask == 255]
    assert np.any(defect_pixels > 128), "Highlight region expected"
    assert np.any(defect_pixels < 128), "Shadow region expected"


def test_dent_frequency_scaling() -> None:
    canvas = _make_canvas()
    gen = DentGenerator()
    res = gen.generate(canvas, "test.png", random_seed=42, severity=0.5, frequency=3.0)

    assert res.metadata["num_dents"] >= 3
