"""Unit tests for surface lighting, texture, and geometry randomization."""

from pathlib import Path

import numpy as np

from synthline_ai.config.models import DefectType, GenerationConfig, ImageInfo
from synthline_ai.generation.pipeline import run_generation
from synthline_ai.generation.randomization import (
    apply_geometry_variation,
    apply_lighting_variation,
    apply_texture_variation,
)


def test_lighting_variation() -> None:
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    rng = np.random.RandomState(42)

    lit = apply_lighting_variation(img, rng, intensity=0.3)
    assert lit.shape == img.shape
    assert lit.dtype == np.uint8
    assert not np.array_equal(lit, img)

    # Determinism
    rng1 = np.random.RandomState(99)
    rng2 = np.random.RandomState(99)
    lit1 = apply_lighting_variation(img, rng1, intensity=0.2)
    lit2 = apply_lighting_variation(img, rng2, intensity=0.2)
    np.testing.assert_array_equal(lit1, lit2)

    # Grayscale
    gray = np.full((64, 64), 128, dtype=np.uint8)
    lit_gray = apply_lighting_variation(gray, np.random.RandomState(1), intensity=0.2)
    assert lit_gray.shape == (64, 64)


def test_texture_variation() -> None:
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    rng = np.random.RandomState(42)

    textured = apply_texture_variation(img, rng, intensity=0.2)
    assert textured.shape == img.shape
    assert textured.dtype == np.uint8
    assert not np.array_equal(textured, img)

    # Zero intensity returns copy
    zero = apply_texture_variation(img, rng, intensity=0.0)
    np.testing.assert_array_equal(zero, img)


def test_geometry_variation() -> None:
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:30, 20:30] = 255

    rng = np.random.RandomState(77)
    warped_img, warped_mask = apply_geometry_variation(
        img,
        mask,
        rng,
        max_rotation_deg=5.0,
        max_shift_px=3,
    )

    assert warped_img.shape == img.shape
    assert warped_mask.shape == mask.shape
    assert warped_mask.dtype == np.uint8
    assert set(np.unique(warped_mask)).issubset({0, 255})
    assert np.count_nonzero(warped_mask) > 0


def test_pipeline_with_variations(tmp_path: Path) -> None:
    img = np.full((64, 64, 3), 140, dtype=np.uint8)
    info = ImageInfo(
        path=tmp_path / "seed.png",
        width=64,
        height=64,
        channels=3,
        mean_brightness=140.0,
        blur_score=100.0,
    )

    config = GenerationConfig(
        seeds_dir=tmp_path,
        defect_type=DefectType.SCRATCH,
        count=4,
        output_dir=tmp_path / "out",
        random_seed=123,
        enable_variations=True,
        lighting_intensity=0.3,
        texture_intensity=0.2,
        geometry_intensity=0.4,
    )

    results = run_generation(config, [img], [info])
    assert len(results) == 4
    for res in results:
        assert res.image.shape == (64, 64, 3)
        assert res.mask.shape == (64, 64)
        assert set(np.unique(res.mask)).issubset({0, 255})
