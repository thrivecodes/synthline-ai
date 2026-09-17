from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from synthline_ai.config.models import DefectType, GenerationConfig, ImageInfo
from synthline_ai.generation.pipeline import run_generation
from synthline_ai.generation.procedural.scratch import ScratchGenerator
from synthline_ai.generation.registry import get_generator


def test_scratch_generates_valid_output() -> None:
    rng = np.random.RandomState(42)
    image = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    generator = ScratchGenerator()
    result = generator.generate(image, "test.png", random_seed=42, severity=0.5)

    assert result.image.shape == image.shape
    assert result.image.dtype == np.uint8
    assert result.mask.shape == (64, 64)
    assert result.mask.dtype == np.uint8

    unique_vals = set(np.unique(result.mask))
    assert unique_vals.issubset({0, 255})
    assert len(unique_vals) > 1, "Mask should have non-zero pixels"

    assert result.defect_type == "scratch"


def test_scratch_deterministic() -> None:
    rng = np.random.RandomState(123)
    image = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    generator = ScratchGenerator()
    result1 = generator.generate(image, "test.png", random_seed=10, severity=0.7)
    result2 = generator.generate(image, "test.png", random_seed=10, severity=0.7)

    np.testing.assert_array_equal(result1.image, result2.image)
    np.testing.assert_array_equal(result1.mask, result2.mask)


def test_scratch_different_seeds() -> None:
    rng = np.random.RandomState(123)
    image = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    generator = ScratchGenerator()
    result1 = generator.generate(image, "test.png", random_seed=10, severity=0.7)
    result2 = generator.generate(image, "test.png", random_seed=20, severity=0.7)

    assert not np.array_equal(result1.image, result2.image) or not np.array_equal(
        result1.mask, result2.mask
    )


def test_scratch_severity_range() -> None:
    rng = np.random.RandomState(123)
    image = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    generator = ScratchGenerator()
    result_0 = generator.generate(image, "test.png", random_seed=10, severity=0.0)
    result_1 = generator.generate(image, "test.png", random_seed=10, severity=1.0)

    assert result_0.image.dtype == np.uint8
    assert result_1.image.dtype == np.uint8


def test_registry_get_scratch() -> None:
    generator = get_generator("scratch")
    assert isinstance(generator, ScratchGenerator)


def test_registry_unknown() -> None:
    with pytest.raises(KeyError):
        get_generator("nonexistent")


def test_pipeline_count(tmp_path: Path) -> None:
    rng = np.random.RandomState(42)
    img1 = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img2 = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    info1 = ImageInfo(
        path=Path("img1.png"),
        width=64,
        height=64,
        channels=3,
        mean_brightness=128.0,
        blur_score=100.0,
    )
    info2 = ImageInfo(
        path=Path("img2.png"),
        width=64,
        height=64,
        channels=3,
        mean_brightness=128.0,
        blur_score=100.0,
    )

    config = GenerationConfig(
        seeds_dir=tmp_path / "seeds",
        defect_type=DefectType.SCRATCH,
        count=5,
        output_dir=tmp_path / "out",
        random_seed=42,
        severity=0.5,
    )

    results = run_generation(config, [img1, img2], [info1, info2])
    assert len(results) == 5


def test_pipeline_deterministic(tmp_path: Path) -> None:
    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    info = ImageInfo(
        path=Path("img.png"),
        width=64,
        height=64,
        channels=3,
        mean_brightness=128.0,
        blur_score=100.0,
    )

    config = GenerationConfig(
        seeds_dir=tmp_path / "seeds",
        defect_type=DefectType.SCRATCH,
        count=2,
        output_dir=tmp_path / "out",
        random_seed=42,
        severity=0.5,
    )

    results1 = run_generation(config, [img], [info])
    results2 = run_generation(config, [img], [info])

    for r1, r2 in zip(results1, results2, strict=True):
        np.testing.assert_array_equal(r1.image, r2.image)
        np.testing.assert_array_equal(r1.mask, r2.mask)
