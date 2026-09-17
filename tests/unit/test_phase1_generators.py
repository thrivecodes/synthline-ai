from __future__ import annotations

import numpy as np

from synthline_ai.generation.procedural.discoloration import DiscolorationGenerator
from synthline_ai.generation.procedural.stain import StainGenerator
from synthline_ai.generation.registry import get_generator


def test_stain_generator_output() -> None:
    img = np.full((128, 128, 3), 180, dtype=np.uint8)
    gen = StainGenerator()
    assert gen.defect_type == "stain"

    res = gen.generate(img, "test.png", random_seed=42, severity=0.6, frequency=1.5)
    assert res.image.shape == img.shape
    assert res.image.dtype == np.uint8
    assert res.mask.shape == (128, 128)
    assert np.count_nonzero(res.mask) > 0
    assert set(np.unique(res.mask)).issubset({0, 255})
    assert res.metadata["num_spots"] >= 1


def test_stain_generator_deterministic() -> None:
    img = np.random.RandomState(0).randint(50, 200, (64, 64, 3), dtype=np.uint8)
    gen = StainGenerator()
    res1 = gen.generate(img, "seed.png", random_seed=1234, severity=0.5)
    res2 = gen.generate(img, "seed.png", random_seed=1234, severity=0.5)

    np.testing.assert_array_equal(res1.image, res2.image)
    np.testing.assert_array_equal(res1.mask, res2.mask)


def test_stain_generator_grayscale() -> None:
    img = np.full((64, 64), 160, dtype=np.uint8)
    gen = StainGenerator()
    res = gen.generate(img, "gray.png", random_seed=99, severity=0.8)
    assert res.image.shape == (64, 64)
    assert res.mask.shape == (64, 64)


def test_discoloration_generator_output() -> None:
    img = np.full((128, 128, 3), 170, dtype=np.uint8)
    gen = DiscolorationGenerator()
    assert gen.defect_type == "discoloration"

    res = gen.generate(img, "test.png", random_seed=77, severity=0.7, frequency=1.0)
    assert res.image.shape == img.shape
    assert res.image.dtype == np.uint8
    assert res.mask.shape == (128, 128)
    assert np.count_nonzero(res.mask) > 0
    assert set(np.unique(res.mask)).issubset({0, 255})


def test_discoloration_generator_deterministic() -> None:
    img = np.random.RandomState(42).randint(50, 200, (64, 64, 3), dtype=np.uint8)
    gen = DiscolorationGenerator()
    res1 = gen.generate(img, "seed.png", random_seed=5555, severity=0.4)
    res2 = gen.generate(img, "seed.png", random_seed=5555, severity=0.4)

    np.testing.assert_array_equal(res1.image, res2.image)
    np.testing.assert_array_equal(res1.mask, res2.mask)


def test_discoloration_generator_grayscale() -> None:
    img = np.full((64, 64), 140, dtype=np.uint8)
    gen = DiscolorationGenerator()
    res = gen.generate(img, "gray.png", random_seed=12, severity=0.5)
    assert res.image.shape == (64, 64)


def test_registry_contains_new_generators() -> None:
    stain_gen = get_generator("stain")
    disc_gen = get_generator("discoloration")
    assert isinstance(stain_gen, StainGenerator)
    assert isinstance(disc_gen, DiscolorationGenerator)
