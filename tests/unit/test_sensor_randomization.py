"""Unit tests for industrial camera sensor noise randomization."""

from __future__ import annotations

import numpy as np

from synthline_ai.generation.randomization.sensor import apply_sensor_noise


def test_apply_sensor_noise_basic() -> None:
    rng = np.random.RandomState(42)
    img = np.full((64, 64, 3), 128, dtype=np.uint8)

    noisy = apply_sensor_noise(img, rng, intensity=0.5)

    assert noisy.shape == img.shape
    assert noisy.dtype == np.uint8
    assert not np.array_equal(noisy, img)


def test_apply_sensor_noise_zero_intensity() -> None:
    rng = np.random.RandomState(42)
    img = np.full((64, 64, 3), 128, dtype=np.uint8)

    clean = apply_sensor_noise(img, rng, intensity=0.0)

    np.testing.assert_array_equal(clean, img)


def test_apply_sensor_noise_deterministic() -> None:
    img = np.full((64, 64, 3), 128, dtype=np.uint8)

    noisy1 = apply_sensor_noise(img, np.random.RandomState(99), intensity=0.4)
    noisy2 = apply_sensor_noise(img, np.random.RandomState(99), intensity=0.4)

    np.testing.assert_array_equal(noisy1, noisy2)
