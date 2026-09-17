"""Shared pytest fixtures for SynthLine AI tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture()
def seed_image_factory(tmp_path: Path):
    """Factory fixture that creates synthetic test images.

    Returns a callable that creates a directory of small PNG seed images.
    """

    def _create(count: int = 5, size: tuple[int, int] = (64, 64)) -> Path:
        seed_dir = tmp_path / "seeds"
        seed_dir.mkdir(exist_ok=True)
        for i in range(count):
            rng = np.random.RandomState(i)
            img = rng.randint(100, 200, (*size, 3), dtype=np.uint8)
            cv2.imwrite(str(seed_dir / f"seed_{i:03d}.png"), img)
        return seed_dir

    return _create
