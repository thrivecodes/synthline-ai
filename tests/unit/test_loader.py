"""Unit tests for loader module."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from synthline_ai.ingestion.loader import load_seeds


def test_load_valid_seeds(tmp_path: Path) -> None:
    """Test loading valid seed images."""
    for i in range(3):
        img_path = tmp_path / f"img_{i}.png"
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), img)

    images, arrays, warnings = load_seeds(tmp_path)

    assert len(images) == 3
    assert len(arrays) == 3
    assert len(warnings) == 0
    assert images[0].width == 64
    assert images[0].height == 64
    assert images[0].channels == 3


def test_load_empty_dir(tmp_path: Path) -> None:
    """Test loading an empty directory."""
    with pytest.raises(ValueError, match="No valid seed images found"):
        load_seeds(tmp_path)


def test_load_nonexistent_dir(tmp_path: Path) -> None:
    """Test loading a nonexistent directory."""
    with pytest.raises(FileNotFoundError, match="Seeds directory not found"):
        load_seeds(tmp_path / "nonexistent")


def test_load_skips_unsupported(tmp_path: Path) -> None:
    """Test that unsupported files are skipped."""
    img_path = tmp_path / "img.png"
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    txt_path = tmp_path / "img.txt"
    txt_path.write_text("not an image")

    images, arrays, warnings = load_seeds(tmp_path)

    assert len(images) == 1
    assert len(warnings) == 0


def test_load_corrupt_file(tmp_path: Path) -> None:
    """Test loading a corrupt image file."""
    img_path = tmp_path / "corrupt.png"
    img_path.write_text("this is not a valid png file")

    images, arrays, warnings = load_seeds(tmp_path)

    assert len(images) == 0
    assert len(arrays) == 0
    assert len(warnings) == 1
    assert warnings[0].code == "corrupt"
    assert warnings[0].severity == "error"
