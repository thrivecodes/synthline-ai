"""Unit tests for quality check module."""

from __future__ import annotations

from pathlib import Path

from synthline_ai.config.models import ImageInfo
from synthline_ai.ingestion.quality import check_seed_quality


def create_image_info(
    name: str = "img.png",
    width: int = 100,
    height: int = 100,
    channels: int = 3,
    mean_brightness: float = 127.0,
    blur_score: float = 200.0,
) -> ImageInfo:
    return ImageInfo(
        path=Path(name),
        width=width,
        height=height,
        channels=channels,
        mean_brightness=mean_brightness,
        blur_score=blur_score,
    )


def test_normal_images() -> None:
    """Test that normal images produce no warnings."""
    images = [create_image_info(f"img_{i}.png") for i in range(3)]
    report = check_seed_quality(images, [])

    assert report.valid_count == 3
    assert report.rejected_count == 0
    assert len(report.warnings) == 0


def test_dark_image() -> None:
    """Test detection of dark images."""
    images = [create_image_info(mean_brightness=20.0)]
    report = check_seed_quality(images, [])

    assert len(report.warnings) == 1
    assert report.warnings[0].code == "too_dark"


def test_bright_image() -> None:
    """Test detection of bright images."""
    images = [create_image_info(mean_brightness=230.0)]
    report = check_seed_quality(images, [])

    assert len(report.warnings) == 1
    assert report.warnings[0].code == "too_bright"


def test_blurry_image() -> None:
    """Test detection of blurry images."""
    images = [create_image_info(blur_score=50.0)]
    report = check_seed_quality(images, [])

    assert len(report.warnings) == 1
    assert report.warnings[0].code == "blurry"


def test_odd_dimensions() -> None:
    """Test detection of images with odd dimensions compared to the median."""
    images = [
        create_image_info("img_1.png", width=100, height=100),
        create_image_info("img_2.png", width=100, height=100),
        create_image_info("img_3.png", width=100, height=100),
        create_image_info("img_large.png", width=200, height=200),
    ]
    report = check_seed_quality(images, [])

    assert len(report.warnings) == 1
    assert report.warnings[0].code == "odd_dimensions"
    assert report.warnings[0].path == Path("img_large.png")
