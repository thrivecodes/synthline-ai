"""Tests for enhanced validation: duplicate output, split leakage, and brightness distribution."""

from __future__ import annotations

import numpy as np

from synthline_ai.generation.base import GenerationResult
from synthline_ai.validation.checks import (
    check_brightness_distribution,
    check_output_duplicates,
    check_split_leakage,
)
from synthline_ai.validation.statistics import compute_split_statistics


def create_result(
    image_val: int = 128,
    seed: str = "test.png",
    split: str = "train",
    defect: str = "scratch",
) -> GenerationResult:
    image = np.full((64, 64, 3), image_val, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    return GenerationResult(
        image=image,
        mask=mask,
        source_seed=seed,
        defect_type=defect,
        split=split,
    )


def test_check_output_duplicates_none() -> None:
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(99)
    img1 = rng1.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    img2 = rng2.randint(0, 256, (64, 64, 3), dtype=np.uint8)

    res1 = GenerationResult(image=img1, mask=np.zeros((64, 64), dtype=np.uint8))
    res2 = GenerationResult(image=img2, mask=np.zeros((64, 64), dtype=np.uint8))

    out = check_output_duplicates([res1, res2])
    assert out["duplicate_pairs"] == 0


def test_check_output_duplicates_found() -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32] = 255

    res1 = GenerationResult(image=img, mask=np.zeros((64, 64), dtype=np.uint8))
    res2 = GenerationResult(image=img, mask=np.zeros((64, 64), dtype=np.uint8))

    out = check_output_duplicates([res1, res2])
    assert out["duplicate_pairs"] == 1


def test_check_split_leakage_clean() -> None:
    results = [
        create_result(seed="1.png", split="train"),
        create_result(seed="1.png", split="train"),
        create_result(seed="2.png", split="val"),
    ]
    out = check_split_leakage(results)
    assert not out["has_leakage"]


def test_check_split_leakage_detected() -> None:
    results = [
        create_result(seed="1.png", split="train"),
        create_result(seed="1.png", split="val"),
    ]
    out = check_split_leakage(results)
    assert out["has_leakage"]
    leaked = out["leaked_seeds"]
    assert isinstance(leaked, list)
    assert isinstance(leaked[0], dict)
    assert leaked[0]["seed"] == "1.png"


def test_check_brightness_distribution_normal() -> None:
    results = [
        create_result(image_val=120),
        create_result(image_val=125),
        create_result(image_val=130),
    ]
    out = check_brightness_distribution(results)
    assert out["outlier_count"] == 0


def test_check_brightness_distribution_outlier() -> None:
    results = [create_result(image_val=100) for _ in range(10)]
    results.append(create_result(image_val=255))
    out = check_brightness_distribution(results)
    outlier_count = out["outlier_count"]
    assert isinstance(outlier_count, int)
    assert outlier_count > 0
    outliers = out["outliers"]
    assert isinstance(outliers, list)
    assert isinstance(outliers[0], dict)
    assert "brightness" in str(outliers[0]["reason"])


def test_compute_split_statistics_basic() -> None:
    results = [
        create_result(split="train", image_val=100),
        create_result(split="train", image_val=100),
        create_result(split="val", image_val=200),
    ]
    out = compute_split_statistics(results)
    assert out["total"] == 3
    splits = out["splits"]
    assert isinstance(splits, dict)
    assert splits["train"]["count"] == 2
    assert splits["val"]["count"] == 1


def test_compute_split_statistics_balanced() -> None:
    # 10 train, 2 val, 2 test -> balanced
    results = [create_result(split="train") for _ in range(10)]
    results.extend([create_result(split="val") for _ in range(2)])
    results.extend([create_result(split="test") for _ in range(2)])
    out = compute_split_statistics(results)
    assert out["is_balanced"] is True

    # 10 train, 1 val -> 1/11 < 10% -> unbalanced
    results_unbalanced = [create_result(split="train") for _ in range(10)]
    results_unbalanced.append(create_result(split="val"))
    out_unb = compute_split_statistics(results_unbalanced)
    assert out_unb["is_balanced"] is False
