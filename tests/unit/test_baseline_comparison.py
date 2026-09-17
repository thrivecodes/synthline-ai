"""Unit tests for probe model baseline comparison (Real vs Synthetic vs Augmented)."""

from __future__ import annotations

import numpy as np
import pytest

from synthline_ai.validation.probe_models import compare_synthetic_vs_real_baselines


def _make_dummy_dataset(
    count: int,
    size: int = 64,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    rng = np.random.RandomState(seed)
    images = []
    masks = []
    labels = []

    for i in range(count):
        img = rng.randint(80, 180, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        label = 1 if i % 2 == 1 else 0
        if label == 1:
            mask[15:45, 15:45] = 255
            img[15:45, 15:45] = (img[15:45, 15:45] * 0.4).astype(np.uint8)
        images.append(img)
        masks.append(mask)
        labels.append(label)

    return images, masks, labels


def test_compare_baselines_basic() -> None:
    syn_imgs, syn_msks, syn_lbls = _make_dummy_dataset(12, seed=10)
    real_imgs, real_msks, real_lbls = _make_dummy_dataset(8, seed=20)

    result = compare_synthetic_vs_real_baselines(
        synthetic_images=syn_imgs,
        synthetic_masks=syn_msks,
        synthetic_labels=syn_lbls,
        real_images=real_imgs,
        real_masks=real_msks,
        real_labels=real_lbls,
        test_ratio=0.5,
        random_seed=42,
    )

    assert "real_only" in result
    assert "synthetic_only" in result
    assert "augmented" in result
    assert "f1_lift" in result
    assert "accuracy_lift" in result
    assert "synthetic_standalone_relative_f1" in result
    assert "is_beneficial" in result

    assert "f1" in result["real_only"]
    assert "f1" in result["synthetic_only"]
    assert "f1" in result["augmented"]
    assert isinstance(result["f1_lift"], float)
    assert isinstance(result["accuracy_lift"], float)


def test_compare_baselines_insufficient_real() -> None:
    syn_imgs, syn_msks, syn_lbls = _make_dummy_dataset(10)
    real_imgs, real_msks, real_lbls = _make_dummy_dataset(3)

    with pytest.raises(ValueError, match="At least 4 real images are required"):
        compare_synthetic_vs_real_baselines(
            synthetic_images=syn_imgs,
            synthetic_masks=syn_msks,
            synthetic_labels=syn_lbls,
            real_images=real_imgs,
            real_masks=real_msks,
            real_labels=real_lbls,
        )


def test_compare_baselines_empty_synthetic() -> None:
    real_imgs, real_msks, real_lbls = _make_dummy_dataset(6)

    with pytest.raises(ValueError, match="Synthetic dataset cannot be empty"):
        compare_synthetic_vs_real_baselines(
            synthetic_images=[],
            synthetic_masks=[],
            synthetic_labels=[],
            real_images=real_imgs,
            real_masks=real_msks,
            real_labels=real_lbls,
        )
