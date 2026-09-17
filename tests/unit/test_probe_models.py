"""Tests for probe model evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from synthline_ai.generation.base import GenerationResult
from synthline_ai.validation.probe_models import (
    evaluate_dataset_quality,
    extract_features,
    train_probe,
)


def create_mock_data(
    n: int,
    label: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Create mock image-mask-label triplets for testing."""
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    for _ in range(n):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)

        if label == 1:
            y, x = np.random.randint(10, 50, 2)
            mask[y : y + 10, x : x + 10] = 255
            img[y : y + 10, x : x + 10] = 255

        images.append(img)
        masks.append(mask)
        labels.append(label)

    return images, masks, labels


def test_extract_features_shape() -> None:
    images, masks, _ = create_mock_data(5, label=1)
    features = extract_features(images, masks)
    assert features.shape == (5, 16)


def test_extract_features_values() -> None:
    images, masks, _ = create_mock_data(2, label=1)
    features = extract_features(images, masks)
    assert np.all(np.isfinite(features))
    assert not np.any(np.isnan(features))


def test_evaluate_dataset_quality_no_real() -> None:
    images, masks, _ = create_mock_data(3, label=1)
    results = [
        GenerationResult(image=img, mask=m, metadata={})
        for img, m in zip(images, masks, strict=True)
    ]

    metrics = evaluate_dataset_quality(results)

    assert "feature_diversity" in metrics
    assert "intra_class_variance" in metrics
    assert "probe_error" not in metrics


def test_evaluate_dataset_quality_with_real() -> None:
    syn_images, syn_masks, syn_labels = create_mock_data(4, label=1)
    syn_images0, syn_masks0, syn_labels0 = create_mock_data(4, label=0)
    syn_images.extend(syn_images0)
    syn_masks.extend(syn_masks0)
    syn_labels.extend(syn_labels0)

    results = [
        GenerationResult(image=img, mask=m, metadata={})
        for img, m in zip(syn_images, syn_masks, strict=True)
    ]

    real_images, real_masks, real_labels = create_mock_data(4, label=1)
    real_images0, real_masks0, real_labels0 = create_mock_data(4, label=0)
    real_images.extend(real_images0)
    real_masks.extend(real_masks0)
    real_labels.extend(real_labels0)

    # Skip if sklearn is not installed
    sklearn_spec = __import__("importlib").util.find_spec("sklearn")
    if sklearn_spec is None:
        pytest.skip("scikit-learn not installed")

    metrics = evaluate_dataset_quality(
        results,
        real_images=real_images,
        real_masks=real_masks,
        real_labels=real_labels,
    )

    assert "feature_diversity" in metrics
    assert "synthetic_cv_accuracy" in metrics
    assert "real_precision" in metrics
    assert "real_recall" in metrics
    assert "real_f1" in metrics
    assert "real_accuracy" in metrics
    assert "feature_importances" in metrics


def test_train_probe_returns_metrics() -> None:
    sklearn_spec = __import__("importlib").util.find_spec("sklearn")
    if sklearn_spec is None:
        pytest.skip("scikit-learn not installed")

    syn_images, syn_masks, syn_labels = create_mock_data(4, label=1)
    syn_images0, syn_masks0, syn_labels0 = create_mock_data(4, label=0)
    syn_images.extend(syn_images0)
    syn_masks.extend(syn_masks0)
    syn_labels.extend(syn_labels0)

    real_images, real_masks, real_labels = create_mock_data(2, label=1)
    real_images0, real_masks0, real_labels0 = create_mock_data(2, label=0)
    real_images.extend(real_images0)
    real_masks.extend(real_masks0)
    real_labels.extend(real_labels0)

    metrics = train_probe(
        syn_images,
        syn_masks,
        syn_labels,
        real_images,
        real_masks,
        real_labels,
    )

    expected_keys = {
        "synthetic_cv_accuracy",
        "real_precision",
        "real_recall",
        "real_f1",
        "real_accuracy",
        "feature_importances",
        "sim_to_real_gap",
        "model_params",
    }

    assert set(metrics.keys()) == expected_keys


def test_feature_diversity_computed() -> None:
    images, masks, _ = create_mock_data(3, label=1)
    results = [
        GenerationResult(image=img, mask=m, metadata={})
        for img, m in zip(images, masks, strict=True)
    ]

    metrics = evaluate_dataset_quality(results)
    assert "feature_diversity" in metrics
    assert isinstance(metrics["feature_diversity"], float)
