from __future__ import annotations

from pathlib import Path

import numpy as np

from synthline_ai.config.models import (
    DatasetSplit,
    DefectType,
    GenerationConfig,
    ImageInfo,
    SplitRatio,
)
from synthline_ai.generation.base import GenerationResult
from synthline_ai.generation.pipeline import partition_seeds, run_generation
from synthline_ai.labeling.boxes import mask_to_yolo_bbox
from synthline_ai.labeling.export import export_yolo


def test_mask_to_yolo_bbox() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:60, 30:70] = 255  # x: 30..70 (w=40), y: 20..60 (h=40)
    xc, yc, w, h = mask_to_yolo_bbox(mask, img_width=100, img_height=100)
    assert abs(xc - 0.5) < 1e-4
    assert abs(yc - 0.4) < 1e-4
    assert abs(w - 0.4) < 1e-4
    assert abs(h - 0.4) < 1e-4


def test_partition_seeds_no_leakage() -> None:
    infos = [
        ImageInfo(
            path=Path(f"seed_{i}.png"),
            width=64,
            height=64,
            channels=3,
            mean_brightness=128.0,
            blur_score=100.0,
        )
        for i in range(10)
    ]
    config = GenerationConfig(
        seeds_dir=Path("seeds"),
        output_dir=Path("output"),
        enable_split=True,
        split_ratio=SplitRatio(train=0.6, val=0.2, test=0.2),
        random_seed=42,
    )
    splits = partition_seeds(infos, config)
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])

    # Strict disjoint partitions
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == 10


def test_run_generation_with_splits() -> None:
    infos = [
        ImageInfo(
            path=Path(f"seed_{i}.png"),
            width=64,
            height=64,
            channels=3,
            mean_brightness=128.0,
            blur_score=100.0,
        )
        for i in range(5)
    ]
    images = [np.full((64, 64, 3), 150, dtype=np.uint8) for _ in range(5)]
    config = GenerationConfig(
        seeds_dir=Path("seeds"),
        defect_type=DefectType.STAIN,
        count=15,
        output_dir=Path("output"),
        enable_split=True,
        split_ratio=SplitRatio(train=0.6, val=0.2, test=0.2),
        random_seed=123,
    )
    results = run_generation(config, images, infos)
    assert len(results) == 15

    splits_found = {r.split for r in results}
    assert DatasetSplit.TRAIN.value in splits_found
    assert DatasetSplit.VAL.value in splits_found


def test_export_yolo(tmp_path: Path) -> None:
    results: list[GenerationResult] = []
    for i in range(4):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        split = "train" if i < 2 else "val"
        res = GenerationResult(
            image=img,
            mask=mask,
            defect_type="stain",
            source_seed=f"seed_{i}.png",
            random_seed=i,
            split=split,
        )
        results.append(res)

    output_dir = tmp_path / "dataset"
    config = GenerationConfig(
        seeds_dir=tmp_path / "seeds",
        output_dir=output_dir,
        enable_split=True,
    )
    data_yaml = export_yolo(results, output_dir, config)
    assert data_yaml.exists()
    assert (output_dir / "yolo" / "images" / "train").exists()
    assert (output_dir / "yolo" / "labels" / "train").exists()
    assert (output_dir / "yolo" / "images" / "val").exists()
    assert (output_dir / "yolo" / "labels" / "val").exists()
