from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthline_ai.config.models import DefectType, GenerationConfig
from synthline_ai.generation.base import GenerationResult
from synthline_ai.labeling.export import export_coco


def test_export_coco_creates_structure(tmp_path: Path) -> None:
    results = []
    for i in range(3):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:20, 10:20] = 255
        res = GenerationResult(
            image=img,
            mask=mask,
            metadata={"test_meta": i},
            source_seed=f"seed_{i}",
            defect_type=DefectType.SCRATCH,
            random_seed=i,
        )
        results.append(res)

    export_dir = tmp_path / "export"
    config = GenerationConfig(
        seeds_dir=tmp_path / "seeds",
        defect_type=DefectType.SCRATCH,
        count=3,
        output_dir=export_dir,
        random_seed=42,
    )

    anno_path = export_coco(results, export_dir, config)

    assert (export_dir / "images").exists()
    assert len(list((export_dir / "images").glob("*.png"))) == 3

    assert (export_dir / "masks").exists()
    assert len(list((export_dir / "masks").glob("*.png"))) == 3

    assert anno_path.exists()

    with open(anno_path) as f:
        coco = json.load(f)

    assert len(coco["images"]) == 3
    assert len(coco["annotations"]) == 3

    assert (export_dir / "config.json").exists()

    meta_lines = (export_dir / "metadata.jsonl").read_text().splitlines()
    assert len(meta_lines) == 3


def test_export_coco_annotation_fields(tmp_path: Path) -> None:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    res = GenerationResult(
        image=img,
        mask=mask,
        metadata={},
        source_seed="seed",
        defect_type=DefectType.SCRATCH,
        random_seed=42,
    )

    export_dir = tmp_path / "export"
    config = GenerationConfig(
        seeds_dir=tmp_path / "seeds",
        defect_type=DefectType.SCRATCH,
        count=1,
        output_dir=export_dir,
        random_seed=42,
    )

    anno_path = export_coco([res], export_dir, config)

    with open(anno_path) as f:
        coco = json.load(f)

    anno = coco["annotations"][0]
    expected_fields = {
        "id",
        "image_id",
        "category_id",
        "bbox",
        "area",
        "segmentation",
        "iscrowd",
    }
    assert expected_fields.issubset(set(anno.keys()))
