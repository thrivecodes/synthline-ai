"""Unit tests for compound multi-defect generation and mixed defect distribution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthline_ai.config.models import DefectType, ExportFormat, GenerationConfig, ImageInfo
from synthline_ai.generation.pipeline import run_generation
from synthline_ai.labeling.export import export_coco, export_yolo


def _make_dummy_dataset(tmp_path: Path, count: int = 3) -> tuple[list[np.ndarray], list[ImageInfo]]:
    images = []
    infos = []
    for i in range(count):
        arr = np.full((100, 100, 3), 120 + i * 10, dtype=np.uint8)
        images.append(arr)
        p = tmp_path / f"seed_{i}.png"
        p.touch()
        infos.append(
            ImageInfo(
                path=p,
                width=100,
                height=100,
                channels=3,
                mean_brightness=float(120 + i * 10),
                blur_score=150.0,
            )
        )
    return images, infos


def test_mixed_defect_generation(tmp_path: Path) -> None:
    images, infos = _make_dummy_dataset(tmp_path, count=2)
    cfg = GenerationConfig(
        seeds_dir=tmp_path,
        output_dir=tmp_path / "out_mixed",
        defect_type=DefectType.MIXED,
        count=12,
        random_seed=42,
    )
    results = run_generation(cfg, images, infos)

    assert len(results) == 12
    # In 12 samples of mixed mode, we should see multiple distinct defect types
    defect_types = {r.defect_type for r in results}
    assert len(defect_types) >= 2


def test_compound_defect_generation(tmp_path: Path) -> None:
    images, infos = _make_dummy_dataset(tmp_path, count=2)
    cfg = GenerationConfig(
        seeds_dir=tmp_path,
        output_dir=tmp_path / "out_compound",
        defect_type=DefectType.SCRATCH,
        compound_defects=True,
        defects_per_image=2,
        count=4,
        random_seed=123,
    )
    results = run_generation(cfg, images, infos)

    assert len(results) == 4
    for r in results:
        assert r.defect_type == "compound"
        assert len(r.instances) == 2
        for inst in r.instances:
            assert "defect_type" in inst
            assert "mask" in inst
            assert "bbox" in inst
            assert "area" in inst
            assert inst["area"] > 0
        # Check overall mask is non-empty
        assert np.any(r.mask == 255)


def test_compound_coco_export(tmp_path: Path) -> None:
    images, infos = _make_dummy_dataset(tmp_path, count=2)
    out_dir = tmp_path / "coco_compound"
    cfg = GenerationConfig(
        seeds_dir=tmp_path,
        output_dir=out_dir,
        defect_type=DefectType.MIXED,
        compound_defects=True,
        defects_per_image=3,
        count=2,
        random_seed=77,
    )
    results = run_generation(cfg, images, infos)
    anno_file = export_coco(results, out_dir, cfg)

    assert anno_file.exists()
    coco_data = json.loads(anno_file.read_text())

    # 2 images, each with 3 instances -> exactly 6 annotations
    assert len(coco_data["images"]) == 2
    assert len(coco_data["annotations"]) == 6

    # Verify annotation image_ids map to the images
    image_ids = [a["image_id"] for a in coco_data["annotations"]]
    assert image_ids.count(1) == 3
    assert image_ids.count(2) == 3


def test_compound_yolo_export(tmp_path: Path) -> None:
    images, infos = _make_dummy_dataset(tmp_path, count=2)
    out_dir = tmp_path / "yolo_compound"
    cfg = GenerationConfig(
        seeds_dir=tmp_path,
        output_dir=out_dir,
        defect_type=DefectType.MIXED,
        compound_defects=True,
        defects_per_image=2,
        count=2,
        export_format=ExportFormat.YOLO,
        random_seed=88,
    )
    results = run_generation(cfg, images, infos)
    data_yaml = export_yolo(results, out_dir, cfg)

    assert data_yaml.exists()
    lbl_files = list((out_dir / "yolo" / "labels").glob("*.txt"))
    assert len(lbl_files) == 2

    for lbl in lbl_files:
        lines = [line.strip() for line in lbl.read_text().splitlines() if line.strip()]
        # Each image should have 2 bounding box lines
        assert len(lines) == 2


def test_compound_with_variations(tmp_path: Path) -> None:
    images, infos = _make_dummy_dataset(tmp_path, count=2)
    cfg = GenerationConfig(
        seeds_dir=tmp_path,
        output_dir=tmp_path / "out_vars",
        defect_type=DefectType.MIXED,
        compound_defects=True,
        defects_per_image=2,
        count=2,
        enable_variations=True,
        geometry_intensity=0.4,
        lighting_intensity=0.3,
        sensor_intensity=0.2,
        random_seed=55,
    )
    results = run_generation(cfg, images, infos)
    assert len(results) == 2
    for r in results:
        assert r.image.dtype == np.uint8
        assert r.mask.dtype == np.uint8
        assert set(np.unique(r.mask)).issubset({0, 255})
        assert len(r.instances) == 2
