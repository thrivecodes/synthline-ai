"""End-to-end integration test for the full generation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from synthline_ai.config.models import DefectType, GenerationConfig
from synthline_ai.generation.pipeline import run_generation
from synthline_ai.ingestion.loader import load_seeds
from synthline_ai.ingestion.quality import check_seed_quality
from synthline_ai.labeling.export import export_coco
from synthline_ai.validation.checks import validate_results
from synthline_ai.validation.previews import create_contact_sheet
from synthline_ai.validation.statistics import compute_statistics


def test_generate_e2e(seed_image_factory: Any, tmp_path: Path) -> None:
    """Full pipeline: seeds -> generate -> export -> validate."""
    seeds_dir = seed_image_factory(count=3)
    output_dir = tmp_path / "output"

    config = GenerationConfig(
        seeds_dir=seeds_dir,
        defect_type=DefectType.SCRATCH,
        count=10,
        output_dir=output_dir,
        random_seed=12345,
    )

    # Load and check seeds
    infos, arrays, warnings = load_seeds(seeds_dir)
    assert len(infos) == 3
    assert len(arrays) == 3

    seed_report = check_seed_quality(infos, arrays)
    assert seed_report.valid_count == 3

    # Generate
    results = run_generation(config, arrays, infos)
    assert len(results) == 10

    # Export
    coco_path = export_coco(results, output_dir, config)
    assert coco_path.exists()
    assert (output_dir / "images").is_dir()
    assert (output_dir / "masks").is_dir()
    assert len(list((output_dir / "images").iterdir())) == 10
    assert len(list((output_dir / "masks").iterdir())) == 10
    assert (output_dir / "config.json").exists()
    assert (output_dir / "metadata.jsonl").exists()

    # Validate COCO JSON structure
    with open(coco_path) as f:
        coco_data = json.load(f)
    assert len(coco_data["images"]) == 10
    assert len(coco_data["annotations"]) == 10
    assert len(coco_data["categories"]) >= 1
    assert coco_data["categories"][0]["name"] == "scratch"

    # Each annotation has required fields
    for ann in coco_data["annotations"]:
        assert "id" in ann
        assert "image_id" in ann
        assert "category_id" in ann
        assert "bbox" in ann
        assert "area" in ann
        assert "segmentation" in ann
        assert "iscrowd" in ann

    # Preview
    contact_path = output_dir / "contact-sheet.jpg"
    create_contact_sheet(results, contact_path)
    assert contact_path.exists()

    # Statistics
    stats = compute_statistics(results)
    assert stats["total_images"] == 10
    assert stats["empty_mask_count"] == 0

    # Validation checks
    validation = validate_results(results)
    assert validation["invalid_count"] == 0

    # Metadata JSONL has 10 lines
    with open(output_dir / "metadata.jsonl") as f:
        lines = f.readlines()
    assert len(lines) == 10
    for line in lines:
        record = json.loads(line)
        assert "image" in record
        assert "source_seed" in record
        assert "defect_type" in record
        assert "random_seed" in record


def test_generate_e2e_reproducibility(seed_image_factory: Any, tmp_path: Path) -> None:
    """Same config and seeds produce identical results."""
    seeds_dir = seed_image_factory(count=2)

    config = GenerationConfig(
        seeds_dir=seeds_dir,
        defect_type=DefectType.SCRATCH,
        count=5,
        output_dir=tmp_path / "out1",
        random_seed=99999,
    )

    infos, arrays, _ = load_seeds(seeds_dir)

    results1 = run_generation(config, arrays, infos)
    results2 = run_generation(config, arrays, infos)

    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2, strict=True):
        np.testing.assert_array_equal(r1.image, r2.image)

        np.testing.assert_array_equal(r1.mask, r2.mask)
        assert r1.source_seed == r2.source_seed
        assert r1.random_seed == r2.random_seed
        assert r1.metadata == r2.metadata
