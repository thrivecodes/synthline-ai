"""Unit tests for dataset fusion and multi-run dataset merging."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from synthline_ai.labeling.fusion import merge_coco_annotations, merge_datasets


def test_merge_coco_annotations(tmp_path: Path) -> None:
    """Verify COCO annotations merge cleanly with ID re-indexing and category mapping."""
    coco1 = {
        "images": [
            {"id": 1, "file_name": "image_0001_scratch.png", "width": 64, "height": 64},
            {"id": 2, "file_name": "image_0002_scratch.png", "width": 64, "height": 64},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "segmentation": [],
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [15, 15, 10, 10],
                "area": 100,
                "segmentation": [],
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "scratch", "supercategory": "defect"}],
    }

    coco2 = {
        "images": [
            {"id": 1, "file_name": "image_0001_crack.png", "width": 64, "height": 64},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [5, 5, 15, 15],
                "area": 225,
                "segmentation": [],
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "crack", "supercategory": "defect"}],
    }

    p1 = tmp_path / "coco1.json"
    p2 = tmp_path / "coco2.json"
    p1.write_text(json.dumps(coco1), encoding="utf-8")
    p2.write_text(json.dumps(coco2), encoding="utf-8")

    maps = [
        {"image_0001_scratch.png": "run1_image_0001_scratch.png"},
        {"image_0001_crack.png": "run2_image_0001_crack.png"},
    ]

    merged = merge_coco_annotations([p1, p2], maps)

    assert len(merged["images"]) == 3
    assert len(merged["annotations"]) == 3
    assert len(merged["categories"]) == 2

    # Image IDs are sequential 1, 2, 3
    img_ids = [img["id"] for img in merged["images"]]
    assert img_ids == [1, 2, 3]

    # Category IDs and names unified
    cat_names = {c["name"] for c in merged["categories"]}
    assert cat_names == {"scratch", "crack"}

    # Annotation IDs sequential 1, 2, 3
    anno_ids = [a["id"] for a in merged["annotations"]]
    assert anno_ids == [1, 2, 3]


def test_merge_datasets(tmp_path: Path) -> None:
    """Verify end-to-end dataset directory fusion."""
    run1 = tmp_path / "run_1"
    run2 = tmp_path / "run_2"
    out_dir = tmp_path / "fused_dataset"

    for r_dir, defect_name in [(run1, "scratch"), (run2, "crack")]:
        img_dir = r_dir / "images"
        mask_dir = r_dir / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        img_file = img_dir / f"image_0001_{defect_name}.png"
        mask_file = mask_dir / f"mask_0001_{defect_name}.png"
        img_arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        mask_arr = np.zeros((64, 64), dtype=np.uint8)
        mask_arr[10:20, 10:20] = 255

        cv2.imwrite(str(img_file), img_arr)
        cv2.imwrite(str(mask_file), mask_arr)

        coco_data = {
            "images": [{"id": 1, "file_name": img_file.name, "width": 64, "height": 64}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 10, 10],
                    "area": 100,
                    "segmentation": [],
                    "iscrowd": 0,
                }
            ],
            "categories": [{"id": 1, "name": defect_name, "supercategory": "defect"}],
        }
        (r_dir / "annotations.coco.json").write_text(json.dumps(coco_data), encoding="utf-8")
        meta_line = json.dumps({"image": img_file.name, "defect_type": defect_name})
        (r_dir / "metadata.jsonl").write_text(meta_line + "\n", encoding="utf-8")

    summary = merge_datasets([run1, run2], out_dir)

    assert summary["merged_runs_count"] == 2
    assert summary["total_images"] == 2
    assert summary["total_annotations"] == 2
    assert set(summary["categories"]) == {"scratch", "crack"}

    # Check filesystem outputs
    assert (out_dir / "images" / "run1_image_0001_scratch.png").exists()
    assert (out_dir / "images" / "run2_image_0001_crack.png").exists()
    assert (out_dir / "masks" / "run1_mask_0001_scratch.png").exists()
    assert (out_dir / "masks" / "run2_mask_0001_crack.png").exists()
    assert (out_dir / "annotations.coco.json").exists()
    assert (out_dir / "metadata.jsonl").exists()
    assert (out_dir / "fusion_summary.json").exists()


def test_merge_datasets_empty_raises(tmp_path: Path) -> None:
    """Verify merge_datasets raises ValueError when empty list provided."""
    with pytest.raises(ValueError, match="At least one source dataset directory"):
        merge_datasets([], tmp_path / "out")
