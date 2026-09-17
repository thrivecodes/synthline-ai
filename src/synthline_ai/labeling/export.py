from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2

from synthline_ai.config.models import GenerationConfig
from synthline_ai.generation.base import GenerationResult
from synthline_ai.labeling.boxes import mask_to_bbox
from synthline_ai.labeling.masks import mask_area, mask_to_polygon


def export_coco(
    results: list[GenerationResult],
    output_dir: Path,
    config: GenerationConfig,
) -> Path:
    """Write generated images, masks, and COCO annotations to output_dir.

    Creates:
    - output_dir/images/ — generated images as PNG
    - output_dir/masks/ — binary masks as PNG
    - output_dir/annotations.coco.json — COCO format annotations
    - output_dir/config.json — serialized GenerationConfig
    - output_dir/metadata.jsonl — one JSON line per image with metadata

    Returns path to annotations.coco.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    # Save config
    config_file = output_dir / "config.json"
    config_file.write_text(config.model_dump_json(indent=2))

    metadata_file = output_dir / "metadata.jsonl"

    from typing import Any

    coco_data: dict[str, Any] = {
        "info": {
            "description": "SynthLine AI generated dataset",
            "version": "0.1.0",
            "date_created": datetime.now(UTC).isoformat(),
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }


    category_map: dict[str, int] = {}

    with open(metadata_file, "w") as f_meta:
        for idx, res in enumerate(results, start=1):
            def_type = (
                res.defect_type.value if hasattr(res.defect_type, "value") else str(res.defect_type)
            )
            if def_type not in category_map:
                cat_id = len(category_map) + 1
                category_map[def_type] = cat_id
                coco_data["categories"].append(
                    {"id": cat_id, "name": def_type, "supercategory": "defect"}
                )
            else:
                cat_id = category_map[def_type]

            img_name = f"image_{idx:04d}_{def_type}.png"
            mask_name = f"mask_{idx:04d}_{def_type}.png"

            img_path = images_dir / img_name
            mask_path = masks_dir / mask_name

            # Save images and masks
            cv2.imwrite(str(img_path), res.image)
            cv2.imwrite(str(mask_path), res.mask)

            h, w = res.image.shape[:2]

            coco_data["images"].append(
                {"id": idx, "file_name": img_name, "width": int(w), "height": int(h)}
            )

            bbox = mask_to_bbox(res.mask)
            area = mask_area(res.mask)
            poly = mask_to_polygon(res.mask)

            coco_data["annotations"].append(
                {
                    "id": idx,
                    "image_id": idx,
                    "category_id": cat_id,
                    "bbox": list(bbox),
                    "area": area,
                    "segmentation": poly,
                    "iscrowd": 0,
                }
            )

            meta_line = {
                "image": img_name,
                "source_seed": res.source_seed,
                "defect_type": def_type,
                "random_seed": res.random_seed,
            }
            meta_line.update(res.metadata)
            f_meta.write(json.dumps(meta_line) + "\n")

    anno_file = output_dir / "annotations.coco.json"
    with open(anno_file, "w") as f:
        json.dump(coco_data, f, indent=2)

    return anno_file
