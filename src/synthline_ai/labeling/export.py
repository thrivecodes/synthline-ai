from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from synthline_ai.config.models import GenerationConfig
from synthline_ai.generation.base import GenerationResult
from synthline_ai.labeling.boxes import mask_to_bbox, mask_to_yolo_bbox
from synthline_ai.labeling.masks import mask_area, mask_to_polygon


def export_coco(
    results: list[GenerationResult],
    output_dir: Path,
    config: GenerationConfig,
) -> Path:
    """Write generated images, masks, and COCO annotations to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    # Save config
    config_file = output_dir / "config.json"
    config_file.write_text(config.model_dump_json(indent=2))

    metadata_file = output_dir / "metadata.jsonl"

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
                "split": getattr(res, "split", "train"),
            }
            meta_line.update(res.metadata)
            f_meta.write(json.dumps(meta_line) + "\n")

    anno_file = output_dir / "annotations.coco.json"
    with open(anno_file, "w") as f:
        json.dump(coco_data, f, indent=2)

    return anno_file


def export_yolo(
    results: list[GenerationResult],
    output_dir: Path,
    config: GenerationConfig,
) -> Path:
    """Export dataset in YOLO object detection format.

    Creates:
    - output_dir/yolo/images/{train,val,test}/ (or flat images/ if splits disabled)
    - output_dir/yolo/labels/{train,val,test}/
    - output_dir/yolo/data.yaml (dataset definition)
    """
    yolo_dir = output_dir / "yolo"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    # Category mapping (0-indexed for YOLO)
    categories = sorted(
        list(
            {
                res.defect_type.value if hasattr(res.defect_type, "value") else str(res.defect_type)
                for res in results
            }
        )
    )
    class_map = {name: idx for idx, name in enumerate(categories)}

    has_splits = config.enable_split and any(
        getattr(r, "split", "") in ("train", "val", "test") for r in results
    )

    for idx, res in enumerate(results, start=1):
        def_type = (
            res.defect_type.value if hasattr(res.defect_type, "value") else str(res.defect_type)
        )
        split = getattr(res, "split", "train") if has_splits else ""

        if split:
            img_dest_dir = yolo_dir / "images" / split
            lbl_dest_dir = yolo_dir / "labels" / split
        else:
            img_dest_dir = yolo_dir / "images"
            lbl_dest_dir = yolo_dir / "labels"

        img_dest_dir.mkdir(parents=True, exist_ok=True)
        lbl_dest_dir.mkdir(parents=True, exist_ok=True)

        img_name = f"image_{idx:04d}_{def_type}.png"
        txt_name = f"image_{idx:04d}_{def_type}.txt"

        cv2.imwrite(str(img_dest_dir / img_name), res.image)

        h, w = res.image.shape[:2]
        x_c, y_c, norm_w, norm_h = mask_to_yolo_bbox(res.mask, img_width=w, img_height=h)

        class_id = class_map[def_type]
        lbl_path = lbl_dest_dir / txt_name
        with open(lbl_path, "w") as f_lbl:
            if norm_w > 0 and norm_h > 0:
                f_lbl.write(f"{class_id} {x_c:.6f} {y_c:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    # Generate data.yaml
    yaml_lines = [
        f"path: {yolo_dir.as_posix()}",
        "train: images/train" if has_splits else "train: images",
        "val: images/val" if has_splits else "val: images",
    ]
    if has_splits and any(getattr(r, "split", "") == "test" for r in results):
        yaml_lines.append("test: images/test")

    yaml_lines.append(f"nc: {len(categories)}")
    yaml_lines.append(f"names: {categories}")

    yaml_file = yolo_dir / "data.yaml"
    yaml_file.write_text("\n".join(yaml_lines) + "\n")

    return yaml_file
