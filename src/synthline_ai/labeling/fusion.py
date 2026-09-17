"""Dataset fusion and multi-run dataset merging engine."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def merge_coco_annotations(
    coco_paths: list[Path],
    image_name_maps: list[dict[str, str]],
) -> dict[str, Any]:
    """Merge multiple COCO JSON annotation files into a unified dataset structure.

    Args:
        coco_paths: List of paths to annotations.coco.json files.
        image_name_maps: List of dicts mapping original filename to destination filename.

    Returns:
        Unified COCO dictionary with re-indexed image and annotation IDs.
    """
    merged_images: list[dict[str, Any]] = []
    merged_annotations: list[dict[str, Any]] = []
    category_map: dict[str, int] = {}
    next_cat_id = 1

    next_image_id = 1
    next_anno_id = 1

    for c_path, name_map in zip(coco_paths, image_name_maps, strict=False):
        if not c_path.exists():
            continue
        data = json.loads(c_path.read_text(encoding="utf-8"))

        # Category mapping for this file
        file_cat_id_to_unified: dict[int, int] = {}
        for cat in data.get("categories", []):
            cat_name = cat["name"]
            if cat_name not in category_map:
                category_map[cat_name] = next_cat_id
                next_cat_id += 1
            file_cat_id_to_unified[cat["id"]] = category_map[cat_name]

        # Image ID mapping for this file
        file_img_id_to_unified: dict[int, int] = {}
        for img in data.get("images", []):
            orig_name = img["file_name"]
            dest_name = name_map.get(orig_name, orig_name)
            unified_img_id = next_image_id
            next_image_id += 1
            file_img_id_to_unified[img["id"]] = unified_img_id

            img_entry = dict(img)
            img_entry["id"] = unified_img_id
            img_entry["file_name"] = dest_name
            merged_images.append(img_entry)

        # Annotations re-indexing
        for anno in data.get("annotations", []):
            old_img_id = anno["image_id"]
            if old_img_id not in file_img_id_to_unified:
                continue
            old_cat_id = anno["category_id"]
            new_cat_id = file_cat_id_to_unified.get(old_cat_id, 1)

            anno_entry = dict(anno)
            anno_entry["id"] = next_anno_id
            next_anno_id += 1
            anno_entry["image_id"] = file_img_id_to_unified[old_img_id]
            anno_entry["category_id"] = new_cat_id
            merged_annotations.append(anno_entry)

    categories_list = [
        {"id": cid, "name": name, "supercategory": "defect"}
        for name, cid in sorted(category_map.items(), key=lambda item: item[1])
    ]

    return {
        "info": {
            "description": "SynthLine AI fused dataset",
            "version": "1.0",
        },
        "images": merged_images,
        "annotations": merged_annotations,
        "categories": categories_list,
    }


def merge_datasets(
    dataset_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Merge multiple generated dataset runs into a single unified benchmark dataset.

    Handles images, masks, COCO annotations, YOLO labels, and metadata.
    """
    if not dataset_dirs:
        raise ValueError("At least one source dataset directory is required for fusion.")

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    coco_paths: list[Path] = []
    image_name_maps: list[dict[str, str]] = []
    metadata_lines: list[str] = []

    total_images_copied = 0

    for run_idx, d_dir in enumerate(dataset_dirs, start=1):
        src_images = d_dir / "images"
        src_masks = d_dir / "masks"
        src_coco = d_dir / "annotations.coco.json"
        src_meta = d_dir / "metadata.jsonl"

        name_map: dict[str, str] = {}

        if src_images.exists():
            for img_file in sorted(src_images.glob("*.png")):
                dest_name = f"run{run_idx}_{img_file.name}"
                name_map[img_file.name] = dest_name
                shutil.copy2(img_file, images_dir / dest_name)
                total_images_copied += 1

                mask_orig_name = img_file.name.replace("image_", "mask_")
                mask_file = src_masks / mask_orig_name
                if mask_file.exists():
                    shutil.copy2(mask_file, masks_dir / f"run{run_idx}_{mask_orig_name}")

        image_name_maps.append(name_map)

        if src_coco.exists():
            coco_paths.append(src_coco)

        if src_meta.exists():
            for line in src_meta.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        record = json.loads(line)
                        if "image" in record and record["image"] in name_map:
                            record["image"] = name_map[record["image"]]
                        metadata_lines.append(json.dumps(record))
                    except json.JSONDecodeError:
                        metadata_lines.append(line)

    # Write unified COCO
    merged_coco = merge_coco_annotations(coco_paths, image_name_maps)
    coco_output_path = output_dir / "annotations.coco.json"
    coco_output_path.write_text(json.dumps(merged_coco, indent=2), encoding="utf-8")

    # Write unified metadata
    if metadata_lines:
        (output_dir / "metadata.jsonl").write_text(
            "\n".join(metadata_lines) + "\n", encoding="utf-8"
        )

    summary = {
        "merged_runs_count": len(dataset_dirs),
        "total_images": len(merged_coco["images"]),
        "total_annotations": len(merged_coco["annotations"]),
        "categories": [c["name"] for c in merged_coco["categories"]],
        "output_directory": str(output_dir.resolve()),
    }
    (output_dir / "fusion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
