from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from synthline_ai import __version__
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
            "version": __version__,
            "date_created": datetime.now(UTC).isoformat(),
        },
        "images": [],
        "annotations": [],
        "categories": [],
    }

    category_map: dict[str, int] = {}
    annotation_id = 1

    with open(metadata_file, "w") as f_meta:
        for idx, res in enumerate(results, start=1):
            def_type = (
                res.defect_type.value if hasattr(res.defect_type, "value") else str(res.defect_type)
            )

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

            if res.instances:
                for inst in res.instances:
                    inst_type = str(inst.get("defect_type", def_type))
                    if inst_type not in category_map:
                        cat_id = len(category_map) + 1
                        category_map[inst_type] = cat_id
                        coco_data["categories"].append(
                            {"id": cat_id, "name": inst_type, "supercategory": "defect"}
                        )
                    else:
                        cat_id = category_map[inst_type]

                    inst_mask = inst.get("mask")
                    if isinstance(inst_mask, np.ndarray):
                        bbox = mask_to_bbox(inst_mask)
                        area = mask_area(inst_mask)
                        poly = mask_to_polygon(inst_mask)
                    else:
                        bbox = mask_to_bbox(res.mask)
                        area = mask_area(res.mask)
                        poly = mask_to_polygon(res.mask)

                    if area > 0:
                        coco_data["annotations"].append(
                            {
                                "id": annotation_id,
                                "image_id": idx,
                                "category_id": cat_id,
                                "bbox": list(bbox),
                                "area": area,
                                "segmentation": poly,
                                "iscrowd": 0,
                            }
                        )
                        annotation_id += 1
            else:
                if def_type not in category_map:
                    cat_id = len(category_map) + 1
                    category_map[def_type] = cat_id
                    coco_data["categories"].append(
                        {"id": cat_id, "name": def_type, "supercategory": "defect"}
                    )
                else:
                    cat_id = category_map[def_type]

                bbox = mask_to_bbox(res.mask)
                area = mask_area(res.mask)
                poly = mask_to_polygon(res.mask)

                coco_data["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": idx,
                        "category_id": cat_id,
                        "bbox": list(bbox),
                        "area": area,
                        "segmentation": poly,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

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

    # Category mapping across top-level defect types and instances
    cat_set = set()
    for res in results:
        if res.instances:
            for inst in res.instances:
                cat_set.add(str(inst.get("defect_type", res.defect_type)))
        else:
            cat_set.add(
                res.defect_type.value if hasattr(res.defect_type, "value") else str(res.defect_type)
            )
    categories = sorted(list(cat_set))
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
        lbl_path = lbl_dest_dir / txt_name
        with open(lbl_path, "w") as f_lbl:
            if res.instances:
                for inst in res.instances:
                    inst_type = str(inst.get("defect_type", def_type))
                    inst_mask = inst.get("mask")
                    if isinstance(inst_mask, np.ndarray):
                        x_c, y_c, norm_w, norm_h = mask_to_yolo_bbox(
                            inst_mask, img_width=w, img_height=h
                        )
                    else:
                        x_c, y_c, norm_w, norm_h = mask_to_yolo_bbox(
                            res.mask, img_width=w, img_height=h
                        )
                    if norm_w > 0 and norm_h > 0 and inst_type in class_map:
                        class_id = class_map[inst_type]
                        f_lbl.write(f"{class_id} {x_c:.6f} {y_c:.6f} {norm_w:.6f} {norm_h:.6f}\n")
            else:
                x_c, y_c, norm_w, norm_h = mask_to_yolo_bbox(res.mask, img_width=w, img_height=h)
                if norm_w > 0 and norm_h > 0 and def_type in class_map:
                    class_id = class_map[def_type]
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


def export_voc(
    results: list[GenerationResult],
    output_dir: Path,
    config: GenerationConfig,
) -> Path:
    """Export dataset in Pascal VOC XML annotation format.

    Creates:
    - output_dir/voc/JPEGImages/
    - output_dir/voc/Annotations/
    - output_dir/voc/ImageSets/Main/{train.txt, val.txt, test.txt} (or all.txt)
    """
    voc_dir = output_dir / "voc"
    images_dir = voc_dir / "JPEGImages"
    annos_dir = voc_dir / "Annotations"
    sets_dir = voc_dir / "ImageSets" / "Main"

    images_dir.mkdir(parents=True, exist_ok=True)
    annos_dir.mkdir(parents=True, exist_ok=True)
    sets_dir.mkdir(parents=True, exist_ok=True)

    split_stems: dict[str, list[str]] = {
        "train": [],
        "val": [],
        "test": [],
        "default": [],
    }

    has_splits = config.enable_split and any(
        getattr(r, "split", "") in ("train", "val", "test") for r in results
    )

    for idx, res in enumerate(results, start=1):
        def_type = (
            res.defect_type.value if hasattr(res.defect_type, "value") else str(res.defect_type)
        )
        stem = f"image_{idx:04d}_{def_type}"
        img_name = f"{stem}.png"
        img_path = images_dir / img_name
        cv2.imwrite(str(img_path), res.image)

        split = getattr(res, "split", "train") if has_splits else "default"
        if split in split_stems:
            split_stems[split].append(stem)

        h, w = res.image.shape[:2]
        c = res.image.shape[2] if res.image.ndim == 3 else 1

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "JPEGImages"
        ET.SubElement(root, "filename").text = img_name
        ET.SubElement(root, "path").text = str(img_path.resolve())

        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "SynthLine AI"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = str(c)

        ET.SubElement(root, "segmented").text = "1"

        instances_to_write = []
        if res.instances:
            for inst in res.instances:
                inst_type = str(inst.get("defect_type", def_type))
                inst_mask = inst.get("mask")
                if isinstance(inst_mask, np.ndarray):
                    x, y, bw, bh = mask_to_bbox(inst_mask)
                else:
                    x, y, bw, bh = mask_to_bbox(res.mask)
                instances_to_write.append((inst_type, x, y, bw, bh))
        else:
            x, y, bw, bh = mask_to_bbox(res.mask)
            instances_to_write.append((def_type, x, y, bw, bh))

        for itype, x, y, bw, bh in instances_to_write:
            if bw > 0 and bh > 0:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = itype
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                bndbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(max(0, x))
                ET.SubElement(bndbox, "ymin").text = str(max(0, y))
                ET.SubElement(bndbox, "xmax").text = str(min(w, x + bw))
                ET.SubElement(bndbox, "ymax").text = str(min(h, y + bh))

        ET.indent(root, space="  ")
        tree = ET.ElementTree(root)
        tree.write(str(annos_dir / f"{stem}.xml"), encoding="utf-8", xml_declaration=True)

    # Write split lists
    if has_splits:
        for s_name in ("train", "val", "test"):
            lines = split_stems[s_name]
            if lines:
                (sets_dir / f"{s_name}.txt").write_text("\n".join(lines) + "\n")
    else:
        all_lines = split_stems["default"]
        (sets_dir / "all.txt").write_text("\n".join(all_lines) + "\n")

    return voc_dir
