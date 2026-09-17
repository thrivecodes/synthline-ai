from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from synthline_ai.config.models import DefectType, GenerationConfig
from synthline_ai.generation.base import GenerationResult
from synthline_ai.labeling.export import export_voc


def test_export_voc_creates_structure(tmp_path: Path) -> None:
    results = []
    for i in range(2):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 15:45] = 255
        res = GenerationResult(
            image=img,
            mask=mask,
            metadata={"idx": i},
            source_seed=f"seed_{i}.png",
            defect_type=DefectType.SCRATCH,
            random_seed=i,
        )
        results.append(res)

    export_dir = tmp_path / "export_voc"
    config = GenerationConfig(
        seeds_dir=tmp_path / "seeds",
        defect_type=DefectType.SCRATCH,
        count=2,
        output_dir=export_dir,
        random_seed=42,
    )

    voc_dir = export_voc(results, export_dir, config)
    assert voc_dir.exists()

    jpeg_dir = voc_dir / "JPEGImages"
    annos_dir = voc_dir / "Annotations"
    sets_dir = voc_dir / "ImageSets" / "Main"

    assert jpeg_dir.exists()
    assert len(list(jpeg_dir.glob("*.png"))) == 2

    assert annos_dir.exists()
    xml_files = list(annos_dir.glob("*.xml"))
    assert len(xml_files) == 2

    # Check set text file
    assert (sets_dir / "all.txt").exists()
    lines = (sets_dir / "all.txt").read_text().splitlines()
    assert len(lines) == 2

    # Parse first XML annotation
    tree = ET.parse(xml_files[0])
    root = tree.getroot()
    assert root.tag == "annotation"

    folder = root.find("folder")
    assert folder is not None and folder.text == "JPEGImages"

    size = root.find("size")
    assert size is not None
    assert size.find("width") is not None and size.find("width").text == "64"
    assert size.find("height") is not None and size.find("height").text == "64"

    objects = root.findall("object")
    assert len(objects) == 1
    obj = objects[0]
    assert obj.find("name") is not None and obj.find("name").text == "scratch"

    bndbox = obj.find("bndbox")
    assert bndbox is not None
    assert bndbox.find("xmin").text == "15"
    assert bndbox.find("ymin").text == "10"
    assert bndbox.find("xmax").text == "45"
    assert bndbox.find("ymax").text == "30"


def test_export_voc_with_splits(tmp_path: Path) -> None:
    results = []
    splits = ["train", "val", "test"]
    for i, s in enumerate(splits):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[5:15, 5:15] = 255
        res = GenerationResult(
            image=img,
            mask=mask,
            metadata={},
            source_seed=f"seed_{i}.png",
            defect_type=DefectType.STAIN,
            random_seed=i,
            split=s,
        )
        results.append(res)

    export_dir = tmp_path / "export_voc_splits"
    config = GenerationConfig(
        seeds_dir=tmp_path / "seeds",
        defect_type=DefectType.STAIN,
        count=3,
        output_dir=export_dir,
        enable_split=True,
    )

    voc_dir = export_voc(results, export_dir, config)
    sets_dir = voc_dir / "ImageSets" / "Main"

    assert (sets_dir / "train.txt").exists()
    assert (sets_dir / "val.txt").exists()
    assert (sets_dir / "test.txt").exists()

    assert len((sets_dir / "train.txt").read_text().splitlines()) == 1
    assert len((sets_dir / "val.txt").read_text().splitlines()) == 1
    assert len((sets_dir / "test.txt").read_text().splitlines()) == 1


def test_export_voc_compound_instances(tmp_path: Path) -> None:
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask1 = np.zeros((50, 50), dtype=np.uint8)
    mask1[2:10, 2:10] = 255
    mask2 = np.zeros((50, 50), dtype=np.uint8)
    mask2[20:30, 20:30] = 255
    combined = mask1 | mask2

    res = GenerationResult(
        image=img,
        mask=combined,
        metadata={"compound": True},
        source_seed="seed.png",
        defect_type="compound",
        random_seed=1,
        instances=[
            {"defect_type": "scratch", "mask": mask1},
            {"defect_type": "dent", "mask": mask2},
        ],
    )

    export_dir = tmp_path / "export_voc_compound"
    config = GenerationConfig(
        seeds_dir=tmp_path / "seeds",
        defect_type=DefectType.MIXED,
        compound_defects=True,
        count=1,
        output_dir=export_dir,
    )

    voc_dir = export_voc([res], export_dir, config)
    xml_files = list((voc_dir / "Annotations").glob("*.xml"))
    assert len(xml_files) == 1

    tree = ET.parse(xml_files[0])
    objects = tree.getroot().findall("object")
    assert len(objects) == 2
    names = [o.find("name").text for o in objects if o.find("name") is not None]
    assert "scratch" in names
    assert "dent" in names
