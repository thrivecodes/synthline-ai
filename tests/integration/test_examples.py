"""Integration tests for examples and procedural seed generator."""

from __future__ import annotations

from pathlib import Path

from examples.generate_sample_seeds import (
    generate_brushed_metal,
    generate_ceramic_tile,
    generate_matte_polymer,
)
from examples.quickstart_pipeline import run_quickstart_demo


def test_procedural_seed_generators() -> None:
    metal = generate_brushed_metal(size=64)
    assert metal.shape == (64, 64, 3)
    assert metal.dtype.name == "uint8"

    ceramic = generate_ceramic_tile(size=64)
    assert ceramic.shape == (64, 64, 3)
    assert ceramic.dtype.name == "uint8"

    polymer = generate_matte_polymer(size=64)
    assert polymer.shape == (64, 64, 3)
    assert polymer.dtype.name == "uint8"


def test_quickstart_pipeline(tmp_path: Path) -> None:
    output_dir = tmp_path / "quickstart_test"
    result_path = run_quickstart_demo(output_dir)

    assert result_path.exists()
    assert (result_path / "annotations.coco.json").exists()
    assert (result_path / "yolo" / "data.yaml").exists()
    assert (result_path / "preview.html").exists()
    assert len(list((result_path / "images").glob("*.png"))) == 6
