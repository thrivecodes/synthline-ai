from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from synthline_ai.cli import app

runner = CliRunner()


def test_cli_info() -> None:
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "SynthLine AI" in result.output
    assert "scratch" in result.output
    assert "stain" in result.output
    assert "discoloration" in result.output


def test_cli_generate_e2e(tmp_path: Path) -> None:
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    for i in range(3):
        img = np.full((64, 64, 3), 150 + i * 10, dtype=np.uint8)
        cv2.imwrite(str(seeds_dir / f"seed_{i:02d}.png"), img)

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "generate",
            "--seeds",
            str(seeds_dir),
            "--count",
            "5",
            "--output",
            str(output_dir),
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0
    assert (output_dir / "annotations.coco.json").exists()
    assert (output_dir / "contact-sheet.jpg").exists()
    assert (output_dir / "report.json").exists()
    assert (output_dir / "metadata.jsonl").exists()
    assert len(list((output_dir / "images").glob("*.png"))) == 5
    assert len(list((output_dir / "masks").glob("*.png"))) == 5


def test_cli_generate_yolo_and_split(tmp_path: Path) -> None:
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    for i in range(4):
        img = np.full((64, 64, 3), 120 + i * 15, dtype=np.uint8)
        cv2.imwrite(str(seeds_dir / f"seed_{i:02d}.png"), img)

    output_dir = tmp_path / "yolo_output"
    result = runner.invoke(
        app,
        [
            "generate",
            "--seeds",
            str(seeds_dir),
            "--defect",
            "stain",
            "--count",
            "6",
            "--output",
            str(output_dir),
            "--format",
            "all",
            "--split",
            "--severity",
            "0.8",
            "--frequency",
            "1.5",
        ],
    )
    assert result.exit_code == 0
    assert (output_dir / "annotations.coco.json").exists()
    assert (output_dir / "yolo" / "data.yaml").exists()


def test_cli_invalid_defect(tmp_path: Path) -> None:
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    cv2.imwrite(str(seeds_dir / "seed.png"), np.zeros((64, 64, 3), dtype=np.uint8))

    output_dir = tmp_path / "output"
    result = runner.invoke(
        app,
        [
            "generate",
            "--seeds",
            str(seeds_dir),
            "--defect",
            "unknown_defect",
            "--output",
            str(output_dir),
        ],
    )
    assert result.exit_code != 0
    assert "Unknown defect type" in result.output


def test_cli_ui_help() -> None:
    result = runner.invoke(app, ["ui", "--help"])
    assert result.exit_code == 0
    assert "Launch the SynthLine AI Local Studio" in result.output
    assert "--host" in result.output
    assert "--port" in result.output
