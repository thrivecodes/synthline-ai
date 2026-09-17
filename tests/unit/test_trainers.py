"""Unit tests for ML framework boilerplate exporters (PyTorch & Ultralytics)."""

from __future__ import annotations

from pathlib import Path

from synthline_ai.export.trainers import (
    export_pytorch_dataset,
    export_training_boilerplates,
    export_yolo_train_script,
)


def test_export_pytorch_dataset(tmp_path: Path) -> None:
    dataset_file = export_pytorch_dataset(tmp_path)
    assert dataset_file.exists()
    content = dataset_file.read_text(encoding="utf-8")
    assert "class SynthLineDataset(Dataset):" in content
    assert "def __getitem__" in content
    assert "def get_dataloader" in content


def test_export_yolo_train_script(tmp_path: Path) -> None:
    train_file = export_yolo_train_script(tmp_path)
    assert train_file.exists()
    content = train_file.read_text(encoding="utf-8")
    assert "from ultralytics import YOLO" in content
    assert "def main() -> None:" in content
    assert "data.yaml" in content


def test_export_training_boilerplates(tmp_path: Path) -> None:
    out_dir = tmp_path / "boilerplates"
    out_dir.mkdir()
    res = export_training_boilerplates(out_dir)
    assert "pytorch" in res
    assert "yolo" in res
    assert res["pytorch"].exists()
    assert res["yolo"].exists()
