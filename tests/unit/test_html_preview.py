"""Tests for interactive HTML preview generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synthline_ai.generation.base import GenerationResult
from synthline_ai.validation.html_preview import generate_html_preview


def test_generate_html_preview_creates_file(tmp_path: Path) -> None:
    results: list[GenerationResult] = []
    for i in range(4):
        img = np.full((64, 64, 3), 120 + i * 20, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        results.append(
            GenerationResult(
                image=img,
                mask=mask,
                source_seed=f"seed_{i}.png",
                defect_type="scratch" if i % 2 == 0 else "stain",
                split="train" if i < 2 else "val",
            )
        )

    out_file = tmp_path / "preview.html"
    returned_path = generate_html_preview(results, out_file, title="Test Run Gallery")

    assert returned_path.exists()
    assert returned_path == out_file
    content = out_file.read_text(encoding="utf-8")
    assert "Test Run Gallery" in content
    assert "data:image/jpeg;base64," in content
    assert "badge-defect" in content
    assert "splitFilterGroup" in content


def test_generate_html_preview_empty(tmp_path: Path) -> None:
    out_file = tmp_path / "empty_preview.html"
    returned_path = generate_html_preview([], out_file)
    assert returned_path.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Total Generated" in content
