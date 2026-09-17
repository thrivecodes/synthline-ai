"""Unit tests for benchmark fixtures and runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synthline_ai.benchmarks.fixtures import (
    create_benchmark_suite,
    generate_brushed_metal,
    generate_ceramic_tile,
    generate_smooth_plastic,
    generate_woven_fabric,
)
from synthline_ai.benchmarks.runner import run_benchmark_suite


def test_benchmark_fixture_generators() -> None:
    metal = generate_brushed_metal(size=64)
    tile = generate_ceramic_tile(size=64)
    fabric = generate_woven_fabric(size=64)
    plastic = generate_smooth_plastic(size=64)

    for img in (metal, tile, fabric, plastic):
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        assert not np.isnan(img).any()


def test_create_benchmark_suite(tmp_path: Path) -> None:
    suite = create_benchmark_suite(tmp_path, image_size=64, random_seed=42)

    seeds_dir = suite["seeds"]
    real_dir = suite["real_heldout"]

    assert seeds_dir.exists()
    assert real_dir.exists()
    assert (real_dir / "masks").exists()

    seed_files = list(seeds_dir.glob("*.png"))
    assert len(seed_files) == 4

    real_files = [f for f in real_dir.glob("*.png") if f.is_file()]
    assert len(real_files) == 12


def test_run_benchmark_suite(tmp_path: Path) -> None:
    out_dir = tmp_path / "bench_run"
    summary = run_benchmark_suite(out_dir, per_defect_count=2, random_seed=42)

    assert "throughput" in summary
    assert "dataset_quality" in summary
    assert "baseline_comparison" in summary
    assert (out_dir / "benchmark_report.json").exists()

    throughput = summary["throughput"]
    assert "scratch" in throughput
    assert "stain" in throughput
    assert "discoloration" in throughput
