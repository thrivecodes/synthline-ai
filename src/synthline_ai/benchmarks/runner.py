"""Benchmark execution runner evaluating generation throughput and sim-to-real metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from synthline_ai.benchmarks.fixtures import create_benchmark_suite
from synthline_ai.config.models import DefectType, GenerationConfig
from synthline_ai.generation.base import GenerationResult
from synthline_ai.generation.pipeline import run_generation
from synthline_ai.ingestion.loader import load_seeds
from synthline_ai.validation.probe_models import (
    compare_synthetic_vs_real_baselines,
    evaluate_dataset_quality,
)


def run_benchmark_suite(
    output_dir: Path,
    per_defect_count: int = 30,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Execute the full SynthLine AI benchmark suite.

    1. Scaffolds benchmark fixtures (metal, ceramic, fabric, polymer).
    2. Measures generation throughput across scratch, stain, and discoloration.
    3. Runs sim-to-real probe evaluation comparing Real-only vs Synthetic-only vs Augmented.

    Args:
        output_dir: Working directory to store benchmark artifacts.
        per_defect_count: Number of synthetic variants to generate per defect type.
        random_seed: Deterministic random seed.

    Returns:
        Structured dictionary of throughput benchmarks and sim-to-real baseline results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    bench_fixture = create_benchmark_suite(
        output_dir / "fixtures",
        image_size=128,
        random_seed=random_seed,
    )

    seeds_dir = bench_fixture["seeds"]
    real_dir = bench_fixture["real_heldout"]
    real_masks_dir = real_dir / "masks"

    # Ingest seeds
    image_infos, seed_arrays, _ = load_seeds(seeds_dir)

    # 1. Throughput Benchmarks
    throughput_results: dict[str, dict[str, float]] = {}
    all_synthetic_results: list[GenerationResult] = []

    # Include clean normal seeds as baseline class 0
    for seed_arr, seed_inf in zip(seed_arrays, image_infos, strict=False):
        empty_mask = np.zeros(seed_arr.shape[:2], dtype=np.uint8)
        all_synthetic_results.append(
            GenerationResult(
                image=seed_arr,
                mask=empty_mask,
                defect_type="normal",
                source_seed=seed_inf.path.name,
            )
        )

    for defect_type in (DefectType.SCRATCH, DefectType.STAIN, DefectType.DISCOLORATION):
        config = GenerationConfig(
            seeds_dir=seeds_dir,
            output_dir=output_dir / "temp_run",
            defect_type=defect_type,
            count=per_defect_count,
            random_seed=random_seed,
            enable_variations=True,
            lighting_intensity=0.2,
            texture_intensity=0.15,
            geometry_intensity=0.3,
        )

        start = time.perf_counter()
        results = run_generation(config, seed_arrays, image_infos)
        elapsed = time.perf_counter() - start

        fps = len(results) / max(elapsed, 0.0001)
        throughput_results[defect_type.value] = {
            "count": float(len(results)),
            "elapsed_seconds": round(elapsed, 3),
            "images_per_second": round(fps, 1),
        }
        all_synthetic_results.extend(results)

    # 2. Ingest held-out real data
    real_images: list[np.ndarray] = []
    real_masks: list[np.ndarray] = []
    real_labels: list[int] = []

    for img_path in sorted(real_dir.iterdir()):
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue
        mask_path = real_masks_dir / img_path.name
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if img is not None and mask is not None:
            real_images.append(img)
            real_masks.append(mask)
            real_labels.append(1 if (mask > 0).any() else 0)

    # 3. Overall Dataset Quality Probe
    quality_metrics = evaluate_dataset_quality(
        all_synthetic_results,
        real_images=real_images,
        real_masks=real_masks,
        real_labels=real_labels,
        random_seed=random_seed,
    )

    # 4. Compare Baselines: Real-only vs Synthetic-only vs Augmented
    syn_imgs = [
        r.image for r in all_synthetic_results if r.image is not None and r.mask is not None
    ]
    syn_msks = [r.mask for r in all_synthetic_results if r.image is not None and r.mask is not None]
    syn_lbls = [1 if (m > 0).any() else 0 for m in syn_msks]

    baseline_comparison: dict[str, Any] = {}
    try:
        baseline_comparison = compare_synthetic_vs_real_baselines(
            synthetic_images=syn_imgs,
            synthetic_masks=syn_msks,
            synthetic_labels=syn_lbls,
            real_images=real_images,
            real_masks=real_masks,
            real_labels=real_labels,
            test_ratio=0.4,
            random_seed=random_seed,
        )
    except Exception as exc:
        baseline_comparison = {"error": str(exc)}

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "total_synthetic_generated": len(all_synthetic_results),
        "total_real_heldout": len(real_images),
        "throughput": throughput_results,
        "dataset_quality": quality_metrics,
        "baseline_comparison": baseline_comparison,
    }

    report_path = output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
