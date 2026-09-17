"""Benchmark synthetic data generation throughput and sim-to-real probe evaluation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from examples.generate_sample_seeds import (  # noqa: E402
    generate_brushed_metal,
    generate_ceramic_tile,
)

from synthline_ai.config.models import (  # noqa: E402
    DefectType,
    GenerationConfig,
    ImageInfo,
)
from synthline_ai.generation.base import GenerationResult  # noqa: E402
from synthline_ai.generation.pipeline import run_generation  # noqa: E402
from synthline_ai.validation.probe_models import evaluate_dataset_quality  # noqa: E402


def run_throughput_benchmark(count: int = 50) -> dict[str, object]:
    """Measure generation speed (images per second) across all defect types."""
    results: dict[str, object] = {}

    rng = np.random.RandomState(42)
    seeds = [
        generate_brushed_metal(256, rng=rng),
        generate_ceramic_tile(256, rng=rng),
    ]
    infos = [
        ImageInfo(
            path=Path(f"seed_{i}.png"),
            width=256,
            height=256,
            channels=3,
            mean_brightness=float(np.mean(img)),
            blur_score=150.0,
        )
        for i, img in enumerate(seeds)
    ]

    for defect_type in [DefectType.SCRATCH, DefectType.STAIN, DefectType.DISCOLORATION]:
        config = GenerationConfig(
            seeds_dir=Path("."),
            output_dir=Path("."),
            defect_type=defect_type,
            count=count,
            enable_variations=True,
            lighting_intensity=0.2,
            texture_intensity=0.1,
            geometry_intensity=0.3,
        )

        start = time.perf_counter()
        gen_results = run_generation(config, seeds, infos)
        elapsed = time.perf_counter() - start

        fps = len(gen_results) / max(elapsed, 0.001)
        results[defect_type.value] = {
            "count": len(gen_results),
            "elapsed_seconds": round(elapsed, 3),
            "images_per_second": round(fps, 1),
        }
        name = defect_type.value.upper()
        print(f"[{name}] {len(gen_results)} in {elapsed:.2f}s ({fps:.1f} img/s)")

    return results


def run_probe_model_benchmark() -> dict[str, object]:
    """Benchmark probe model evaluation on synthetic dataset vs held-out simulated real dataset."""
    rng = np.random.RandomState(42)

    # Synthetic training set
    syn_results: list[GenerationResult] = []
    for i in range(20):
        img = generate_brushed_metal(128, rng=rng)
        mask = np.zeros((128, 128), dtype=np.uint8)
        if i % 2 == 1:
            # Defect
            mask[40:80, 40:80] = 255
            img[40:80, 40:80] = (img[40:80, 40:80] * 0.4).astype(np.uint8)
        syn_results.append(GenerationResult(image=img, mask=mask))

    # Real evaluation set (held-out with slightly different illumination distribution)
    real_images: list[np.ndarray] = []
    real_masks: list[np.ndarray] = []
    real_labels: list[int] = []
    for i in range(12):
        img = generate_brushed_metal(128, rng=rng)
        img = np.clip(img.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        label = 1 if i % 2 == 1 else 0
        if label == 1:
            mask[35:75, 45:85] = 255
            img[35:75, 45:85] = (img[35:75, 45:85] * 0.35).astype(np.uint8)
        real_images.append(img)
        real_masks.append(mask)
        real_labels.append(label)

    metrics = evaluate_dataset_quality(
        syn_results,
        real_images=real_images,
        real_masks=real_masks,
        real_labels=real_labels,
        random_seed=42,
    )
    print("\n[PROBE SIM-TO-REAL BENCHMARK]")
    print(f"Feature diversity: {metrics.get('feature_diversity', 0.0):.4f}")
    print(f"Synthetic CV accuracy: {metrics.get('synthetic_cv_accuracy', 0.0):.3f}")
    print(f"Real test accuracy: {metrics.get('real_accuracy', 0.0):.3f}")
    print(f"Real F1: {metrics.get('real_f1', 0.0):.3f}")
    print(f"Sim-to-real gap: {metrics.get('sim_to_real_gap', 0.0):.3f}")

    return metrics


def main() -> None:
    print("=== SynthLine AI Benchmark Suite ===\n")
    throughput = run_throughput_benchmark(count=30)
    probe_results = run_probe_model_benchmark()

    out_file = Path("benchmarks/results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "throughput": throughput,
        "probe_benchmark": probe_results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nBenchmark summary saved to {out_file.resolve()}")


if __name__ == "__main__":
    main()
