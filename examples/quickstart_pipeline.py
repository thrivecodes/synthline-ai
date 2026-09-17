"""End-to-end Python programmatic workflow using SynthLine AI."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.generate_sample_seeds import generate_brushed_metal  # noqa: E402

from synthline_ai.config.models import (  # noqa: E402
    DefectType,
    ExportFormat,
    GenerationConfig,
    SplitRatio,
)
from synthline_ai.generation.pipeline import run_generation  # noqa: E402
from synthline_ai.ingestion.loader import load_seeds  # noqa: E402
from synthline_ai.ingestion.quality import check_seed_quality  # noqa: E402
from synthline_ai.labeling.export import export_coco, export_yolo  # noqa: E402
from synthline_ai.validation.checks import (  # noqa: E402
    check_brightness_distribution,
    check_output_duplicates,
    check_split_leakage,
    validate_results,
)
from synthline_ai.validation.html_preview import generate_html_preview  # noqa: E402
from synthline_ai.validation.probe_models import evaluate_dataset_quality  # noqa: E402
from synthline_ai.validation.statistics import compute_statistics  # noqa: E402


def run_quickstart_demo(output_dir: Path | None = None) -> Path:
    """Demonstrate the entire pipeline programmatically with zero external dependencies."""
    target_dir = output_dir or Path(tempfile.mkdtemp(prefix="synthline_quickstart_"))
    seeds_dir = target_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare sample seeds
    import cv2

    for i in range(3):
        img = generate_brushed_metal(size=256)
        cv2.imwrite(str(seeds_dir / f"seed_metal_{i + 1:02d}.png"), img)

    print(f"[1/6] Created 3 procedural seed images in {seeds_dir}")

    # 2. Ingest and QA
    image_infos, arrays, warnings = load_seeds(seeds_dir)
    seed_report = check_seed_quality(image_infos, arrays)
    print(f"[2/6] Ingested {len(arrays)} seeds with {len(seed_report.warnings)} warnings")

    # 3. Configure generation
    config = GenerationConfig(
        seeds_dir=seeds_dir,
        defect_type=DefectType.SCRATCH,
        count=6,
        output_dir=target_dir / "dataset",
        random_seed=42,
        severity=0.6,
        frequency=1.2,
        enable_split=True,
        split_ratio=SplitRatio(train=0.6, val=0.2, test=0.2),
        export_format=ExportFormat.ALL,
        enable_variations=True,
        lighting_intensity=0.2,
        texture_intensity=0.1,
        geometry_intensity=0.3,
    )

    # 4. Generate
    results = run_generation(config, arrays, image_infos)
    print(f"[3/6] Generated {len(results)} defect variations with environmental randomization")

    # 5. Export COCO & YOLO
    ds_dir = config.output_dir
    ds_dir.mkdir(parents=True, exist_ok=True)
    coco_path = export_coco(results, ds_dir, config)
    yolo_path = export_yolo(results, ds_dir, config)
    print(f"[4/6] Exported COCO ({coco_path.name}) and YOLO ({yolo_path.name})")

    # 6. Quality Validation & Preview
    validation = validate_results(results)
    stats = compute_statistics(results)
    dup_check = check_output_duplicates(results)
    leakage_check = check_split_leakage(results)
    bright_check = check_brightness_distribution(results)
    probe_metrics = evaluate_dataset_quality(results)

    preview_path = ds_dir / "preview.html"
    generate_html_preview(results, preview_path, title="Quickstart Dataset Gallery")
    print(
        f"[5/6] Validation: valid_masks={validation['valid_count']}/{len(results)}, "
        f"leakage={leakage_check['has_leakage']}, duplicates={dup_check['duplicate_pairs']}, "
        f"diversity={probe_metrics.get('feature_diversity', 0.0):.2f}, "
        f"outliers={bright_check['outlier_count']}, total={stats['total_images']}"
    )
    print(f"[6/6] Interactive HTML gallery saved to {preview_path}")

    return ds_dir


if __name__ == "__main__":
    out = run_quickstart_demo(Path("runs/quickstart_demo"))
    print(f"\nQuickstart demo completed successfully! Output: {out.resolve()}")
