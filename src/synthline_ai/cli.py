"""CLI entry point for SynthLine AI."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from synthline_ai import __version__
from synthline_ai.config.models import (
    DefectType,
    ExportFormat,
    GenerationConfig,
    SplitRatio,
)
from synthline_ai.export.trainers import export_training_boilerplates
from synthline_ai.generation.pipeline import run_generation
from synthline_ai.ingestion.loader import load_seeds
from synthline_ai.ingestion.quality import check_seed_quality
from synthline_ai.labeling.export import export_coco, export_voc, export_yolo
from synthline_ai.labeling.fusion import merge_datasets
from synthline_ai.recipes.models import GenerationRecipe
from synthline_ai.recipes.presets import get_preset, list_presets
from synthline_ai.validation.checks import (
    check_brightness_distribution,
    check_output_duplicates,
    check_split_leakage,
    validate_results,
)
from synthline_ai.validation.heatmap import generate_defect_heatmap
from synthline_ai.validation.html_preview import generate_html_preview
from synthline_ai.validation.previews import create_contact_sheet
from synthline_ai.validation.statistics import compute_split_statistics, compute_statistics

app = typer.Typer(
    name="synthline-ai",
    help="Synthetic visual-data generation for computer vision.",
    no_args_is_help=True,
)

recipe_app = typer.Typer(
    name="recipe",
    help="Industry domain recipe presets and declarative workflows.",
    no_args_is_help=True,
)

dataset_app = typer.Typer(
    name="dataset",
    help="Dataset operations, merging, and fusion.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def generate(
    seeds: Path = typer.Option(..., help="Directory of seed images"),
    defect: str = typer.Option(
        "scratch",
        help="Defect type (scratch, stain, discoloration, crack, pinhole, dent, mixed)",
    ),
    count: int = typer.Option(100, min=1, max=10000, help="Number of images to generate"),
    output: Path = typer.Option(..., help="Output directory"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    severity: float = typer.Option(0.5, min=0.0, max=1.0, help="Defect severity (0.0-1.0)"),
    frequency: float = typer.Option(1.0, min=0.1, max=5.0, help="Defect frequency/density"),
    format: str = typer.Option("coco", help="Export format (coco, yolo, voc, all)"),
    split: bool = typer.Option(False, help="Enable train/val/test dataset partitioning"),
    train_ratio: float = typer.Option(0.7, min=0.0, max=1.0, help="Train split proportion"),
    val_ratio: float = typer.Option(0.2, min=0.0, max=1.0, help="Validation split proportion"),
    test_ratio: float = typer.Option(0.1, min=0.0, max=1.0, help="Test split proportion"),
    variations: bool = typer.Option(False, help="Enable surface environmental variations"),
    lighting: float = typer.Option(0.2, min=0.0, max=1.0, help="Lighting variation intensity"),
    texture: float = typer.Option(0.15, min=0.0, max=1.0, help="Texture variation intensity"),
    geometry: float = typer.Option(0.5, min=0.0, max=1.0, help="Geometric affine jitter intensity"),
    sensor: float = typer.Option(0.1, min=0.0, max=1.0, help="Sensor noise intensity"),
    compound: bool = typer.Option(
        False, help="Enable compound multi-defect generation per workpiece"
    ),
    defects_per_image: int = typer.Option(
        1, min=1, max=5, help="Number of defect instances per workpiece (1-5)"
    ),
    auto_roi: bool = typer.Option(
        False, help="Constrain defects strictly within workpiece boundary"
    ),
    preset: str | None = typer.Option(
        None,
        "--preset",
        help="Industry domain preset to load parameters from",
    ),
) -> None:
    """Generate synthetic defect images from seed images."""
    start_time = time.time()

    console.print(f"\n[bold]SynthLine AI[/bold] v{__version__}\n")

    # Apply preset defaults if specified
    if preset is not None:
        try:
            recipe = get_preset(preset)
            console.print(
                f"[cyan]Applied Preset:[/cyan] [bold]{recipe.title}[/bold] ({recipe.domain})"
            )
        except KeyError as err:
            console.print(f"[red]Error:[/red] {err}")
            raise typer.Exit(code=1) from None

        if defect == "scratch" and recipe.defect_type:
            defect = recipe.defect_type.value
        if severity == 0.5:
            severity = recipe.severity
        if frequency == 1.0:
            frequency = recipe.frequency
        if not variations and recipe.enable_variations:
            variations = recipe.enable_variations
            lighting = recipe.lighting_intensity
            texture = recipe.texture_intensity
            geometry = recipe.geometry_intensity
            sensor = recipe.sensor_intensity
        if not compound and recipe.compound_defects:
            compound = recipe.compound_defects
            defects_per_image = recipe.defects_per_image
        if not auto_roi and recipe.auto_roi:
            auto_roi = recipe.auto_roi
        if format == "coco" and recipe.export_format:
            format = recipe.export_format.value
        if not split and recipe.enable_split:
            split = recipe.enable_split
            train_ratio = recipe.split_ratio.train
            val_ratio = recipe.split_ratio.val
            test_ratio = recipe.split_ratio.test

    # Validate defect type
    try:
        defect_type = DefectType(defect)
    except ValueError:
        console.print(f"[red]Error:[/red] Unknown defect type '{defect}'.")
        console.print(f"Available types: {[d.value for d in DefectType]}")
        raise typer.Exit(code=1) from None

    # Validate export format
    try:
        export_fmt = ExportFormat(format.lower())
    except ValueError:
        console.print(f"[red]Error:[/red] Unknown export format '{format}'.")
        console.print(f"Available formats: {[f.value for f in ExportFormat]}")
        raise typer.Exit(code=1) from None

    split_config = SplitRatio(train=train_ratio, val=val_ratio, test=test_ratio)

    config = GenerationConfig(
        seeds_dir=seeds,
        defect_type=defect_type,
        count=count,
        output_dir=output,
        random_seed=seed,
        severity=severity,
        frequency=frequency,
        enable_split=split,
        split_ratio=split_config,
        export_format=export_fmt,
        enable_variations=variations,
        lighting_intensity=lighting if variations else 0.0,
        texture_intensity=texture if variations else 0.0,
        geometry_intensity=geometry if variations else 0.0,
        sensor_intensity=sensor if variations else 0.0,
        compound_defects=compound,
        defects_per_image=defects_per_image,
        auto_roi=auto_roi,
    )

    # Step 1: Load seeds
    console.print(f"[bold]Seeds:[/bold]     Loading from {seeds}")
    try:
        image_infos, arrays, load_warnings = load_seeds(seeds)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    # Step 2: Quality checks
    seed_report = check_seed_quality(image_infos, arrays)
    console.print(f"           {len(image_infos)} images loaded")
    if seed_report.warnings or load_warnings:
        total_warnings = len(seed_report.warnings) + len(load_warnings)
        console.print(f"           [yellow]{total_warnings} warning(s)[/yellow]")
        for w in load_warnings:
            console.print(f"             - {w.path.name}: {w.message}")
        for w in seed_report.warnings:
            console.print(f"             - {w.path.name}: {w.message}")

    console.print(f"[bold]Generator:[/bold] {defect} (severity={severity}, frequency={frequency})")
    console.print(f"[bold]Format:[/bold]    {export_fmt.value} (split={'yes' if split else 'no'})")
    console.print(f"[bold]Output:[/bold]    {output}\n")

    # Step 3: Generate
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating", total=count)
        results = run_generation(config, arrays, image_infos)
        progress.update(task, completed=count)

    # Step 4: Export Datasets
    console.print("\nExporting dataset...")
    coco_path: Path | None = None
    yolo_path: Path | None = None
    voc_path: Path | None = None

    if export_fmt in (ExportFormat.COCO, ExportFormat.ALL):
        coco_path = export_coco(results, output, config)
    if export_fmt in (ExportFormat.YOLO, ExportFormat.ALL):
        yolo_path = export_yolo(results, output, config)
    if export_fmt in (ExportFormat.VOC, ExportFormat.ALL):
        voc_path = export_voc(results, output, config)

    # Step 5: Contact sheet, Spatial Heatmap & Interactive HTML preview
    contact_path = output / "contact-sheet.jpg"
    create_contact_sheet(results, contact_path, max_samples=16)
    heatmap_path = output / "heatmap.png"
    _, spatial_metrics = generate_defect_heatmap(results, heatmap_path)
    html_preview_path = output / "preview.html"
    title = f"SynthLine AI — {defect.capitalize()} Dataset"
    generate_html_preview(results, html_preview_path, title=title)

    # Step 6: ML Training Scaffolding (PyTorch Dataset & Ultralytics YOLO)
    export_training_boilerplates(output, config)

    # Step 7: Statistics and validation
    stats = compute_statistics(results)
    validation = validate_results(results)
    dup_check = check_output_duplicates(results)
    brightness_check = check_brightness_distribution(results)
    leakage_check = check_split_leakage(results)
    split_stats = compute_split_statistics(results) if split else {}

    # Write report
    report: dict[str, object] = {
        "statistics": stats,
        "validation": validation,
        "output_duplicates": dup_check,
        "brightness_distribution": brightness_check,
        "split_leakage": leakage_check,
        "spatial_distribution": spatial_metrics,
    }
    if split_stats:
        report["split_statistics"] = split_stats
    report_path = output / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary
    elapsed = time.time() - start_time

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("Images", str(stats["total_images"]))
    table.add_row("Valid masks", str(validation["valid_count"]))
    table.add_row("Empty masks", str(stats["empty_mask_count"]))

    mask_stats = stats.get("mask_area_stats", {})
    if isinstance(mask_stats, dict):
        mean_area = mask_stats.get("mean", 0.0)
        table.add_row("Avg mask area", f"{mean_area:.1f}%")

    if coco_path:
        table.add_row("COCO annotations", str(coco_path))
    if yolo_path:
        table.add_row("YOLO data.yaml", str(yolo_path))
    if voc_path:
        table.add_row("Pascal VOC XML", str(voc_path))
    if config.auto_roi:
        table.add_row("Workpiece ROI", "[green]Enforced[/green]")

    dup_pairs = dup_check.get("duplicate_pairs", 0)
    if isinstance(dup_pairs, int) and dup_pairs > 0:
        table.add_row("Output duplicates", f"[yellow]{dup_pairs} pair(s)[/yellow]")

    if leakage_check.get("has_leakage"):
        table.add_row("Split leakage", "[red]DETECTED[/red]")

    outlier_count = brightness_check.get("outlier_count", 0)
    if isinstance(outlier_count, int) and outlier_count > 0:
        table.add_row("Brightness outliers", f"[yellow]{outlier_count}[/yellow]")

    cov = spatial_metrics.get("coverage_ratio", 0.0) * 100
    table.add_row("Surface coverage", f"{cov:.1f}%")
    dispersion = spatial_metrics.get("spatial_entropy", 0.0)
    table.add_row("Spatial dispersion", f"{dispersion:.2f} / 1.0")

    table.add_row("Contact sheet", str(contact_path))
    table.add_row("Spatial heatmap", str(heatmap_path))
    table.add_row("HTML preview", str(html_preview_path))
    table.add_row("Report", str(report_path))

    console.print("\n[bold green]Results:[/bold green]")
    console.print(table)
    console.print(f"\n[dim]Done in {elapsed:.1f}s[/dim]\n")


@app.command()
def info() -> None:
    """Show version and available generators."""
    from synthline_ai.generation.registry import available_generators

    console.print(f"\n[bold]SynthLine AI[/bold] v{__version__}")
    console.print(f"[bold]Available generators:[/bold] {', '.join(available_generators())}\n")


@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", help="Host interface to bind to"),
    port: int = typer.Option(8000, help="Port to run the studio server on"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    """Launch the SynthLine AI Local Studio web interface."""
    import uvicorn

    console.print(f"\n[bold]SynthLine AI[/bold] Studio v{__version__}")
    console.print(f"Starting server at [link=http://{host}:{port}]http://{host}:{port}[/link]\n")
    uvicorn.run("synthline_ai.web.app:app", host=host, port=port, reload=reload)


@app.command()
def probe(
    run_dir: Path = typer.Option(..., help="Path to a completed generation run directory"),
    real_dir: Path = typer.Option(
        None, help="Directory of real labeled images (with masks/ subdir)"
    ),
    seed: int = typer.Option(42, help="Random seed for probe model training"),
    compare_baselines: bool = typer.Option(
        False,
        "--compare-baselines",
        help="Compare Real, Synthetic, and Augmented models on held-out real data",
    ),
) -> None:
    """Evaluate synthetic data quality with an optional probe-model sim-to-real test."""
    import cv2

    from synthline_ai.generation.base import GenerationResult
    from synthline_ai.validation.probe_models import (
        compare_synthetic_vs_real_baselines,
        evaluate_dataset_quality,
    )

    console.print(f"\n[bold]SynthLine AI[/bold] Probe Evaluation v{__version__}\n")

    # Load generated images and masks
    images_dir = run_dir / "images"
    masks_dir = run_dir / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        console.print("[red]Error:[/red] Run directory must contain images/ and masks/ subdirs.")
        raise typer.Exit(code=1)

    results: list[GenerationResult] = []
    for img_path in sorted(images_dir.iterdir()):
        mask_path = masks_dir / img_path.name
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        if img is not None and mask is not None:
            results.append(GenerationResult(image=img, mask=mask))

    console.print(f"Loaded [bold]{len(results)}[/bold] synthetic image-mask pairs")

    # Load real data if provided
    real_images_list = None
    real_masks_list = None
    real_labels_list = None
    if real_dir and real_dir.exists():
        real_images_list = []
        real_masks_list = []
        real_labels_list = []
        real_masks_path = real_dir / "masks"
        for img_path in sorted(real_dir.iterdir()):
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            real_mask_path: Path | None = (
                real_masks_path / img_path.name if real_masks_path.exists() else None
            )
            if real_mask_path is not None and real_mask_path.exists():
                mask = cv2.imread(str(real_mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    real_images_list.append(img)
                    real_masks_list.append(mask)
                    label = 1 if mask.max() > 0 else 0
                    real_labels_list.append(label)
        console.print(f"Loaded [bold]{len(real_images_list)}[/bold] real image-mask pairs")

    metrics = evaluate_dataset_quality(
        results,
        real_images=real_images_list,
        real_masks=real_masks_list,
        real_labels=real_labels_list,
        random_seed=seed,
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Feature diversity", f"{metrics.get('feature_diversity', 0.0):.4f}")
    table.add_row("Intra-class variance", f"{metrics.get('intra_class_variance', 0.0):.4f}")

    if "synthetic_cv_accuracy" in metrics:
        table.add_row("Synthetic CV accuracy", f"{metrics['synthetic_cv_accuracy']:.3f}")
        table.add_row("Real accuracy", f"{metrics['real_accuracy']:.3f}")
        table.add_row("Real F1", f"{metrics['real_f1']:.3f}")
        gap = metrics.get("sim_to_real_gap", 0.0)
        gap_color = "green" if isinstance(gap, float) and abs(gap) < 0.1 else "yellow"
        table.add_row("Sim-to-real gap", f"[{gap_color}]{gap:.3f}[/{gap_color}]")

    if "probe_error" in metrics:
        table.add_row("Probe status", f"[yellow]{metrics['probe_error']}[/yellow]")

    console.print("\n[bold green]Probe Results:[/bold green]")
    console.print(table)

    # Baseline comparison if requested
    if compare_baselines and real_images_list and len(real_images_list) >= 4:
        syn_imgs = [r.image for r in results if r.image is not None and r.mask is not None]
        syn_msks = [r.mask for r in results if r.image is not None and r.mask is not None]
        syn_lbls = [1 if (m > 0).any() else 0 for m in syn_msks]

        try:
            comp = compare_synthetic_vs_real_baselines(
                synthetic_images=syn_imgs,
                synthetic_masks=syn_msks,
                synthetic_labels=syn_lbls,
                real_images=real_images_list,
                real_masks=real_masks_list,
                real_labels=real_labels_list,
                random_seed=seed,
            )
            metrics["baseline_comparison"] = comp

            comp_table = Table(
                title="Baseline Comparison (Held-out Real Test)",
                box=None,
                padding=(0, 2),
            )
            comp_table.add_column("Model Baseline", style="bold")
            comp_table.add_column("Accuracy")
            comp_table.add_column("Precision")
            comp_table.add_column("Recall")
            comp_table.add_column("F1 Score")

            r_m = comp.get("real_only")
            s_m = comp.get("synthetic_only")
            a_m = comp.get("augmented")

            if isinstance(r_m, dict) and isinstance(s_m, dict) and isinstance(a_m, dict):
                comp_table.add_row(
                    "Real-Only Baseline",
                    f"{float(r_m['accuracy']):.3f}",
                    f"{float(r_m['precision']):.3f}",
                    f"{float(r_m['recall']):.3f}",
                    f"{float(r_m['f1']):.3f}",
                )
                comp_table.add_row(
                    "Synthetic-Only",
                    f"{float(s_m['accuracy']):.3f}",
                    f"{float(s_m['precision']):.3f}",
                    f"{float(s_m['recall']):.3f}",
                    f"{float(s_m['f1']):.3f}",
                )
                comp_table.add_row(
                    "Real + Synthetic (Augmented)",
                    f"{float(a_m['accuracy']):.3f}",
                    f"{float(a_m['precision']):.3f}",
                    f"{float(a_m['recall']):.3f}",
                    f"[bold green]{float(a_m['f1']):.3f}[/bold green]",
                )
                console.print("\n[bold green]Baseline Comparison Matrix:[/bold green]")
                console.print(comp_table)
                lift_val = comp.get("f1_lift", 0.0)
                f1_lift = float(lift_val) if isinstance(lift_val, (int, float)) else 0.0
                lift_str = f"+{f1_lift:.3f}" if f1_lift >= 0 else f"{f1_lift:.3f}"
                console.print(
                    f"Synthetic Augmentation F1 Lift: [bold green]{lift_str}[/bold green]\n"
                )
        except Exception as err:
            console.print(f"[yellow]Warning: Could not compute baseline comparison: {err}[/yellow]")

    probe_report_path = run_dir / "probe-report.json"
    with open(probe_report_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    console.print(f"\n[dim]Report saved to {probe_report_path}[/dim]\n")


@app.command()
def benchmark(
    count: int = typer.Option(20, min=1, max=500, help="Number of variants per defect"),
    output: Path = typer.Option(
        Path("./benchmark_results"), help="Directory for benchmark results"
    ),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Run the standardized SynthLine AI benchmark suite across surfaces and defects."""
    from synthline_ai.benchmarks.runner import run_benchmark_suite

    console.print(f"\n[bold]SynthLine AI[/bold] Benchmark Suite v{__version__}\n")
    console.print(f"Executing surface fixtures benchmark ({count} variants/defect)...")

    summary = run_benchmark_suite(output, per_defect_count=count, random_seed=seed)

    table = Table(title="Throughput Benchmarks", box=None, padding=(0, 2))
    table.add_column("Defect Type", style="bold")
    table.add_column("Generated")
    table.add_column("Elapsed")
    table.add_column("Throughput (img/s)", style="bold cyan")

    for defect_name, data in summary.get("throughput", {}).items():
        table.add_row(
            defect_name.capitalize(),
            str(int(data["count"])),
            f"{data['elapsed_seconds']:.2f}s",
            f"{data['images_per_second']:.1f}",
        )
    console.print(table)

    comp = summary.get("baseline_comparison", {})
    if comp and "real_only" in comp:
        comp_table = Table(
            title="\nSim-to-Real Transfer Baseline Comparison",
            box=None,
            padding=(0, 2),
        )
        comp_table.add_column("Model Baseline", style="bold")
        comp_table.add_column("Accuracy")
        comp_table.add_column("Precision")
        comp_table.add_column("Recall")
        comp_table.add_column("F1 Score")

        for label, key in [
            ("Real-Only Baseline", "real_only"),
            ("Synthetic-Only", "synthetic_only"),
            ("Real + Synthetic (Augmented)", "augmented"),
        ]:
            m = comp.get(key)
            if isinstance(m, dict):
                comp_table.add_row(
                    label,
                    f"{float(m['accuracy']):.3f}",
                    f"{float(m['precision']):.3f}",
                    f"{float(m['recall']):.3f}",
                    f"{float(m['f1']):.3f}",
                )
        console.print(comp_table)
        lift_val = comp.get("f1_lift", 0.0)
        f1_lift = float(lift_val) if isinstance(lift_val, (int, float)) else 0.0
        lift_str = f"+{f1_lift:.3f}" if f1_lift >= 0 else f"{f1_lift:.3f}"
        console.print(f"Synthetic Augmentation F1 Lift: [bold green]{lift_str}[/bold green]\n")

    report_file = output / "benchmark_report.json"
    console.print(f"[dim]Complete benchmark report saved to {report_file}[/dim]\n")


@app.command()
def export(
    run: Path = typer.Option(..., help="Path to an existing generation run directory"),
    format: str = typer.Option("coco", help="Export format (coco, yolo, all)"),
    output: Path = typer.Option(..., help="Target directory for the exported dataset"),
) -> None:
    """Export an existing generation run into standard dataset formats."""
    import cv2

    from synthline_ai.generation.base import GenerationResult

    console.print(f"\n[bold]SynthLine AI[/bold] Export v{__version__}\n")

    try:
        export_fmt = ExportFormat(format.lower())
    except ValueError:
        console.print(f"[red]Error:[/red] Unknown export format '{format}'.")
        console.print(f"Available formats: {[f.value for f in ExportFormat]}")
        raise typer.Exit(code=1) from None

    images_dir = run / "images"
    masks_dir = run / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        console.print(
            "[red]Error:[/red] Run directory must contain 'images/' and 'masks/' folders."
        )
        raise typer.Exit(code=1)

    # Load metadata if present
    meta_by_name: dict[str, dict[str, object]] = {}
    meta_path = run / "metadata.jsonl"
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    img_name = str(item.get("image", ""))
                    if img_name:
                        meta_by_name[img_name] = item
        except Exception:
            pass

    # Load config if present
    config_path = run / "config.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)
                config = GenerationConfig.model_validate(config_data)
                config.output_dir = output
                config.export_format = export_fmt
        except Exception:
            config = GenerationConfig(
                seeds_dir=run,
                output_dir=output,
                export_format=export_fmt,
            )
    else:
        config = GenerationConfig(
            seeds_dir=run,
            output_dir=output,
            export_format=export_fmt,
        )

    results: list[GenerationResult] = []
    for img_file in sorted(images_dir.iterdir()):
        if img_file.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        mask_candidates = [
            masks_dir / img_file.name,
            masks_dir / img_file.name.replace("image_", "mask_", 1),
        ]
        mask_file = next((p for p in mask_candidates if p.exists()), None)
        if mask_file is None:
            continue

        img = cv2.imread(str(img_file))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        item_meta = meta_by_name.get(img_file.name, {})
        defect_type_val = str(item_meta.get("defect_type", config.defect_type.value))
        source_seed_val = str(item_meta.get("source_seed", "unknown"))
        split_val = str(item_meta.get("split", "train"))
        rnd_val = item_meta.get("random_seed", 0)
        rnd_seed_val = rnd_val if isinstance(rnd_val, int) else 0

        results.append(
            GenerationResult(
                image=img,
                mask=mask,
                source_seed=source_seed_val,
                defect_type=defect_type_val,
                random_seed=rnd_seed_val,
                split=split_val,
                metadata=item_meta,
            )
        )

    if not results:
        console.print("[red]Error:[/red] No valid image-mask pairs found to export.")
        raise typer.Exit(code=1)

    console.print(f"Loaded [bold]{len(results)}[/bold] images from {run}")
    console.print(f"Exporting to format: [bold]{export_fmt.value}[/bold] -> {output}\n")

    coco_path = None
    yolo_path = None
    if export_fmt in (ExportFormat.COCO, ExportFormat.ALL):
        coco_path = export_coco(results, output, config)
    if export_fmt in (ExportFormat.YOLO, ExportFormat.ALL):
        yolo_path = export_yolo(results, output, config)

    # Contact sheet and HTML preview
    contact_path = output / "contact-sheet.jpg"
    create_contact_sheet(results, contact_path, max_samples=16)
    html_preview_path = output / "preview.html"
    export_title = f"SynthLine AI — {export_fmt.value.upper()} Export"
    generate_html_preview(results, html_preview_path, title=export_title)

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Exported items", str(len(results)))
    if coco_path:
        table.add_row("COCO annotations", str(coco_path))
    if yolo_path:
        table.add_row("YOLO data.yaml", str(yolo_path))
    table.add_row("Contact sheet", str(contact_path))
    table.add_row("HTML preview", str(html_preview_path))

    console.print("[bold green]Export Complete:[/bold green]")
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Recipe Subcommands
# ---------------------------------------------------------------------------


@recipe_app.command("list")
def recipe_list() -> None:
    """List all available pre-configured industry domain recipes."""
    presets = list_presets()
    table = Table(
        title="SynthLine AI — Industry Domain Presets",
        box=None,
        padding=(0, 1),
        collapse_padding=True,
    )
    table.add_column("Preset Slug", style="bold cyan", no_wrap=True)
    table.add_column("Domain", style="green")
    table.add_column("Title")
    table.add_column("Defect", style="dim")
    table.add_column("ROI")
    table.add_column("Compound")

    for p in presets:
        compound_val = (
            f"[green]Yes ({p.defects_per_image})[/green]" if p.compound_defects else "[dim]No[/dim]"
        )
        table.add_row(
            p.name,
            p.domain,
            p.title,
            p.defect_type.value,
            "[green]Yes[/green]" if p.auto_roi else "[dim]No[/dim]",
            compound_val,
        )
    console.print()
    console.print(table)
    console.print("\n[dim]Run 'synthline generate --preset <slug>' to execute a preset.[/dim]\n")


@recipe_app.command("inspect")
def recipe_inspect(
    name: str = typer.Argument(..., help="Preset name or path to recipe JSON file"),
) -> None:
    """Display the detailed configuration of a domain recipe or preset."""
    recipe: GenerationRecipe
    p = Path(name)
    if p.exists() and p.is_file():
        try:
            recipe = GenerationRecipe.load(p)
        except Exception as exc:
            console.print(f"[red]Error loading recipe file:[/red] {exc}")
            raise typer.Exit(code=1) from None
    else:
        try:
            recipe = get_preset(name)
        except KeyError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from None

    console.print(f"\n[bold]{recipe.title}[/bold] ([green]{recipe.domain}[/green])")
    if recipe.description:
        console.print(f"[italic]{recipe.description}[/italic]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Parameter", style="bold")
    table.add_column("Value")

    table.add_row("Slug", recipe.name)
    table.add_row("Defect type", recipe.defect_type.value)
    table.add_row("Severity", str(recipe.severity))
    table.add_row("Frequency", str(recipe.frequency))
    table.add_row(
        "Environmental variations",
        "[green]Enabled[/green]" if recipe.enable_variations else "[dim]Disabled[/dim]",
    )
    if recipe.enable_variations:
        table.add_row("  Lighting intensity", str(recipe.lighting_intensity))
        table.add_row("  Texture intensity", str(recipe.texture_intensity))
        table.add_row("  Geometry jitter", str(recipe.geometry_intensity))
        table.add_row("  Sensor noise", str(recipe.sensor_intensity))
    table.add_row(
        "Compound defects",
        f"[green]Yes ({recipe.defects_per_image} per image)[/green]"
        if recipe.compound_defects
        else "[dim]No[/dim]",
    )
    table.add_row(
        "Workpiece boundary (Auto ROI)",
        "[green]Enforced[/green]" if recipe.auto_roi else "[dim]Disabled[/dim]",
    )
    table.add_row("Export format", recipe.export_format.value)
    table.add_row(
        "Dataset split",
        (
            f"train={recipe.split_ratio.train}, "
            f"val={recipe.split_ratio.val}, "
            f"test={recipe.split_ratio.test}"
        )
        if recipe.enable_split
        else "[dim]Disabled[/dim]",
    )

    console.print(table)
    console.print()


@recipe_app.command("export")
def recipe_export(
    preset: str = typer.Argument(..., help="Preset slug to export"),
    output: Path = typer.Option(..., "--output", "-o", help="Target JSON file path"),
) -> None:
    """Export an industry domain preset to a JSON recipe file for custom editing."""
    try:
        recipe = get_preset(preset)
    except KeyError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    saved_path = recipe.save(output)
    console.print(f"[green]Exported recipe:[/green] {recipe.title} -> {saved_path}")


@recipe_app.command("run")
def recipe_run(
    recipe: str = typer.Argument(..., help="Preset slug or path to recipe JSON file"),
    seeds: Path = typer.Option(..., "--seeds", "-s", help="Directory of seed images"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    count: int = typer.Option(100, min=1, max=10000, help="Number of images to generate"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """Execute dataset generation directly using an industry recipe or preset."""
    rec: GenerationRecipe
    p = Path(recipe)
    if p.exists() and p.is_file():
        try:
            rec = GenerationRecipe.load(p)
        except Exception as exc:
            console.print(f"[red]Error loading recipe file:[/red] {exc}")
            raise typer.Exit(code=1) from None
    else:
        try:
            rec = get_preset(recipe)
        except KeyError as exc:
            console.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(code=1) from None

    generate(
        seeds=seeds,
        defect=rec.defect_type.value,
        count=count,
        output=output,
        seed=seed,
        severity=rec.severity,
        frequency=rec.frequency,
        format=rec.export_format.value,
        split=rec.enable_split,
        train_ratio=rec.split_ratio.train,
        val_ratio=rec.split_ratio.val,
        test_ratio=rec.split_ratio.test,
        variations=rec.enable_variations,
        lighting=rec.lighting_intensity,
        texture=rec.texture_intensity,
        geometry=rec.geometry_intensity,
        sensor=rec.sensor_intensity,
        compound=rec.compound_defects,
        defects_per_image=rec.defects_per_image,
        auto_roi=rec.auto_roi,
    )


# ---------------------------------------------------------------------------
# Dataset Fusion Subcommands
# ---------------------------------------------------------------------------


@dataset_app.command("merge")
def dataset_merge(
    runs: list[Path] = typer.Argument(..., help="Dataset run directories to merge"),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Destination directory for merged dataset"
    ),
) -> None:
    """Merge multiple generated dataset runs into a unified benchmark dataset."""
    console.print(f"\n[bold]SynthLine AI — Dataset Fusion[/bold] v{__version__}\n")
    console.print(f"Merging [bold]{len(runs)}[/bold] dataset runs -> {output}\n")

    for r in runs:
        if not r.exists() or not r.is_dir():
            console.print(f"[red]Error:[/red] Directory not found: {r}")
            raise typer.Exit(code=1)

    try:
        summary = merge_datasets(runs, output)
    except Exception as exc:
        console.print(f"[red]Fusion Error:[/red] {exc}")
        raise typer.Exit(code=1) from None

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Merged runs", str(summary["merged_runs_count"]))
    table.add_row("Total images", str(summary["total_images"]))
    table.add_row("Total annotations", str(summary["total_annotations"]))
    table.add_row("Categories", ", ".join(summary["categories"]))
    table.add_row("Output directory", summary["output_directory"])

    console.print("[bold green]Fusion Complete:[/bold green]")
    console.print(table)
    console.print()


app.add_typer(recipe_app, name="recipe")
app.add_typer(dataset_app, name="dataset")


if __name__ == "__main__":
    app()
