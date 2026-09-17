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
from synthline_ai.config.models import DefectType, GenerationConfig
from synthline_ai.generation.pipeline import run_generation
from synthline_ai.ingestion.loader import load_seeds
from synthline_ai.ingestion.quality import check_seed_quality
from synthline_ai.labeling.export import export_coco
from synthline_ai.validation.checks import validate_results
from synthline_ai.validation.previews import create_contact_sheet
from synthline_ai.validation.statistics import compute_statistics

app = typer.Typer(
    name="synthline-ai",
    help="Synthetic visual-data generation for computer vision.",
    no_args_is_help=True,
)

console = Console()


@app.command()
def generate(
    seeds: Path = typer.Option(..., help="Directory of seed images"),
    defect: str = typer.Option("scratch", help="Defect type to generate"),
    count: int = typer.Option(100, min=1, max=10000, help="Number of images to generate"),
    output: Path = typer.Option(..., help="Output directory"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    severity: float = typer.Option(0.5, min=0.0, max=1.0, help="Defect severity (0.0-1.0)"),
) -> None:
    """Generate synthetic defect images from seed images."""
    start_time = time.time()

    console.print(f"\n[bold]SynthLine AI[/bold] v{__version__}\n")

    # Validate defect type
    try:
        defect_type = DefectType(defect)
    except ValueError:
        console.print(f"[red]Error:[/red] Unknown defect type '{defect}'.")
        console.print(f"Available types: {[d.value for d in DefectType]}")
        raise typer.Exit(code=1) from None

    config = GenerationConfig(
        seeds_dir=seeds,
        defect_type=defect_type,
        count=count,
        output_dir=output,
        random_seed=seed,
        severity=severity,
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

    console.print(f"[bold]Generator:[/bold] {defect} (severity={severity})")
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

        # Run generation (currently synchronous, update progress after)
        results = run_generation(config, arrays, image_infos)
        progress.update(task, completed=count)

    # Step 4: Export COCO
    console.print("\nExporting dataset...")
    coco_path = export_coco(results, output, config)

    # Step 5: Contact sheet
    contact_path = output / "contact-sheet.jpg"
    create_contact_sheet(results, contact_path, max_samples=16)

    # Step 6: Statistics and validation
    stats = compute_statistics(results)
    validation = validate_results(results)

    # Write report
    report = {
        "statistics": stats,
        "validation": validation,
    }
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

    table.add_row("COCO annotations", str(coco_path))
    table.add_row("Contact sheet", str(contact_path))
    table.add_row("Report", str(report_path))

    console.print("\n[bold green]Results:[/bold green]")
    console.print(table)
    console.print(f"\n[dim]Done in {elapsed:.1f}s[/dim]\n")


@app.command()
def info() -> None:
    """Show version and available generators."""
    # Import registry to trigger auto-registration
    from synthline_ai.generation.registry import available_generators

    console.print(f"\n[bold]SynthLine AI[/bold] v{__version__}")
    console.print(f"[bold]Available generators:[/bold] {', '.join(available_generators())}\n")


if __name__ == "__main__":
    app()
