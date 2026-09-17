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
    assert (output_dir / "preview.html").exists()
    assert (output_dir / "report.json").exists()
    assert (output_dir / "metadata.jsonl").exists()
    assert len(list((output_dir / "images").glob("*.png"))) == 5
    assert len(list((output_dir / "masks").glob("*.png"))) == 5


def test_cli_export(tmp_path: Path) -> None:
    # First generate a run
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    for i in range(2):
        img = np.full((64, 64, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(seeds_dir / f"seed_{i}.png"), img)

    run_dir = tmp_path / "run_source"
    res_gen = runner.invoke(
        app,
        ["generate", "--seeds", str(seeds_dir), "--count", "2", "--output", str(run_dir)],
    )
    assert res_gen.exit_code == 0

    # Now test export command
    export_dir = tmp_path / "run_exported_yolo"
    res_exp = runner.invoke(
        app,
        ["export", "--run", str(run_dir), "--format", "yolo", "--output", str(export_dir)],
    )
    assert res_exp.exit_code == 0
    assert (export_dir / "yolo" / "data.yaml").exists()
    assert (export_dir / "contact-sheet.jpg").exists()
    assert (export_dir / "preview.html").exists()


def test_cli_probe(tmp_path: Path) -> None:
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    img = np.full((64, 64, 3), 140, dtype=np.uint8)
    cv2.imwrite(str(seeds_dir / "seed.png"), img)

    run_dir = tmp_path / "run_for_probe"
    res_gen = runner.invoke(
        app,
        ["generate", "--seeds", str(seeds_dir), "--count", "3", "--output", str(run_dir)],
    )
    assert res_gen.exit_code == 0

    res_probe = runner.invoke(
        app,
        ["probe", "--run-dir", str(run_dir)],
    )
    assert res_probe.exit_code == 0
    assert (run_dir / "probe-report.json").exists()


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
    import re

    result = runner.invoke(app, ["ui", "--help"], color=False)
    assert result.exit_code == 0
    clean_output = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", result.output)
    assert "Launch the SynthLine AI Local Studio" in clean_output
    assert "--host" in clean_output
    assert "--port" in clean_output


def test_cli_probe_and_benchmark_help() -> None:
    import re

    res_probe = runner.invoke(app, ["probe", "--help"], color=False)
    assert res_probe.exit_code == 0
    clean_probe = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", res_probe.output)
    assert "--compare-baselines" in clean_probe
    assert "--run-dir" in clean_probe

    res_bench = runner.invoke(app, ["benchmark", "--help"], color=False)
    assert res_bench.exit_code == 0
    clean_bench = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", res_bench.output)
    assert "benchmark suite" in clean_bench
    assert "--count" in clean_bench


def test_cli_benchmark_execution(tmp_path: Path) -> None:
    bench_dir = tmp_path / "bench_out"
    result = runner.invoke(
        app,
        ["benchmark", "--count", "2", "--output", str(bench_dir), "--seed", "42"],
    )
    assert result.exit_code == 0
    assert (bench_dir / "benchmark_report.json").exists()


def test_cli_recipe_list() -> None:
    result = runner.invoke(app, ["recipe", "list"])
    assert result.exit_code == 0
    assert "automotive_stamping" in result.output
    assert "semiconductor_wafer" in result.output
    assert "pcb_electronics" in result.output


def test_cli_recipe_inspect() -> None:
    result = runner.invoke(app, ["recipe", "inspect", "automotive_stamping"])
    assert result.exit_code == 0
    assert "Automotive Sheet Metal Stamping" in result.output
    assert "automotive" in result.output
    assert "Severity" in result.output


def test_cli_recipe_export_and_inspect_file(tmp_path: Path) -> None:
    out_file = tmp_path / "custom_recipe.json"
    res_exp = runner.invoke(
        app,
        ["recipe", "export", "pcb_electronics", "--output", str(out_file)],
    )
    assert res_exp.exit_code == 0
    assert out_file.exists()

    res_insp = runner.invoke(app, ["recipe", "inspect", str(out_file)])
    assert res_insp.exit_code == 0
    assert "PCB & SMT Surface Mount" in res_insp.output


def test_cli_generate_with_preset(tmp_path: Path) -> None:
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    for i in range(2):
        img = np.full((64, 64, 3), 130 + i * 20, dtype=np.uint8)
        cv2.imwrite(str(seeds_dir / f"seed_{i}.png"), img)

    out_dir = tmp_path / "preset_gen_out"
    result = runner.invoke(
        app,
        [
            "generate",
            "--seeds",
            str(seeds_dir),
            "--preset",
            "automotive_stamping",
            "--count",
            "2",
            "--output",
            str(out_dir),
            "--seed",
            "42",
        ],
    )
    assert result.exit_code == 0
    assert (out_dir / "annotations.coco.json").exists()
    assert (out_dir / "report.json").exists()


def test_cli_dataset_merge(tmp_path: Path) -> None:
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    img = np.full((64, 64, 3), 150, dtype=np.uint8)
    cv2.imwrite(str(seeds_dir / "seed.png"), img)

    run1 = tmp_path / "run_1"
    run2 = tmp_path / "run_2"
    runner.invoke(
        app,
        ["generate", "--seeds", str(seeds_dir), "--count", "2", "--output", str(run1)],
    )
    runner.invoke(
        app,
        ["generate", "--seeds", str(seeds_dir), "--count", "2", "--output", str(run2)],
    )

    fused_out = tmp_path / "fused_out"
    result = runner.invoke(
        app,
        ["dataset", "merge", str(run1), str(run2), "--output", str(fused_out)],
    )
    assert result.exit_code == 0
    assert (fused_out / "annotations.coco.json").exists()
    assert (fused_out / "fusion_summary.json").exists()
    assert len(list((fused_out / "images").glob("*.png"))) == 4
