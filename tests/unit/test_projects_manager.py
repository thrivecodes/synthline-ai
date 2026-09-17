"""Unit tests for ProjectManager filesystem storage and orchestration."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from synthline_ai.config.models import DefectType, ExportFormat
from synthline_ai.projects.manager import ProjectManager
from synthline_ai.projects.models import ProjectCreate, RunCreate


def _create_dummy_seed(path: Path) -> None:
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_project_lifecycle(tmp_path: Path) -> None:
    manager = ProjectManager(workspace_root=tmp_path)
    assert len(manager.list_projects()) == 0

    # Create project
    proj = manager.create_project(
        ProjectCreate(name="Bottle Cap Inspection", description="Caps line 4")
    )
    assert proj.name == "Bottle Cap Inspection"
    assert proj.id.startswith("proj_")
    assert proj.seeds_count == 0
    assert proj.runs_count == 0

    # List projects
    projects = manager.list_projects()
    assert len(projects) == 1
    assert projects[0].id == proj.id

    # Get project detail
    detail = manager.get_project(proj.id)
    assert detail is not None
    assert detail.id == proj.id
    assert detail.name == "Bottle Cap Inspection"
    assert len(detail.seeds) == 0

    # Non-existent project
    assert manager.get_project("proj_invalid") is None


def test_save_seed_and_quality_warnings(tmp_path: Path) -> None:
    manager = ProjectManager(workspace_root=tmp_path)
    proj = manager.create_project(ProjectCreate(name="Quality Test Project"))

    # Save a normal seed
    img = np.full((64, 64, 3), 120, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", img)
    seed_path = manager.save_seed(proj.id, "sample_1.png", encoded.tobytes())
    assert seed_path.exists()

    # Detail should reflect seed
    detail = manager.get_project(proj.id)
    assert detail is not None
    assert detail.seeds_count == 1
    assert len(detail.seeds) == 1
    assert detail.seeds[0].path.name == "sample_1.png"


def test_create_run_requires_seeds(tmp_path: Path) -> None:
    manager = ProjectManager(workspace_root=tmp_path)
    proj = manager.create_project(ProjectCreate(name="Empty Seed Project"))

    run_payload = RunCreate(
        defect_type=DefectType.SCRATCH,
        count=5,
    )
    with pytest.raises(ValueError, match="No seed images"):
        manager.create_run(proj.id, run_payload)


def test_create_run_and_retrieve(tmp_path: Path) -> None:
    manager = ProjectManager(workspace_root=tmp_path)
    proj = manager.create_project(ProjectCreate(name="Run Test Project"))

    # Upload seed
    img = np.full((64, 64, 3), 150, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", img)
    manager.save_seed(proj.id, "seed_ok.png", encoded.tobytes())

    # Trigger generation run
    run_payload = RunCreate(
        defect_type=DefectType.SCRATCH,
        count=4,
        severity=0.5,
        frequency=1.0,
        export_format=ExportFormat.ALL,
        enable_split=True,
    )
    run = manager.create_run(proj.id, run_payload)
    assert run.id.startswith("run_")
    assert run.project_id == proj.id
    assert run.count == 4
    assert run.defect_type == "scratch"
    assert run.status == "completed"

    # Verify generated artifacts on disk
    run_dir = Path(run.output_dir)
    assert (run_dir / "contact-sheet.jpg").exists()
    assert (run_dir / "preview.html").exists()
    assert (run_dir / "report.json").exists()
    assert (run_dir / "run.json").exists()
    assert (run_dir / "annotations.coco.json").exists()
    assert (run_dir / "yolo" / "data.yaml").exists()

    # Test get_run_image_path
    img_path = manager.get_run_image_path(proj.id, run.id, "contact-sheet.jpg")
    assert img_path is not None and img_path.exists()
    assert manager.get_run_image_path(proj.id, run.id, "missing_img.jpg") is None

    # Test get_seed_path and delete_seed
    seed_path = manager.get_seed_path(proj.id, "seed_ok.png")
    assert seed_path is not None and seed_path.exists()
    assert manager.get_seed_path(proj.id, "nonexistent.png") is None

    assert manager.delete_seed(proj.id, "seed_ok.png") is True
    assert manager.delete_seed(proj.id, "seed_ok.png") is False

    # Test get_run
    fetched_run = manager.get_run(proj.id, run.id)
    assert fetched_run is not None
    assert fetched_run.id == run.id
    assert fetched_run.count == 4

    # Project detail updated
    detail = manager.get_project(proj.id)
    assert detail is not None
    assert detail.runs_count == 1
    assert len(detail.runs) == 1

    # Delete project
    assert manager.delete_project(proj.id) is True
    assert manager.get_project(proj.id) is None
    assert manager.delete_project("proj_invalid") is False
