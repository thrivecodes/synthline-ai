"""Filesystem-backed project and run store."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from synthline_ai.config.models import (
    ExportFormat,
    GenerationConfig,
    ImageInfo,
    QualityWarning,
)
from synthline_ai.generation.pipeline import run_generation
from synthline_ai.ingestion.loader import load_seeds
from synthline_ai.ingestion.quality import check_seed_quality
from synthline_ai.labeling.export import export_coco, export_yolo
from synthline_ai.projects.models import (
    Project,
    ProjectCreate,
    ProjectDetail,
    Run,
    RunCreate,
)
from synthline_ai.validation.checks import validate_results
from synthline_ai.validation.previews import create_contact_sheet
from synthline_ai.validation.statistics import compute_statistics


class ProjectManager:
    """Manages project workspaces, seed storage, and generation runs."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        if workspace_root is None:
            self.root = Path.cwd() / "synthline_workspace"
        else:
            self.root = workspace_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.projects_dir = self.root / "projects"
        self.projects_dir.mkdir(exist_ok=True)

    def _project_dir(self, project_id: str) -> Path:
        return self.projects_dir / project_id

    def list_projects(self) -> list[Project]:
        projects: list[Project] = []
        for p_dir in self.projects_dir.iterdir():
            if p_dir.is_dir() and (p_dir / "project.json").exists():
                data = json.loads((p_dir / "project.json").read_text())
                seeds_dir = p_dir / "seeds"
                runs_dir = p_dir / "runs"
                seeds_cnt = len(list(seeds_dir.glob("*.*"))) if seeds_dir.exists() else 0
                runs_cnt = len(list(runs_dir.iterdir())) if runs_dir.exists() else 0
                projects.append(
                    Project(
                        id=data["id"],
                        name=data["name"],
                        description=data.get("description", ""),
                        created_at=data.get("created_at", ""),
                        seeds_count=seeds_cnt,
                        runs_count=runs_cnt,
                    )
                )
        return sorted(projects, key=lambda x: x.created_at, reverse=True)

    def create_project(self, payload: ProjectCreate) -> Project:
        p_id = f"proj_{uuid.uuid4().hex[:8]}"
        p_dir = self._project_dir(p_id)
        p_dir.mkdir(parents=True, exist_ok=True)
        (p_dir / "seeds").mkdir(exist_ok=True)
        (p_dir / "runs").mkdir(exist_ok=True)

        now = datetime.now(UTC).isoformat()
        data = {
            "id": p_id,
            "name": payload.name,
            "description": payload.description,
            "created_at": now,
        }
        (p_dir / "project.json").write_text(json.dumps(data, indent=2))
        return Project(
            id=p_id,
            name=payload.name,
            description=payload.description,
            created_at=now,
            seeds_count=0,
            runs_count=0,
        )

    def get_project(self, project_id: str) -> ProjectDetail | None:
        p_dir = self._project_dir(project_id)
        if not p_dir.exists() or not (p_dir / "project.json").exists():
            return None

        data = json.loads((p_dir / "project.json").read_text())
        seeds_dir = p_dir / "seeds"
        runs_dir = p_dir / "runs"

        # Load seed metadata if seeds exist
        seed_infos: list[ImageInfo] = []
        seed_warnings: list[QualityWarning] = []
        if seeds_dir.exists() and any(seeds_dir.iterdir()):
            try:
                seed_infos, arrays, load_warnings = load_seeds(seeds_dir)
                report = check_seed_quality(seed_infos, arrays)
                seed_warnings = list(load_warnings) + list(report.warnings)
            except Exception:
                pass

        # Load runs
        runs: list[Run] = []
        if runs_dir.exists():
            for r_dir in sorted(runs_dir.iterdir(), reverse=True):
                meta_file = r_dir / "run.json"
                if meta_file.exists():
                    try:
                        r_data = json.loads(meta_file.read_text())
                        runs.append(Run(**r_data))
                    except Exception:
                        pass

        return ProjectDetail(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            seeds_count=len(seed_infos),
            runs_count=len(runs),
            seeds=seed_infos,
            seed_warnings=seed_warnings,
            runs=runs,
        )

    def delete_project(self, project_id: str) -> bool:
        p_dir = self._project_dir(project_id)
        if p_dir.exists():
            shutil.rmtree(p_dir)
            return True
        return False

    def save_seed(self, project_id: str, filename: str, content: bytes) -> Path:
        p_dir = self._project_dir(project_id)
        if not p_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")
        seeds_dir = p_dir / "seeds"
        seeds_dir.mkdir(exist_ok=True)
        target_path = seeds_dir / filename
        target_path.write_bytes(content)
        return target_path

    def create_run(self, project_id: str, payload: RunCreate) -> Run:
        p_dir = self._project_dir(project_id)
        if not p_dir.exists():
            raise FileNotFoundError(f"Project '{project_id}' not found.")

        seeds_dir = p_dir / "seeds"
        if not seeds_dir.exists() or not any(seeds_dir.iterdir()):
            raise ValueError("No seed images uploaded in project. Please upload seeds first.")

        # Load seeds
        seed_infos, arrays, _ = load_seeds(seeds_dir)

        # Create unique run directory
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        run_output_dir = p_dir / "runs" / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

        config = GenerationConfig(
            seeds_dir=seeds_dir,
            defect_type=payload.defect_type,
            count=payload.count,
            output_dir=run_output_dir,
            random_seed=payload.random_seed,
            severity=payload.severity,
            frequency=payload.frequency,
            enable_split=payload.enable_split,
            split_ratio=payload.split_ratio,
            export_format=payload.export_format,
        )

        # Run pipeline
        results = run_generation(config, arrays, seed_infos)

        # Export dataset
        if payload.export_format in (ExportFormat.COCO, ExportFormat.ALL):
            export_coco(results, run_output_dir, config)
        if payload.export_format in (ExportFormat.YOLO, ExportFormat.ALL):
            export_yolo(results, run_output_dir, config)

        # Contact sheet
        contact_path = run_output_dir / "contact-sheet.jpg"
        create_contact_sheet(results, contact_path, max_samples=16)

        # Statistics and checks
        stats = compute_statistics(results)
        validation = validate_results(results)

        report = {
            "statistics": stats,
            "validation": validation,
        }
        (run_output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

        run_obj = Run(
            id=run_id,
            project_id=project_id,
            defect_type=payload.defect_type.value,
            count=len(results),
            severity=payload.severity,
            frequency=payload.frequency,
            created_at=datetime.now(UTC).isoformat(),
            status="completed",
            output_dir=str(run_output_dir),
            statistics=stats,
            validation=validation,
        )

        (run_output_dir / "run.json").write_text(run_obj.model_dump_json(indent=2))
        return run_obj

    def get_run(self, project_id: str, run_id: str) -> Run | None:
        p_dir = self._project_dir(project_id)
        run_file = p_dir / "runs" / run_id / "run.json"
        if not run_file.exists():
            return None
        return Run(**json.loads(run_file.read_text()))
