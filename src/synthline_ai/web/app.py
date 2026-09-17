"""FastAPI web application and REST API for SynthLine AI local workspace."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from synthline_ai import __version__
from synthline_ai.generation.registry import available_generators
from synthline_ai.projects.manager import ProjectManager
from synthline_ai.projects.models import (
    Project,
    ProjectCreate,
    ProjectDetail,
    Run,
    RunCreate,
)

app = FastAPI(
    title="SynthLine AI",
    description="Synthetic visual-data generation for computer vision teams.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ProjectManager()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/info")
def get_info() -> dict[str, object]:
    """Retrieve version and available procedural defect generators."""
    return {
        "version": __version__,
        "generators": available_generators(),
    }


@app.get("/api/projects", response_model=list[Project])
def list_projects() -> list[Project]:
    """List all visual inspection projects in the local workspace."""
    return manager.list_projects()


@app.post("/api/projects", response_model=Project, status_code=201)
def create_project(payload: ProjectCreate) -> Project:
    """Create a new project workspace."""
    return manager.create_project(payload)


@app.get("/api/projects/{project_id}", response_model=ProjectDetail)
def get_project(project_id: str) -> ProjectDetail:
    """Retrieve full project details, seed images, and run history."""
    proj = manager.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    return proj


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict[str, str]:
    """Delete a project workspace and all associated data."""
    if not manager.delete_project(project_id):
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    return {"status": "deleted"}


@app.post("/api/projects/{project_id}/seeds")
async def upload_seeds(
    project_id: str,
    files: list[UploadFile] = File(...),
) -> dict[str, object]:
    """Upload one or more seed images to the project."""
    proj = manager.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    saved_files: list[str] = []
    for f in files:
        if not f.filename:
            continue
        content = await f.read()
        manager.save_seed(project_id, f.filename, content)
        saved_files.append(f.filename)

    # Refresh project to get quality assessment
    refreshed = manager.get_project(project_id)
    return {
        "uploaded_count": len(saved_files),
        "files": saved_files,
        "total_seeds": refreshed.seeds_count if refreshed else len(saved_files),
        "warnings": [w.model_dump() for w in (refreshed.seed_warnings if refreshed else [])],
    }


@app.post("/api/projects/{project_id}/runs", response_model=Run, status_code=201)
def create_run(project_id: str, payload: RunCreate) -> Run:
    """Execute a procedural generation run on project seeds."""
    proj = manager.get_project(project_id)
    if not proj:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")

    try:
        return manager.create_run(project_id, payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects/{project_id}/runs/{run_id}", response_model=Run)
def get_run(project_id: str, run_id: str) -> Run:
    """Get metadata and statistics of a generation run."""
    run = manager.get_run(project_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return run


@app.get("/api/projects/{project_id}/runs/{run_id}/preview")
def get_run_preview(project_id: str, run_id: str) -> FileResponse:
    """Return the contact-sheet preview image of a generation run."""
    run = manager.get_run(project_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    preview_file = Path(run.output_dir) / "contact-sheet.jpg"
    if not preview_file.exists():
        raise HTTPException(status_code=404, detail="Preview contact sheet not found.")
    return FileResponse(str(preview_file), media_type="image/jpeg")


@app.get("/api/projects/{project_id}/runs/{run_id}/report")
def get_run_report(project_id: str, run_id: str) -> FileResponse:
    """Return the validation and statistics report.json."""
    run = manager.get_run(project_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    report_file = Path(run.output_dir) / "report.json"
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Report file not found.")
    return FileResponse(str(report_file), media_type="application/json")


@app.get("/api/projects/{project_id}/runs/{run_id}/download")
def download_run_dataset(project_id: str, run_id: str) -> StreamingResponse:
    """Bundle generated images, annotations, and reports into a downloadable ZIP."""
    run = manager.get_run(project_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    run_dir = Path(run.output_dir)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run output directory not found.")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in run_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(run_dir)
                zf.write(file_path, arcname)

    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=synthline-{run_id}.zip"},
    )


# ---------------------------------------------------------------------------
# UI Workflow Application
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def serve_ui() -> HTMLResponse:
    """Serve the standalone single-page application for local browser workflows."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
