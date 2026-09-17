"""Unit tests for FastAPI Local Studio endpoints."""

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from synthline_ai.projects.manager import ProjectManager
from synthline_ai.web.app import app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    isolated_manager = ProjectManager(workspace_root=tmp_path)
    monkeypatch.setattr("synthline_ai.web.app.manager", isolated_manager)
    return TestClient(app)


def test_api_info_and_ui(client: TestClient) -> None:
    # Test GET /api/info
    res = client.get("/api/info")
    assert res.status_code == 200
    data = res.json()
    assert "version" in data
    assert "scratch" in data["generators"]
    assert "stain" in data["generators"]
    assert "discoloration" in data["generators"]

    # Test GET / (UI)
    res_ui = client.get("/")
    assert res_ui.status_code == 200
    assert "SynthLine AI" in res_ui.text


def test_api_projects_crud(client: TestClient) -> None:
    # List empty
    res = client.get("/api/projects")
    assert res.status_code == 200
    assert res.json() == []

    # Create project
    res_create = client.post(
        "/api/projects",
        json={"name": "Solar Cell QA", "description": "Microcrack detection"},
    )
    assert res_create.status_code == 201
    proj = res_create.json()
    proj_id = proj["id"]
    assert proj["name"] == "Solar Cell QA"

    # Get project detail
    res_get = client.get(f"/api/projects/{proj_id}")
    assert res_get.status_code == 200
    detail = res_get.json()
    assert detail["id"] == proj_id
    assert detail["seeds_count"] == 0

    # Non-existent project
    res_404 = client.get("/api/projects/non_existent_id")
    assert res_404.status_code == 404

    # Delete project
    res_del = client.delete(f"/api/projects/{proj_id}")
    assert res_del.status_code == 200
    assert res_del.json() == {"status": "deleted"}

    res_del_404 = client.delete(f"/api/projects/{proj_id}")
    assert res_del_404.status_code == 404


def test_api_runs_and_downloads(client: TestClient) -> None:
    # Create project
    res_create = client.post(
        "/api/projects",
        json={"name": "Glass Surface QA"},
    )
    assert res_create.status_code == 201
    proj_id = res_create.json()["id"]

    # Trying to run generation without seeds should return 400
    res_no_seeds = client.post(
        f"/api/projects/{proj_id}/runs",
        json={"defect_type": "scratch", "count": 2},
    )
    assert res_no_seeds.status_code == 400

    # Upload seed image
    img = np.full((64, 64, 3), 130, dtype=np.uint8)
    _, encoded = cv2.imencode(".png", img)
    files = [("files", ("seed1.png", encoded.tobytes(), "image/png"))]
    res_upload = client.post(f"/api/projects/{proj_id}/seeds", files=files)
    assert res_upload.status_code == 200
    assert res_upload.json()["uploaded_count"] == 1

    # Trigger generation run
    res_run = client.post(
        f"/api/projects/{proj_id}/runs",
        json={
            "defect_type": "stain",
            "count": 2,
            "severity": 0.5,
            "frequency": 1.0,
            "export_format": "all",
            "enable_split": False,
        },
    )
    assert res_run.status_code == 201
    run_data = res_run.json()
    run_id = run_data["id"]
    assert run_data["count"] == 2
    assert run_data["defect_type"] == "stain"

    # Preview contact sheet
    res_preview = client.get(f"/api/projects/{proj_id}/runs/{run_id}/preview")
    assert res_preview.status_code == 200
    assert res_preview.headers["content-type"] == "image/jpeg"

    # Report json
    res_report = client.get(f"/api/projects/{proj_id}/runs/{run_id}/report")
    assert res_report.status_code == 200
    report_data = res_report.json()
    assert "statistics" in report_data
    assert "validation" in report_data

    # Download dataset ZIP
    res_zip = client.get(f"/api/projects/{proj_id}/runs/{run_id}/download")
    assert res_zip.status_code == 200
    assert res_zip.headers["content-type"] == "application/zip"
    assert len(res_zip.content) > 0

    # 404 tests for run endpoints
    assert client.get(f"/api/projects/{proj_id}/runs/bad_run/preview").status_code == 404
    assert client.get(f"/api/projects/{proj_id}/runs/bad_run/report").status_code == 404
    assert client.get(f"/api/projects/{proj_id}/runs/bad_run/download").status_code == 404
