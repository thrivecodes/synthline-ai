# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Core Package Scaffolding**: Initialized `synthline-ai` package with Python 3.11+ support, Hatchling build system, and typed API (`py.typed`).
- **Data Models & Configuration**: Config schemas for generation parameters (`GenerationConfig`), seed metadata (`ImageInfo`), defect taxonomy (`DefectType`), and quality diagnostics (`SeedSetReport`).
- **Ingestion & QA Engine**: Seed image discovery with support for PNG, JPEG, TIFF, BMP; diagnostic quality checks for brightness anomalies, Laplacian blur scores, corrupt/invalid headers, and dimension outliers.
- **Procedural Defect Engine**:
  - `BaseGenerator` abstract interface and `GenerationResult` standard contract.
  - Procedural `ScratchGenerator` with Bezier/interpolated spline paths, variable curvature, alpha-blending, anti-aliased drawing, and binary mask generation.
  - Dynamic `registry` system with auto-registration of defect generators.
  - Deterministic generation loop with reproducible pseudo-random seeds.
- **Labeling & COCO Export**:
  - Binary mask area computation and polygon extraction.
  - Automatic bounding box derivation conforming to COCO format (`[x, y, w, h]`).
  - Standardized dataset export creating `images/`, `masks/`, `annotations.coco.json`, `metadata.jsonl`, and `config.json`.
- **Validation & Observability**:
  - Dataset statistics computation (mask area coverage ratios, brightness distribution).
  - Post-generation QA validation checks (empty masks, invalid bounds, shape mismatches).
  - Visual contact sheet generation with side-by-side mask overlay blending.
- **CLI**:
  - `synthline-ai generate`: End-to-end dataset generation pipeline with Rich console formatting and progress bars.
  - `synthline-ai info`: Package inspection and registered generator listing.
- **Repository & Engineering Infrastructure**:
  - Comprehensive `.gitignore` covering Python bytecode, virtualenvs, build artifacts, test caches, and synthetic run outputs.
  - Multi-OS GitHub Actions CI matrix workflow (`ubuntu`, `windows`, `macos`) across Python 3.11, 3.12, 3.13 with Ruff, Mypy, and Pytest coverage.
  - GitHub CodeQL security analysis workflow.
  - Automated GitHub release and artifact packaging workflow.
  - GitHub Issue Forms (`bug_report.yml`, `feature_request.yml`), discussion configuration, PR template, `CONTRIBUTING.md`, and `SECURITY.md`.
  - Repository `CODEOWNERS` mapped to `@thrive-spectrexq`.
