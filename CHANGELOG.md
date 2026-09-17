# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Phase 4 Validation & Proof, Probe Models & Defect Variability**:
  - `validation/probe_models.py`: 16-feature computer vision extraction pipeline, lightweight gradient boosting probe classifier measuring sim-to-real performance gap against real held-out inspection samples, and dataset feature diversity/variance scoring.
  - `generation/procedural/variability.py`: `apply_defect_variability` applying per-defect HSV color shifts, opacity jitter, and morphological boundary roughness.
  - Enhanced Quality Checks & Statistics in `validation/checks.py` & `validation/statistics.py`:
    - Output image duplicate and near-duplicate detection via 64-bit dHash perceptual hashing.
    - Distribution analysis for image brightness and contrast with statistical outlier identification (>2σ).
    - Data partition leakage verification ensuring zero cross-split seed contamination.
    - Per-split summary metrics and partition balance verification (`compute_split_statistics`).
  - CLI `synthline-ai probe`: Command for evaluating generated dataset directories against real labeled reference images.
  - `deduplication.py`: 64-bit perceptual difference hashing (`dHash`) and Hamming distance analysis detecting exact and near-duplicate seed images during ingestion QA.
  - `randomization/lighting.py`: Directional illumination gradients, vignetting, and light falloff simulation matching factory line illumination shifts.
  - `randomization/texture.py`: High-frequency sensor noise and low-frequency surface roughness micro-variations.
  - `randomization/geometry.py`: Sub-pixel affine translations and micro-rotations keeping image and defect mask in strict pixel-perfect alignment.
  - CLI options: `--variations`, `--lighting`, `--texture`, `--geometry` for generation customization.
- **Phase 2 Local User Experience & Studio**:
  - `synthline-ai ui`: Local web studio server command launching FastAPI/Uvicorn interface.
  - Interactive Browser Studio UI: Adheres to SynthLine visual design tokens (warm editorial `#f5f4ef` palette, sans-serif typography, responsive sidebar/workspace grid).
  - Project Workspace Management: Creation, deletion, listing, and inspection of defect datasets via `ProjectManager`.
  - Drag-and-Drop Ingestion: Direct browser seed uploads with automatic Laplacian blur, brightness, and dimension quality checks.
  - Interactive Generation Configuration: Real-time defect selection, count, severity, frequency, dataset splitting, and export format controls.
  - Visual Contact Sheet Previews: Direct rendered contact sheets with mask overlay previews inside the studio interface.
  - 1-Click ZIP Dataset Bundles: Complete ZIP archive downloads including images, binary masks, COCO annotations, YOLO datasets (`data.yaml`), and `report.json`.
  - Comprehensive REST API endpoints (`/api/projects`, `/api/projects/{id}/seeds`, `/api/projects/{id}/runs`, `/api/projects/{id}/runs/{run_id}/download`, `/api/info`).
- **Phase 1 Procedural Defect Generators**:
  - `StainGenerator`: Organic splatter and liquid blotch generation with deformed polygonal geometry, edge-feathering Gaussian blur, natural color shifts, and precise binary masks.
  - `DiscolorationGenerator`: Thermal discoloration, oxidation patina, and color bleaching via HSV color space transforms and elliptic gradient profiles.
  - Frequency & Severity Controls: Dynamic stroke/spot density control (`frequency`) and defect strength scaling (`severity`).
- **Seed-Aware Dataset Splitting (Leak Prevention)**:
  - Source seeds are partitioned into disjoint `train`/`val`/`test` sets *before* generation variants are created, strictly preventing data leakage between splits.
- **YOLO Dataset Export**:
  - `export_yolo`: Normalized coordinate bounding boxes (`[x_center, y_center, width, height]`), partitioned image/label directories, and `data.yaml` specification.
  - CLI support for format selection (`--format coco`, `--format yolo`, `--format all`).
- **CLI Enhancements**:
  - Added CLI options: `--defect {scratch,stain,discoloration}`, `--frequency`, `--format`, `--split`, `--train-ratio`, `--val-ratio`, `--test-ratio`.
- **Phase 0 Foundation (Initial)**:
  - Core ingestion engine with Laplacian blur analysis and brightness checks.
  - Procedural scratch generator with curve interpolation.
  - COCO annotation and metadata export.
  - Contact sheet previews and dataset-level statistical validation.
  - Comprehensive `.gitignore`, GitHub CI/CD matrix workflows, CodeQL security checks, release automation, and `CODEOWNERS` mapping.
