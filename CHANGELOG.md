# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] - 2026-09-17

### Added
- **Declarative Recipe Engine & Industry Domain Presets (`synthline_ai.recipes`)**:
  - Pydantic-based `GenerationRecipe` supporting JSON serialization (`to_json`, `save`, `load`) and `.to_config()` conversion.
  - 6 domain presets: `automotive_stamping`, `semiconductor_wafer`, `pcb_electronics`, `pharmaceutical_packaging`, `textile_fabric`, and `glass_optics`.
  - Preset CLI suite: `synthline recipe list`, `synthline recipe inspect <name>`, `synthline recipe export <name>`, `synthline recipe run <name>`.
  - Added `--preset <name>` flag to `synthline generate` for 1-step domain-optimized synthesis.
- **Dataset Fusion & Multi-Run Dataset Merging (`synthline_ai.labeling.fusion`)**:
  - `merge_datasets` engine combining multiple runs into a unified benchmark.
  - Global category unification, sequential re-indexing of image IDs and annotation IDs, and run-prefixed image/mask filenames preventing collisions.
  - CLI command: `synthline dataset merge <run_dirs...> --output <out_dir>`.
- **Local Studio UI Domain Preset Integration**:
  - Added "Industry Domain Preset" dropdown in the "Generate Defect" tab with instant configuration presets and contextual domain badges.
  - Added `GET /api/recipes` endpoint returning available industry presets.

## [0.2.2] - 2026-09-17

### Added
- **Defect Spatial Heatmap & Dispersion Analysis (`validation/heatmap.py`)**:
  - Accumulates defect occurrences into normalized 2D density heatmaps (`heatmap.png`) overlaid on the workpiece.
  - Computes spatial metrics including surface coverage ratio (%), weighted defect centroid coordinates, and normalized Shannon spatial dispersion entropy (0.0 to 1.0).
  - Integrated into CLI summary tables, validation reports, and Studio UI run inspection cards.
- **Zero-Boilerplate ML Framework Integration (`export/trainers.py`)**:
  - `SynthLineDataset` (`dataset_pytorch.py`): Ready-to-import PyTorch / torchvision `Dataset` and `DataLoader` supporting paired image and binary defect mask loading.
  - Ultralytics YOLO Training Script (`train_yolo.py`): Standalone training script pre-configured with `data.yaml`, image resolution, and hyperparameter flags.
- **Production OCI Docker Container & Compose Setup**:
  - Minimal multi-stage `Dockerfile` based on `python:3.12-slim` with OpenCV system libraries and non-root security.
  - Instant 1-click `docker-compose.yml` with persistent volume mappings.
  - Automated GitHub Container Registry (`ghcr.io/thrivecodes/synthline-ai`) build and push workflow.

## [0.2.1] - 2026-09-17

### Added
- **Workpiece Foreground ROI Engine (`generation/roi.py` & `--auto-roi`)**:
  - Adaptive Otsu workpiece segmentation distinguishing physical parts from background conveyor belts, inspection tables, and bench tooling.
  - Defect boundary constraint (`apply_roi_constraint`) ensuring synthetic scratches, cracks, and blemishes are strictly confined to the workpiece surface.
  - Fully integrated into generation pipeline (`config.auto_roi`), CLI (`--auto-roi`), and Studio UI.
- **Pascal VOC XML Export Format (`ExportFormat.VOC` & `labeling/export.py`)**:
  - Standard Pascal VOC format outputting `JPEGImages/`, `Annotations/` (`.xml`), and `ImageSets/Main/` partition listings (`train.txt`, `val.txt`, `test.txt`, or `all.txt`).
  - Supports single defects as well as compound multi-defect instances with complete `<object>`, `<name>`, and `<bndbox>` elements.
  - Supported across CLI (`--format voc`), Studio UI, and Python SDK.
- **Fast Package Management Documentation**:
  - Added comprehensive `uv` and `uvx` setup guides to `README_PYPI.md` for lightning-fast installs and zero-install CLI execution (`uvx synthline-ai ui`).

## [0.2.0] - 2026-09-17

### Added
- **Procedural Crack Generator (`generation/procedural/crack.py`)**:
  - Dendritic step-wise fracture paths with tortuous main fissures and randomized branching sub-cracks.
  - Dark shadow core profile with subtle outer stress halos simulating brittle fracture.
- **Procedural Pinhole & Void Generator (`generation/procedural/pinhole.py`)**:
  - Porosity voids, welding blowholes, and coating pinholes arranged in natural Poisson-like clusters.
  - Realistic depth shading with dark crater cores and bright rim highlights.
- **Procedural Dent & Impression Generator (`generation/procedural/dent.py`)**:
  - 3D surface impressions and sheet-metal dents with directional lighting shading.
  - Accurate crest specular highlights facing illumination source and trough shadow occlusion opposite the light.
- **Industrial Camera & Optical Sensor Noise (`generation/randomization/sensor.py`)**:
  - Coupled Poisson shot noise and Gaussian thermal read noise.
  - Chromatic aberration simulating lateral dispersion across RGB channels.
  - High-speed conveyor motion blur and optical lens defocusing.
- **Balanced Multi-Class Dataset Generation (`DefectType.MIXED`)**:
  - Generate balanced multi-class visual inspection datasets across all 6 defect families in a single run.
  - Dynamic COCO multi-category registration and YOLO multi-class label mappings.
- **Multi-Defect Compound Mode (`compound_defects` & `defects_per_image`)**:
  - Sequential multi-defect generation on the same workpiece (e.g. scratch + stain, crack + pinholes).
  - Multi-instance COCO segmentation exports (multiple annotation entries per image).
  - Multi-bounding box YOLO exports (multiple class lines per label text file).
- **Baseline Comparison Engine & Probe Baselines (`validation/baseline_comparison.py`)**:
  - `compare_synthetic_vs_real_baselines`: Train probe models on synthetic data, real data, and combined data to compute sim-to-real gap, augmentation gain, and statistical parity.
  - `synthline-ai probe --compare-baselines`: Direct CLI comparison command.
- **Benchmark Suite & Runner (`benchmarks/`)**:
  - `synthline-ai benchmark`: Built-in throughput benchmark evaluating generator speed (images/sec) and probe quality across all defect types.
- **Expanded Local Web Studio UI**:
  - Interactive tabs: Seeds, Generate, Runs, and System Health.
  - Direct seed image thumbnail gallery, seed deletion, and real-time upload status.
  - Defect cards for all 6 defect classes + mixed mode.
  - Sliders and toggles for compound multi-defect mode, defects per image, camera sensor noise, and affine geometric jitter.

## [0.1.0] - 2026-09-17

### Added
- **Production Packaging & PyPI Release Readiness**:
  - Full PyPI specification with standard metadata, Trove classifiers, Apache-2.0 license, and project URLs.
  - Sdist and wheel distribution packages verified clean by `twine check`.
- **Standalone Dataset Exporter (`synthline-ai export`)**:
  - CLI command converting existing generation runs into COCO JSON, YOLO format (`data.yaml`), or both, without re-executing generation.
  - Automatic mask resolution supporting both direct and prefixed naming conventions.
- **Interactive HTML Visual Inspector (`preview.html`)**:
  - Self-contained, responsive dataset inspection gallery generated with each run.
  - Interactive mode toggles (Generated Image vs. Mask Overlay), split filtering (Train, Val, Test), and defect metric badges.
- **Curated Benchmarks & Example Utilities**:
  - `examples/generate_sample_seeds.py`: Pure procedural generators for realistic industrial surfaces (brushed metal, polished ceramic, matte polymer).
  - `examples/quickstart_pipeline.py`: Complete zero-mock Python API demonstration from seed loading to export and evaluation.
  - `benchmarks/benchmark_sim2real.py`: Reproducible throughput and probe-model sim-to-real evaluation suite.
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
