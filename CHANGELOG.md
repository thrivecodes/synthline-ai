# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
