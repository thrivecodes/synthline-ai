# SynthLine AI

[![PyPI version](https://img.shields.io/pypi/v/synthline-ai.svg?color=087f8c)](https://pypi.org/project/synthline-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/synthline-ai.svg)](https://pypi.org/project/synthline-ai/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/thrivecodes/synthline-ai/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-thrivecodes%2Fsynthline--ai-17212b.svg)](https://github.com/thrivecodes/synthline-ai)

> **Turn a few normal images into a labeled, reproducible computer-vision dataset.**

**SynthLine AI** creates controlled synthetic defect datasets from a small set of normal seed images. Instead of manually annotating rare factory defects or waiting months to collect real failures, SynthLine AI generates defects **with exact segmentation masks and bounding boxes derived by construction**.

---

## Key Capabilities

- **Labels by Construction**: Segmentation masks are generated natively with each defect. Bounding boxes are derived directly from the binary mask contours.
- **Procedural Defect Engines**:
  - `scratch`: Realistic polyline trajectory scratches with curvature, depth modulation, and variable thickness.
  - `stain`: Multi-layer organic fluid contours with transparency gradients and irregular boundaries.
  - `discoloration`: Surface oxidation, thermal spots, and tonal shifts with smooth radial falloff.
- **Defect Appearance Variability**: Per-defect HSV color shifts, opacity jitter, and morphological boundary roughness.
- **Surface Domain Randomization**: Directional lighting gradients, sensor noise, surface micro-roughness, and sub-pixel affine transformations.
- **Data Contamination Prevention**: Automatic train/validation/test seed-aware splitting ensuring zero cross-split seed leakage.
- **Standards-Compliant Exports**: Direct export to **COCO JSON** (`annotations.coco.json`), **YOLO** (`data.yaml` + normalized coordinates), and compact ZIP bundles.
- **Quality & Sim-to-Real Proof**:
  - Statistical distribution checks (brightness, contrast, >2σ outliers).
  - 64-bit difference perceptual hashing (`dHash`) detecting exact and near-duplicate images.
  - Built-in lightweight probe classifier (`GradientBoosting`) measuring sim-to-real gap on real inspection samples.
- **Local-First & Private**: Runs 100% offline on your machine. No mandatory cloud uploads or API keys required.
- **Interactive Visual Inspection**: Standalone HTML gallery (`preview.html`) and built-in Local Studio browser interface (`synthline-ai ui`).

---

## Installation

### Using `uv` (Recommended — 10-100x Faster)
```bash
# Direct install
uv pip install synthline-ai

# Or add to your uv project
uv add synthline-ai

# With Sim-to-Real probe model evaluation
uv add "synthline-ai[probe]"
```

### Zero-Install Instant Execution (`uvx`)
Run SynthLine AI directly without installing into your local Python environment:
```bash
# Launch interactive local studio UI immediately
uvx synthline-ai ui

# Run benchmark suite instantly
uvx synthline-ai benchmark
```

### Using Standard `pip`
```bash
# Standard installation
pip install synthline-ai

# With Sim-to-Real probe evaluation
pip install "synthline-ai[probe]"
```

---

## Quickstart CLI

### 1. Generate Synthetic Defect Dataset
```bash
synthline-ai generate \
  --seeds ./data/seeds \
  --defect scratch \
  --count 100 \
  --output ./runs/scratch_dataset \
  --format all \
  --split \
  --variations \
  --seed 42
```

Outputs produced in `./runs/scratch_dataset`:
```text
scratch_dataset/
├── images/                   # Generated images (train/val/test)
├── masks/                    # Binary segmentation masks
├── annotations.coco.json     # COCO format annotations
├── yolo/                     # YOLO dataset structure with data.yaml
├── metadata.jsonl            # Lineage tracking metadata
├── contact-sheet.jpg         # Visual overview contact sheet
├── preview.html              # Interactive inspection gallery
└── report.json               # Statistical validation report
```

### 2. Standalone Dataset Re-Export
Export an existing run into another format without regenerating images:
```bash
synthline-ai export \
  --run ./runs/scratch_dataset \
  --format yolo \
  --output ./exports/yolo_export
```

### 3. Evaluate Sim-to-Real Quality Gap
Train a lightweight probe model to benchmark your synthetic data against held-out real images:
```bash
synthline-ai probe \
  --run-dir ./runs/scratch_dataset \
  --real-dir ./data/real_val
```

### 4. Launch Local Web Studio
```bash
synthline-ai ui --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser for drag-and-drop ingestion, interactive parameter tuning, and 1-click dataset downloads.

---

## Python API Usage

```python
import cv2
from pathlib import Path
from synthline_ai.config.models import GenerationConfig, DefectType, ExportFormat
from synthline_ai.generation.pipeline import run_generation
from synthline_ai.ingestion.loader import load_seeds
from synthline_ai.labeling.export import export_coco, export_yolo
from synthline_ai.validation.html_preview import generate_html_preview

# 1. Ingest seed images
infos, arrays, warnings = load_seeds(Path("data/seeds"))

# 2. Configure generation
config = GenerationConfig(
    seeds_dir=Path("data/seeds"),
    output_dir=Path("runs/metal_inspection"),
    defect_type=DefectType.SCRATCH,
    count=50,
    severity=0.7,
    enable_variations=True,
    enable_split=True,
    export_format=ExportFormat.ALL,
    random_seed=42,
)

# 3. Generate defects with exact labels
results = run_generation(config, arrays, infos)

# 4. Export datasets and preview
output_dir = Path("runs/metal_inspection")
export_coco(results, output_dir, config)
export_yolo(results, output_dir, config)
generate_html_preview(results, output_dir / "preview.html")

print(f"Dataset generated: {len(results)} samples")
```

---

## Supported Defect Types

| Defect Type | Description | Key Controls |
| :--- | :--- | :--- |
| `scratch` | Sharp linear scratches, tool marks, and gouges | Severity (width/alpha), Frequency, Curve Jitter |
| `stain` | Fluid contamination, oil spills, and residue marks | Organic perimeter, multi-layer opacity, color tint |
| `discoloration` | Thermal hotspots, oxidation, and surface fading | Radial gradient falloff, saturation/brightness shifts |

---

## Project Information

- **Repository**: [github.com/thrivecodes/synthline-ai](https://github.com/thrivecodes/synthline-ai)
- **Documentation & Architecture**: [README.md on GitHub](https://github.com/thrivecodes/synthline-ai/blob/main/README.md)
- **Issue Tracker**: [github.com/thrivecodes/synthline-ai/issues](https://github.com/thrivecodes/synthline-ai/issues)
- **Changelog**: [CHANGELOG.md](https://github.com/thrivecodes/synthline-ai/blob/main/CHANGELOG.md)
- **License**: Apache-2.0
