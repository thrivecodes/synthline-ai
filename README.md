# SynthLine

**Synthetic defect-data generation for manufacturing computer vision.** SynthLine turns a small set of real “good part” images into a larger, labeled dataset of realistic defects for training and evaluating inspection models.

> **Status:** private repository, pre-MVP. The current priority is proving the synthetic-to-real transfer loop with a reproducible procedural generation pipeline.

## Product goal

Manufacturing teams need many labeled examples of defects, but real defects are often rare, expensive to collect, and difficult to annotate. SynthLine lets an integrator or manufacturer:

1. Upload 10–50 images of a single good part.
2. Define a small defect taxonomy, such as scratches, dents, or discoloration.
3. Generate synthetic defect images with exact masks and bounding boxes.
4. Preview and validate the generated dataset.
5. Download a standard COCO-style dataset for an existing computer-vision pipeline.

SynthLine is a **data-generation tool**, not an end-to-end inspection-model hosting product.

## MVP scope

### In scope

- One part type per project.
- 2D images only.
- 10–50 real good-part seed images.
- Three to six customer-defined defect types.
- Procedural defect generation as the first and most predictable strategy.
- Basic lighting, texture, and background randomization.
- Automatic segmentation masks and bounding boxes.
- COCO-style JSON plus images and masks.
- A preview gallery and dataset-quality report.
- A lightweight validation harness with optional probe-model training.
- Reproducible generation using recorded configuration and random seeds.

### Explicitly out of scope for v1

- 3D/CAD ingestion and digital-twin simulation.
- Video or temporal defects.
- Non-visual signals such as sound, vibration, or thermal data.
- Multi-part projects in a single generation run.
- Training and hosting a customer’s production inspection model.
- Fine-grained parameter tuning beyond defect type, severity, and frequency.

## Implementation plan

The first release should be a vertical slice:

```text
Seed images
    → ingestion and QA
    → procedural defect synthesis
    → masks and bounding boxes
    → COCO export
    → visual preview and statistics
    → optional probe-model validation
```

The initial implementation should be local-first. A Python library and CLI will make the generation pipeline testable and useful before a browser UI or hosted job system is introduced.

### Recommended stack

- **Python 3.11+** for the generation and validation pipeline.
- **OpenCV, NumPy, Pillow, and scikit-image** for image processing.
- **Albumentations** for augmentations and domain randomization.
- **Pydantic** for project and generation configuration.
- **FastAPI** for the API once the core pipeline is stable.
- **Gradio or Streamlit** for the first UI prototype; a separate React/Next.js UI can follow.
- **SQLite and local filesystem** for the initial metadata and artifact store.
- A Redis-backed worker system only when generation jobs need to run asynchronously at scale.

Do not introduce microservices, Kubernetes, or cloud-specific infrastructure until the generation and evaluation loop demonstrates value.

## Generation strategies

The generation engine should expose a common strategy interface so techniques can be compared without changing ingestion, labeling, export, or validation.

### Procedural perturbation — first implementation

Procedural generation is the v1 default because it is fast, deterministic, explainable, and produces labels by construction.

Initial defect types:

- **Scratch:** curved or jagged variable-width paths blended into the source image.
- **Discoloration/stain:** irregular blurred regions with localized hue, saturation, or brightness changes.
- **Dent:** a shaded irregular or elliptical region with a brightened rim; initially experimental because realistic dents depend heavily on lighting.

Every generator should return the modified image, one or more masks, defect parameters, severity, and the random seed used.

### Future strategies

- Diffusion-based local inpainting.
- More advanced lighting-aware geometry and texture simulation.
- Customer-specific generation strategies selected from validation results.

Generative inpainting should be added only after procedural generation establishes a measurable baseline.

## System architecture

```text
                         ┌────────────────────┐
  Good-part images ─────▶│ Ingestion and QA    │
  Defect configuration ─▶│ Project management  │
                         └─────────┬──────────┘
                                   ▼
                         ┌────────────────────┐
                         │ Generation engine  │
                         │ procedural first   │
                         │ randomization      │
                         │ inpainting later   │
                         └─────────┬──────────┘
                                   ▼
                         ┌────────────────────┐
                         │ Labeling and export │
                         │ masks, boxes, COCO  │
                         └─────────┬──────────┘
                                   ▼
                         ┌────────────────────┐
                         │ Validation harness  │
                         │ reports and preview │
                         └────────────────────┘
```

### Proposed repository structure

```text
synthline/
├── pyproject.toml
├── README.md
├── src/
│   └── synthline/
│       ├── ingestion/          # image loading, validation, and QA
│       ├── generation/
│       │   ├── procedural/     # scratch, dent, discoloration generators
│       │   ├── inpainting/     # future diffusion-based edits
│       │   └── randomization/  # lighting, texture, and transforms
│       ├── labeling/           # masks, boxes, and COCO export
│       ├── validation/         # statistics, previews, and probe models
│       ├── projects/           # configuration and artifact storage
│       └── cli.py
├── api/                        # FastAPI application, after the core pipeline
├── web/                        # browser UI
├── tests/                      # unit, integration, and fixture tests
├── benchmarks/                 # public dataset fixtures and evaluations
├── docs/
└── scripts/
```

## Data model and reproducibility

A generation must preserve the configuration that produced it. At minimum, record:

- project and part type;
- source seed image for every generated image;
- defect type, severity, and parameters;
- generation method;
- random seed;
- image dimensions and artifact paths;
- dataset split.

Example metadata:

```json
{
  "image": "part_0001_scratch.png",
  "source_seed": "seed_004.png",
  "part_type": "bracket_v3",
  "generation_method": "procedural_perturbation",
  "random_seed": 884321,
  "defects": [
    {
      "type": "scratch",
      "severity": "moderate",
      "bbox": [124, 88, 212, 101],
      "mask": "masks/part_0001_scratch_mask.png"
    }
  ]
}
```

Generated images should be reproducible from the project configuration and seed. This is essential for debugging, benchmark comparisons, and customer support.

## Ingestion and seed-set QA

The ingestion layer should check and report:

- unsupported or corrupted files;
- image dimensions and color modes;
- extreme brightness, darkness, or blur;
- duplicate and near-duplicate images;
- inconsistent framing or orientation;
- whether the part can be localized reasonably.

Warnings should not block early experimentation unless an image is unusable. A QA report might identify inconsistent lighting or dimensions while still allowing the user to generate a dataset.

For the first prototype, assume the customer supplies consistently framed or cropped images. Automatic part segmentation can be added once the core generation loop is validated.

## Labels and export

Masks should be generated internally for every defect because they are the source of truth. Bounding boxes can then be derived from masks. The default export should contain:

```text
export/
├── images/
├── masks/
├── annotations.json       # COCO-style JSON
├── metadata.jsonl         # per-image generation metadata
└── report.html or report.json
```

COCO annotations should include image IDs, category IDs, bounding boxes, areas, and segmentation data. Customers should be able to request either bounding-box-only output or images with segmentation masks.

### Avoiding split leakage

Do not randomly split generated images when variants derived from the same source seed appear in both training and validation. Prefer splitting the original seed images first, then generating each dataset split from its own seed subset. If too few seeds are available, record the relationship and label the validation result as an estimate.

## Validation methodology

The central product risk is the sim-to-real gap. Visual plausibility alone is not evidence that generated data will improve a real inspection model.

### Automated dataset checks

- Empty or nearly empty masks.
- Invalid boxes and segmentation geometry.
- Defect area and severity distributions.
- Class balance and generation failure rate.
- Brightness, contrast, and texture distributions.
- Duplicate or near-duplicate outputs.

### Visual checks

Generate contact sheets showing:

- original/generated pairs;
- mask overlays;
- examples grouped by defect type and severity;
- smallest and largest generated defects.

### Probe-model checks

Where real defect examples are available, train a small fixed baseline model on generated data and evaluate it on held-out real images. Report precision, recall, and F1 separately for each defect type. Clearly distinguish dataset-health checks from evidence of transfer to real defects.

Public industrial datasets such as MVTec AD can be used for internal benchmarks. Customer images should remain customer-controlled and should not be used to train shared models without explicit consent.

## Initial CLI target

The first end-to-end prototype should support a command like:

```bash
synthline generate \
  --seeds ./data/good \
  --defect scratch \
  --count 500 \
  --output ./runs/bracket-scratch \
  --seed 12345
```

The command should produce images, masks, COCO annotations, metadata, a preview contact sheet, and a basic quality report.

## API and UI workflow

After the CLI works, expose the same library through a small API:

```text
POST   /projects
POST   /projects/{id}/seeds
POST   /projects/{id}/defects
POST   /projects/{id}/generations
GET    /generations/{id}
GET    /generations/{id}/preview
GET    /generations/{id}/report
GET    /generations/{id}/download
DELETE /projects/{id}
```

The first UI should follow a simple flow:

```text
Create project → upload seeds → define defects → generate → preview → download
```

Long-running generation should initially use a simple background task. Add Redis and a dedicated worker only when job duration or concurrent users require it.

## Development phases

### Phase 0 — technical spike

- Establish the Python package and test setup.
- Implement image loading and basic QA.
- Implement procedural scratches.
- Generate masks and bounding boxes.
- Export COCO annotations.
- Create previews and dataset statistics.

### Phase 1 — procedural MVP

- Add discoloration and experimental dents.
- Add severity and frequency controls.
- Record reproducible configurations and seeds.
- Implement seed-aware train/validation/test splits.
- Add unit and integration tests.

### Phase 2 — browser workflow

- Add project creation and image upload.
- Add defect configuration.
- Show progress and generated previews.
- Add downloadable ZIP exports and reports.

### Phase 3 — pilot readiness

- Add public benchmark fixtures.
- Add probe-model training and per-defect metrics.
- Add deletion, retention, and access controls.
- Test with two or three design partners.

### Phase 4 — post-MVP

- Diffusion-based inpainting.
- 3D/CAD ingestion.
- Multi-part projects.
- Hosted downstream model training.

## Security and privacy

Customer images may contain confidential product designs. The MVP should commit to:

- using images only for the customer’s requested generation;
- not training shared models on customer images without explicit opt-in;
- allowing project and artifact deletion;
- keeping customer artifacts isolated by project;
- documenting retention behavior before pilots begin.

Audit logging, stronger tenant isolation, and configurable retention should be added before onboarding larger manufacturers.

## Success metrics

- A first-time user can go from seed images to a downloadable dataset in under one hour.
- Generated datasets contain valid, correctly aligned labels with a low generation failure rate.
- Synthetic-trained models perform meaningfully above baseline on real defects for at least two simple defect types.
- At least one pilot partner evaluates or deploys a model trained substantially on SynthLine data within 60 days.

The most important metric is not how realistic an image looks in isolation; it is whether generated data helps detect real production defects.

## Getting started

The runtime and package setup are being established during Phase 0. The recommended starting environment is Python 3.11+ with a virtual environment and a `pyproject.toml`-managed package.

Once the initial pipeline is in place:

```bash
git clone https://github.com/ThriveCodes/synthline.git
cd synthline
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

No external API keys should be required for the procedural generation pipeline. Any future hosted or diffusion-based provider must document its credentials and data-handling implications.

## Contributing

This is an early-stage private repository. Keep changes focused on one component and add or update tests with implementation changes. Update this README when a design decision changes the architecture, scope, schema, or validation methodology.

Prefer branching and pull requests for normal development. Experimental generation changes should include example outputs and the configuration used to produce them.

## License

Private and confidential. All rights reserved. This project is not currently licensed for external use, distribution, or reproduction.

## Glossary

- **Anomaly detection:** learning normal appearance and flagging deviations rather than classifying known defect categories.
- **Domain randomization:** varying non-essential visual factors such as lighting, background, and angle to improve generalization.
- **Seed set:** the small collection of real good-part images used as the source for generation.
- **Sim-to-real gap:** the performance difference between training or evaluating with synthetic data and performance on real production images.
