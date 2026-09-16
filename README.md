# SynthLine

**Synthetic visual-data generation for everyone.** SynthLine turns a small set of real “good” images into a larger, labeled dataset of realistic defects, variations, and anomalies for computer-vision experimentation, training, and evaluation.

> **Status:** private repository, pre-MVP. The current priority is proving a reproducible generation and validation loop that is useful to individuals, students, researchers, developers, creators, and organizations.

## Product vision

SynthLine is a general-purpose tool for creating synthetic visual data from a small collection of real images. Anyone should be able to use it for learning, prototyping, research, dataset augmentation, or computer-vision projects without needing a large dataset or specialized simulation team.

The product will be especially valuable to companies that need repeatable, private, and scalable data generation for inspection, quality assurance, product development, robotics, security, retail, agriculture, and other visual workflows.

A typical user can:

1. Upload a small set of real “good” or normal images.
2. Define one or more defect, anomaly, or variation types.
3. Generate synthetic images with exact masks and bounding boxes.
4. Preview and validate the generated dataset.
5. Download a standard dataset for an existing computer-vision pipeline.

SynthLine is a **data-generation tool**, not an end-to-end inspection-model hosting product. It should work for individual users while offering stronger privacy, reproducibility, collaboration, and scale for organizations.

## Who it is for

### Everyone

- Students learning computer vision.
- Independent developers and makers.
- Researchers testing computer-vision ideas.
- Data scientists who need more examples for a prototype.
- Educators creating practical training datasets.
- Creators building image-analysis tools.

### Organizations and companies

- Manufacturing and quality-inspection teams.
- Machine-vision integrators and consultancies.
- Robotics and automation companies.
- Retail, logistics, and agriculture teams.
- Security and infrastructure teams.
- Product, engineering, and research groups with limited labeled data.

The general product should remain simple and accessible. Company-focused capabilities can provide larger generation limits, private storage, team workspaces, audit logs, retention controls, support, and integrations without making the basic workflow difficult for individual users.

## MVP scope

### In scope

- One image domain or project per generation run.
- 2D images only.
- A small set of real seed images.
- User-defined defect, anomaly, or variation types.
- Procedural generation as the first and most predictable strategy.
- Basic lighting, texture, background, and geometric randomization.
- Automatic segmentation masks and bounding boxes.
- COCO-style JSON plus images and masks.
- A preview gallery and dataset-quality report.
- A lightweight validation harness with optional probe-model training.
- Reproducible generation using recorded configuration and random seeds.
- A local CLI and a simple browser workflow.

### Explicitly out of scope for v1

- 3D/CAD ingestion and digital-twin simulation.
- Video or temporal defects.
- Non-visual signals such as sound, vibration, or thermal data.
- Multi-domain projects in a single generation run.
- Training and hosting a user’s production model.
- Fine-grained parameter tuning beyond type, severity, frequency, and basic variation controls.

## Implementation plan

The first release should be a complete vertical slice:

```text
Seed images
    → ingestion and QA
    → procedural defect/anomaly synthesis
    → masks and bounding boxes
    → COCO export
    → visual preview and statistics
    → optional probe-model validation
```

The initial implementation should be local-first. A Python library and CLI will make the generation pipeline testable and useful before a browser UI or hosted job system is introduced.

### Recommended stack

- **Python 3.11+** for the generation and validation pipeline.
- **OpenCV, NumPy, Pillow, and scikit-image** for image processing.
- **Albumentations** for augmentation and domain randomization.
- **Pydantic** for project and generation configuration.
- **FastAPI** for the API once the core pipeline is stable.
- **Gradio or Streamlit** for the first UI prototype; a separate React/Next.js UI can follow.
- **SQLite and local filesystem** for initial metadata and artifacts.
- A Redis-backed worker system only when generation jobs need to run asynchronously at scale.

Do not introduce microservices, Kubernetes, or cloud-specific infrastructure until the generation and evaluation loop demonstrates value.

## Generation strategies

The generation engine should expose a common strategy interface so techniques can be compared without changing ingestion, labeling, export, or validation.

### Procedural perturbation — first implementation

Procedural generation is the v1 default because it is fast, deterministic, explainable, and produces labels by construction.

Initial generation types:

- **Scratch or line mark:** curved or jagged variable-width paths blended into the source image.
- **Discoloration or stain:** irregular blurred regions with localized hue, saturation, or brightness changes.
- **Dent or deformation:** shaded irregular regions with a brightened rim; initially experimental because realistic results depend heavily on lighting.
- **Texture and appearance variation:** controlled changes to lighting, contrast, noise, background, orientation, and other non-essential factors.

Every generator should return the modified image, one or more masks, parameters, severity, and the random seed used.

### Future strategies

- Diffusion-based local inpainting.
- More advanced lighting-aware geometry and texture simulation.
- Domain-specific generation packs for different use cases.
- User-provided reference images for guiding generation.
- Organization-specific generation strategies selected from validation results.

Generative inpainting should be added only after procedural generation establishes a measurable baseline.

## System architecture

```text
                         ┌────────────────────┐
  Seed images ──────────▶│ Ingestion and QA    │
  User configuration ───▶│ Project management  │
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

- project and domain;
- source seed image for every generated image;
- generation or anomaly type, severity, and parameters;
- generation method;
- random seed;
- image dimensions and artifact paths;
- dataset split.

Example metadata:

```json
{
  "image": "image_0001_scratch.png",
  "source_seed": "seed_004.png",
  "domain": "product_surface",
  "generation_method": "procedural_perturbation",
  "random_seed": 884321,
  "labels": [
    {
      "type": "scratch",
      "severity": "moderate",
      "bbox": [124, 88, 212, 101],
      "mask": "masks/image_0001_scratch_mask.png"
    }
  ]
}
```

Generated images should be reproducible from the project configuration and seed. This is essential for debugging, benchmark comparisons, collaboration, and customer support.

## Ingestion and seed-set QA

The ingestion layer should check and report:

- unsupported or corrupted files;
- image dimensions and color modes;
- extreme brightness, darkness, or blur;
- duplicate and near-duplicate images;
- inconsistent framing or orientation;
- whether the subject or region can be localized reasonably.

Warnings should not block early experimentation unless an image is unusable. A QA report might identify inconsistent lighting or dimensions while still allowing the user to generate a dataset.

For the first prototype, assume users supply consistently framed or cropped images. Automatic subject or part segmentation can be added once the core generation loop is validated.

## Labels and export

Masks should be generated internally for every synthetic region because they are the source of truth. Bounding boxes can then be derived from masks. The default export should contain:

```text
export/
├── images/
├── masks/
├── annotations.json       # COCO-style JSON
├── metadata.jsonl         # per-image generation metadata
└── report.html or report.json
```

COCO annotations should include image IDs, category IDs, bounding boxes, areas, and segmentation data. Users should be able to request either bounding-box-only output or images with segmentation masks.

### Avoiding split leakage

Do not randomly split generated images when variants derived from the same source seed appear in both training and validation. Prefer splitting the original seed images first, then generating each dataset split from its own seed subset. If too few seeds are available, record the relationship and label the validation result as an estimate.

## Validation methodology

The central technical risk is the sim-to-real gap. Visual plausibility alone is not evidence that generated data will improve a real computer-vision model.

### Automated dataset checks

- Empty or nearly empty masks.
- Invalid boxes and segmentation geometry.
- Label area and severity distributions.
- Class balance and generation failure rate.
- Brightness, contrast, and texture distributions.
- Duplicate or near-duplicate outputs.

### Visual checks

Generate contact sheets showing:

- original/generated pairs;
- mask overlays;
- examples grouped by generation type and severity;
- smallest and largest generated regions.

### Probe-model checks

Where real labeled examples are available, train a small fixed baseline model on generated data and evaluate it on held-out real images. Report precision, recall, and F1 separately for each label type. Clearly distinguish dataset-health checks from evidence of transfer to real-world data.

Public datasets can be used for internal benchmarks. User and company images should remain user-controlled and should not be used to train shared models without explicit consent.

## Initial CLI target

The first end-to-end prototype should support a command like:

```bash
synthline generate \
  --seeds ./data/good \
  --defect scratch \
  --count 500 \
  --output ./runs/example-scratch \
  --seed 12345
```

The command should produce images, masks, COCO annotations, metadata, a preview contact sheet, and a basic quality report.

## API and UI workflow

After the CLI works, expose the same library through a small API:

```text
POST   /projects
POST   /projects/{id}/seeds
POST   /projects/{id}/labels
POST   /projects/{id}/generations
GET    /generations/{id}
GET    /generations/{id}/preview
GET    /generations/{id}/report
GET    /generations/{id}/download
DELETE /projects/{id}
```

The first UI should follow a simple flow:

```text
Create project → upload seeds → define variations → generate → preview → download
```

Long-running generation should initially use a simple background task. Add Redis and a dedicated worker only when job duration or concurrent users require it.

## Development phases

### Phase 0 — technical spike

- Establish the Python package and test setup.
- Implement image loading and basic QA.
- Implement procedural scratches and simple visual variations.
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
- Add user-friendly generation configuration.
- Show progress and generated previews.
- Add downloadable ZIP exports and reports.

### Phase 3 — organization and pilot readiness

- Add public benchmark fixtures.
- Add probe-model training and per-label metrics.
- Add deletion, retention, and access controls.
- Add private projects and team workspaces.
- Test with individual users and two or three organizations.

### Phase 4 — post-MVP

- Diffusion-based inpainting.
- 3D/CAD ingestion.
- Multi-domain projects.
- Hosted downstream model training.
- Organization-level integrations, audit logs, and advanced collaboration.

## Security and privacy

Images may contain private personal, product, or business information. The MVP should commit to:

- using images only for the user’s requested generation;
- not training shared models on user or company images without explicit opt-in;
- allowing project and artifact deletion;
- keeping artifacts isolated by project;
- documenting retention behavior before hosted pilots begin.

Individual users should be able to work locally where possible. Companies may require private deployment, stronger tenant isolation, audit logging, role-based access, and configurable retention before adoption.

## Product and business model direction

The core tool should remain available to everyone, with a useful local or free experience for individuals. Organization-focused plans can provide:

- higher generation limits;
- private hosted workspaces or self-hosted deployment;
- team collaboration and permissions;
- audit logs and retention policies;
- API access and integrations;
- priority support and custom generation strategies.

The product should validate usage patterns before locking in pricing. Usage-based, project-based, and organization subscriptions can be evaluated during pilots.

## Success metrics

- A first-time user can go from seed images to a downloadable dataset in under one hour.
- Individual users can run the core workflow without specialized infrastructure.
- Generated datasets contain valid, correctly aligned labels with a low generation failure rate.
- Synthetic-trained models perform meaningfully above baseline on real examples for at least two simple generation types.
- At least one individual or research user and one organization evaluate the generated data in a real project.
- Company pilots demonstrate value through faster experimentation, fewer labeling requirements, or improved model performance.

The most important metric is not how realistic an image looks in isolation; it is whether generated data helps users solve real computer-vision problems.

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

This is an early-stage private repository. Keep changes focused on one component and add or update tests with implementation changes. Update this README when a design decision changes the architecture, scope, schema, target users, or validation methodology.

Prefer branching and pull requests for normal development. Experimental generation changes should include example outputs and the configuration used to produce them.

## License

Private and confidential. All rights reserved. This project is not currently licensed for external use, distribution, or reproduction.

## Glossary

- **Anomaly detection:** learning normal appearance and flagging deviations rather than classifying known defect categories.
- **Domain randomization:** varying non-essential visual factors such as lighting, background, and angle to improve generalization.
- **Seed set:** the small collection of real images used as the source for generation.
- **Sim-to-real gap:** the performance difference between training or evaluating with synthetic data and performance on real-world images.
