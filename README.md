# SynthLine AI

Synthetic visual-data generation for real computer-vision teams, researchers, and builders.

SynthLine AI turns a small set of real “good” images into a larger, labeled dataset of realistic defects, variations, and anomalies for visual inspection, quality control, and model development workflows.

It helps teams build better image datasets without waiting for rare defects, outsourcing expensive labeling, or relying on fragile heuristics.

> Status: pre-MVP / local-first prototype
>
> The current focus is a reproducible generation and validation workflow that is useful for individual developers, students, researchers, and small teams working with real-world defect data.

---

## Why this product exists

Computer-vision projects often fail because the data is the bottleneck.

The real problem is usually not the model architecture. It is:

- too few defect examples
- too little labeled data
- rare failure cases that are difficult to capture
- inconsistent or expensive manual labeling
- privacy concerns around uploading sensitive images
- poor reproducibility when experiments are repeated later
- synthetic data that looks good but does not help a real model

SynthLine AI was created to solve that gap directly.

The goal is not to generate “pretty” images. The goal is to generate useful data:
- images with realistic defects
- exact masks
- bounding boxes
- metadata and reproducibility
- validation reports
- exportable dataset outputs that fit existing vision workflows

---

## Product promise

Given a small collection of normal images, SynthLine AI helps a user:

1. Validate the seed set.
2. Choose a defect or visual variation type.
3. Configure generation parameters.
4. Generate synthetic variations with labels.
5. Inspect the outputs visually.
6. Validate dataset quality.
7. Export the result to a standard dataset format.
8. Reproduce or refine the run later.

A user should be able to go from “a folder of good images” to “a labeled synthetic dataset ready for model experimentation” in a clear, controlled workflow.

SynthLine AI is a synthetic data generation and validation tool.

It is not:
- a general-purpose AI image generator
- a hosted model-training platform
- a production inspection product by itself
- a replacement for real-world validation data
- a guarantee that synthetic data will improve every model

---

## Who it is for

### Independent developers and makers
- building prototypes with limited defect data
- testing ideas before buying a larger data collection pipeline
- validating image-based models without large labeling operations

### Researchers and students
- experimenting with synthetic training data
- studying sim-to-real gaps
- creating reproducible dataset-generation experiments

### Small engineering teams
- quality inspection workflows
- process monitoring
- product-surface validation
- prototype model evaluation before large-scale data investments

### Manufacturing and machine-vision teams
- surface inspection
- defect detection
- packaging quality checks
- product verification and early-stage QA research

### Data and ML teams
- augmenting limited datasets
- testing segmentation and detection setups
- preparing benchmark data for experiments

---

## Core product concept

SynthLine AI makes synthetic visual data generation practical and measurable.

A typical workflow:

```text
Normal seed images
    ↓
Image quality checks
    ↓
Configurable defect generation
    ↓
Masks + bounding boxes
    ↓
Preview + validation
    ↓
COCO / YOLO / ZIP export
```

The output includes:
- generated images
- segmentation masks
- derived bounding boxes
- metadata and reproducibility records
- dataset reports
- preview sheets and visual validation views

---

## Initial target use cases

The first product focus is quality-inspection-style synthetic data for visually consistent domains.

Examples:
- scratches on manufactured surfaces
- stains or discoloration
- appearance variation
- localized defect-like patterns
- product-surface quality tasks
- prototype defect datasets for robotics or inspection workflows

This is intentionally narrower than “all image generation.”

The product should win in a clear niche before expanding across unrelated image-generation use cases.

---

## Product principles

### 1. Local-first
The core pipeline should run locally without external API keys.

That makes the product:
- easier to try
- more private
- more reproducible
- better for researchers and students
- better for private images and confidential products

### 2. Reproducibility
Every generation run must be traceable:
- the seed images
- the generator
- the parameters
- the random seed
- the metadata
- the version of the generation logic

If a user cannot reproduce a run, the system is not useful for real workflows.

### 3. Labels by construction
Masks should be generated as part of the process.

Bounding boxes should be derived from those masks rather than created independently.

This keeps labels aligned and easier to trust.

### 4. Validation over aesthetics
The real question is not:
- “Does it look realistic?”

The real question is:
- “Does the synthetic data help on real data?”

This is one of the most important product distinctions.

### 5. Simplicity first
The first version should expose only a few critical controls:
- defect type
- severity
- frequency
- output count
- random seed
- output format

No need for a complicated parameter maze before the product proves value.

---

## MVP scope

### In scope
- Python package and CLI
- local project execution
- image ingestion and quality checks
- small seed image sets
- 2D image generation
- procedural defect generation
- basic noise, brightness, contrast, and texture variation
- segmentation masks
- bounding boxes
- COCO-style export
- YOLO-style export
- metadata records
- preview gallery
- dataset-quality reports
- reproducible generation config
- quality checks and validation
- test suite and reproducibility checks

### Out of scope for v1
- 3D CAD ingestion
- digital twins
- video or temporal defect generation
- multi-domain projects in one run
- model-hosting platform
- production deployment infrastructure
- general-purpose inpainting as the primary mode
- broad enterprise identity / permission systems
- microservices or distributed cloud architecture

---

## Implementation strategy

The first release should be a complete vertical slice:

```text
Seed images
    → ingestion and QA
    → procedural defect generation
    → masks and bounding boxes
    → validation and preview
    → dataset export
    → reproducible metadata
```

The initial implementation should be local-first and library-driven.

A Python package plus CLI gives the fastest path to:
- deterministic behavior
- easier testing
- reproducibility
- research usefulness
- product validation

After that, a browser app or API can be layered on top without redesigning the generation engine.

---

## Recommended technology stack

- Python 3.11+
- NumPy
- OpenCV
- Pillow
- scikit-image
- Albumentations
- Pydantic
- pytest
- Click or Typer for CLI
- FastAPI after core pipeline stability
- SQLite or local storage for metadata
- Local filesystem for generated artifacts
- Optional Gradio or Streamlit for early UI
- React / Next.js later for a serious browser workflow

Do not introduce microservices, Kubernetes, or cloud architecture until real usage requires it.

---

## Generation strategies

SynthLine AI should expose a common generator interface so different generation strategies can be compared without changing the rest of the pipeline.

### Procedural generation — first implementation
This is the v1 default because it is:
- fast
- explainable
- reproducible
- label-friendly
- easier to validate

Initial generation types:
- scratch or line mark
- discoloration or stain
- texture or appearance variation
- mild lighting and surface variation

These should return:
- the modified image
- one or more masks
- bounding boxes
- generator parameters
- severity
- random seed
- metadata

### Future strategies
- diffusion-based local inpainting
- lighting-aware simulation
- domain-specific generation packs
- user-provided reference styles
- organization-specific generation recipes

Generative inpainting should be added only after the procedural baseline proves useful.

---

## System architecture

```text
                    ┌─────────────────────┐
   Seed images  ────▶│ Ingestion and QA   │
   Config / params ─▶│ Project management │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Generation engine   │
                    │ procedural first    │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Labeling + export   │
                    │ masks, boxes, COCO  │
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Validation and QA   │
                    │ reports + preview   │
                    └─────────────────────┘
```

---

## Proposed repository structure

```text
synthline-ai/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── synthline_ai/
│       ├── cli.py
│       ├── config/
│       │   ├── models.py
│       │   └── defaults.py
│       ├── ingestion/
│       │   ├── loader.py
│       │   ├── quality.py
│       │   └── deduplication.py
│       ├── generation/
│       │   ├── base.py
│       │   ├── pipeline.py
│       │   ├── procedural/
│       │   │   ├── scratch.py
│       │   │   ├── discoloration.py
│       │   │   └── variability.py
│       │   ├── randomization/
│       │   │   ├── lighting.py
│       │   │   ├── texture.py
│       │   │   └── geometry.py
│       │   └── registry.py
│       ├── labeling/
│       │   ├── masks.py
│       │   ├── boxes.py
│       │   └── export.py
│       ├── validation/
│       │   ├── checks.py
│       │   ├── statistics.py
│       │   ├── previews.py
│       │   └── probe_models.py
│       ├── projects/
│       │   ├── models.py
│       │   ├── storage.py
│       │   └── runs.py
│       └── version.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── golden/
├── benchmarks/
├── examples/
├── docs/
├── scripts/
├── web/
└── .github/
```

---

## Data model and reproducibility

Every generation run should preserve the exact configuration used to create it.

At minimum, record:
- project name
- source seed images
- generator type
- severity
- frequency
- parameters
- random seed
- output count
- image paths
- mask paths
- bounding boxes
- split assignment
- generation timestamp
- generator version

Example:

```json
{
  "image": "image_0001_scratch.png",
  "source_seed": "seed_004.png",
  "domain": "product_surface",
  "generation_method": "procedural",
  "generator_version": "0.1.0",
  "random_seed": 884321,
  "split": "train",
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

This is essential for:
- debugging
- reproducing runs
- contamination checks
- experiment comparison
- support and user trust

---

## Ingestion and seed-set QA

The ingestion layer should inspect and report:
- unsupported or corrupt files
- dimensions and color mode
- dark or overly bright images
- blur or low-quality samples
- duplicate and near-duplicate images
- inconsistent framing
- orientation issues
- object localization feasibility
- inconsistent visual distribution across the seed set

Example warnings:
- image too dark
- near-duplicate detected
- image dimensions differ from seed median
- suspicious blur level

Warnings should not block generation unless the image is unusable.

The MVP should assume users provide consistently framed images.

---

## Labeling and export

Masks are the source of truth.

Bounding boxes should be derived from masks.

Default export structure:

```text
export/
├── images/
├── masks/
├── annotations.coco.json
├── annotations.yolo/
├── metadata.jsonl
├── config.json
├── preview.html
├── contact-sheet.jpg
└── report.json
```

### COCO export
Include:
- image IDs
- categories
- annotation IDs
- bounding boxes
- segmentation masks
- areas
- image dimensions

### YOLO export
Support the common object-detection format with normalized coordinates.

### Safe dataset splitting
Do not randomly split generated variants when they originate from the same seed image.

Use:
- split source seeds first
- generate variants within each split
- prevent leakage between train/validation/test

---

## Validation methodology

The central technical risk is the sim-to-real gap.

Synthetic data is only useful if it helps a model perform better or provide a more useful experiment on real data.

### Automated checks
- empty or nearly empty masks
- invalid boxes
- invalid segmentation geometry
- mask area distribution
- class balance
- generation failures
- duplicates and near-duplicates
- brightness and contrast distribution
- split leakage
- suspicious output anomalies

### Visual checks
Generate preview sheets showing:
- original/generated pairs
- mask overlays
- bounding-box overlays
- examples grouped by defect type
- examples grouped by severity
- smallest and largest defects
- failed or suspicious samples

### Probe-model checks
If real labeled examples are available:
- train a small baseline model on generated data
- evaluate on held-out real images
- compare against real-only baseline
- report precision, recall, and F1

This is the product’s practical truth test.

---

## Initial CLI target

The first end-to-end prototype should support a command like:

```bash
synthline-ai generate \
  --seeds ./data/good \
  --defect scratch \
  --count 500 \
  --output ./runs/example-scratch \
  --seed 12345
```

The command should produce:
- images
- masks
- metadata
- COCO annotations
- preview contact sheet
- basic quality report

---

## Browser workflow and API vision

After the CLI is stable, expose the same functionality through an API and a small browser workflow.

### Proposed API
```text
POST   /projects
GET    /projects/{id}
POST   /projects/{id}/seeds
POST   /projects/{id}/runs
GET    /runs/{id}
GET    /runs/{id}/preview
GET    /runs/{id}/report
GET    /runs/{id}/download
DELETE /projects/{id}
```

### Browser workflow
```text
Create project → upload seeds → define variations → generate → preview → validate → export
```

Long-running tasks can initially use a background queue. Do not introduce Redis or complex orchestration until real demand requires it.

---

## Development phases

### Phase 0 — technical spike
- establish package and test setup
- implement image loading and quality checks
- implement procedural scratch generation
- generate masks and boxes
- export COCO
- create previews and dataset statistics

### Phase 1 — procedural MVP
- add discoloration and stain generation
- add appearance variation
- add severity and frequency controls
- implement reproducible metadata
- add seed-aware train/validation/test splits
- add unit and integration tests

### Phase 2 — local user experience
- project creation and seed upload
- simple UI for generation config
- progress and preview
- downloadable exports and reports

### Phase 3 — validation and proof
- public benchmark fixtures
- probe-model evaluation
- compare synthetic vs real-data baselines
- test with real users and small orgs

### Phase 4 — post-MVP
- advanced inpainting
- 3D/CAD ingestion
- multi-domain generation
- organization-level collaboration
- hosted private workspaces

---

## Security and privacy

Images may contain private, sensitive, or proprietary information.

The MVP should commit to:
- keeping local generation local
- not training shared models on user images without explicit consent
- allowing project deletion
- isolating artifacts by project
- documenting retention and storage behavior
- designing hosted features with privacy in mind

For organizations, private deployment and stronger controls may eventually be needed:
- retention policies
- access controls
- audit logs
- team workspaces
- self-hosted deployment

---

## Business model direction

The product should stay useful to individuals and small teams.

A sensible future model:
- local free or low-cost workflow
- paid hosted workspaces
- paid retry/compute jobs
- team and org features
- API access
- private deployment options

Do not lock pricing before validating the workflow. The product should prove value with real users first.

---

## Success metrics

The most important metric is not how realistic an image looks.

The most important metric is whether the generated data helps in a real vision workflow.

### Product metrics
- a user can go from seed images to dataset export in under one hour
- the local workflow works without infrastructure
- config and run reproduction works
- exported data loads in common vision tools

### Technical metrics
- low generation failure rate
- valid masks and boxes
- low invalid-output rate
- no seed leakage
- deterministic generation by seed

### Outcome metrics
- generated data helps on held-out real images
- a real user uses it for valid quality-inspection or detection work
- mixed synthetic and real data outperforms a weaker baseline in real tests

---

## Getting started

The project is in early development. The recommended environment is Python 3.11+.

```bash
git clone https://github.com/thrivecodes/synthline-ai.git
cd synthline-ai

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pytest
```

No external API key should be required for the procedural generation pipeline.

Any future hosted or external provider must clearly document:
- credentials
- retention behavior
- data usage
- privacy implications
- provider responsibilities

---

## Contributing

This is an early-stage private project.

When contributing:
- keep changes focused
- add or update tests with implementation changes
- include example outputs for visual generator changes
- keep generation config reproducible
- update docs when architecture changes
- prefer pull requests for normal work
- keep experimental logs and example outputs with changes

---

## License

This repository is currently private and confidential.

All rights reserved. This project is not currently licensed for external use, distribution, or reproduction.

---

## Glossary

- Anomaly detection: learning normal appearance and flagging deviations.
- Domain randomization: varying lighting, background, and texture to improve robustness.
- Seed set: a small set of real images used as the source for synthetic generation.
- Sim-to-real gap: the difference between synthetic-data performance and real-world performance.
- Reproducible run: a generation output that can be recreated from stored configuration and seed.
- Label by construction: masks and boxes are created as part of generation rather than manually after the fact.

---

## Final positioning

SynthLine AI is not trying to be “AI image generation for everyone.”

It is trying to be the most practical tool for:
- generating realistic defect data from a small number of normal images
- producing labels automatically
- validating whether the synthetic dataset is actually useful
- exporting clean, reproducible data into standard vision workflows

That is a much stronger, clearer, and more defensible product position.

---

## Quick summary

SynthLine AI helps teams turn a small set of real images into a dataset they can trust.

It is designed for:
- small-batch quality inspection
- model prototyping
- research and education
- realistic defect generation
- reproducible, labeled synthetic data

It is not designed to be a generic AI art tool.

It is designed to be a serious dataset-generation tool for real computer-vision workflows.
