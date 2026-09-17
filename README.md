# SynthLine

**Generate reproducible, labeled defect datasets from a small set of normal images.**

SynthLine is a local-first synthetic visual-data generation workbench for computer-vision developers, researchers, students, and small machine-vision teams.

It transforms a small collection of real “good” images into a larger dataset containing controlled scratches, stains, discoloration, texture variations, and other visual anomalies—with segmentation masks, bounding boxes, metadata, previews, and dataset-quality reports generated automatically.

```text
Normal seed images
        ↓
Image quality checks
        ↓
Defect and variation configuration
        ↓
Synthetic image generation
        ↓
Masks and bounding boxes
        ↓
Validation and visual review
        ↓
COCO / YOLO / ZIP export
```

> **Status:** Pre-MVP  
> The initial goal is to prove that a reproducible procedural generation and validation workflow can create useful data for real computer-vision problems.

---

## Why SynthLine exists

Computer-vision projects often fail because teams do not have enough labeled images.

Collecting more real-world defect images can be:

- Expensive.
- Slow.
- Difficult to reproduce.
- Dependent on rare failures.
- Risky when products or facilities are confidential.
- Difficult for students and independent developers.
- Operationally expensive when every image must be labeled manually.

SynthLine provides a controlled way to create additional training and evaluation data from a small set of normal images.

The goal is not to generate visually impressive images for their own sake. The goal is to help users test computer-vision systems with less manual data collection and labeling.

---

## Product promise

Given a small set of consistently framed normal images, SynthLine should allow a user to:

1. Validate the quality of the input images.
2. Define one or more defect or variation types.
3. Generate synthetic images deterministically.
4. Produce pixel-level masks and bounding boxes automatically.
5. Review generated samples and statistics.
6. Export a standard dataset for an existing computer-vision pipeline.
7. Reproduce or modify the generation run later.
8. Measure whether synthetic data improves performance on real images.

SynthLine is a **synthetic data generation and validation tool**.

It is not initially intended to be:

- A complete defect-detection product.
- A hosted model-training platform.
- A replacement for real-world validation data.
- A guarantee that synthetic data will improve model performance.
- A general-purpose image-generation platform.

---

## Initial target users

SynthLine is designed initially for:

- Independent computer-vision developers.
- Machine-learning researchers.
- Students and educators.
- Makers and technical founders.
- Small manufacturing and quality-inspection teams.
- Machine-vision integrators and consultants.
- Teams prototyping defect-detection systems with limited labeled data.

The first product wedge is **surface-defect dataset generation**, especially for objects or materials where users can provide consistently framed normal images.

Examples include:

- Product surfaces.
- Manufactured parts.
- Packaging.
- Printed materials.
- Wood, metal, plastic, ceramic, or painted surfaces.
- Simple inspection scenes with relatively stable framing.

---

## Example workflow

A typical project looks like this:

```text
Create a project
    ↓
Add normal seed images
    ↓
Review image-quality warnings
    ↓
Select "scratch"
    ↓
Choose count, severity, frequency, and seed
    ↓
Generate the dataset
    ↓
Review image/mask overlays and contact sheets
    ↓
Inspect the quality report
    ↓
Export to COCO, YOLO, or ZIP
    ↓
Train or evaluate a separate computer-vision model
```

Example command:

```bash
synthline generate \
  --seeds ./data/good \
  --defect scratch \
  --count 500 \
  --output ./runs/example-scratch \
  --seed 12345
```

The output should include:

```text
runs/example-scratch/
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

---

## Core principles

### 1. Local-first

The core generation pipeline should run locally without an external API key.

This makes SynthLine:

- Easy to try.
- Suitable for private images.
- Useful in classrooms and research environments.
- Compatible with offline workflows.
- Easier to test and benchmark.
- Less expensive to operate during the early product stage.

A hosted version can be added later for collaboration, storage, and larger jobs.

### 2. Reproducibility

Every generation run must record the configuration and random seed used to produce it.

A user should be able to reproduce a run from:

- The project configuration.
- The source seed images.
- The generator version.
- The generation strategy.
- The random seed.
- The requested output count.
- The environment or package version where practical.

### 3. Labels by construction

Masks should be generated as part of the image-generation process.

Bounding boxes should be derived from masks rather than drawn independently.

This keeps the image, mask, segmentation, and bounding-box labels aligned.

### 4. Validation over visual appeal

A realistic-looking image is not enough.

SynthLine should report:

- Whether masks are valid.
- Whether labels are balanced.
- Whether generated images are duplicates.
- Whether generated regions are too small or too large.
- Whether the generated distribution is materially different from the seed distribution.
- Whether synthetic data improves performance on real validation images when those images are available.

### 5. Simple defaults

The first user should not need to understand every low-level generation parameter.

The interface should expose a small number of useful controls:

- Defect type.
- Quantity.
- Severity.
- Frequency.
- Random seed.
- Output format.

Advanced parameters can be added later.

---

## MVP scope

### Included in the MVP

- Python package and CLI.
- Local project configuration.
- Importing common image formats.
- Seed-image quality checks.
- Procedural scratch generation.
- Procedural stain and discoloration generation.
- Basic lighting and texture variation.
- Severity and frequency controls.
- Deterministic random seeds.
- Pixel masks.
- Bounding boxes derived from masks.
- COCO-style export.
- YOLO-style export.
- Per-image generation metadata.
- Preview contact sheets.
- HTML or JSON quality reports.
- Seed-aware train, validation, and test splitting.
- Unit and integration tests.
- Optional baseline model evaluation when real labeled images are available.

### Explicitly out of scope for the first release

- 3D or CAD ingestion.
- Digital-twin simulation.
- Video generation.
- Temporal defects.
- Audio, vibration, thermal, or other non-visual signals.
- Production model hosting.
- End-to-end model deployment.
- Automatic segmentation of arbitrary complex objects.
- Multi-domain generation in one run.
- Enterprise identity management.
- Kubernetes or microservice infrastructure.
- Diffusion-based generation as the primary method.
- A marketplace for datasets or generation strategies.

---

## Initial generation strategies

SynthLine should expose a common generation interface so that different strategies can be compared without changing ingestion, labeling, export, or validation.

Conceptually:

```python
class Generator:
    name: str

    def generate(
        self,
        image,
        config,
        random_state,
    ) -> GenerationResult:
        ...
```

Each generator should return:

- The modified image.
- One or more masks.
- Bounding boxes derived from the masks.
- Defect or variation type.
- Severity.
- Parameters used.
- Random seed.
- Generator version.
- Failure or warning information when applicable.

### Scratch and line marks

The first generator should support:

- Curved scratches.
- Jagged scratches.
- Variable-width lines.
- Multiple scratches per image.
- Different opacity levels.
- Dark, bright, or mixed scratch appearance.
- Blur and edge-softening controls.
- Orientation and length variation.

Scratch geometry should be generated before compositing the visual appearance so that the mask remains exact.

### Stains and discoloration

The initial stain generator should support:

- Irregular connected regions.
- Blurred edges.
- Hue changes.
- Saturation changes.
- Brightness changes.
- Multiple stain sizes.
- Different opacity levels.
- Localized and diffuse discoloration.

### Texture and appearance variation

Non-defect variation can help prevent models from memorizing irrelevant visual properties.

Initial variations may include:

- Brightness.
- Contrast.
- Color temperature.
- Noise.
- Blur.
- Small rotations.
- Small translations.
- Background variation where safe.
- Mild scale variation.
- Texture intensity.

These variations should not accidentally change the semantic label or create invalid masks.

### Dents and deformations

Dents are intentionally experimental.

Realistic dents depend on:

- Object geometry.
- Material properties.
- Lighting direction.
- Camera position.
- Surface reflectance.
- Scene context.

They should not be marketed as reliable until benchmark results demonstrate that they are useful.

---

## Generation configuration

A generation configuration should be explicit, versioned, and serializable.

Example:

```json
{
  "project_name": "metal-surface-demo",
  "domain": "product_surface",
  "source_directory": "./data/good",
  "generation_method": "procedural",
  "generators": [
    {
      "type": "scratch",
      "count": 1,
      "severity": "moderate",
      "frequency": 0.8
    },
    {
      "type": "discoloration",
      "count": 1,
      "severity": "low",
      "frequency": 0.35
    }
  ],
  "output_count": 500,
  "image_format": "png",
  "random_seed": 12345,
  "split_strategy": "seed_aware",
  "export_formats": ["coco", "yolo"]
}
```

The configuration should be saved with every generation run.

---

## Data model and metadata

Every generated image must retain its relationship to the source seed image.

Example metadata record:

```json
{
  "image": "images/image_0001_scratch.png",
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
      "area": 1674,
      "mask": "masks/image_0001_scratch_mask.png",
      "parameters": {
        "width": 4.2,
        "opacity": 0.72,
        "length": 112.0
      }
    }
  ]
}
```

At minimum, SynthLine should record:

- Project identifier.
- Generation-run identifier.
- Source seed image.
- Generated image path.
- Generation method.
- Generator version.
- Defect or variation type.
- Severity.
- Generator parameters.
- Random seed.
- Image dimensions.
- Mask path.
- Bounding box.
- Mask area.
- Dataset split.
- Creation timestamp.
- Warnings or generation failures.

---

## Input-image quality checks

The ingestion layer should inspect and report:

- Unsupported file formats.
- Corrupted images.
- Image dimensions.
- Color modes.
- Alpha channels.
- Extreme brightness or darkness.
- Excessive blur.
- Duplicate images.
- Near-duplicate images.
- Inconsistent aspect ratios.
- Inconsistent framing.
- Orientation differences.
- Images that are too small for the requested generation.
- Images that contain unexpected subjects or backgrounds where detectable.

Warnings should not block experimentation unless the image is unusable.

For example:

```text
WARNING: seed_017.png is significantly darker than the seed-set median.
WARNING: seed_023.png is a near-duplicate of seed_004.png.
ERROR: seed_031.jpg could not be decoded.
```

The first version should assume that users provide consistently framed or cropped images. Automatic object or subject segmentation can be added later.

---

## Labels and export formats

Masks are the source of truth.

Bounding boxes should be computed from masks using a consistent convention:

```text
[x_min, y_min, width, height]
```

The default export should include:

```text
export/
├── images/
├── masks/
├── annotations.coco.json
├── annotations.yolo/
├── metadata.jsonl
├── config.json
├── preview.html
└── report.json
```

### COCO export

COCO annotations should include:

- Image IDs.
- Image dimensions.
- Category IDs.
- Category names.
- Bounding boxes.
- Areas.
- Segmentation data.
- Annotation IDs.
- Dataset metadata.

### YOLO export

YOLO export should support the common object-detection format:

```text
class_id center_x center_y width height
```

Coordinates should be normalized according to the YOLO specification.

### Future export formats

Potential future integrations include:

- Ultralytics datasets.
- Roboflow-compatible ZIP exports.
- Pascal VOC.
- Mask R-CNN formats.
- Hugging Face dataset layouts.
- Custom export templates.

Export compatibility is important for distribution because users should be able to move from SynthLine into tools they already use.

---

## Avoiding data leakage

Generated images derived from the same source image must not be randomly distributed across training and validation sets.

The preferred process is:

```text
Split original seed images first
        ↓
Generate variants independently within each split
```

For example:

```text
seed_001.png → train
seed_002.png → train
seed_003.png → validation
seed_004.png → test
```

All generated variants derived from `seed_003.png` must remain in the validation split.

This prevents a model from appearing to generalize when it has effectively seen the same source image during training.

---

## Validation and quality reports

Every generation run should produce machine-readable and human-readable validation results.

### Automated checks

The validation system should check:

- Empty masks.
- Nearly empty masks.
- Masks outside image boundaries.
- Invalid bounding boxes.
- Negative or zero-sized boxes.
- Invalid segmentation geometry.
- Missing annotations.
- Mismatched image and mask dimensions.
- Class imbalance.
- Label-area distributions.
- Severity distributions.
- Generation failure rates.
- Duplicate outputs.
- Near-duplicate outputs.
- Unusual brightness or contrast.
- Excessive image corruption.
- Split leakage.
- Unexpected output dimensions.

### Visual checks

The preview should include:

- Original and generated image pairs.
- Mask overlays.
- Bounding-box overlays.
- Examples grouped by defect type.
- Examples grouped by severity.
- Smallest generated regions.
- Largest generated regions.
- Failed or suspicious outputs.
- Randomly selected examples.
- Seed-image coverage.

### Probe-model evaluation

When real labeled images are available, SynthLine should support an optional baseline experiment:

```text
Train a small fixed model on generated data
        ↓
Evaluate on held-out real images
        ↓
Compare against a real-data-only baseline
```

The report should include:

- Precision.
- Recall.
- F1 score.
- Per-class metrics.
- Confusion matrix where applicable.
- Number of real training images.
- Number of synthetic training images.
- Generator configuration.
- Random seed.
- Model configuration.
- Comparison with baseline.

Synthetic data should be considered useful only when it produces measurable value on a relevant real-world evaluation set.

---

## Recommended architecture

The first implementation should be a Python package with a CLI. The browser interface should use the same application services as the CLI rather than implementing a separate generation path.

```text
                    ┌─────────────────────┐
                    │ CLI                  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Application services │
                    │ projects / runs      │
                    └──────────┬──────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       │                       │                       │
┌──────▼──────┐        ┌───────▼────────┐       ┌──────▼──────┐
│ Ingestion   │        │ Generation     │       │ Validation  │
│ and QA      │        │ strategies     │       │ and reports │
└──────┬──────┘        └───────┬────────┘       └──────┬──────┘
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Labeling and export │
                    └─────────────────────┘
```

### Recommended technology

- Python 3.11+.
- NumPy.
- OpenCV.
- Pillow.
- scikit-image.
- Albumentations where appropriate.
- Pydantic for configuration and validation.
- Typer or Click for the CLI.
- pytest for testing.
- FastAPI after the core pipeline is stable.
- SQLite for local project metadata.
- Local filesystem for image artifacts.
- Gradio or Streamlit for an early UI prototype.
- React or Next.js only after the workflow has been validated.

Do not introduce microservices, Kubernetes, Redis, cloud-specific infrastructure, or a distributed job system until real usage requires them.

---

## Proposed repository structure

```text
synthline/
├── pyproject.toml
├── README.md
├── LICENSE
├── src/
│   └── synthline/
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
│       │   │   └── deformation.py
│       │   ├── randomization/
│       │   │   ├── lighting.py
│       │   │   ├── texture.py
│       │   │   └── geometry.py
│       │   └── registry.py
│       ├── labeling/
│       │   ├── masks.py
│       │   ├── boxes.py
│       │   └── segmentation.py
│       ├── export/
│       │   ├── coco.py
│       │   ├── yolo.py
│       │   └── archive.py
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
└── web/
```

---

## CLI design

The CLI should make common tasks straightforward.

### Create a project

```bash
synthline project init ./projects/metal-surface
```

### Inspect seed images

```bash
synthline inspect \
  --seeds ./data/good \
  --report ./reports/seed-quality.html
```

### Generate a dataset

```bash
synthline generate \
  --project ./projects/metal-surface \
  --defect scratch \
  --count 500 \
  --severity moderate \
  --frequency 0.8 \
  --output ./runs/scratch-500 \
  --seed 12345
```

### Preview a run

```bash
synthline preview \
  --run ./runs/scratch-500 \
  --output ./runs/scratch-500/preview.html
```

### Validate a run

```bash
synthline validate \
  --run ./runs/scratch-500
```

### Export a run

```bash
synthline export \
  --run ./runs/scratch-500 \
  --format coco \
  --output ./exports/scratch-coco
```

### Reproduce a run

```bash
synthline reproduce \
  --config ./runs/scratch-500/config.json \
  --output ./runs/scratch-500-reproduced
```

---

## API and browser workflow

The API should be added after the local generation library is stable.

A future API may expose:

```text
POST   /projects
GET    /projects/{id}
DELETE /projects/{id}

POST   /projects/{id}/seeds
GET    /projects/{id}/seeds

POST   /projects/{id}/runs
GET    /runs/{id}
GET    /runs/{id}/preview
GET    /runs/{id}/report
GET    /runs/{id}/download
DELETE /runs/{id}
```

The initial browser workflow should remain simple:

```text
Create project
    → upload seed images
    → review seed quality
    → choose defect type
    → configure generation
    → generate
    → review previews
    → inspect report
    → download dataset
```

Long-running generation can initially use a simple background task. A dedicated worker system should be introduced only when generation time or concurrent usage justifies it.

---

## Hosted product direction

The local product should remain useful on its own.

A hosted version may later provide:

- Private project storage.
- Browser-based generation.
- Team workspaces.
- Background generation jobs.
- Usage history.
- Shareable reports.
- Dataset versioning.
- API access.
- Role-based permissions.
- Audit logs.
- Configurable retention.
- Organization-level integrations.
- Private deployment options.

The hosted version should not be required to validate the core product idea.

---

## Distribution strategy

Synthetic-data tooling is difficult to distribute if it is positioned as a broad platform before users have a specific reason to try it.

SynthLine should use a developer-first distribution strategy.

### Free local CLI

The CLI should be:

- Free to install.
- Usable without an account.
- Usable without an API key.
- Useful with a small local dataset.
- Documented with copy-and-paste examples.

This reduces the friction between discovering SynthLine and experiencing its value.

### Public examples

Provide downloadable example projects containing:

- Seed images with a permissive license.
- A generation configuration.
- Generated outputs.
- Quality reports.
- Example model-evaluation results.
- Reproduction instructions.

Every example should answer:

> What problem did the generated data help solve?

### Reproducible recipes

Users should be able to publish or share:

- Seed-set descriptions.
- Generator configuration.
- Random seeds.
- Output statistics.
- Preview reports.
- Benchmark results.

A shareable recipe is more useful for adoption than a screenshot of generated images.

### Integrations

Prioritize exports and documentation for tools developers already use:

- COCO.
- YOLO.
- Ultralytics.
- PyTorch.
- TensorFlow.
- Roboflow.
- Hugging Face datasets.
- Standard Python data pipelines.

### Content and search strategy

Useful documentation topics include:

- How to generate scratch-defect datasets.
- How to train a defect detector with synthetic images.
- How to avoid synthetic-data leakage.
- How to validate synthetic images against real images.
- When procedural generation is better than diffusion.
- How to create segmentation masks automatically.
- How to test whether synthetic data improves model recall.

The distribution message should focus on outcomes, not on the internal architecture.

### Community and benchmark loop

Build a public benchmark around small, reproducible examples.

Encourage users to share:

- Seed-set characteristics.
- Generation configurations.
- Real validation results.
- Failure cases.
- Generator improvements.
- Domain-specific recipes.

Trust is especially important for synthetic data. Publishing negative results is valuable because it shows that SynthLine is measuring utility rather than promising that synthetic data always works.

---

## Business model direction

Do not lock in pricing before observing actual usage.

A possible future model is:

### Free local edition

- Local CLI.
- Core procedural generators.
- Basic exports.
- Local reports.
- Small public examples.

### Hosted individual edition

- Browser workflow.
- Private project storage.
- Larger generation limits.
- Run history.
- Shareable reports.
- Additional exports.

### Team and organization edition

- Team workspaces.
- Permissions.
- Audit logs.
- Retention controls.
- API access.
- Private deployment.
- Priority support.
- Custom generation strategies.
- Higher generation limits.

The value metric should eventually be tied to something users understand, such as generated images, compute usage, projects, or workspace capacity.

---

## Development roadmap

### Phase 0: Technical spike

- Establish the Python package.
- Add configuration models.
- Implement image loading.
- Implement seed-image QA.
- Implement scratch generation.
- Generate masks and bounding boxes.
- Export COCO annotations.
- Generate preview contact sheets.
- Add basic statistics.
- Create fixture-based tests.

### Phase 1: Procedural MVP

- Add discoloration and stain generation.
- Add appearance randomization.
- Add severity and frequency controls.
- Record reproducible configurations.
- Implement seed-aware dataset splitting.
- Add validation reports.
- Add YOLO export.
- Add generation failure handling.
- Add golden-image tests where practical.

### Phase 2: Local user experience

- Add project initialization.
- Add a simple local UI.
- Add progress reporting.
- Add configuration templates.
- Add downloadable HTML reports.
- Add example projects.
- Improve error messages and documentation.

### Phase 3: Real-world validation

- Add public benchmark fixtures.
- Add optional probe-model training.
- Compare synthetic-only, real-only, and mixed-data baselines.
- Test with independent developers and researchers.
- Test with one or two small organizations.
- Measure time saved and model-performance impact.

### Phase 4: Hosted workflow

- Add user accounts.
- Add private project storage.
- Add asynchronous generation jobs.
- Add project and artifact deletion.
- Add retention controls.
- Add team workspaces.
- Add API access.
- Add usage limits and billing experiments.

### Phase 5: Advanced generation

- Diffusion-based local inpainting.
- Lighting-aware deformation.
- Domain-specific generation packs.
- User-provided reference images.
- 3D or CAD integration.
- Advanced collaboration.
- Organization-specific generation strategies.

---

## Success metrics

The most important metric is not the number of images generated.

The most important question is whether generated data helps users solve real computer-vision problems.

### Product metrics

- A new user can create a downloadable dataset in under one hour.
- A user can run the core workflow without specialized infrastructure.
- The CLI can be installed and used without an API key.
- Users can reproduce a previous run successfully.
- Exported datasets work in at least two common computer-vision frameworks.
- Users return to modify or reproduce generation configurations.

### Technical metrics

- Low generation failure rate.
- Valid masks for generated images.
- Valid bounding boxes for generated images.
- No seed leakage between dataset splits.
- Stable output for a fixed configuration and random seed.
- Useful seed-image quality warnings.
- Reasonable class and severity distributions.

### Outcome metrics

- Synthetic-trained models improve over a real-data-only baseline in at least one initial benchmark.
- Mixed real and synthetic data improves recall or F1 on held-out real images.
- At least one independent developer or researcher uses SynthLine in a real project.
- At least one organization evaluates SynthLine against an existing data-collection workflow.
- Users report reduced time spent collecting or labeling initial training data.

---

## Security and privacy

Images may contain private personal, product, or business information.

The local-first workflow should ensure that images remain under the user’s control.

For any hosted product, SynthLine should commit to:

- Using uploaded images only for the requested product functionality.
- Not training shared models on user images without explicit opt-in.
- Supporting project and artifact deletion.
- Isolating artifacts between projects and tenants.
- Clearly documenting retention behavior.
- Protecting access to generated datasets and reports.
- Providing appropriate controls for business and confidential data.

Organization-focused deployments may eventually require:

- Private networking.
- Self-hosting.
- Role-based access.
- Audit logs.
- Configurable retention.
- Encryption at rest and in transit.
- Workspace-level access controls.

---

## Limitations

SynthLine cannot guarantee that synthetic images will improve a computer-vision model.

Synthetic data can fail when:

- The generated visual patterns are too artificial.
- The defect appearance is incorrectly modeled.
- Real images contain lighting or backgrounds not represented by the seed set.
- The generated class distribution is unrealistic.
- The model learns generator artifacts.
- The source images are inconsistent or unrepresentative.
- Training and evaluation data share source-image variants.
- The defect requires 3D geometry or physical simulation.

Real-world validation remains necessary.

SynthLine should help users measure these limitations rather than hide them.

---

## Getting started

The project is currently in pre-MVP development.

The intended development environment is Python 3.11 or newer.

```bash
git clone https://github.com/thrivecodes/synthline.git
cd synthline

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pytest
```

The procedural generation pipeline should not require external API keys.

Future hosted or diffusion-based features must document:

- Required credentials.
- Expected infrastructure.
- Image-processing behavior.
- Data retention.
- Privacy implications.
- Additional operating costs.

---

## Development principles

- Keep the first vertical slice small and complete.
- Prefer deterministic behavior over visual complexity.
- Add tests with every generator.
- Preserve generation configurations.
- Treat masks as the source of truth.
- Split source images before generating variants.
- Measure real-world usefulness.
- Do not introduce infrastructure before usage requires it.
- Keep the local workflow useful without a hosted account.
- Document known failures and limitations.
- Include example outputs with generator changes.
- Keep product claims narrower than the evidence supports.

---

## Contributing

SynthLine is an early-stage private project.

When contributing:

1. Keep changes focused on one component.
2. Add or update tests with implementation changes.
3. Include configuration files for experimental generation changes.
4. Include example outputs where visual behavior changes.
5. Record random seeds for reproducibility.
6. Update documentation when architecture or product behavior changes.
7. Prefer branches and pull requests for normal development.
8. Avoid adding infrastructure that is not required by a demonstrated use case.

---

## License

This repository is private and confidential.

All rights reserved. The project is not currently licensed for external use, distribution, or reproduction.

---

## Glossary

**Anomaly detection**  
Learning normal appearance and identifying deviations rather than classifying only known defect categories.

**Bounding box**  
A rectangular region surrounding a labeled object or defect.

**Domain randomization**  
Varying non-essential visual factors such as lighting, texture, background, and orientation.

**Generation run**  
One reproducible execution of the generation pipeline using a specific configuration and random seed.

**Seed image**  
A real input image used as the source for one or more generated images.

**Sim-to-real gap**  
The performance difference between models trained or evaluated on synthetic data and their performance on real-world data.

**Synthetic data**  
Artificially generated data created to resemble or extend real-world data.

**Validation set**  
A group of images used to measure performance during development.

**Test set**  
A held-out group of images used for final evaluation. It should remain isolated from generation and model-tuning decisions.

---

## Project status

SynthLine is currently focused on proving one complete workflow:

```text
Small set of normal images
    → procedural defect generation
    → exact masks and bounding boxes
    → reproducible dataset export
    → visual and automated validation
    → measurement on real images
```

The project should expand only after this workflow demonstrates measurable value.
