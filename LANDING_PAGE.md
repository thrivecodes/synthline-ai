# SynthLine Landing Page

This document defines the visual direction, page structure, copy, interaction model, and implementation guidance for the SynthLine marketing landing page.

The landing page should present SynthLine as a serious developer and research tool for generating useful, labeled computer-vision data. It should feel considered and distinctive—not like a generic AI startup template, dashboard template, or automatically generated SaaS interface.

> Important: Do not use grid lines, blueprint backgrounds, circuit patterns, excessive gradients, floating glass cards, or decorative AI imagery. The interface should be clean, editorial, tactile, and evidence-led.

---

## Landing page objective

The landing page must communicate the following within the first few seconds:

1. SynthLine starts with a small set of normal images.
2. It creates controlled synthetic defects and variations.
3. Masks, bounding boxes, metadata, and reports are generated with the images.
4. The output can be exported to existing computer-vision workflows.
5. The core workflow is local-first and reproducible.

The primary call to action is not “generate AI images.” It is:

> Create a useful, labeled computer-vision dataset from the images you already have.

The page should prioritize clarity, proof, and trust over visual novelty.

---

## Visual direction

### Overall character

SynthLine should feel:

- Precise but approachable.
- Technical without being cold.
- Calm rather than hyperactive.
- Practical rather than promotional.
- Research-informed rather than speculative.
- Designed for people who inspect data carefully.

The visual language should resemble a well-designed specialist software product: strong typography, deliberate spacing, useful image comparisons, restrained color, and clear hierarchy.

### Do not use

The following patterns are explicitly discouraged:

- Grid-line backgrounds.
- Blueprint or circuit-board motifs.
- Neon cyberpunk styling.
- Large glowing blobs behind every section.
- Excessive glassmorphism.
- Repeated floating cards with identical rounded corners.
- Fake terminal panels used only as decoration.
- Generic robot, brain, neural-network, or hologram illustrations.
- Random abstract AI-generated artwork.
- Excessive pill-shaped buttons and labels.
- Overuse of gradients.
- Huge text that occupies most of the first screen without explaining the product.
- Animated counters with no meaningful data.
- Stock photography of people looking at dashboards.
- Claims that are not supported by a benchmark or implemented feature.

### Layout principles

- Use generous whitespace and a strong reading column.
- Use an asymmetric editorial composition where appropriate.
- Let real product imagery provide the visual interest.
- Use horizontal rules sparingly and only as structural separators.
- Prefer image frames, captions, labels, and annotations over decorative patterns.
- Keep the number of simultaneous visual elements low.
- Make every section answer a specific user question.
- Avoid making every section look like a three-column card grid.

---

## Suggested visual theme

A light, warm technical theme is recommended for the public landing page. It differentiates SynthLine from the common dark AI-product aesthetic and allows image comparisons to remain the visual focus.

```css
:root {
  --color-page: #f5f4ef;
  --color-surface: #fbfaf6;
  --color-surface-muted: #ebeae4;
  --color-ink: #17212b;
  --color-ink-muted: #65717a;
  --color-line: #d8d8d0;

  --color-accent: #087f8c;
  --color-accent-dark: #075d68;
  --color-accent-soft: #d8eeee;
  --color-highlight: #e6b85c;
  --color-success: #287856;
  --color-warning: #9b6817;
}
```

The colors are suggestions, not a rigid requirement. The important qualities are warmth, legibility, restraint, and contrast.

### Typography

Use one expressive but highly readable display face and one neutral text face, or use a single excellent variable sans-serif with careful weight changes.

Recommended characteristics:

- Headings: compact, confident, and not oversized.
- Body copy: relaxed line height and readable measure.
- Metadata: small uppercase labels or restrained monospace text.
- Code: a practical monospace font used only where it communicates technical detail.

Avoid pairing multiple trendy fonts or using a monospace font for the entire interface.

### Buttons

Use rectangular buttons with modest corner radii, clear labels, and enough padding. Primary actions should look dependable rather than playful.

```text
[Try locally]
[See how it works]
[Read the documentation]
```

Avoid vague calls to action such as “Unlock the future,” “Start creating,” or “Experience AI.”

---

## Page structure

```text
Landing page
├── Navigation
├── Hero: a few images to labeled data
├── Product evidence visual
├── Capability statement
├── The data problem
├── Four-step workflow
├── Interactive or recorded image comparison
├── Validation and reproducibility
├── Export compatibility
├── Local-first privacy
├── Focused use cases
├── Developer quick start
├── Transparent project status
├── Final call to action
└── Footer
```

The page should not be a long sequence of identical feature cards. Vary the composition between sections: use a split layout, a full-width comparison, a short editorial block, a code example, and a compact list where appropriate.

---

## 1. Navigation

The navigation should be quiet and functional.

```text
SynthLine       Product    How it works    Developers    Benchmarks    Docs
                                                            GitHub   Try locally
```

### Navigation behavior

- Keep the logo and product name left-aligned.
- Keep one primary action visible on desktop.
- Use a compact menu on mobile.
- Do not make the navigation sticky unless testing shows it improves usability.
- Do not add a pricing link until pricing exists.
- Do not imply that hosted accounts are available if the product is still local-first.

Suggested links:

- Product.
- How it works.
- Developers.
- Benchmarks.
- Documentation.
- GitHub.
- Try locally.

If a destination is not yet available, omit the link rather than displaying a dead or misleading route.

---

## 2. Hero section

### Recommended headline

```text
Turn a few normal images into labeled computer-vision data.
```

### Recommended supporting copy

```text
SynthLine creates reproducible synthetic defects, masks, bounding boxes,
and dataset reports from a small set of real images—so you can prototype
and validate vision systems without waiting for rare defects or expensive labeling.
```

### Primary actions

```text
[Try SynthLine locally]    [See how it works]
```

Secondary links may include:

```text
Read the documentation
View benchmark status
Explore the repository
```

### Hero composition

Use a broad, quiet layout with the copy on one side and a real product-data composition on the other. Do not use a dashboard screenshot as the hero.

The hero visual should show a single image moving through a clear transformation:

```text
Normal seed image → Generated scratch → Mask overlay → Export metadata
```

A possible composition:

```text
┌───────────────────────────────┬─────────────────────────┐
│                               │                         │
│  Headline and explanation     │  Image comparison       │
│                               │                         │
│  Actions and small proof      │  Original / generated   │
│                               │  Mask / annotation      │
└───────────────────────────────┴─────────────────────────┘
```

The visible image should be a real example asset, not an abstract illustration. Add a small caption beneath it:

```text
One source image, one controlled variation, labels generated by construction.
```

A restrained metadata strip can show:

```text
TYPE       scratch
SEVERITY   moderate
SEED       12345
MASK       valid
FORMAT     COCO
```

Do not animate all of these values. If the example is animated, use one slow, purposeful transition between the original and labeled result.

---

## 3. Proof strip

Immediately below the hero, show four concise product properties.

```text
Local-first        Reproducible runs        Exact masks        Portable exports
```

Each item should have a short explanation revealed on hover or visible below it on smaller screens:

```text
Local-first
Run the procedural workflow without uploading private images.

Reproducible runs
Record the configuration, generator version, and random seed.

Exact masks
Create masks during generation and derive boxes from them.

Portable exports
Use COCO, YOLO, images, masks, metadata, and reports elsewhere.
```

Use simple line icons only if they clarify meaning. Text should remain understandable without icons.

---

## 4. The data problem

### Suggested heading

```text
Real defects are expensive to collect.
```

### Suggested copy

```text
Computer-vision teams often have plenty of normal images and very few examples
of the failures they need to detect. SynthLine helps turn that starting point
into a controlled experiment without pretending that synthetic data replaces reality.
```

Use a two-column editorial layout rather than a card grid.

#### Left column: the constraints

- Defects may be rare.
- Manual labeling is slow.
- Product images may be confidential.
- Experiments need to be repeatable.
- A visually plausible image is not proof of usefulness.

#### Right column: the SynthLine approach

```text
Start with normal images.
Define a controlled variation.
Generate the image and labels together.
Inspect the result.
Test it against real data when available.
```

This section should establish honesty: SynthLine is a tool for creating and evaluating experiments, not a promise that every generated image will improve a model.

---

## 5. Workflow section

### Suggested heading

```text
A direct path from seed images to an exportable dataset.
```

Present the workflow as a numbered vertical sequence or a horizontal process line without background grid lines.

### 01 — Add seed images

```text
Import a small set of real, normal images. SynthLine checks dimensions,
color, blur, duplicates, and consistency before generation starts.
```

### 02 — Define a variation

```text
Choose a generator such as scratch or discoloration, then set quantity,
severity, frequency, and random seed.
```

### 03 — Generate and inspect

```text
Create images, masks, boxes, metadata, previews, and a quality report in one run.
```

### 04 — Export or evaluate

```text
Export to a familiar dataset format, or compare a synthetic-data experiment
against held-out real images when validation data is available.
```

Each step should have one visual artifact: a file thumbnail, configuration fragment, mask overlay, or report excerpt.

---

## 6. Product evidence visual

The landing page should include one substantial image-review section. This is more valuable than several decorative feature illustrations.

### Recommended heading

```text
See the image, the label, and the configuration together.
```

Use a large comparison viewer with four states:

```text
Original    Generated    Mask overlay    Bounding box
```

The controls should be understated tabs or text buttons, not oversized pill controls.

The viewer should support:

- Original and generated image comparison.
- Mask visibility.
- Bounding-box visibility.
- Defect-type selection.
- Severity selection where example assets exist.
- A small metadata drawer.

The first version should use static, pre-generated examples. Do not run generation in the browser or imply that the public demo is a hosted generation service.

### Example metadata drawer

```json
{
  "source_seed": "seed_004.png",
  "generator": "scratch",
  "severity": "moderate",
  "random_seed": 884321,
  "mask_area": 1674,
  "export": "COCO"
}
```

The metadata should be readable and useful. Avoid displaying arbitrary technical values only to make the product look sophisticated.

---

## 7. Core capabilities

### Suggested heading

```text
Built for dataset work, not image-generation theater.
```

Use a varied two-column layout with short sections rather than six identical cards.

### Procedural generators

Create explainable scratches, stains, discoloration, and controlled appearance variations.

### Labels by construction

Create pixel masks as part of the generation process and derive bounding boxes from those masks.

### Reproducible runs

Record source images, configuration, generator version, parameters, and random seed.

### Dataset QA

Check masks, boxes, duplicates, class balance, image statistics, failure rates, and split leakage.

### Standard exports

Produce COCO, YOLO, images, masks, JSONL metadata, and reports for existing pipelines.

### Real-world evaluation

When real labels are available, compare a synthetic-data experiment against an appropriate baseline.

Each capability should be paired with an actual artifact or screenshot from the product. If an artifact does not exist yet, describe the feature as planned rather than presenting a fabricated interface.

---

## 8. Validation section

This section should be prominent because it is a key product differentiator.

### Suggested heading

```text
A generated image is not automatically useful data.
```

### Suggested copy

```text
SynthLine helps you inspect what was generated, verify that the labels are valid,
and measure whether synthetic data helps on real images.
```

Show a compact report excerpt, for example:

```text
Generation success rate       98.7%
Invalid masks                 0
Duplicate outputs             3
Seed leakage                 0
Average mask area             2.8%
```

Only show measured values from a real run. Until benchmarks exist, use explicit status labels:

```text
Benchmark status: in progress
```

Never invent model-performance numbers, customer logos, user counts, or accuracy improvements.

### Report interaction

A “View sample report” link can open a real static report or a documented example. The report should show:

- Label validity.
- Mask-area distribution.
- Severity distribution.
- Duplicate detection.
- Seed coverage.
- Split leakage.
- Generation failures.
- Probe-model results when available.

---

## 9. Export compatibility

### Suggested heading

```text
Use the output in the tools you already have.
```

The page should show export formats as a simple text row or a compact list:

```text
COCO JSON    YOLO    Ultralytics    PyTorch    TensorFlow    Roboflow
```

Avoid using logos unless the relevant brand usage is permitted and the integration has actually been tested.

Show a copyable command:

```bash
synthline export \
  --run ./runs/scratch-500 \
  --format yolo \
  --output ./exports/scratch-yolo
```

Supporting copy:

```text
SynthLine produces portable artifacts rather than locking your dataset into a proprietary platform.
```

---

## 10. Local-first and privacy section

### Suggested heading

```text
Keep private images under your control.
```

### Suggested copy

```text
The core procedural workflow is designed to run locally without an API key.
That makes SynthLine useful for confidential product images, research datasets,
classrooms, and offline development environments.
```

Use a restrained checklist:

```text
No API key for the core workflow
No upload required for local generation
Reproducible local artifacts
Project-level output organization
Explicit deletion and retention plans for future hosted workflows
```

Do not claim enterprise security certifications, hosted encryption, tenant isolation, or compliance features until they exist and have been verified.

Future hosted capabilities may include:

- Private workspaces.
- Team permissions.
- Retention controls.
- Audit logs.
- API access.
- Self-hosted deployment options.

Clearly label these as planned if they are not implemented.

---

## 11. Focused use cases

Keep the use-case section narrow and practical.

### Surface inspection

Create controlled scratch and stain examples for product-surface inspection prototypes.

### Research and education

Build reproducible datasets for computer-vision experiments, assignments, and demonstrations.

### Rapid prototyping

Test a model architecture before investing in a larger real-world data-collection effort.

### Machine-vision pilots

Explore defect categories and labeling strategies before a production deployment.

Use short descriptions and real example images. Do not list every possible industry. SynthLine will appear more credible if it demonstrates depth in a small number of use cases.

---

## 12. Developer quick start

The landing page should include a practical installation block.

```bash
git clone https://github.com/thrivecodes/synthline.git
cd synthline

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

synthline generate \
  --seeds ./data/good \
  --defect scratch \
  --count 100 \
  --output ./runs/demo \
  --seed 12345
```

Include links to:

```text
Read the developer documentation
View example projects
Explore the CLI reference
```

The first successful experience should be as short as possible. The landing page must not send users through account creation, API-key setup, or a sales form to try the local workflow.

---

## 13. Transparent product status

Use a compact status section near the bottom of the page.

```text
Current status

✓ Product direction defined
✓ Procedural generation architecture defined
✓ Reproducible metadata model defined
○ Core implementation in progress
○ Public benchmark in progress
○ Browser workflow planned
○ Hosted collaboration planned
```

Update this section as the project progresses. Transparency is part of the product’s credibility, especially because synthetic data is often oversold.

---

## 14. Final call to action

### Suggested heading

```text
Start with the images you already have.
```

### Suggested copy

```text
Generate labeled experiments, validate them against reality,
and build your next computer-vision prototype with less manual data preparation.
```

### Actions

```text
[Try locally]    [Read the documentation]    [View benchmark status]
```

The final call to action should repeat the product’s actual next step. Do not use a generic “Get started with AI” message.

---

## 15. Footer

Suggested footer structure:

```text
SynthLine
Synthetic visual-data generation for computer vision.

Product
How it works
Benchmarks
Roadmap

Developers
Documentation
CLI reference
Examples
GitHub

Trust
Privacy
Data handling
Security

© SynthLine. All rights reserved.
```

Because the project is currently private, omit public links that do not exist. A simple footer is preferable to a large sitemap full of placeholder pages.

---

## Product application relationship

The marketing page and product application should share design tokens but should not look identical.

The landing page should be editorial and explanatory. The application should be focused and operational.

### Application navigation

```text
Overview
Seed images
Generation runs
Validation
Exports
Project settings
```

### Application layout

```text
┌───────────────┬─────────────────────────────────────────┐
│ Project nav   │ Workspace                               │
│               │                                         │
│ Overview      │ Configuration and image review          │
│ Seed images   │                                         │
│ Runs          │                                         │
│ Validation    │                                         │
│ Exports       │                                         │
└───────────────┴─────────────────────────────────────────┘
```

The product UI should avoid carrying over marketing decorations. It should emphasize:

- Clear status.
- Useful previews.
- Configuration visibility.
- Warnings before long jobs.
- Reproducibility.
- Export actions.

---

## UX principles

### Progressive disclosure

Show the few controls needed for a safe first generation. Place advanced parameters behind an explicit advanced section.

### Explainable actions

Every generated output should be traceable to a source image, configuration, generator, and random seed.

### Fast feedback

Show image-quality warnings and invalid configuration errors before generation starts.

### Safe defaults

The default configuration should produce valid, reviewable examples without requiring expert knowledge.

### Evidence before animation

Use motion only when it explains a transformation, such as an original image becoming a labeled generated image. Avoid continuous ambient animation.

### Accessible visual design

Support:

- Keyboard navigation.
- Visible focus states.
- Adequate color contrast.
- Non-color indicators for warnings and labels.
- Responsive layouts.
- Clear loading and error states.
- Descriptive alternative text for image comparisons.
- Reduced-motion preferences.

### Honest status communication

Use clear states:

```text
Ready
Generating
Completed
Completed with warnings
Failed
Cancelled
```

Avoid vague messages such as “Something went wrong” or “Optimizing.”

---

## Implementation guidance

The landing page should be implemented only after the core generation workflow has real screenshots, example outputs, and a documented configuration format.

Do not build a highly polished marketing site around unimplemented promises. Start with a small static page using verified artifacts, then expand as the product earns more evidence.

Recommended frontend structure:

```text
web/
├── app/
│   ├── (marketing)/
│   │   ├── page.tsx
│   │   ├── product/
│   │   ├── how-it-works/
│   │   ├── benchmarks/
│   │   └── docs/
│   └── (app)/
│       ├── projects/
│       ├── generations/
│       ├── validation/
│       └── exports/
├── components/
│   ├── marketing/
│   ├── image-review/
│   ├── reports/
│   └── ui/
├── styles/
│   ├── tokens.css
│   └── globals.css
└── public/
    ├── examples/
    ├── screenshots/
    └── brand/
```

Recommended future stack:

- Next.js or React.
- TypeScript.
- Tailwind CSS or a small custom design system.
- Accessible component primitives.
- Lightweight charting for validation reports.
- Static example assets for the public page.
- FastAPI for the backend interface.

The browser should not contain a separate generation implementation. It should call the same application services used by the CLI.

---

## Pre-launch checklist

Before publishing the landing page, verify:

- The headline accurately describes the current product.
- Every screenshot comes from the real product or is clearly marked as a concept.
- Every benchmark number has a reproducible source.
- No unsupported integrations are presented as complete.
- Local installation instructions work on a clean environment.
- The primary call to action works.
- Images have useful alternative text.
- The page works without animation.
- Mobile layouts are tested.
- No grid-line or blueprint background has been added.
- The page does not resemble a generic AI SaaS template.
- Future features are explicitly labeled as planned.
