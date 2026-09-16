# SynthLine (working name)

**Synthetic defect-data generation for manufacturing computer vision — turn a handful of "good part" images into a large, labeled dataset of realistic defects, without needing real defective parts.**

---

## 1. Problem

Any team trying to build a custom visual inspection model faces the same chicken-and-egg problem: supervised defect-detection models need many labeled examples of defects to train well, but defects are (thankfully) rare and inconsistent in real production. A line might run for weeks without producing a single example of a given defect type, and even when defects do occur, photographing and labeling them consistently is rarely anyone's job. Waiting to accumulate enough real defective-part photos can take months, and by the time enough exist, the product line may have already changed — a new part revision, a new supplier, a new tolerance spec — making the collected data partly obsolete.

Manufacturers and machine-vision integrators currently have three unsatisfying options: fall back to expensive full digital-twin simulation platforms built for large enterprises with dedicated simulation teams; manually stage and photograph fake defects (slow, inconsistent, and limited to defects someone thought to simulate); or simply accept a weaker model trained on whatever thin real-world data exists and hope it generalizes.

## 2. Solution

A tool that takes a small number of real "good part" images (or a 3D scan/CAD file, if available) plus a defect taxonomy the customer defines (scratch, dent, discoloration, misalignment, missing component, etc.) and generates a large, labeled synthetic dataset of that part with realistic defect variations — ready to drop into a training pipeline. No digital twin required, no waiting for real defects to occur, and no dedicated simulation engineer needed on the customer's side.

Core generation approaches (to prototype and compare in v1):
- **Procedural perturbation** — programmatic texture/geometry edits (scratches, dents, discoloration) applied directly to real "good" images with randomized parameters (location, size, severity, orientation).
- **Generative inpainting** — diffusion-based local edits that insert realistic defect regions into real images while preserving lighting, material appearance, and surrounding context.
- **Domain randomization** — varying lighting, background, and camera angle across generated samples so downstream models generalize better to real, imperfect factory-floor conditions rather than overfitting to one clean capture setup.

Output: labeled images (with bounding boxes or segmentation masks per defect) in a format that plugs directly into standard CV training pipelines, plus a small held-out validation split generated the same way.

## 3. Target customer (ICP for MVP)

- **Primary — machine-vision integrators and consultancies.** Firms that design and deploy custom inspection systems for multiple manufacturing clients, typically 5–50 people, who hit the exact same cold-start data problem on every new client engagement. One relationship here can validate the tool across several real projects, which is faster and lower-friction than convincing one manufacturer at a time.
- **Secondary — small-to-mid manufacturers with an in-house data/ML person** (or a part-time contractor) trying to build their own inspection model in-house rather than buying a full vendor solution, who are stuck on data rather than on modeling know-how.
- **Tertiary — CV teams at larger inspection vendors** who want to supplement thin real-world defect datasets for edge cases (rare defect types, new product lines) rather than replace their whole data pipeline.

**Buyer persona:** typically an engineering lead or ML/data engineer, not a plant manager — this is a technical-buyer sale, not an ops/floor sale, which changes the pitch: lead with model accuracy and time-to-usable-dataset, not with "reduce defects on your line" messaging.

## 4. Why now

- Synthetic data tooling for industrial visual inspection is called out as a genuine, underfilled gap: large-enterprise tools like NVIDIA Omniverse solve this with full digital twins, but smaller manufacturers and integrators need something far simpler that doesn't require building a 3D simulation of the whole line.
- The broader trend in visual QC is toward needing fewer labeled samples and faster deployment — synthetic data generation is a direct answer to that trend rather than a side feature bolted onto an inspection product.
- Generative image models (diffusion-based inpainting/editing) have gotten good enough at small, localized, realistic edits that this is newly practical for a small team to build well, rather than requiring a large research group or a custom-rendered 3D pipeline.
- Machine vision adoption in manufacturing is accelerating broadly (market roughly doubling by 2030, with most manufacturers planning AI-based visual inspection deployments soon) — every one of those new deployments hits the same data cold-start problem this product solves.

## 5. Competitive landscape

- **Full digital-twin / simulation platforms** (e.g., NVIDIA Omniverse-based workflows): powerful, photorealistic, but require 3D asset creation and simulation expertise most small teams don't have in-house. We compete by being usable from 2D photos alone, with no simulation background required.
- **End-to-end inspection vendors** (Landing AI, Elementary, Covision Quality, Instrumental, and similar): sell a full inspection product including the model and deployment, not just the training data. We are not competing with them directly — we could plausibly become a data supplier *to* smaller vendors or integrators building something similar, rather than a rival to the big platforms.
- **Manual data augmentation libraries** (standard CV augmentation tooling — flips, rotations, color jitter): widely used but only vary existing images, they don't synthesize entirely new defect instances. We're solving a different, harder problem: generating defects that don't exist yet in the seed set at all.
- **Our wedge:** nobody targeting the *small manufacturer / integrator* segment specifically with a "no 3D, no simulation team, no real defect data required" product. That's the gap this MVP is built to test.

## 6. MVP scope

**In scope (v1):**
- Single part type per project, 2D images only (no 3D/CAD ingestion yet).
- A small defect taxonomy per project (3–6 defect types defined by the customer, e.g., scratch / dent / discoloration).
- Upload a handful (10–50) of real "good part" images as the seed set.
- Generate a labeled synthetic dataset (target: hundreds to low thousands of images) with bounding-box or mask labels per defect instance.
- Export in a standard format (e.g., COCO-style JSON + images) that plugs into common training pipelines.
- A simple web UI: upload seed images → define defect types (with reference examples or text description) → generate → preview → download.
- A basic internal validation harness (see §11) that reports estimated dataset quality before the customer even downloads it.

**Explicitly out of scope for v1:**
- 3D/CAD-based synthesis or full digital-twin simulation.
- Training or hosting the downstream inspection model ourselves — v1 is a data tool, not an end-to-end inspection product.
- Multi-part-type projects in a single run.
- Fine-grained control over generation parameters beyond defect type and rough severity/frequency — keep the interface simple for v1.
- Video/temporal defects (e.g., intermittent mechanical faults visible only over time) — static image defects only.
- Non-visual defect signals (sound, vibration, thermal) — camera-visible defects only.

## 7. System architecture

```
                         ┌─────────────────────┐
 [Seed "good part"       │   Ingestion Service   │
  images, 10–50] ───────▶│  (validate, store,     │
                         │   basic QA checks)     │
                         └──────────┬───────────┘
                                    ▼
 [Defect taxonomy,        ┌─────────────────────┐
  reference examples] ───▶│  Generation Engine    │
                         │  - procedural           │
                         │  - generative inpaint   │
                         │  - domain randomization │
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  Labeling / Export      │
                         │  (auto-generated boxes/ │
                         │   masks + metadata)     │
                         └──────────┬───────────┘
                                    ▼
                         ┌─────────────────────┐
                         │  Validation Harness     │──▶ Quality report to customer
                         │  (train small probe      │    (est. accuracy signal)
                         │   model, test vs. real   │
                         │   benchmark defects)     │
                         └─────────────────────┘
```

**Components:**
- **Ingestion service** — accepts uploaded images, runs basic sanity checks (resolution, lighting consistency, part visibility), and flags seed sets that are too inconsistent to generate good synthetic data from before the customer wastes a generation run on bad input.
- **Generation engine** — the core IP. Houses the three generation approaches (§2) as swappable strategies; v1 ships with procedural perturbation as the default (fastest, most predictable), generative inpainting as an opt-in for defect types that need more visual realism.
- **Labeling/export** — since defects are synthetically inserted, we know exactly where they are — labels are generated automatically alongside the images rather than requiring a separate annotation step, which is a meaningful speed advantage over any workflow that starts from real photos needing manual labeling.
- **Validation harness** — trains a small internal probe model on a sample of the generated data and reports an estimated quality signal before the customer downloads anything. This is as much a product-trust feature as an engineering one (see §11).

## 8. Data / export schema (illustrative)

```json
{
  "image": "part_0001_scratch.png",
  "part_type": "bracket_v3",
  "defects": [
    {
      "type": "scratch",
      "bbox": [124, 88, 212, 101],
      "mask": "part_0001_scratch_mask.png",
      "severity": "moderate"
    }
  ],
  "source": "synthetic",
  "generation_method": "procedural_perturbation"
}
```

- Exports ship as COCO-style JSON plus an image/mask folder by default, since that's the format most CV training pipelines already expect — a customer's existing training code should need minimal changes to consume our output.
- Every synthetic image is tagged with its generation method and defect metadata, so customers (and we, internally) can later analyze which generation technique produces the most useful training data per defect type.

## 9. Data strategy (cold start)

This concept is specifically designed to not need our own proprietary dataset:

1. Internal development and testing use public industrial datasets (e.g., MVTec AD and similar anomaly-detection benchmarks) purely to validate that models trained on our synthetic output perform reasonably against real defect examples.
2. Each customer brings their own seed images (their real "good parts") — we never need to acquire or own a manufacturing dataset ourselves.
3. Design partners (2–3 manufacturers or integrators) get early access in exchange for feedback on how well the generated data performs when they actually train their own models on it — this is our real-world validation loop, not a data acquisition play.
4. Over time, with explicit customer consent, anonymized performance signal (not the images themselves) — e.g., "procedural perturbation worked well for scratch defects on metal parts" — can become an internal knowledge base that improves generation quality across customers without ever pooling anyone's actual images.

## 10. Security & data privacy

- Customer part images are frequently sensitive (unreleased product designs, proprietary tooling) — this needs to be treated as seriously as any B2B SaaS handling confidential IP from day one, not bolted on later.
- v1 commitments to design partners: images are used only to generate that customer's dataset, are not used to train any shared/cross-customer model without explicit opt-in, and are deletable on request.
- Longer-term (post-MVP): customer-specific storage isolation, audit logging of who accessed what, and a clear data retention policy are likely required before larger manufacturers will engage — worth flagging early even though it's not a v1 build item, since it affects how we structure storage from the start.

## 11. Testing & validation methodology

The core technical risk (sim-to-real gap) needs a real measurement, not just an assumption that generated data "looks right":

1. **Benchmark validation (pre-launch):** train small models on synthetic data generated from public benchmark "good" images, then test against the same benchmark's real defect examples. This gives an internal, repeatable accuracy signal before any customer is involved.
2. **Per-generation quality report:** every generated dataset ships with a lightweight internal report (via the validation harness in §7) — not a guarantee of real-world performance, but an early signal so customers aren't flying blind.
3. **Pilot-partner ground truth:** the real test is whether a model trained substantially on our synthetic data catches real defects when deployed by a design partner — this is the metric that actually matters and the one used for the go/no-go decision on continuing past MVP.
4. **Per-defect-type tracking:** track validation results separately per defect type (scratch vs. dent vs. discoloration, etc.) rather than as one aggregate number — some defect types will transfer far better than others, and knowing which is more useful than a single blended accuracy figure.

## 12. Business model (early thinking)

- **Pricing shape:** usage-based or per-project, not per-seat — customers use this in bursts (per new part type or product line), not continuously, so a flat monthly seat license is likely a mismatch for v1. A "per generated dataset" or tiered monthly credit model is worth testing with pilot partners.
- **Free pilot → paid conversion:** first dataset generation free (or heavily discounted) per design partner, converting to paid once they've validated it against their own model.
- **Integrator channel potential:** if machine-vision integrators become the primary customer, a channel/reseller motion (they use it across their own client projects) could scale faster than direct-to-manufacturer sales — worth testing which motion pilot partners actually prefer.
- Pricing specifics are explicitly not locked in for MVP — the pilot phase is partly about learning what customers are actually willing to pay for and how they'd rather be billed.

## 13. Pilot / go-to-market plan

- Identify 5–10 candidate machine-vision integrators and consultancies (smaller firms are more accessible and iterate faster than large ones).
- Offer a free pilot: generate a dataset for one of their live customer projects, and compare a model trained on it against their current approach or timeline.
- Success criteria for converting a pilot to paid: a model trained substantially on our synthetic data reaches usable accuracy meaningfully faster than their current data-collection timeline, on at least one real defect type.
- In parallel, run 1–2 pilots directly with manufacturers who have an in-house ML person, to validate both customer segments rather than assuming the integrator channel is the only viable path.

## 14. MVP success metrics

- **Technical:** a model trained on generated data reaches meaningfully-above-baseline detection accuracy on real defect examples for at least the 2–3 simplest defect types (scratch, discoloration) in the taxonomy.
- **Business:** at least 1 of 2–3 pilot partners uses generated data in a real model they deploy or seriously evaluate within 60 days.
- **Product:** seed-images-to-downloadable-dataset in under an hour for a first-time user, with no manual scripting required.
- **Trust/adoption signal:** at least one pilot partner is willing to be referenced or used as a case study — a proxy for whether the product delivered real, defensible value rather than just a technically interesting demo.

## 15. Roadmap

- **Phase 0 (weeks 1–4):** build and validate the generation pipeline against public benchmarks; confirm the sim-to-real gap is small enough to be useful before doing any customer outreach.
- **Phase 1 (weeks 5–10):** run 2–3 free pilots with integrators or manufacturers on real (but non-critical) inspection projects; instrument how well their models perform on our generated data.
- **Phase 2 (weeks 11–16):** convert pilots to paid; tighten the UI, export formats, and pricing model based on what actually blocked pilot users.
- **Phase 3 (post-MVP, not in scope now):** 3D/CAD ingestion for richer geometry-aware defects, multi-part-type projects, and a hosted "train the model for you" option as a natural upsell once the core data-generation quality is proven.

_(Timeframes are rough planning targets, not commitments — the real pacing depends on how fast the validation loop in §11 converges.)_

## 16. Key risks / open questions

- **Sim-to-real gap is the central risk:** if models trained on our synthetic data don't transfer to real production defects, the product has no value regardless of how good the images look to a human. This needs to be validated against real defect data as early as possible, not assumed.
- **Defect taxonomy limits:** some defect types (subtle dimensional deviations, complex multi-part assembly errors) may be much harder to synthesize convincingly than surface-level scratches or discoloration — v1 should lean into the defect types that are genuinely tractable rather than promising full generality.
- **Customer trust in synthetic data:** some manufacturers may be skeptical of training a QC model on generated rather than real defect images — the pilot's job is as much about building that trust with evidence as it is about the technology itself.
- **IP/confidentiality sensitivity:** part images can reveal proprietary product details, which may slow down design-partner recruitment more than expected — worth planning for NDAs as a normal part of pilot outreach, not an edge case.
- **Defensibility:** generation techniques themselves are not unique long-term; the moat has to come from generation quality per defect type, ease of use, and (eventually) an accumulated understanding of which techniques transfer well to which industries — worth being honest with the team about this from day one.

## 17. Team / roles (placeholder — fill in)

- Generative/CV modeling: _TBD_
- Pipeline/infra (generation + export): _TBD_
- Product/UI: _TBD_
- Pilot outreach & validation: _TBD_

## 18. Repository structure (proposed)

```
synthline/
├── ingestion/          # upload handling, seed-set QA checks
├── generation/          
│   ├── procedural/      # perturbation-based defect synthesis
│   ├── inpainting/       # diffusion-based generative edits
│   └── randomization/    # lighting/background/angle variation
├── labeling/            # auto-label generation, export formatting
├── validation/           # probe-model training + benchmark scoring
├── web/                  # upload → configure → generate → download UI
├── benchmarks/            # public dataset fixtures for internal testing
└── docs/
```

## 19. Getting started (dev environment — placeholder)

```bash
# clone repo
git clone <repo-url>
cd synthline

# generation pipeline
# ...

# web UI
# ...

# benchmark/validation scripts
# ...
```

_(Fill in once the initial stack is chosen — recommend prototyping the procedural-perturbation approach first since it's the fastest to get working end-to-end, then layering in generative inpainting for defect types that need more realism.)_

## 20. Glossary

- **Anomaly detection:** a modeling approach that learns what "normal" looks like and flags deviations, as opposed to classifying specific known defect categories.
- **Domain randomization:** varying non-essential visual factors (lighting, background, angle) during synthetic data generation so a model trained on it generalizes better to real-world conditions.
- **Sim-to-real gap:** the performance drop (if any) a model experiences when trained on synthetic data and tested on real-world data — the central risk this product needs to keep measured and small.
- **Seed set:** the small number of real "good part" images a customer provides as the starting point for synthetic generation.
