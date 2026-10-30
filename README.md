# AI Image Generation Dataset — Cloudflare Workers AI

## 1. Project Overview

This repository contains the automated pipeline and dataset metadata for a controlled 288-image research collection generated via **Cloudflare Workers AI**. 

The dataset evaluates generative performance across **48 common object categories** under two contrasting visual paradigms:
- **Photorealistic Studio Photography:** Clear, commercial studio compositions with natural textures and realistic lighting.
- **Hand-Drawn Sketches:** Authentic graphite, pencil, or ink sketches featuring contouring, hatching, and expressive linework.

Each category includes exactly three photorealistic images and three hand-drawn sketch images (6 images per category × 48 categories = **288 total images**). The entire generation process is fully automated using a stateless Python client that communicates with Cloudflare's serverless AI platform.

---

## 2. Image Generation Model

Image synthesis was conducted using **FLUX.2 Klein 4B**, hosted directly on Cloudflare Workers AI:

- **Model Identifier:** `@cf/black-forest-labs/flux-2-klein-4b`
- **Developer:** Black Forest Labs
- **Architecture & Parameters:** Rectified flow transformer (~4 billion parameters) optimized for fast inference and high prompt fidelity.
- **Serving Infrastructure:** Serverless inference running entirely on Cloudflare's distributed GPU infrastructure. No local GPU hardware is utilized for inference; requests are dispatched via standard HTTPS REST calls.
- **Image Resolution:** 1024 × 1024 pixels (1:1 square aspect ratio).
- **Native Output Format:** Original JPEG encoding returned directly by the model. The pipeline captures and writes the original bytes directly to disk without intermediate PNG transcoding or recompression.

Official Documentation: [Cloudflare Workers AI — FLUX.2 Klein 4B](https://developers.cloudflare.com/workers-ai/models/flux-2-klein-4b/) | [Black Forest Labs](https://blackforestlabs.ai/)

---

## 3. Master Prompt Templates

To maintain scientific consistency and eliminate prompt drift, all generations are driven by two invariant prompt templates:
1. `prompts/photorealistic.txt`
2. `prompts/sketch.txt`

### Dynamic Object Substitution
Each template establishes strict guidelines for subject clarity, composition, lighting, style, and square 1024 × 1024 resolution. During execution, the Python automation pipeline dynamically updates only the first line of the prompt to inject the target category:

```text
OBJECT: dog
OBJECT: airplane
OBJECT: chair
```

All subsequent instructions, constraints, and negative styling directives remain completely identical across every object in that style.

<details>
<summary><strong>View Complete Master Prompt: Photorealistic Template</strong></summary>

```text
OBJECT: dog

Generate exactly one high-quality, photorealistic AI image representing the object specified above.

IMAGE GENERATION REQUIREMENTS:

1. Subject:
   - The image must clearly represent the specified object as the primary subject.
   - The subject should be easily recognizable.
   - Allow natural variation in the subject's appearance, characteristics, and presentation.

2. Composition and Environment:
   - The subject may appear in any realistic setting or situation.
   - Use a natural, realistic environment or background appropriate for the subject.
   - Additional objects, environmental elements, and contextual details are allowed.
   - Allow creative freedom in the composition, positioning, camera perspective, and surrounding environment.
   - Avoid extreme close-ups or unusual camera angles that make the primary subject difficult to recognize.

3. Visual Quality:
   - Generate a photorealistic image with natural colors, realistic textures, and high visual quality.
   - Use realistic lighting appropriate for the environment.
   - Ensure that the primary subject is visually clear and recognizable.
   - Maintain a natural and visually coherent scene.

4. Technical Specifications:
   - Aspect ratio: 1:1 (square).
   - Target resolution: 1024 × 1024 pixels.
   - Generate exactly ONE image.
   - Maintain high image quality and visual clarity.

5. General Instructions:
   - Allow natural variation in the subject's appearance, colors, surroundings, and overall composition.
   - Do not generate collages, split-screen images, or multiple panels.

FINAL INSTRUCTION:

Generate one high-quality, photorealistic, square image representing the object specified above, following all the specifications while allowing natural creativity in the scene, background, and composition.
```

</details>

<details>
<summary><strong>View Complete Master Prompt: Hand-Drawn Sketch Template</strong></summary>

```text
OBJECT: dog

Generate exactly one high-quality, authentic hand-drawn sketch image representing the object specified above.

IMAGE GENERATION REQUIREMENTS:

1. Subject:
   - The image must clearly represent the specified object as the primary subject.
   - The subject should be easily recognizable.
   - Allow natural variation in the subject's appearance, characteristics, and presentation.

2. Composition and Environment:
   - The subject may appear in any fitting setting, contextual environment, or clean background.
   - Use a natural, expressive sketched environment or background appropriate for the subject.
   - Additional sketched objects, environmental elements, and contextual details are allowed.
   - Allow creative freedom in the composition, positioning, perspective, and surrounding sketch elements.
   - Avoid extreme close-ups or unusual angles that make the primary subject difficult to recognize.

3. Visual Style and Artistic Quality:
   - Generate an authentic hand-drawn sketch with expressive pencil, graphite, charcoal, or ink linework.
   - Use natural hatching, cross-hatching, contour lines, and realistic shading appropriate for a hand-drawn artwork.
   - Ensure that the primary subject is visually clear, distinct, and recognizable.
   - Maintain an organic, coherent, and artistic hand-drawn presentation.

4. Technical Specifications:
   - Aspect ratio: 1:1 (square).
   - Target resolution: 1024 × 1024 pixels.
   - Generate exactly ONE image.
   - Maintain high artistic quality and visual clarity.

5. General Instructions:
   - Allow natural variation in the subject's pose, linework style, shading depth, and overall composition.
   - Do not generate collages, split-screen images, or multiple panels.

FINAL INSTRUCTION:

Generate one high-quality, authentic hand-drawn, square sketch image representing the object specified above, following all the specifications while allowing natural creativity in the scene, background, and artistic composition.
```

</details>

---

## 4. Automated Generation Workflow

The pipeline is implemented in `generate_images.py`. It operates as a fully stateless, deterministic batch processor designed for autonomous execution, quota management, and cryptographic tracking.

```
[categories.json] (48 Categories)
        │
        ├──> Select Master Template (Photorealistic / Sketch)
        │
        ├──> Format Prompt ("OBJECT: <category>")
        │
        ├──> Send Stateless REST API Request ──> [Cloudflare Workers AI]
        │                                             │ (FLUX.2 Klein 4B)
        │    Validate JPEG Header & 1024×1024 <───────┘ (Direct Base64/Bytes)
        │
        ├──> Atomic Write: images/<style>/<category>/<category>_xx.jpg
        │
        └──> Checkpoint Record in dataset_manifest.json (SHA-256 + dHash)
```

### End-to-End Execution Sequence
1. **Category Ingestion:** Loads the 48 standardized object categories from `categories.json`.
2. **Template Selection:** Selects the corresponding style template (`prompts/photorealistic.txt` or `prompts/sketch.txt`).
3. **Prompt Composition:** Dynamically substitutes line 1 with the target category while locking all remaining instructions.
4. **Stateless API Invocation:** Issues an isolated HTTPS POST request to Cloudflare Workers AI (`ai/run/@cf/black-forest-labs/flux-2-klein-4b`). Each call is completely independent—no conversation history or previous image contexts are shared.
5. **Direct Payload Extraction:** Decodes the base64/binary image payload without decompression or format re-encoding.
6. **Technical Validation:** Asserts that bytes start with genuine JPEG markers (`\xff\xd8\xff`) and that dimensions are strictly 1024 × 1024 pixels.
7. **Atomic Disk Persistence:** Directs image bytes to atomic temporary files before finalizing paths under `images/<style>/<category>/<category>_<index>.jpg`.
8. **Manifest Ledger Update:** Calculates SHA-256 cryptographic checksums and perceptual difference hashes (dHash), appending an immutable record to `dataset_manifest.json`.

### Six-Batch Execution Architecture
To respect provider quota boundaries and distribute rate allowances cleanly, the 288 tasks are structured into six discrete production batches:

| Batch | Style | Image Index | Categories | Target Images | Assigned Account |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Batch 1** | Photorealistic | `01` | 48 | 48 | `ACCOUNT_1` |
| **Batch 2** | Photorealistic | `02` | 48 | 48 | `ACCOUNT_2` |
| **Batch 3** | Photorealistic | `03` | 48 | 48 | `ACCOUNT_3` |
| **Batch 4** | Sketch | `01` | 48 | 48 | `ACCOUNT_4` |
| **Batch 5** | Sketch | `02` | 48 | 48 | `ACCOUNT_5` |
| **Batch 6** | Sketch | `03` | 48 | 48 | `ACCOUNT_6` |

---

## 5. How to Run the Pipeline

The automation pipeline provides a command-line interface in `generate_images.py` with support for full execution, single-batch processing, status inspections, dry runs, and mock simulations.

### 1. Generating the Complete Dataset (All 288 Images)
To run all six production batches sequentially with automated credential rotation and manifest checkpointing:
```bash
python3 generate_images.py --batch all
```
*Note: The generator checks existing files before making any API request. If interrupted, re-running this command safely resumes from the exact point of interruption without re-generating or overwriting completed images.*

### 2. Generating an Individual Batch
To execute a specific batch (e.g., Batch 1 for photorealistic index 01, or Batch 4 for sketch index 01):
```bash
python3 generate_images.py --batch 1
python3 generate_images.py --batch 4
```

### 3. Checking Dataset Progress and Status
To inspect the live progress across all six batches, including completed, pending, and blocked task counts:
```bash
python3 generate_images.py --status
```

### 4. Executing the Offline Test Suite
To run the comprehensive 11-domain automated test suite (verifying prompt invariance, byte preservation, image validators, error hierarchies, and manifest logic):
```bash
python3 test_offline.py
```

---

## 6. Project Architecture & File Descriptions

| File / Directory | Purpose & Description |
| :--- | :--- |
| **`generate_images.py`** | **Core Production Engine:** Manages CLI execution, multi-account authentication, prompt templating, REST API communication with Cloudflare Workers AI, original JPEG byte preservation, 1024 × 1024 image validation, atomic disk writing, and manifest checkpointing. |
| **`test_offline.py`** | **Offline Test Suite:** Implements 11 comprehensive unit and integration test domains. Validates prompt invariance, byte preservation, error hierarchies, and resume behavior locally without requiring network access or consuming API budget. |
| **`recover_blocked_tasks.py`** | **Controlled Recovery Tool:** Scans `dataset_manifest.json` for tasks flagged with safety moderation code 3030, isolates candidate tasks, protects existing completed files, and executes audited re-submissions under manual confirmation (`--confirm-recovery`). |
| **`generate_single_task.py`** | **Targeted Task Generator:** Helper script designed for granular single-task generation. Supports custom account delegation and semantic prompt disambiguation (used to resolve `photorealistic:airplane:01` with `OBJECT: civilian cargo airplane`). |
| **`categories.json`** | **Standard Category Manifest:** Defines the 48 standardized research object categories (e.g., `dog`, `airplane`, `chair`, `apple`, `train`, `car`, etc.) and batch target counts. |
| **`dataset_manifest.json`** | **Cryptographic Ledger & Audit History:** The single source of truth for the dataset. Tracks each image's completion status, output path, SHA-256 hash, perceptual difference hash (`dHash`), generation latency, technical validation metadata, and prior attempt history. |

---

## 7. Generated Dataset Summary

| Property | Value |
| :--- | :--- |
| **Total Categories** | 48 |
| **Photorealistic Images per Category** | 3 |
| **Sketch Images per Category** | 3 |
| **Total Photorealistic Sub-Dataset** | 144 images |
| **Total Sketch Sub-Dataset** | 144 images |
| **Total Production Dataset** | **288 images (100.0% complete)** |
| **Image Resolution** | 1024 × 1024 pixels |
| **Aspect Ratio** | 1:1 (Square) |
| **Native Storage Encoding** | Genuine JPEG |
| **Total Storage Footprint** | ~224.08 MB (234,961,064 bytes) |
| **Manifest Checkpoint File** | `dataset_manifest.json` |

### Output Directory Structure
```
images/
├── photorealistic/
│   ├── airplane/
│   │   ├── airplane_01.jpg
│   │   ├── airplane_02.jpg
│   │   └── airplane_03.jpg
│   ├── dog/
│   │   ├── dog_01.jpg
│   │   ├── dog_02.jpg
│   │   └── dog_03.jpg
│   └── ... (46 additional categories)
└── sketch/
    ├── airplane/
    │   ├── airplane_01.jpg
    │   ├── airplane_02.jpg
    │   └── airplane_03.jpg
    ├── dog/
    │   ├── dog_01.jpg
    │   ├── dog_02.jpg
    │   └── dog_03.jpg
    └── ... (46 additional categories)
```

---

## 8. Implementation Highlights

- **Stateless 1:1 API Requests:** Every image is generated via an independent HTTP request, preventing prompt contamination, latent state carryover, or conversational drift.
- **Original Format Preservation:** The pipeline retains the native JPEG bytes generated by Cloudflare Workers AI. No intermediate PNG conversions or quality-degrading re-encodings occur.
- **Strict Pre-Write Validation:** Every generated byte stream is inspected for genuine JPEG signatures (`\xff\xd8\xff`) and exact 1024 × 1024 resolution before being saved to disk.
- **Immutable Manifest & Checkpointing:** Progress is tracked per image in `dataset_manifest.json`, storing timestamps, response latencies, cryptographic SHA-256 hashes, and perceptual hashes (dHash).
- **Resume Protection:** If execution is paused or restarted, the script verifies existing files and resumes from the exact point of interruption without overwriting or re-generating valid images.
- **Fault-Tolerant Resilience:** Features automated exponential backoff for transient HTTP network errors while capturing automated downstream moderation events with comprehensive audit trails.

---

## 9. Official References

- **Cloudflare Workers AI Platform:** [https://developers.cloudflare.com/workers-ai/](https://developers.cloudflare.com/workers-ai/)
- **FLUX.2 Klein 4B Model Reference:** [https://developers.cloudflare.com/workers-ai/models/flux-2-klein-4b/](https://developers.cloudflare.com/workers-ai/models/flux-2-klein-4b/)
- **Black Forest Labs:** [https://blackforestlabs.ai/](https://blackforestlabs.ai/)
