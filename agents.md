# 🤖 AGENTS.MD — Master Rules, Protocols & Operating Procedures

> **Project:** OpenMed (AI Hospital) — Multi-Organ Multi-Modal Medical Diagnostic Platform  
> **Repository:** `/home/honeypot/Projects/FAST_API/OpenMed/`  
> **Documentation Vault:** `/home/honeypot/Obsidian/~Honeypot/OpenMed/`  
> **Project Specification:** `/home/honeypot/Obsidian/~Honeypot/IDEAS/fyp.md`  
> **Lead Developer:** Muhammad Fezan (H0NEYP0T-466)  
> **Purpose:** Final Year Project (FYP) — 7th & 8th Semester  

---

## 🧭 Core Philosophy & Academic Standards

This project is a high-stakes university **Final Year Project (FYP)**. It is not a quick prototype, a hackathon submission, or an unverified demo. Every AI coding agent operating in this repository must function as a rigorous senior research engineer and pair programmer: meticulous, academically honest, technically precise, and dedicated to production-grade engineering.

---

## 📜 The 10 Mandatory Agent Rules

### 1. Visual & Design Consistency (Zero Drift)
* **Canonical Design System — Atelier Zero:** The frontend implements the **Atelier Zero** editorial design system (vault: `opendesign/design-systems/atelier-zero/`, contract documented in `[[design-system-atelier-zero]]` in the OpenMed Obsidian vault). The app presents as a printed medical annual: warm paper canvas (`#efe7d2`), ink hierarchy (`#15140f` → `#8b8676`), bone card surfaces (`#f7f1de`), 3D viewports styled as dark ink "plates" with corner brackets and `Plate Nº` captions, Roman-numeral section rules, and a **single coral accent (`#ed6f5c`)** per viewport.
* **Never invent new styles:** Strictly adhere to the established visual language, theme, and component patterns of the application.
* **Palette Preservation:** All colors must come from the token block in `src/index.css` (which copies the Atelier Zero schema exactly: `--paper`, `--bone`, `--ink*`, `--accent`, `--mustard`, `--olive`). Do not add, remove, tweak, or substitute theme colors, border radii, typography sizes, or UI palettes unless explicitly instructed by the user. Mustard is jewelry (≤1%, one nav ★); never a CTA fill. No pure white, no pure black, no glassmorphism, no emoji in UI copy.
* **Typography:** Inter Tight (display), Playfair Display Italic 500 (emotional nouns, Roman numerals, `fin.`), Inter (body), JetBrains Mono (coordinates, plate numbers, data rows) — all self-hosted via Fontsource, imported in `src/main.tsx`.
* **Component Reuse:** Reuse existing UI primitives from `src/components/` and the shared classes in `src/App.css` (`.plate`, `.eyebrow`, `.sec-rule`, `.hud-chip`, `.btn-primary/.btn-ghost`, `.pill`, `.dot`). Do not create one-off ad-hoc styled variants when a canonical component already exists.

---

### 2. Strict File & Folder Architecture
The codebase must remain impeccably organized as organs and modalities expand. Never place files haphazardly in root directories.

```
OpenMed/
├── agents.md                               # This master rules document
├── README.md                               # Root project overview
├── src/                                    # Frontend application (React 19 + TypeScript)
│   ├── assets/                             # Static assets, icons, anatomy SVGs
│   ├── components/                         # Shared & design-system UI components
│   │   ├── common/                         # Buttons, Modals, Loaders, ErrorBoundaries
│   │   ├── medical/                        # Medical image viewer, Canvas mask overlays
│   │   ├── ocr/                            # Document viewer, text extractor, entity tags
│   │   └── report/                         # Clinical report display and export
│   ├── features/                           # Feature modules
│   │   └── organs/                         # Organ-specific frontend views & hooks
│   │       ├── brain/
│   │       ├── lungs/
│   │       └── ... [N organs]
│   ├── types/                              # Central TypeScript definitions
│   └── utils/                              # Shared frontend utility functions
│
├── backend/                                # Backend services (FastAPI + Python 3.11+)
│   ├── app/
│   │   ├── api/v1/                         # API route controllers
│   │   ├── core/                           # Config, logging, security, middleware
│   │   └── organs/                         # Organ modular backends (Strict Isolation)
│   │       ├── brain/
│   │       │   ├── classification/         # Dataloaders, model wrappers, inference
│   │       │   ├── segmentation/           # Dataloaders, model wrappers, inference
│   │       │   └── README.md               # Brain-specific documentation
│   │       ├── lungs/
│   │       │   ├── classification/
│   │       │   ├── segmentation/
│   │       │   └── README.md
│   │       └── ... [N organs]
│   └── tests/                              # Comprehensive test suites
│
└── storage/                                # Ephemeral / local cache (GIT-IGNORED)
```

Every single organ module added must follow the exact structure:
```
<organ_name>/
├── classification/
│   ├── dataset.py                          # Dedicated dataset & dataloader logic
│   ├── preprocessor.py                     # Image preprocessing & augmentation
│   ├── model.py                            # Model architecture definition/loading
│   └── pipeline.py                         # Single-image inference pipeline
├── segmentation/
│   ├── dataset.py
│   ├── preprocessor.py
│   ├── model.py
│   └── pipeline.py
└── README.md                               # Organ-specific technical documentation
```

---

### 3. Proactive Git Hygiene (Zero-Prompt Gitignore)
* **Never ask permission:** Immediately and automatically append any large file patterns, model checkpoints, medical volumes, or cache directories to `.gitignore`.
* **Prohibited Git Files:** Under no circumstances should weights (`*.pt`, `*.pth`, `*.onnx`, `*.safetensors`, `*.bin`, `*.ckpt`), datasets (`*.nii`, `*.nii.gz`, `*.dcm`, raw image zip files), or virtual environments ever be committed.
* Always verify `git status` or inspect `.gitignore` before performing file writes that involve binary assets.

---

### 4. Research-First Protocol (No Outdated Assumptions)
* **Never rely solely on stale training memory:** For any modern deep learning model (e.g., DINOv2, SAM 2, SegFormer, ConvNeXt V2, U-Mamba, MedMamba, nnU-Net v2, TrOCR, PaddleOCR) or cutting-edge library APIs, **agents must explicitly research first**.
* Check official repositories, documentation, and research papers for the exact API signatures, pre-trained weight keys, normalization constants, and input tensor requirements before writing implementation code.

---

### 5. Clean, Readable, and Explicit Code
* **Readability over Brevity:** Always prioritize clean, maintainable, explicit, and self-documenting code over clever one-liners or compressed syntax, even if it requires significantly more lines.
* **Type Safety:**
  * **Frontend:** Strict TypeScript without exceptions. `any` is strictly banned. Use proper generic interfaces, unions, and Discriminated Unions.
  * **Backend:** Strict Python type hints and Pydantic v2 schemas for all payloads and returns.
* **Defensive Error Handling:** Always handle edge cases: missing files, empty tensors, zero division in metrics, invalid image dimensions, and out-of-memory (OOM) risks.

---

### 6. Strict Organ Independence (Zero Inter-Organ Coupling)
* **Total Autonomy:** Each organ package (e.g., `brain/`, `lungs/`, `retina/`) must be completely self-contained. It must implement its own:
  * Dataset parsers and data loaders.
  * Preprocessing, resizing, and normalization pipelines.
  * Model initializers and inference routines.
* **No Cascading Failures:** A bug, refactor, or experiment inside `brain/` must never affect, break, or require changes in `lungs/` or any other organ. Shared logic is allowed only for truly generic helpers (e.g., generic array-to-tensor converters, standard HTTP wrappers).

---

### 7. Training & Inference Domain Equivalence
A critical flaw in medical AI is distribution shift caused by preprocessing mismatches. Agents must guarantee that inference preprocessing is mathematically identical to training preprocessing:
* **Resolution & Interpolation:** If a model was trained on $224 \times 224$ with Bicubic interpolation, inference must use $224 \times 224$ Bicubic (not Bilinear or Nearest).
* **Normalization Constants:** Strict adherence to exact channel means and standard deviations (e.g., ImageNet vs. domain-specific medical statistics).
* **Color Space & Channel Order:** Enforce RGB vs. BGR vs. Grayscale consistency (e.g., OpenCV loads BGR, PIL loads RGB).
* **CT/MRI Hounsfield Windowing:** For CT scans, apply the exact Hounsfield Unit (HU) windowing (e.g., Lung window: $[-1000, 400]$, Brain window: $[0, 80]$) used during training before feeding tensors to the model.

---

### 8. The FYP Academic Duty: Critical Pushback
Because this is a Final Year Project with significant academic and technical weight:
* **Voice Critical Concerns:** If the user requests an architectural choice, dataset split, model selection, or pipeline design that is technically flawed, scientifically invalid, or prone to evaluation failure (e.g., data leakage between patient scans, using 2D slice classification for 3D volumetric diagnosis, improper evaluation metrics):
  1. **Directly and respectfully alert the user.**
  2. Clearly explain **why** the decision is risky or incorrect.
  3. Cite current medical research, benchmarks, or standard MICCAI/IEEE best practices.
  4. Provide a sound, robust alternative.
* **User Override:** If the user acknowledges the risk and still explicitly insists on their approach, proceed as instructed, but cleanly document the caveat.

---

### 9. High-Polish Production Quality (Zero AI Slop)
The final project must present as a refined, clinical-grade medical software platform, not an incomplete AI-generated proof-of-concept:
* **Complete Metadata:** Real page titles, OpenGraph tags, semantic HTML5, favicon, and SEO meta tags.
* **Routing & Error Handling:** Custom 404 pages, accessible 500 error boundaries, and connection timeout alerts.
* **Realistic UX States:** Polished skeleton loaders for scan analysis, intuitive drag-and-drop zones, non-blocking asynchronous toasts, empty state placeholders, and mobile/desktop responsive views.
* **Medical Accessibility:** High contrast ratios, legible anatomical labels, and smooth canvas zoom/pan/reset controls.

---

### 10. Documentation Protocol & Obsidian Vault Mirroring
Code files must remain clean—**never clutter code files with comments**. Instead:
* **Per-Organ README Files:** Every organ directory must have its own separate `README.md` file.
* **Obsidian Vault Mirroring (`/home/honeypot/Obsidian/~Honeypot/OpenMed/`):** Every file across the codebase must have its equivalent explanation file in this vault folder. Whenever you create, update, add, delete, append, or modify anything in the codebase, update the corresponding vault documentation file as well. Use mermaid diagrams and clean markdown to explain the code.
* **Index Backlinks:** Whenever you document something in the vault, add a wikilink back to `[[index]]` (and add the file's wikilink into `index.md`) so all files connect properly in the Obsidian Graph View. That's it.

---

## 🛠️ Verification Commands for Agents

Before completing any task, agents must execute and verify:

```bash
# Frontend Type & Build Verification
npm run lint          # oxlint checks
npm run build         # TypeScript type-check (tsc -b) and Vite bundle

# Backend Verification (once backend directory is populated)
pytest tests/         # Unit and integration test passes
ruff check .          # Python style & lint validation
```
