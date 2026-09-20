# Brain Organ — OpenMed

FastAPI service for 9-class brain MRI classification on an EfficientNetV2-B2 backbone, with Grad-CAM attribution and
typical-site mapping onto the frontend's 3D atlas.

```
brain/
├── classification/
│   ├── dataset.py          Dataset discovery, grouped leak-free splitting
│   ├── preprocessor.py     timm-derived transforms (class-aware minority augmentation)
│   ├── model.py            Backbone factory, optim/schedulers, Grad-CAM, overlay
│   ├── label_space.py      Canonical 9-class list + checkpoint/label binding
│   ├── pipeline.py         Single-image inference, fail-closed checkpoint loading
│   ├── brain_regions.py    Location registry, MNI conversion, presentation priors
│   ├── visualization.py    Training artefacts
│   ├── router.py           HTTP contract, validation, readiness
│   ├── brain_kaggle.py     Training entrypoint
│   └── checkpoints/        Model weights (git-ignored)
├── utils/
│   └── dataset_integrity_report.py  Comprehensive SHA-256/pHash integrity auditor
└── segmentation/           Reserved
```

## Invariants

These are enforced in code and covered by `backend/tests`. Breaking them is what
caused the incidents recorded below.

**1. A checkpoint and its class names must agree exactly.**
`label_space.py` owns the only copy of the class list. `resolve_label_space()`
binds a head width to names and raises on any disagreement; the pipeline converts
that into `ModelUnavailableError`, which the router serves as HTTP 503. Nothing
resizes a model to fit a mismatched checkpoint.

**2. The label space travels with the weights.**
Training writes `<checkpoint>.label_space.json` beside the `.pth`, containing
`class_names`, `model_tag` and the run `metrics`. Deploying a checkpoint means
copying **both** files. Without the sidecar the service still runs, using the
canonical list and reporting `label_space_source: "canonical (unverified)"` plus
a `/health` warning.

**3. Splits never share a source scan.**
`get_stratified_splits` groups images before assigning splits and raises if any
group ends up in two splits. Two images are the same group when they are
byte-identical, or when they share a numeric filename stem **and** consecutive
frames are measurably the same picture (median adjacent 32×32 correlation ≥ 0.90).
The correlation test matters: filename stems alone would have wrongly merged
1,254 unrelated scans that merely share the `Siar_healthy_N` prefix.

**4. Explanation rendering is shared code.**
`overlay_cam_on_image` and `GradCAM` are used by both the serving pipeline and the
training artefacts, so a heatmap in a report is the same computation as one in the
API. `GradCAM` keeps hooks attached for its lifetime; use it as a context manager.

**5. Coordinates are millimetres at the boundary.**
The registry is authored in normalised `[-1, 1]³` design units and converted by
`to_mni_mm` before leaving the API, because the viewer labels its readout
"Stereotactic MNI". `locations_3d[].probability` is a presentation prior scaled by
classifier confidence — never a lesion measurement. `localization_basis` says so,
and the frontend renders it.

**6. Train and inference preprocessing are the same object.**
Both come from `resolve_model_data_config(model)`; the resolved input is
208×208 bicubic at `crop_pct` 0.89, not the 512×512 stored on disk.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `OPENMED_BRAIN_CHECKPOINT` | Weights to serve | `classification/checkpoints/brain_best_model.pth` |
| `OPENMED_BRAIN_DATASET` | Dataset root for training and `/model-info` counts | — |
| `OPENMED_BRAIN_OUTPUT` | Run output directory | `.` (use `/kaggle/working` on Kaggle) |
| `OPENMED_CORS_ORIGINS` | Allowed browser origins | local Vite dev origins |
| `VITE_BRAIN_API_BASE` | Frontend API base | `http://localhost:8016/api/brain` |

## Training

```bash
cd backend
python -m app.organs.brain.classification.brain_kaggle \
  --data_root "$OPENMED_BRAIN_DATASET" --output_dir /kaggle/working \
  --epochs 75 --patience 15 --seed 42
```

The calibrated recipe targets overfitting while preventing underfitting:
stochastic depth (`--drop_path 0.15`), gentle MixUp 0.1 / CutMix 0.2 with `mixup_prob 0.3`,
class-weighted soft-target cross entropy, label smoothing 0.05, calibrated RandAugment
(`--auto_augment rand-m5-mstd0.5-inc1`), stronger minority class augmentation
(`--minority_augment rand-m7-mstd0.5-inc1`), and AdamW with norm/bias exempt from decay.
Each epoch validates raw and EMA weights and checkpoints whichever wins.
Pass empty/zero values to disable any of them.

Produces `checkpoints/brain_best_model.pth` (best `val_loss`) plus
`brain_best_acc.pth`, a `split_manifest.csv` recording every assignment, and
`artifacts/` with figures, `metrics.json` and the annotated report. Grouping costs
about 90 s of hashing and signature extraction on first run, then caches to
`.case_group_cache.json` in the dataset directory, keyed on manifest and image
fingerprints.

## Verification

```bash
cd backend && python -m pytest        # 37 tests: splits, label space, Grad-CAM, HTTP
npm run lint && npm run build         # from the repository root
```

## Known limits

- **2D slices, single labels.** One slice cannot exclude disease in a volume, and
  `location` is inferred from the tumour type rather than measured in the image.
- **Three classes are confined to one split** by the grouping rule
  (`Hemangiopericytoma T1C+`, `Neurocytoma T1`, `Neurocytoma T1C+`): every image of
  those classes belongs to a single source scan, so no honest holdout exists.
- **Seven duplicate clusters (79 samples) carry contradictory labels** — identical
  pixels annotated as two different classes. Grouping cannot resolve this; it needs
  a `DATA.json` correction. The training log reports the count on every run.
- **External Normal and Pituitary scans inherited machine-labelled sequences.** The
  `utils/sequence_tagger.py` MobileNetV3 assigned T1/T1C+/T2 for 4,000 images, and
  those guesses are ground truth downstream. It reports held-out accuracy and
  refuses silently; rows below 0.60 confidence are discarded at export. Its output
  is over-represented in `T1C+` (62% versus 41% in curated data), so sequence
  errors concentrate there.
- **`datasets/brain/archive` mixes two provenances**: 12,626 curated scans with real
  location annotations, and 4,000 derived rows whose `location` was fabricated by
  the earlier tagger export (now changed to empty with
  `localization_annotated: false`).

## History

Recorded so the same class of mistake is not reintroduced.

- **Silent relabelling.** A 39-class checkpoint was served against a 42-name list.
  The pipeline rebuilt the head to fit and kept the longer list, so indices 36–38
  printed the wrong class: every Schwannoma scan was reported as Pituitary with
  0.93–0.9999 confidence, and the 3D viewer highlighted the sella turcica. Invariant
  1 now makes this a 503.
- **Repeated Grad-CAM.** `generate()` detached its own hooks, so reusing one
  instance replayed the first image's activations for every later sample — the
  exported `gradcam_samples.png` was one heatmap printed eight times.
- **Leakage.** Stratification on `location_class` was documented as preventing
  anatomy leakage but performs no grouping: 1,046 byte-identical copies straddled
  split boundaries and 96% of curated samples sat in multi-split name groups. Test
  accuracy of 0.9699 with a ~1 point train/val gap measured memorisation.
- **Region registry gaps.** 16 of the dataset's 39 location tags were absent and
  dropped silently at debug level; all 39 now resolve through entries or aliases.
- **Fabricated presentation values.** Region confidence came from a hard-coded
  `0.90 − 0.12·rank`, and the frontend returned a mock diagnosis whenever the
  backend errored and the filename contained a preset keyword.
