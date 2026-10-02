# OpenMed — LiteMedSAM fine-tuning (Kaggle)

One script runs the whole pipeline:

```bash
!python trainKaggle.py
```

## What it does, in order

| Stage | What happens |
|---|---|
| 1 | Environment report, dependency check, hardware assertion, LiteMedSAM weights fetched from Google Drive and **verified with `strict=True` + a real forward pass** |
| 2 | Hygiene audit of both datasets — pairing, geometry, SHA256 duplicates, pHash near-duplicates, mask alignment, morphology |
| 3 | Cross-source overlap check, with every perceptual hit confirmed at pixel level |
| 4 | Split construction with group-level leakage verification |
| 5 | DDP training on 2 × T4, fp16, early stopping at patience 10 |
| 6 | Final test evaluation, per-source breakdown, prompt-stress run, plots |
| 7 | `TRAINING_REPORT.txt` and a ZIP of everything in `/kaggle/working` |

Any stage can be run alone — see **Debugging** below.

## Expected Kaggle inputs

```
/kaggle/input/datasets/h0neyp0t/fypseg                     OpenMed segmentation
/kaggle/input/datasets/nikhilroxtomar/brain-tumor-segmentation   figshare/BTSC
```

Kaggle nests mounts inconsistently, so the resolver walks `/kaggle/input` to
depth 4 and tries three strategies in order:

1. a direct child whose name matches, i.e. `/kaggle/input/<slug>`
2. **any depth, matched by directory name** — this is what handles the
   `/kaggle/input/datasets/<owner>/<slug>/` shape above
3. name-independent fallback: BTSC is recognised by numeric stems
   (`1.png … 3064.png`), OpenMed by descriptive ones

Layout inside each root is detected too — `images/` + `masks/`, a
`png_dataset/` of `<stem>.png` + `<stem>_mask.png`, or any single folder of
paired `_mask` files. If none match, the error prints the top-level listing.

Override with `--openmed-root` / `--btsc-root`. When resolution fails the error
lists every dataset-shaped directory it found, each tagged with what it looks
like, so the fix is a copy-paste:

```
  Searched under /kaggle/input (depth 4). Dataset-shaped directories found:
    /kaggle/input/datasets/h0neyp0t/fypseg  -> looks like: openmed
    /kaggle/input/datasets/nikhi.../brain-tumor-segmentation  -> looks like: btsc
```

**Accelerator must be GPU T4 ×2.** The script asserts this and stops in the
first seconds rather than silently running on CPU for twelve hours.

## Files

```
trainKaggle.py                 entry point — orchestrates all seven stages
medsam_training/
  config.py                    every hyper-parameter; serialised to config.json
  bootstrap.py                 environment, dependencies, weight download + verify
  audit.py                     hygiene checks, pHash, cross-source overlap
  data.py                      group keys, splits, Dataset, augmentation
  engine.py                    metrics, loss, DDP loop, early stopping
  reporting.py                 plots, text report, ZIP export
  medsam_model.py              architecture snapshot (trainable)
  preprocessor.py              preprocessing snapshot — serving-identical
```

The last two are byte-identical to
`backend/app/organs/brain/segmentation/{model,preprocessor}.py`. That is the
point: training and inference must build the same tensor, or the fine-tuned
weights are being served inputs they never saw.

## Serving compatibility

The trained checkpoint is saved as a **bare, CPU, unwrapped state dict** at
`best/lite_medsam.pth`. To deploy:

```bash
cp /kaggle/working/<run>/best/lite_medsam.pth \
   backend/app/organs/brain/segmentation/checkpoints/lite_medsam.pth
```

No code change is needed. The backend loads it with `strict=True`, so a
mismatch fails loudly at startup rather than silently producing bad masks.

Preprocessing is shared, not reimplemented: the training Dataset calls
`preprocessor.preprocess_image` directly. Same resize (longest side → 256,
`INTER_AREA`), same per-image min-max normalisation (not `/255`), same
right/bottom zero-pad, same 256-space box coordinates.

## Design decisions worth knowing

**fp16, not bf16.** T4 is sm_75 — fp16 has tensor cores, bf16 does not. Losses
are still computed in float32, because a Dice ratio in fp16 loses precision
exactly where these masks are hardest (foreground is 1–3 % of the frame).

**Prompt encoder frozen, BatchNorm frozen.** With box-only prompts the point
and dense-mask branches never receive a gradient, so training them is dead
work. Frozen BatchNorm stops each rank accumulating its own running statistics
over its own half of the batch — with two GPUs those would silently diverge.

**Encoder LR is 10× lower** (1e-5 vs 1e-4). TinyViT carries the pretrained
prior worth keeping; the decoder is what needs to learn this dataset.

**Validation is rank-strided, not `DistributedSampler`.** A distributed sampler
pads the last batch, so some images get counted twice and the metric skews.
Each rank evaluates a disjoint stride and the results are gathered.

**Early stopping is decided on rank 0 and broadcast**, so both processes exit
together instead of one rank hanging on a collective.

**Non-finite losses are detected collectively.** A NaN on one rank would
desynchronise every later collective and hang until the session died.

## Split policy — read this before quoting a number

**OpenMed** filenames encode lesion and slice
(`T1C+ - Cystic glioblastoma occipital , ventricle 005`). Stripping the
sequence prefix and the trailing number gives a **lesion key**, so all slices
of one lesion stay on one side of the split. Verified on the real data:
11,194 images → **348 lesion groups**, of which **300 appear under more than one
sequence prefix** — exactly the duplication that a naive file-level split would
have leaked across train and validation.

**BTSC** filenames are `1.png … 3064.png`. The figshare source carries a patient
id inside its original `.mat` files, but the PNG re-export drops it and the
Kaggle copy has no metadata file. There is no patient key to group on.

So BTSC is split into **contiguous index blocks** rather than randomly. This
keeps runs of neighbouring slices together, but it is an improvement, not a
proof: each block boundary is a place where one scan could straddle two splits.
The report measures exactly how many slices sit at such a boundary and prints
it. That number is the honest limitation — not zero.

The manifest records `group_is_subject_verified: false` for BTSC so nothing
downstream can mistake the block key for a patient key.

## Outputs

```
<out>/
  config.json                 exact configuration of the run
  best/lite_medsam.pth        deployable weights
  last.pt                     resumable (optimizer, scheduler, scaler, RNG)
  metrics/epochs.csv          one row per epoch
  metrics/history.json        per-epoch curves
  metrics/baseline_val.json   pretrained model, before any training
  metrics/summary.json        final test metrics, in-domain and cross-source
  metrics/test_per_image.json per-image Dice, IoU, HD95
  splits/{train,val,test}.json
  plots/                      01 curves · 02 LR · 03 distributions
                              04 calibration · 05 baseline-vs-finetuned
                              06 qualitative overlays
  logs/TRAINING_REPORT.txt    the artefact for the SRS
  rank0.log rank1.log
```

Plus `openmed_litemedsam_<timestamp>.zip` in `/kaggle/working`.

## Metrics, and what they do not mean

Reported per source and pooled: **Dice, IoU, precision, recall, HD95**,
plus IoU-head calibration.

- **HD95 is in pixels.** No validated pixel spacing exists for these slices, so
  no millimetre figure is given.
- **HD95 is `nan`** when exactly one of prediction and ground truth is empty.
  Substituting a large finite number would look like a measurement.
- **The prompt is not end-to-end.** Training and evaluation boxes come from the
  ground-truth mask. The deployed pipeline prompts with a Grad-CAM box, which is
  looser and sometimes off-centre. These numbers are *conditional segmentation
  quality*. The perturbed-prompt run (`±6 px`) shows how much the score depends
  on prompt accuracy — it is a sensitivity check, not a simulation of Grad-CAM.
- **Validation selects the model; test is evaluated once**, on the best
  checkpoint. No test-based model selection.

## Debugging

```bash
!python trainKaggle.py --audit-only              # hygiene + split, no training
!python trainKaggle.py --skip-audit              # reuse cached audit.json
!python trainKaggle.py --epochs 30 --batch 8
!python trainKaggle.py --gpus 1                  # single GPU
!python trainKaggle.py --grad-checkpoint         # if VRAM is tight
!python trainKaggle.py --resume <out>/last.pt
!python trainKaggle.py --smoke                   # 2 steps/epoch, wiring only
```

Each module also imports standalone, so a stage can be driven from a notebook
cell without re-running the ones before it.

## Known limitations

1. Prompt is not end-to-end (see above).
2. HD95 in pixels, undefined cases reported as `nan`.
3. BTSC has no patient identifier — block split, residual boundary exposure
   quantified in the report.
4. OpenMed group keys are lesion keys derived from filenames, not verified
   patient identifiers.
5. Two sources only. Both appear in train and test, so a cross-source number is
   available, but neither is a genuinely external cohort.
6. The `medsam_model.py` snapshot has `@torch.no_grad()` removed from
   `MedSAM_Lite.forward` so gradients can flow. That matches the working-tree
   change in the backend; if the backend is reverted, the two diverge.
