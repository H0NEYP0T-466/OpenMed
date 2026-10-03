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
   backend/app/organs/brain/segmentation/experimental_lab/checkpoints/lite_medsam.pth
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
- **Measured cost of the CAM-derived prompt, on real masks.** Evaluated against
  the `openmed_seg_clean` held-out test split (50 sampled non-Normal stems, one
  seed), swapping only the prompt source:

  | prompt source | mean Dice | median Dice | mean IoU |
  | :--- | ---: | ---: | ---: |
  | ground-truth box (oracle) | 0.8776 | 0.9218 | 0.8027 |
  | served Grad-CAM box | 0.3134 | 0.2348 | 0.2281 |

  The CAM prompt cost **0.56 mean Dice**, and the served path matched or beat the
  oracle box on **1 of 49** cases. So the ~0.88 in the tables above is reachable
  only with ground-truth prompts; the served pipeline is far below it, and the
  box prompt — not the mask decoder — is the bottleneck. Two caveats, both of
  which make this figure *generous*: the sample is small, and 306 of the 1,112
  held-out seg stems also appear in the classifier's image pool, so the CAM is
  optimistically good on those. Treat 0.31 as an upper bound on served quality.
- **How good the box has to be.** Scaling the ground-truth box about its centre
  gives boxes of known quality, which separates "the box is misplaced" from "the
  decoder cannot cope with a misplaced box" (16 samples, 112 runs):

  | prompt box | box IoU vs truth | Dice |
  | :--- | ---: | ---: |
  | truth ×0.6 (clips the lesion) | 0.360 | 0.5978 |
  | truth ×0.8 | 0.640 | 0.8336 |
  | truth ×1.0 (oracle) | 1.000 | 0.8852 |
  | truth ×1.3 (slightly loose) | 0.592 | **0.8937** |
  | truth ×1.8 | 0.309 | 0.7209 |
  | truth ×2.5 | 0.162 | 0.4046 |
  | **actual CAM box** | **0.123** | **0.2348** |

  Three things follow. The CAM box overlaps the true lesion at only **0.123 IoU** —
  it is barely on target, which is the whole failure. A box **slightly larger than
  the lesion scores best** (×1.3 beats the exact box, 0.8937 vs 0.8852), so an
  over-generous margin is the right instinct. But it falls off a cliff past that:
  ×1.8 costs 0.17 Dice and ×2.5 costs 0.49. An earlier adaptive-margin box landed
  around ×1.8 and made things worse for exactly this reason — too loose, not
  "loose is wrong". Binned by quality: Dice 0.89 at box IoU ≥ 0.7, 0.66 at
  0.3–0.4, **0.075 at 0.0–0.1**. Target for any improvement to the prompt is
  **box IoU ≥ 0.5**, which is worth ~0.86 Dice.
- **The assistive fix that does work: let the clinician supply the box.** Since
  the box is the bottleneck and the clinician knows where the lesion is, the
  workspace now supports click-to-segment (`POST /api/brain/segment-click`): the
  click becomes a small box centred on it, and no classification runs. Measured
  on the held-out split with the click placed inside the ground-truth mask
  (48px box, n=15): **mean Dice 0.795, median 0.917** — versus 0.31 for the CAM
  box and 0.88 for the oracle. Two caveats. Sending the click as a *bare point*
  scored only 0.254 mean (the weights were trained on boxes, not points), and
  adding the point alongside the box made every box size worse. Response time on
  CPU was ~4.5s p50; the forward pass is fast, the rest is decode and encode, so
  this is not yet real-time on a CPU-only box.
- **The raw serving mode returns an empty mask.** The pipeline can also be asked
  to segment with no prompt at all (`POST /api/brain/segment` with
  `use_heatmap_prompt=false`), in which case the decoder runs from the image
  embedding alone. Measured against the released weights on the same scan: **0
  foreground pixels unprompted vs 26,548 with the Grad-CAM box prompt**. These
  weights only ever saw boxes, so with no prompt the decoder emits nothing at
  all. The mode exists to *demonstrate* that the prompt is load-bearing — it is
  not a scoreable baseline, and the metrics above do not describe it. Note also
  that the IoU head still reports a number (0.457 in that run) for an empty mask;
  do not put it in a results table.
- **The serving box uses a fixed threshold and margin, deliberately.** Training
  and evaluation boxes are tight ground-truth boxes; serving thresholds the
  classifier's Grad-CAM at 0.5 with a 10px pad. An adaptive
  percentile/morphology box was trialled and reverted: it measured a wash against
  the decoder's own IoU head (0.6868 vs 0.6803) while producing roughly 3× larger
  masks (32,084 vs 10,508 mean foreground px), and in the workspace it made the
  decoder segment well beyond the lesion. It remains available via
  `adaptive_box=True` for low-contrast scans where a fixed cut finds nothing.
- **Grad-CAM++ was trialled and reverted.** It measured a tighter hot area (8.2%
  vs 10.2% of frame above half-peak) but rendered a fragmented map on real scans —
  scattered hot blobs rather than one coherent region over the lesion. Plain
  Grad-CAM is the default; `OPENMED_CAM_METHOD=gradcam++` opts in. Note that this
  map is both the workspace's displayed explanation and the source of the box
  prompt, so a change here is visible in two places at once.
- **The peak-point prompt is off by default because it measured worse.** Sending
  a positive point at the peak activation alongside the box cost 0.051 mean IoU
  (0.6868 → 0.6355) and was worse on five of eight samples. It is opt-in
  (`use_peak_point=True`), not a default.
- **Validation selects the model; test is evaluated once**, on the best
  checkpoint. No test-based model selection.

## Debugging

```bash
!python trainKaggle.py --audit-only              # hygiene + split, no training
!python trainKaggle.py --force-audit             # re-run the audit (cached by default)
!python trainKaggle.py --epochs 30 --batch 8
!python trainKaggle.py --gpus 1                  # single GPU
!python trainKaggle.py --grad-checkpoint         # if VRAM is tight
!python trainKaggle.py --resume <out>/last.pt
!python trainKaggle.py --smoke                   # 2 steps/epoch, wiring only
```

Each module also imports standalone, so a stage can be driven from a notebook
cell without re-running the ones before it.

## Watching a run

The notebook cell will only show the tail of stdout when the process exits —
that is a Kaggle display limitation, not something the script can change. Read
the files instead:

```python
!tail -n 40 /kaggle/working/openmed_seg_run/rank0.log
import pandas as pd
pd.read_csv("/kaggle/working/openmed_seg_run/metrics/epochs.csv").tail(10)
```

`epochs.csv` gets a row the moment an epoch ends.

## Testing

```bash
python test_medsam_pkg.py
```

88 checks over the pure logic: metric mathematics (including the `nan` HD95
convention), the loss tensor shapes, group keys, box geometry, augmentation,
the Dataset end-to-end on synthetic files, split construction with a **negative
control** for the leakage check, and the config contract.

No model construction, no forward passes, no CUDA, no training — importing
torch is the heaviest thing it does. It runs in about 15 seconds.

Static analysis, with the two serving snapshots excluded because they are
byte-identical copies of code that is not ours to restyle:

```bash
python -m ruff check --select F,B,SIM,RUF100 \
  --exclude medsam_training/medsam_model.py,medsam_training/preprocessor.py .
```

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
