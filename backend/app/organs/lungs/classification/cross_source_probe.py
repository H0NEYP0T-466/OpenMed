"""Cross-source probe — did v2 actually remove the shortcut?

The v1 failure was a TB head that fired 0.9144 / 0.6036 / 0.9969 on COVID
images with no TB evidence. v2 added source-matched negatives and uniform
resampling, but the global and within-source AUCs stayed near 1.0000. Those
numbers cannot detect a surviving source shortcut, because inside-collection
positives vs normals is the exact pair the head was trained to separate.

This script scores the two heads on *every* source group in the held-out test
split, especially groups the head was never meant to fire on:

  TB head    on nih, covid, covid_normal   -> should be near 0
  COVID head on nih, tb,    tb_normal     -> should be near 0

If those groups score high, the model is still a source detector and the v2 fix
did not do its job. If they score near 0, the near-perfect within-source AUC is
a genuinely easy public-dataset pair rather than a leaked cue.

Reads the v2 corpus straight out of the zip (no 5 GB extraction), samples a
fixed number of test rows per source, and writes a JSON report.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Run as a module from backend/ (python -m app.organs.lungs.classification.cross_source_probe).
# A plain script run puts classification/ on sys.path[0], which makes the local
# model.py shadow torchxrayvision's internal `model.utils` package.
from .model import FINDINGS, build_model

ROOT = Path(__file__).resolve().parent
DEFAULT_ZIP = Path("/home/honeypot/Desktop/lungs_class_v2.zip")
DEFAULT_CKPT = ROOT / "checkpoints" / "xrv_best.pth"
DEFAULT_OUT = ROOT / "artifacts" / "cross_source_probe.json"

COVID_INDEX = FINDINGS.index("COVID")
TB_INDEX = FINDINGS.index("TB")

# Every row in these groups carries no evidence for the corresponding head.
# nih rows are labelled 0/0; the other heads are -1 (not assessed) in the v2
# label matrix, so a high score on them is a false alarm rather than a miss.
GROUPS = ("nih", "covid", "covid_normal", "tb", "tb_normal")


def load_model(checkpoint: Path) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    classes = payload.get("classes") if isinstance(payload, dict) else None
    if classes and tuple(classes) != tuple(FINDINGS):
        raise RuntimeError("checkpoint class order does not match the serving order")
    model = build_model(cache_dir=str(checkpoint.parent / "xrv"))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def to_tensor(raw: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(raw)) as handle:
        gray = handle.convert("L").resize((224, 224), Image.Resampling.LANCZOS)
    array = np.asarray(gray, dtype=np.float32) / 255.0
    array = array * 2048.0 - 1024.0
    return torch.from_numpy(array).unsqueeze(0)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), 4),
        "median": round(float(np.median(array)), 4),
        "p90": round(float(np.percentile(array, 90)), 4),
        "frac_above_0_5": round(float((array > 0.5).mean()), 4),
        "frac_above_0_9": round(float((array > 0.9).mean()), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--per_group", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    torch.set_num_threads(min(8, os.cpu_count() or 4))
    rng = random.Random(args.seed)

    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint)

    print(f"Reading labels from: {args.zip}")
    with zipfile.ZipFile(args.zip) as archive:
        with archive.open("lungs_class_v2/labels.csv") as handle:
            import csv
            rows = list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))

    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "test" and row["source"] in GROUPS:
            buckets[row["source"]].append(row)
    for source in GROUPS:
        rng.shuffle(buckets[source])
        buckets[source] = buckets[source][: args.per_group]
        print(f"  {source:14s} sampled {len(buckets[source])}")

    scores: dict[str, dict[str, list[float]]] = {
        source: {"COVID": [], "TB": []} for source in GROUPS
    }

    with zipfile.ZipFile(args.zip) as archive:
        for source in GROUPS:
            tensors = []
            for row in buckets[source]:
                member = f"lungs_class_v2/images/{row['image']}"
                tensors.append(to_tensor(archive.read(member)))

            with torch.no_grad():
                for offset in range(0, len(tensors), args.batch_size):
                    batch = torch.stack(tensors[offset : offset + args.batch_size])
                    probabilities = torch.sigmoid(model(batch))
                    for line in probabilities:
                        scores[source]["COVID"].append(float(line[COVID_INDEX]))
                        scores[source]["TB"].append(float(line[TB_INDEX]))
            print(f"  scored {source:14s} ({len(tensors)} images)")

    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_kind": "ema best" if "xrv_best" in args.checkpoint.name else "other",
        "seed": args.seed,
        "per_group": args.per_group,
        "split": "test",
        "sources": {g: {h: summarize(v) for h, v in scores[g].items()} for g in GROUPS},
        # The decisive controls: heads firing on groups with no evidence for them.
        "controls": {
            "TB_head_on_nih": summarize(scores["nih"]["TB"]),
            "TB_head_on_covid": summarize(scores["covid"]["TB"]),
            "TB_head_on_covid_normal": summarize(scores["covid_normal"]["TB"]),
            "COVID_head_on_nih": summarize(scores["nih"]["COVID"]),
            "COVID_head_on_tb": summarize(scores["tb"]["COVID"]),
            "COVID_head_on_tb_normal": summarize(scores["tb_normal"]["COVID"]),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {args.out}\n")

    print(f"{'source':14s} {'COVID mean':>11s} {'COVID >.5':>10s} {'TB mean':>10s} {'TB >.5':>9s}")
    for source in GROUPS:
        c, t = report["sources"][source]["COVID"], report["sources"][source]["TB"]
        print(f"{source:14s} {c['mean']:11.4f} {c['frac_above_0_5']:10.3f} {t['mean']:10.4f} {t['frac_above_0_5']:9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
