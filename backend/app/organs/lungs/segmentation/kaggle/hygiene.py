"""Stage A: dataset hygiene, before any split exists.

A split built on a dirty dataset is worse than no split, so this runs first. One
parallel pass reads every item once (33,920 lung-tree pairs plus the 5,826
infection-tree pairs, 39,746 in all) and records everything the checks need; the
checks then run on those rows. Each is logged as PASS / WARN / FAIL / INFO into
``hygiene.json``. Nothing is dropped silently: every removal is a row in
``dropped_files.csv`` with a reason.

  H01 pair integrity            every image has exactly one mask in its tree
  H02 readability               no unreadable or truncated files
  H03 geometry                  256 x 256, image and mask agree, resolution census
  H04 channels                  census; grayscale-safe (single channel or R == G == B)
  H05 mask values               strictly {0, 255}; intermediate-value census
  H06 lung mask extent          no empty / full-frame mask; floor 2,725 px
  H07 lung mask morphology      components, holes, border contact, solidity
  H08 lung mask alignment       foreground on anatomy, not padding or the border
  H09 infection mask extent     COVID-19 never empty; Normal and Non-COVID always empty
  H10 infection inside lung     lesion pixels outside the lung mask
  H11 image content             blank, low-contrast, saturated frames
  H12 exact duplicates          SHA-256 of raw bytes and of decoded pixels
  H13 conflicting duplicates    identical image carrying different labels or masks
  H14 near duplicates           DCT pHash, pixel-confirmed before acting
  H15 group integrity           class and stem parse; cluster size census
  H16 tree consistency          infection-tree files equal their lung-tree twins
  H17 shipped-split leakage     cross-split duplicate pairs (the "before" numbers)

Policy
    Unusable files (unreadable, wrong geometry, empty / full-frame / off-anatomy lung
    mask, blank image) are dropped. Exact pixel duplicates collapse to one member,
    preferring the one that carries head-B supervision. A cluster containing the same
    image under different classes or masks is dropped entirely: nothing says which
    label is right. Near duplicates are *not* dropped; they are clustered so the split
    keeps them together. Violations the data cannot explain (H01, H09, H16) are FAIL
    and stop the run unless ``--continue-on-fail`` drops the offenders.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from ..labels import CLASS_COVID
from .catalog import Catalog, Record, parse_bids_subject, parse_stem
from .clusters import ClusterResult, build_clusters, phash64
from .config import TrainConfig

logger = logging.getLogger("hygiene")

ROWS_VERSION = 1
LIST_CAP = 25            # offenders listed inline in hygiene.json; the CSV has them all


# ------------------------------------------------------------------------ check log


@dataclass
class Check:
    id: str
    name: str
    status: str
    detail: str
    data: dict[str, Any] = field(default_factory=dict)
    fail_uids: list[int] = field(default_factory=list)


class CheckLog:
    def __init__(self, scope: str) -> None:
        self.scope = scope
        self.checks: list[Check] = []

    def add(self, check_id: str, name: str, status: str, detail: str,
            data: dict[str, Any] | None = None, fail_uids: list[int] | None = None) -> Check:
        if status not in {"PASS", "WARN", "FAIL", "INFO"}:
            raise ValueError(f"bad check status {status!r}")
        check = Check(check_id, name, status, detail, data or {}, fail_uids or [])
        self.checks.append(check)
        log = logger.error if status == "FAIL" else logger.warning if status == "WARN" else logger.info
        log("%s %-5s %-30s %s", check_id, status, name, detail)
        return check

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.status == "FAIL"]

    def to_list(self) -> list[dict[str, Any]]:
        return [{"id": c.id, "name": c.name, "status": c.status, "detail": c.detail,
                 "data": c.data} for c in self.checks]

    def tally(self) -> dict[str, int]:
        return dict(Counter(c.status for c in self.checks))


class HygieneFailure(RuntimeError):
    """One or more FAIL checks. The report has been written before this is raised."""


# --------------------------------------------------------------------- per-item pass


def _sha_pixels(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(f"{array.shape}|{array.dtype}".encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _decode(buffer: bytes, flag: int) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(buffer, dtype=np.uint8), flag)


def _read(path: str) -> tuple[bytes | None, str | None]:
    try:
        with open(path, "rb") as handle:
            return handle.read(), None
    except OSError as exc:
        return None, str(exc)


def _channel_info(raw: np.ndarray) -> tuple[int, bool]:
    if raw.ndim == 2:
        return 1, True
    channels = raw.shape[2]
    if channels < 3:
        return channels, True
    return channels, bool(np.array_equal(raw[:, :, 0], raw[:, :, 1])
                          and np.array_equal(raw[:, :, 1], raw[:, :, 2]))


def mask_geometry(mask: np.ndarray, gray: np.ndarray | None, band: int) -> dict[str, Any]:
    """Area, components, holes, border contact, solidity and padding overlap of a binary mask."""
    m = mask.astype(np.uint8)
    height, width = m.shape
    area = int(m.sum())
    out: dict[str, Any] = {"area": area, "components": 0, "significant_components": 0,
                           "holes_px": 0, "hole_components": 0, "border_touch_components": 0,
                           "solidity": None, "border_band_frac": 0.0, "padding_frac": None,
                           "largest_component_px": 0}
    if area == 0:
        return out
    count, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    floor = max(50, int(0.005 * height * width))
    out["components"] = int(count - 1)
    out["significant_components"] = int((areas >= floor).sum())
    out["largest_component_px"] = int(areas.max())
    touch = 0
    for i in range(1, count):
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                  cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            touch += 1
    out["border_touch_components"] = touch
    filled = ndimage.binary_fill_holes(m > 0)
    holes = filled & (m == 0)
    out["holes_px"] = int(holes.sum())
    out["hole_components"] = int(ndimage.label(holes)[1]) if holes.any() else 0
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(np.vstack(contours))
        hull_area = float(cv2.contourArea(hull))
        out["solidity"] = float(area / hull_area) if hull_area > 0 else None
    edge = np.zeros_like(m, dtype=bool)
    edge[:band, :] = edge[-band:, :] = True
    edge[:, :band] = edge[:, -band:] = True
    out["border_band_frac"] = float(((m > 0) & edge).sum() / area)
    if gray is not None and gray.shape == m.shape:
        out["padding_frac"] = float((gray[m > 0] <= 2).mean())
    return out


def analyze_item(task: dict[str, Any]) -> dict[str, Any]:
    """Everything the checks need from one item. Runs in a worker process."""
    size = int(task["size"])
    row: dict[str, Any] = {"key": task["key"], "tree": task["tree"], "uid": task["uid"],
                           "errors": []}
    image_bytes, err = _read(task["image_path"])
    if image_bytes is None:
        row["errors"].append(f"image: {err}")
    mask_bytes, err = _read(task["lung_mask_path"])
    if mask_bytes is None:
        row["errors"].append(f"lung mask: {err}")
    inf_bytes = None
    if task.get("infection_mask_path"):
        inf_bytes, err = _read(task["infection_mask_path"])
        if inf_bytes is None:
            row["errors"].append(f"infection mask: {err}")

    raw = _decode(image_bytes, cv2.IMREAD_UNCHANGED) if image_bytes else None
    gray = _decode(image_bytes, cv2.IMREAD_GRAYSCALE) if image_bytes else None
    lung_raw = _decode(mask_bytes, cv2.IMREAD_UNCHANGED) if mask_bytes else None
    inf_raw = _decode(inf_bytes, cv2.IMREAD_UNCHANGED) if inf_bytes else None
    if image_bytes and (raw is None or gray is None):
        row["errors"].append("image: decode failed")
    if mask_bytes and lung_raw is None:
        row["errors"].append("lung mask: decode failed")
    if inf_bytes and inf_raw is None:
        row["errors"].append("infection mask: decode failed")
    row["readable"] = not row["errors"]
    if not row["readable"]:
        return row

    channels, gray_safe = _channel_info(raw)
    row.update({
        "img_shape": [int(gray.shape[0]), int(gray.shape[1])],
        "img_channels": channels, "img_gray_safe": gray_safe, "img_dtype": str(raw.dtype),
        "sha_bytes": hashlib.sha256(image_bytes).hexdigest(),
        "sha_pixels": _sha_pixels(gray),
        "phash": format(phash64(cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
                                if gray.shape != (size, size) else gray), "016x"),
    })

    flat = gray.reshape(-1).astype(np.float32)
    p1, p99 = np.percentile(flat, (1, 99))
    row["content"] = {"mean": float(flat.mean()), "std": float(flat.std()),
                      "min": int(flat.min()), "max": int(flat.max()),
                      "dynamic_range": float(p99 - p1),
                      "saturated_frac": float((flat >= 250).mean())}

    def mask_plane(arr: np.ndarray) -> np.ndarray:
        return arr[:, :, 0] if arr.ndim == 3 else arr

    lung_plane = mask_plane(lung_raw)
    lung_vals = np.unique(lung_plane)
    row["lung_shape"] = [int(lung_plane.shape[0]), int(lung_plane.shape[1])]
    row["lung_values"] = [int(v) for v in lung_vals[:16]]
    row["lung_ambiguous_px"] = int(((lung_plane != 0) & (lung_plane != 255)).sum())
    row["sha_lung_mask"] = _sha_pixels((lung_plane >= 128).astype(np.uint8))
    lung_bin = lung_plane >= 128
    row["lung"] = mask_geometry(lung_bin, gray if lung_plane.shape == gray.shape else None,
                                int(task["band"]))

    if inf_raw is not None:
        inf_plane = mask_plane(inf_raw)
        inf_bin = inf_plane >= 128
        row["inf_shape"] = [int(inf_plane.shape[0]), int(inf_plane.shape[1])]
        row["inf_values"] = [int(v) for v in np.unique(inf_plane)[:16]]
        row["inf_ambiguous_px"] = int(((inf_plane != 0) & (inf_plane != 255)).sum())
        row["inf_area"] = int(inf_bin.sum())
        row["sha_inf_mask"] = _sha_pixels(inf_bin.astype(np.uint8))
        if inf_bin.shape == lung_bin.shape:
            row["inf_outside_lung_px"] = int((inf_bin & ~lung_bin).sum())
        row["inf_components"] = int(cv2.connectedComponents(inf_bin.astype(np.uint8), 8)[0] - 1)
    return row


# ----------------------------------------------------------------------- the stage


@dataclass
class HygieneResult:
    records: list[Record]                          # survivors
    all_records: list[Record]                      # every catalogued record
    drops: list[dict[str, str]]
    checks: CheckLog
    clusters: ClusterResult
    cluster_of: dict[int, int]
    rows: dict[tuple[str, int], dict[str, Any]]
    report: dict[str, Any]
    cluster_scope: list[int]                       # uids the clusters were built over


def _summary(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), "min": float(values.min()), "p1": float(np.percentile(values, 1)),
            "median": float(np.median(values)), "p99": float(np.percentile(values, 99)),
            "max": float(values.max()), "mean": float(values.mean())}


def _cap(items: list[Any]) -> list[Any]:
    return items[:LIST_CAP]


def run_hygiene(
    catalog: Catalog, cfg: TrainConfig, out_dir: str, *, workers: int | None = None
) -> HygieneResult:
    started = time.time()
    log = CheckLog("hygiene")
    records = catalog.records
    by_uid = {r.uid: r for r in records}
    drops: list[dict[str, str]] = list(catalog.orphan_rows)
    dropped: dict[int, tuple[str, str]] = {}

    def drop(uid: int, reason: str, detail: str = "") -> None:
        if uid not in dropped:
            dropped[uid] = (reason, detail)

    tasks: list[dict[str, Any]] = []
    for r in records:
        tasks.append({"key": f"lung|{r.uid}", "tree": "lung", "uid": r.uid,
                      "image_path": r.image_path, "lung_mask_path": r.lung_mask_path,
                      "infection_mask_path": None, "size": cfg.image_size,
                      "band": cfg.border_band_px})
        if r.in_infection_tree:
            tasks.append({"key": f"infection|{r.uid}", "tree": "infection", "uid": r.uid,
                          "image_path": r.infection_image_path,
                          "lung_mask_path": r.infection_lung_mask_path,
                          "infection_mask_path": r.infection_mask_path, "size": cfg.image_size,
                          "band": cfg.border_band_px})
    n_lung = sum(1 for t in tasks if t["tree"] == "lung")
    n_inf = len(tasks) - n_lung
    logger.info("hygiene pass over %d items (%d lung-tree + %d infection-tree) on %d workers",
                len(tasks), n_lung, n_inf, workers or os.cpu_count() or 1)

    rows: dict[tuple[str, int], dict[str, Any]] = {}
    workers = max(1, workers or os.cpu_count() or 1)
    if workers == 1 or len(tasks) < 64:
        iterator = map(analyze_item, tasks)
        pool = None
    else:
        pool = Pool(workers)
        iterator = pool.imap(analyze_item, tasks, chunksize=64)
    try:
        for done, row in enumerate(iterator, start=1):
            rows[(row["tree"], row["uid"])] = row
            if done % 5000 == 0:
                logger.info("  %d / %d items read (%.0fs)", done, len(tasks), time.time() - started)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    lung_rows = {uid: rows[("lung", uid)] for uid in by_uid}
    inf_rows = {uid: rows[("infection", uid)] for uid in by_uid if ("infection", uid) in rows}

    # ---- H01 pair integrity ----------------------------------------------------------
    pairing = catalog.pairing
    problems = {k: v for k, v in pairing.items() if k != "infection_twin_in_other_split"}
    if problems:
        log.add("H01", "pair integrity", "FAIL",
                "; ".join(f"{k}: {len(v)}" for k, v in problems.items()),
                {"problems": {k: _cap(v) for k, v in problems.items()}})
    else:
        log.add("H01", "pair integrity", "PASS",
                f"{len(records)} lung-tree images each have exactly one lung mask; "
                f"{n_inf} infection-tree images each have a lung mask and an infection mask; "
                "no orphans in either direction",
                {"lung_pairs": len(records), "infection_pairs": n_inf})
    moved = pairing.get("infection_twin_in_other_split", [])
    if moved:
        log.add("H01b", "tree split agreement", "WARN",
                f"{len(moved)} infection-tree files sit in a different split than their lung twin",
                {"files": _cap(moved)})

    # ---- H02 readability -------------------------------------------------------------
    unreadable = [uid for uid in by_uid if not lung_rows[uid]["readable"]
                  or (uid in inf_rows and not inf_rows[uid]["readable"])]
    for uid in unreadable:
        errs = lung_rows[uid]["errors"] + (inf_rows[uid]["errors"] if uid in inf_rows else [])
        drop(uid, "unreadable", "; ".join(errs))
    log.add("H02", "readability", "WARN" if unreadable else "PASS",
            f"{len(unreadable)} unreadable or truncated item(s) dropped" if unreadable
            else f"all {len(tasks)} items decode",
            {"unreadable": len(unreadable), "examples": _cap([by_uid[u].image_path for u in unreadable])})
    live = [uid for uid in by_uid if uid not in dropped]

    # ---- H03 geometry ----------------------------------------------------------------
    census: Counter[str] = Counter()
    wrong_size, mismatch = [], []
    for uid in live:
        for tree_rows in (lung_rows, inf_rows):
            row = tree_rows.get(uid)
            if row is None:
                continue
            census[f"{row['img_shape'][1]}x{row['img_shape'][0]}"] += 1
            shapes = [row["img_shape"], row["lung_shape"]] + ([row["inf_shape"]] if "inf_shape" in row else [])
            if any(s != shapes[0] for s in shapes):
                mismatch.append(uid)
            elif shapes[0] != [cfg.image_size, cfg.image_size]:
                wrong_size.append(uid)
    for uid in sorted(set(mismatch)):
        drop(uid, "geometry_mismatch", "image and mask shapes disagree")
    for uid in sorted(set(wrong_size) - set(mismatch)):
        drop(uid, "geometry_not_256", f"shape {lung_rows[uid]['img_shape']}")
    bad_geo = len(set(mismatch) | set(wrong_size))
    log.add("H03", "geometry", "WARN" if bad_geo else "PASS",
            f"{bad_geo} item(s) dropped for geometry" if bad_geo
            else f"every image and mask is exactly {cfg.image_size}x{cfg.image_size}; "
                 "image and mask shapes agree",
            {"resolution_census": dict(census), "mismatch": len(set(mismatch)),
             "not_expected_size": len(set(wrong_size))})
    live = [uid for uid in live if uid not in dropped]

    # ---- H04 channels ----------------------------------------------------------------
    chan: Counter[str] = Counter()
    not_gray = []
    for uid in live:
        for tree, tree_rows in (("lung", lung_rows), ("infection", inf_rows)):
            row = tree_rows.get(uid)
            if row is None:
                continue
            chan[f"{row['img_channels']}ch {row['img_dtype']}"] += 1
            if not row["img_gray_safe"]:
                not_gray.append((tree, uid))
    log.add("H04", "channels", "WARN" if not_gray else "PASS",
            f"{len(not_gray)} multi-channel image(s) with R != G or G != B" if not_gray
            else "all images are single-channel or R == G == B, so a grayscale decode loses nothing",
            {"channel_census": dict(chan), "not_gray_safe": _cap([by_uid[u].image_path for _, u in not_gray])})

    # ---- H05 mask values -------------------------------------------------------------
    value_census: Counter[str] = Counter()
    ambiguous = []
    for uid in live:
        for tree, tree_rows in (("lung", lung_rows), ("infection", inf_rows)):
            row = tree_rows.get(uid)
            if row is None:
                continue
            value_census[f"lung[{tree}]:{tuple(row['lung_values'])}"] += 1
            if row["lung_ambiguous_px"]:
                ambiguous.append((uid, "lung", row["lung_ambiguous_px"]))
            if "inf_values" in row:
                value_census[f"infection:{tuple(row['inf_values'])}"] += 1
                if row["inf_ambiguous_px"]:
                    ambiguous.append((uid, "infection", row["inf_ambiguous_px"]))
    log.add("H05", "mask values", "WARN" if ambiguous else "PASS",
            f"{len(ambiguous)} mask(s) hold values other than 0 and 255 (binarised at >= 128)"
            if ambiguous else "every mask is strictly {0, 255}",
            {"value_census": dict(value_census), "ambiguous": _cap(
                [{"path": by_uid[u].image_path, "mask": k, "px": n} for u, k, n in ambiguous])})

    # ---- H06 lung mask extent --------------------------------------------------------
    full_px = cfg.full_frame_fraction * cfg.image_size ** 2
    empty = [u for u in live if lung_rows[u]["lung"]["area"] == 0]
    full = [u for u in live if lung_rows[u]["lung"]["area"] >= full_px]
    below = [u for u in live if 0 < lung_rows[u]["lung"]["area"] < cfg.lung_area_floor_px]
    for u in empty:
        drop(u, "lung_mask_empty")
    for u in full:
        drop(u, "lung_mask_full_frame", f"area {lung_rows[u]['lung']['area']} px")
    areas = np.array([lung_rows[u]["lung"]["area"] for u in live if u not in dropped], dtype=np.float64)
    smallest = [{"path": by_uid[u].image_path, "area_px": lung_rows[u]["lung"]["area"]}
                for u in sorted(live, key=lambda x: lung_rows[x]["lung"]["area"])[:LIST_CAP]
                if lung_rows[u]["lung"]["area"] > 0]
    status = "WARN" if (empty or full or below) else "PASS"
    log.add("H06", "lung mask extent", status,
            f"{len(empty)} empty, {len(full)} full-frame (both dropped), {len(below)} below the "
            f"{cfg.lung_area_floor_px}px floor (kept, listed)" if status == "WARN"
            else f"no empty or full-frame masks; smallest lung region is {int(areas.min())} px "
                 f"({areas.min() / cfg.image_size ** 2:.1%} of the frame), floor {cfg.lung_area_floor_px}",
            {"area_px": _summary(areas), "empty": len(empty), "full_frame": len(full),
             "below_floor": len(below), "smallest": smallest,
             "below_floor_examples": _cap([by_uid[u].image_path for u in below])})
    live = [u for u in live if u not in dropped]

    # ---- H07 lung mask morphology ----------------------------------------------------
    comp_hist = Counter(lung_rows[u]["lung"]["significant_components"] for u in live)
    odd = [u for u in live if lung_rows[u]["lung"]["significant_components"] != 2]
    solidity = np.array([lung_rows[u]["lung"]["solidity"] for u in live
                         if lung_rows[u]["lung"]["solidity"] is not None])
    holes = [u for u in live if lung_rows[u]["lung"]["holes_px"] > 0]
    touching = [u for u in live if lung_rows[u]["lung"]["border_touch_components"] > 0]
    morph_path = os.path.join(out_dir, "lung_mask_morphology_exceptions.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(morph_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "class", "shipped_split", "significant_components",
                         "components", "holes_px", "border_touch_components", "solidity"])
        for u in odd:
            g = lung_rows[u]["lung"]
            writer.writerow([by_uid[u].image_path, by_uid[u].cls, by_uid[u].shipped_split,
                             g["significant_components"], g["components"], g["holes_px"],
                             g["border_touch_components"], g["solidity"]])
    log.add("H07", "lung mask morphology", "INFO",
            f"significant components per mask {dict(sorted(comp_hist.items()))}; {len(odd)} mask(s) "
            f"do not have two (listed, not failed); {len(holes)} with holes; {len(touching)} touch "
            "the frame border",
            {"components_hist": {str(k): v for k, v in sorted(comp_hist.items())},
             "solidity": _summary(solidity), "with_holes": len(holes),
             "hole_px": _summary(np.array([lung_rows[u]["lung"]["holes_px"] for u in holes], dtype=float)),
             "touching_border": len(touching), "not_two_components": len(odd),
             "exceptions_csv": os.path.basename(morph_path),
             "examples": _cap([by_uid[u].image_path for u in odd])})

    # ---- H08 lung mask alignment -----------------------------------------------------
    pad_max = 1.0 - cfg.align_inside_min
    padded = [u for u in live if (lung_rows[u]["lung"]["padding_frac"] or 0.0) > pad_max]
    off = [u for u in padded if (lung_rows[u]["lung"]["padding_frac"] or 0.0) > 0.5]
    edge = [u for u in live if lung_rows[u]["lung"]["border_band_frac"] > cfg.border_fraction_warn]
    for u in off:
        drop(u, "lung_mask_off_anatomy",
             f"{lung_rows[u]['lung']['padding_frac']:.0%} of the mask sits on black padding")
    log.add("H08", "lung mask alignment", "WARN" if (padded or edge) else "PASS",
            f"{len(padded)} mask(s) put >{pad_max:.0%} of their foreground on near-black padding "
            f"({len(off)} dropped: >50%); {len(edge)} put >{cfg.border_fraction_warn:.0%} within "
            f"{cfg.border_band_px}px of the frame edge" if (padded or edge)
            else "every lung mask sits on image content, away from padding and the frame edge",
            {"on_padding": len(padded), "dropped_off_anatomy": len(off), "on_frame_edge": len(edge),
             "padding_frac": _summary(np.array([lung_rows[u]["lung"]["padding_frac"] or 0.0 for u in live])),
             "examples": _cap([by_uid[u].image_path for u in padded + edge])})
    live = [u for u in live if u not in dropped]

    # ---- H09 infection mask extent ---------------------------------------------------
    viol: list[int] = []
    inf_stats: dict[str, Counter[str]] = {c: Counter() for c in ("COVID-19", "Non-COVID", "Normal")}
    for u in live:
        row = inf_rows.get(u)
        if row is None:
            continue
        cls = by_uid[u].cls
        empty_inf = row["inf_area"] == 0
        inf_stats[cls]["empty" if empty_inf else "non_empty"] += 1
        if (cls == CLASS_COVID and empty_inf) or (cls != CLASS_COVID and not empty_inf):
            viol.append(u)
    if viol:
        log.add("H09", "infection mask extent", "FAIL",
                f"{len(viol)} violation(s): COVID-19 must never be empty, Normal and Non-COVID must "
                "always be empty",
                {"census": {k: dict(v) for k, v in inf_stats.items()},
                 "examples": _cap([{"path": by_uid[u].infection_mask_path, "class": by_uid[u].cls,
                                    "area": inf_rows[u]["inf_area"]} for u in viol])},
                fail_uids=viol)
    else:
        log.add("H09", "infection mask extent", "PASS",
                "COVID-19 infection masks are never empty; Normal and Non-COVID are always empty "
                "(over the whole infection tree, not a sample)",
                {"census": {k: dict(v) for k, v in inf_stats.items()}})

    # ---- H10 infection inside lung ---------------------------------------------------
    outside = [(u, inf_rows[u]["inf_outside_lung_px"]) for u in live
               if u in inf_rows and inf_rows[u].get("inf_outside_lung_px", 0) > 0]
    outside.sort(key=lambda t: -t[1])
    total_inf = int(sum(inf_rows[u]["inf_area"] for u in live if u in inf_rows))
    log.add("H10", "infection inside lung", "WARN" if outside else "PASS",
            f"{len(outside)} COVID-19 image(s) have infection pixels outside the lung mask "
            f"({sum(n for _, n in outside)} of {total_inf} lesion px; not auto-dropped)" if outside
            else "no infection pixel lies outside the lung mask",
            {"images_with_outside_px": len(outside), "outside_px_total": int(sum(n for _, n in outside)),
             "infection_px_total": total_inf,
             "worst": _cap([{"path": by_uid[u].infection_mask_path, "outside_px": n} for u, n in outside])})

    # ---- H11 image content -----------------------------------------------------------
    blank = [u for u in live if lung_rows[u]["content"]["std"] < cfg.blank_std
             or lung_rows[u]["content"]["dynamic_range"] < 8]
    for u in blank:
        drop(u, "blank_image", f"std {lung_rows[u]['content']['std']:.2f}")
    flat = [u for u in live if u not in dropped and lung_rows[u]["content"]["std"] < cfg.low_contrast_std]
    sat = [u for u in live if u not in dropped
           and lung_rows[u]["content"]["saturated_frac"] > cfg.saturated_fraction]
    log.add("H11", "image content", "WARN" if (blank or flat or sat) else "PASS",
            f"{len(blank)} blank (dropped), {len(flat)} low-contrast, {len(sat)} saturated (kept, listed)"
            if (blank or flat or sat) else "no blank, low-contrast or saturated frames",
            {"blank": len(blank), "low_contrast": len(flat), "saturated": len(sat),
             "std": _summary(np.array([lung_rows[u]["content"]["std"] for u in live])),
             "examples": _cap([by_uid[u].image_path for u in blank + flat + sat])})
    live = [u for u in live if u not in dropped]

    # ---- clusters (H12-H15) ----------------------------------------------------------
    sha_bytes = {u: lung_rows[u]["sha_bytes"] for u in live}
    sha_pixels = {u: lung_rows[u]["sha_pixels"] for u in live}
    phashes = {u: int(lung_rows[u]["phash"], 16) for u in live}
    paths = {u: by_uid[u].image_path for u in live}
    subject_keys = (
        {u: s for u in live if (s := parse_bids_subject(by_uid[u].filename)) is not None}
        if cfg.group_by_subject else {}
    )
    clusters = build_clusters(
        live, sha_bytes, sha_pixels, phashes, paths,
        phash_hamming_max=cfg.phash_hamming_max, pixel_mae=cfg.phash_pixel_mae,
        min_corr=cfg.phash_min_corr, bucket_cap=cfg.phash_bucket_cap,
        subject_keys=subject_keys,
    )

    cluster_scope = list(live)
    exact_groups = clusters.exact_groups
    log.add("H12", "exact duplicates", "WARN" if exact_groups else "PASS",
            f"{len(exact_groups)} pixel-identical group(s) covering "
            f"{sum(len(g) for g in exact_groups)} images; {len(clusters.exact_byte_pairs)} byte-identical "
            f"pair(s), {len(clusters.exact_pixel_pairs)} pixel-identical pair(s) (the difference is "
            "re-encoded copies)" if exact_groups else "no exact duplicates",
            {"pixel_identical_groups": len(exact_groups),
             "images_in_groups": sum(len(g) for g in exact_groups),
             "byte_identical_pairs": len(clusters.exact_byte_pairs),
             "pixel_identical_pairs": len(clusters.exact_pixel_pairs),
             "examples": _cap([[by_uid[u].image_path for u in g] for g in exact_groups])})

    # conflicting duplicates: identical image, different class or masks -> drop the cluster
    conflict_clusters: dict[int, str] = {}
    for group in exact_groups:
        classes = {by_uid[u].cls for u in group}
        lung_hashes = {lung_rows[u]["sha_lung_mask"] for u in group}
        inf_hashes = {inf_rows[u]["sha_inf_mask"] for u in group if u in inf_rows}
        reasons = []
        if len(classes) > 1:
            reasons.append(f"classes {sorted(classes)}")
        if len(lung_hashes) > 1:
            reasons.append("different lung masks")
        if len(inf_hashes) > 1:
            reasons.append("different infection masks")
        if reasons:
            conflict_clusters[clusters.cluster_of[group[0]]] = " + ".join(reasons)
    conflict_uids = [u for cid in conflict_clusters for u in clusters.members[cid]]
    for cid, why in conflict_clusters.items():
        for u in clusters.members[cid]:
            drop(u, "conflicting_duplicate_cluster", f"cluster {cid}: {why}")
    log.add("H13", "conflicting duplicates", "WARN" if conflict_clusters else "PASS",
            f"{len(conflict_clusters)} cluster(s) hold the same image under different labels; "
            f"{len(conflict_uids)} image(s) dropped with them" if conflict_clusters
            else "no identical image carries two different labels or masks",
            {"clusters": len(conflict_clusters), "images_dropped": len(conflict_uids),
             "examples": _cap([{"cluster": cid, "why": why,
                                "paths": [by_uid[u].image_path for u in clusters.members[cid]][:6]}
                               for cid, why in conflict_clusters.items()])})

    # exact collapse: keep one member per pixel-identical group, preferring supervision
    collapsed = 0
    for group in exact_groups:
        survivors = [u for u in group if u not in dropped]
        if len(survivors) < 2:
            continue
        survivors.sort(key=lambda u: (not by_uid[u].in_infection_tree, by_uid[u].image_path))
        keep = survivors[0]
        for u in survivors[1:]:
            drop(u, "exact_duplicate", f"pixel-identical to {by_uid[keep].image_path}")
            collapsed += 1

    near = clusters.near_pairs
    near_cross_class = [(a, b) for (a, b) in near if by_uid[a].cls != by_uid[b].cls]
    log.add("H14", "near duplicates", "WARN" if near else "PASS",
            f"{len(near)} pHash pair(s) within {cfg.phash_hamming_max} bits confirmed at the pixel "
            f"level (MAE <= {cfg.phash_pixel_mae}, corr >= {cfg.phash_min_corr}); clustered, not "
            f"dropped; {clusters.stats.get('phash_rejected', 0)} hash-near candidates rejected by the "
            f"pixel test; {len(near_cross_class)} confirmed pair(s) disagree on class"
            if near else "no confirmed near duplicates",
            {"candidates": clusters.stats.get("phash_candidates", 0), "confirmed": len(near),
             "rejected_by_pixel_test": clusters.stats.get("phash_rejected", 0),
             "skipped_buckets": clusters.stats.get("skipped_buckets", 0),
             "skipped_items": clusters.stats.get("skipped_items", 0),
             "cross_class_pairs": len(near_cross_class),
             "hamming_hist": dict(sorted(Counter(int(v["hamming"]) for v in near.values()).items())),
             "examples": _cap([{"a": by_uid[a].image_path, "b": by_uid[b].image_path, **v}
                               for (a, b), v in near.items()])})

    # ---- H15 group integrity ---------------------------------------------------------
    unparsed = [u for u in live if parse_stem(by_uid[u].filename, by_uid[u].cls) is None]
    disagree = [u for u in live
                if (p := parse_stem(by_uid[u].filename, by_uid[u].cls)) and p[0] != by_uid[u].cls]
    bids = [u for u in live if parse_bids_subject(by_uid[u].filename) is not None]
    subjects = Counter(parse_bids_subject(by_uid[u].filename) for u in bids)
    size_dist = clusters.size_distribution()
    log.add("H15", "group integrity", "WARN" if (unparsed or disagree) else "PASS",
            f"{len(unparsed)} filename(s) do not parse, {len(disagree)} stem(s) disagree with their "
            f"class directory" if (unparsed or disagree)
            else f"class and stem parse for every file; cluster sizes {size_dist}; {len(bids)} "
                 f"BIDS-style file(s) from {len(subjects)} subject id(s) (see subject_groups)",
            {"subject_groups": {"bids_files": len(bids), "subjects": len(subjects),
                                "subjects_with_multiple_files": sum(1 for n in subjects.values() if n > 1),
                                "largest_subject": max(subjects.values(), default=0)},
             "unparsed": _cap([by_uid[u].image_path for u in unparsed]),
             "stem_class_disagree": _cap([by_uid[u].image_path for u in disagree]),
             "cluster_size_distribution": {str(k): v for k, v in size_dist.items()},
             "clusters": clusters.stats["clusters"], "largest_cluster": clusters.stats["largest_cluster"]})

    # ---- H16 tree consistency --------------------------------------------------------
    twin_bad = []
    twin_checked = 0
    for u in live:
        irow = inf_rows.get(u)
        if irow is None:
            continue
        twin_checked += 1
        lrow = lung_rows[u]
        if irow["sha_bytes"] != lrow["sha_bytes"] and irow["sha_pixels"] != lrow["sha_pixels"]:
            twin_bad.append((u, "image differs"))
        elif irow["sha_lung_mask"] != lrow["sha_lung_mask"]:
            twin_bad.append((u, "lung mask differs"))
    if twin_bad:
        log.add("H16", "tree consistency", "FAIL",
                f"{len(twin_bad)} infection-tree file(s) differ from their lung-tree twin; the "
                "infection mask may not align with the image the model sees",
                {"examples": _cap([{"path": by_uid[u].infection_image_path, "why": w}
                                   for u, w in twin_bad])},
                fail_uids=[u for u, _ in twin_bad])
    else:
        log.add("H16", "tree consistency", "PASS",
                f"all {twin_checked} infection-tree images and lung masks equal their lung-tree twins "
                "(checked exhaustively)", {"checked": twin_checked})

    # ---- FAIL handling ---------------------------------------------------------------
    failing = log.failed
    if failing and cfg.continue_on_fail:
        for chk in failing:
            for u in chk.fail_uids:
                drop(u, f"fail_{chk.id}", chk.detail)
        logger.warning("--continue-on-fail: dropped %d offender(s) of %d FAIL check(s)",
                       sum(len(c.fail_uids) for c in failing), len(failing))

    for uid, (reason, detail) in sorted(dropped.items()):
        r = by_uid[uid]
        drops.append({"path": r.image_path, "tree": "lung", "split": r.shipped_split,
                      "class": r.cls, "reason": reason, "detail": detail})
    survivors = [r for r in records if r.uid not in dropped]


    by_reason = Counter(d["reason"] for d in drops)
    report = {
        "rows_version": ROWS_VERSION,
        "elapsed_s": round(time.time() - started, 1),
        "counts": {
            "items_total": len(tasks), "lung_tree_items": n_lung, "infection_tree_items": n_inf,
            "records_catalogued": len(records), "records_kept": len(survivors),
            "records_dropped": len(records) - len(survivors),
        },
        "tree_counts_by_shipped_split": catalog.tree_counts,
        "checks": log.to_list(),
        "tally": log.tally(),
        "drops": {"total": len(drops), "by_reason": dict(by_reason)},
        "clusters": {**clusters.stats,
                     "size_distribution": {str(k): v for k, v in size_dist.items()}},
        "collapsed_exact_duplicates": collapsed,
        "cfg": {k: getattr(cfg, k) for k in (
            "lung_area_floor_px", "phash_hamming_max", "phash_pixel_mae", "phash_min_corr",
            "image_size", "continue_on_fail")},
    }

    result = HygieneResult(
        records=survivors, all_records=records, drops=drops, checks=log, clusters=clusters,
        cluster_of=clusters.cluster_of, rows=rows, report=report, cluster_scope=cluster_scope,
    )
    write_drops(drops, os.path.join(out_dir, "dropped_files.csv"))
    write_hygiene_json(report, os.path.join(out_dir, "hygiene.json"))
    if failing and not cfg.continue_on_fail:
        raise HygieneFailure(
            "hygiene FAIL (report written to hygiene.json): "
            + "; ".join(f"{c.id} {c.name}: {c.detail}" for c in failing)
            + ". Inspect the data, or rerun with --continue-on-fail to drop the offenders."
        )
    return result


# ----------------------------------------------------------------------- writers


def write_drops(drops: list[dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "tree", "split", "class", "reason", "detail"])
        writer.writeheader()
        writer.writerows(drops)


def write_hygiene_json(report: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1, default=str)
        handle.write("\n")
    os.replace(tmp, path)
