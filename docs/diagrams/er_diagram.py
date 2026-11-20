#!/usr/bin/env python3
"""OpenMed SRS — Entity-Relationship diagram (Figure 5.6.2).

Covers the full release-1.0 data model the SRS names: patient records, scan
series, classification labels, segmentation masks, generated reports — plus the
supporting entities the rest of the SRS implies (users, model registry, audit
log, consultation notes, interactive prompts).

Attribute names are taken from the real response contracts, not invented:
  backend/app/organs/brain/classification/router.py   -> ClassificationResponse
  backend/app/organs/brain/segmentation/experimental_lab/router.py
                                                      -> SegmentationResult /
                                                         ClickSegmentationResult

Notation: crow's foot (one = tick, many = three prongs).
The audit gate refuses to write the figure if a relationship line passes
through an entity box or a box collides with another.

Pure stdlib. Run:  python3 er_diagram.py
Out:  er_openmed.svg
"""

import math
from PIL import ImageFont

# ------------------------------------------------------------------ model ---
# id, column, row, name, attributes
# Attribute prefixes: "PK " primary key, "FK " foreign key, "U " unique.
ENTITIES = [
    ("patient", 0, 0, "PATIENT", [
        ("PK", "patient_id"), ("U", "mrn"), ("", "full_name"),
        ("", "date_of_birth"), ("", "sex"), ("", "created_at"),
    ]),
    ("case", 1, 0, "CASE", [
        ("PK", "case_id"), ("FK", "patient_id"), ("FK", "attending_id"),
        ("", "organ_module"), ("", "status"), ("", "opened_at"),
    ]),
    ("series", 2, 0, "SCAN_SERIES", [
        ("PK", "series_id"), ("FK", "case_id"), ("U", "scan_id"),
        ("", "modality"), ("", "organ"), ("", "slice_count"),
    ]),
    ("slice", 3, 0, "SCAN_SLICE", [
        ("PK", "slice_id"), ("FK", "series_id"), ("", "slice_index"),
        ("", "image_path"), ("", "width_px"), ("", "height_px"),
    ]),
    ("user", 0, 1, "APP_USER", [
        ("PK", "user_id"), ("U", "email"), ("", "full_name"),
        ("", "role"), ("", "is_active"),
    ]),
    ("report", 0, 2, "REPORT", [
        ("PK", "report_id"), ("FK", "case_id"), ("FK", "author_id"),
        ("", "status"), ("", "version"), ("", "signed_at"), ("", "pdf_path"),
    ]),
    ("classif", 2, 1, "CLASSIFICATION_RESULT", [
        ("PK", "result_id"), ("FK", "slice_id"), ("FK", "model_id"),
        ("", "predicted_class"), ("", "confidence"), ("", "tumour_type"),
        ("", "label_space"), ("", "gradcam_ref"), ("", "inference_ms"),
    ]),
    ("mask", 3, 1, "SEGMENTATION_MASK", [
        ("PK", "mask_id"), ("FK", "result_id"), ("FK", "model_id"),
        ("", "prompt_mode"), ("", "mask_path"), ("", "iou_pred"),
        ("", "foreground_px"), ("", "area_mm2"), ("", "volume_mm3"),
    ]),
    ("note", 1, 1, "CONSULTATION_NOTE", [
        ("PK", "note_id"), ("FK", "case_id"), ("FK", "author_id"),
        ("", "body"), ("", "created_at"),
    ]),
    ("audit", 1, 2, "AUDIT_LOG", [
        ("PK", "log_id"), ("FK", "user_id"), ("", "action"),
        ("", "entity"), ("", "entity_id"), ("", "occurred_at"),
    ]),
    ("model", 2, 2, "MODEL_REGISTRY", [
        ("PK", "model_id"), ("", "organ"), ("", "task"),
        ("", "architecture"), ("", "version"), ("", "checkpoint_path"),
        ("", "metrics_json"),
    ]),
    ("prompt", 3, 2, "SEGMENT_PROMPT", [
        ("PK", "prompt_id"), ("FK", "mask_id"), ("", "prompt_type"),
        ("", "point_xy"), ("", "box_xywh"), ("", "polarity"),
    ]),
]

# from, to, cardinality at from end ("1"), at to end ("N")
RELS = [
    ("patient", "case",   "1", "N"),   # a patient has many cases
    ("case", "series",    "1", "N"),   # a case holds many scan series
    ("series", "slice",   "1", "N"),   # a series holds many slices
    ("slice", "classif",  "1", "N"),   # each slice can be classified by several models
    ("classif", "mask",   "1", "N"),   # each result can carry several masks
    ("mask", "prompt",    "1", "N"),   # a mask is edited through several prompts
    ("model", "classif",  "1", "N"),   # one registry entry, many results
    ("model", "mask",     "1", "N"),   # one segmenter entry, many masks
    ("user", "case",      "1", "N"),   # attending clinician
    ("user", "report",    "1", "N"),   # signing clinician
    ("user", "note",      "1", "N"),   # author of consultation notes
    ("user", "audit",     "1", "N"),   # actor in the audit trail
    ("case", "report",    "1", "N"),
    ("case", "note",      "1", "N"),
]

BW = 286          # box width
BH_HDR = 40       # header height
ROW_LINE = 27     # attribute line height
COLS = [46, 386, 726, 1066]
ROWS = [130, 470, 810]
PAD = 12

W, H = 1410, 1180
F_NAME = 22
F_ATTR = 19
F_CARD = 20
F_LEGEND = 18


# --------------------------------------------------------------- geometry ---
_BOLD = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"


NAME_SIZE = 21


def name_metrics(name):
    """One uniform header size for every box; long names wrap at an underscore."""
    if ImageFont.truetype(_BOLD, NAME_SIZE).getlength(name) <= BW - 26:
        return NAME_SIZE, [name]
    parts = name.split("_")
    half = max(1, len(parts) // 2)
    lines = ["_".join(parts[:half]), "_".join(parts[half:])]
    size = NAME_SIZE
    while size > 14:
        f = ImageFont.truetype(_BOLD, size)
        if max(f.getlength(l) for l in lines) <= BW - 26:
            return size, lines
        size -= 1
    return size, lines


def box(e):
    """id, col, row, name, attrs -> (x0, y0, x1, y1, name, attrs, size, lines, hdr)"""
    eid, col, row, name, attrs = e
    size, lines = name_metrics(name)
    hdr = BH_HDR + (len(lines) - 1) * (size + 6)
    x0, y0 = COLS[col], ROWS[row]
    h = hdr + len(attrs) * ROW_LINE + PAD
    return x0, y0, x0 + BW, y0 + h, name, attrs, size, lines, hdr


BOXES = {e[0]: box(e) for e in ENTITIES}


CHANNEL = [(COLS[i] + BW + COLS[i + 1]) / 2 for i in range(3)]
ROW_BOTTOM = {}
for r in range(3):
    ROW_BOTTOM[r] = max(b[3] for b in (box(e) for e in ENTITIES) if True
                        and b is not None) if False else 0
ROW_BOTTOM = {r: max(box(e)[3] for e in ENTITIES if e[2] == r) for r in range(3)}
CORRIDOR = {r: (ROW_BOTTOM[r] + ROWS[r + 1]) / 2 for r in range(2)}


def route(a, b):
    """Polyline from a to b that only travels empty space.

    Same row or same column: straight between facing edges. Anything else:
    out of the source sideways into a free channel, down/up that channel,
    across inside the corridor between rows, then into the target. Both the
    channels (column gaps) and the corridors (row gaps) are empty by
    construction, so a routed line can never cross a box.
    """
    A, B = BOXES[a], BOXES[b]
    acol = COLS.index(A[0])
    bcol = COLS.index(B[0])
    arow = ROWS.index(A[1])
    brow = ROWS.index(B[1])
    acx, acy = (A[0] + A[2]) / 2, (A[1] + A[3]) / 2
    bcx, bcy = (B[0] + B[2]) / 2, (B[1] + B[3]) / 2

    if arow == brow:
        if acx < bcx:
            return [(A[2], acy), (B[0], bcy)], "right", "left"
        return [(A[0], acy), (B[2], bcy)], "left", "right"
    if acol == bcol:
        if acy < bcy:
            return [(acx, A[3]), (acx, B[1])], "down", "up"
        return [(acx, A[1]), (acx, B[3])], "up", "down"

    # vertical legs live in channels, horizontal legs in corridors, so a
    # routed line never enters a box it does not belong to
    if bcol > acol:
        ch1, out_edge, out_dir = CHANNEL[min(acol, 2)], A[2], "right"
        ch2, in_edge, in_dir = CHANNEL[max(bcol - 1, 0)], B[0], "left"
    else:
        ch1, out_edge, out_dir = CHANNEL[max(acol - 1, 0)], A[0], "left"
        ch2, in_edge, in_dir = CHANNEL[min(bcol, 2)], B[2], "right"
    corr = CORRIDOR[brow - 1] if brow > arow else CORRIDOR[brow]

    pts = [(out_edge, acy)]
    if abs(ch1 - ch2) < 1:
        pts += [(ch1, acy), (ch1, bcy), (in_edge, bcy)]
    else:
        pts += [(ch1, acy), (ch1, corr), (ch2, corr), (ch2, bcy), (in_edge, bcy)]
    clean = [pts[0]]
    for q in pts[1:]:
        if abs(q[0] - clean[-1][0]) > 0.5 or abs(q[1] - clean[-1][1]) > 0.5:
            clean.append(q)
    return clean, out_dir, in_dir


def seg_hits_box(p1, p2, b, margin=6.0):
    x0, y0, x1, y1 = b[0] + margin, b[1] + margin, b[2] - margin, b[3] - margin
    if x0 >= x1 or y0 >= y1:
        return False
    # Liang-Barsky clipping test
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p1[0] - x0), (dx, x1 - p1[0]),
                 (-dy, p1[1] - y0), (dy, y1 - p1[1])):
        if p == 0:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    return t0 < t1


def segments():
    out = []
    for a, b, ca, cb in RELS:
        pts, d1, d2 = route(a, b)
        out.append((a, b, ca, cb, pts, d1, d2))
    return out


def audit(verbose=True):
    bad = []
    segs = segments()
    for a, b, ca, cb, pts, d1, d2 in segs:
        for i in range(len(pts) - 1):
            for eid, bx in BOXES.items():
                if eid in (a, b):
                    continue
                if seg_hits_box(pts[i], pts[i + 1], bx):
                    bad.append(f"{a}->{b} passes through {eid}")
    ids = list(BOXES)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            A, B = BOXES[ids[i]], BOXES[ids[j]]
            if A[0] < B[2] and B[0] < A[2] and A[1] < B[3] and B[1] < A[3]:
                bad.append(f"{ids[i]} box overlaps {ids[j]} box")
    for eid, b in BOXES.items():
        if b[0] < 0 or b[1] < 0 or b[2] > W or b[3] > H:
            bad.append(f"{eid} escapes the canvas")
    if verbose:
        for x in bad:
            print("  " + x)
    return bad


def line_crossings():
    segs = [(p[i], p[i + 1]) for _, _, _, _, p, _, _ in segments()
            for i in range(len(p) - 1)]
    n = 0
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            (x1, y1), (x2, y2) = segs[i]
            (x3, y3), (x4, y4) = segs[j]
            d = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
            if abs(d) < 1e-9:
                continue
            t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / d
            u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / d
            if 0.08 < t < 0.92 and 0.08 < u < 0.92:
                n += 1
    return n


# ---------------------------------------------------------------- render ---
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def crow_foot(x, y, ang, ink="#33454e"):
    """Three prongs opening toward the box at (x, y) coming from direction ang."""
    out = []
    for off in (-0.42, 0.0, 0.42):
        a = ang + off
        out.append(f'<line x1="{x}" y1="{y}" x2="{x + 20*math.cos(a):.1f}" '
                   f'y2="{y + 20*math.sin(a):.1f}" stroke="{ink}" stroke-width="1.6"/>')
    return "".join(out)


def render(path):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         '<rect width="100%" height="100%" fill="#ffffff"/>',
         '<text x="46" y="52" font-family="DejaVu Sans, sans-serif" font-size="34" '
         'font-weight="700" fill="#101820">OpenMed — Entity-Relationship Diagram, '
         'release 1.0</text>',
         '<text x="46" y="82" font-family="DejaVu Sans, sans-serif" font-size="19" '
         'fill="#4a5b63">Crow\'s foot notation · PK primary key · FK foreign key · '
         'U unique</text>']

    NORM = {"right": (1, 0), "left": (-1, 0), "up": (0, -1), "down": (0, 1)}

    for a, b, ca, cb, pts, d1, d2 in segments():
        dstr = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        o.append(f'<path d="{dstr}" fill="none" stroke="#7d929c" stroke-width="1.6"/>')
        n1 = NORM[d1]
        n2 = NORM[d2]
        # crow's foot at the many end, tick at the one end
        if cb == "N":
            many_pt, many_n = pts[-1], n2
            one_pt, one_n = pts[0], n1
        else:
            many_pt, many_n = pts[0], n1
            one_pt, one_n = pts[-1], n2
        o.append(crow_foot(many_pt[0], many_pt[1],
                           math.atan2(-many_n[1], -many_n[0])))
        cx = one_pt[0] + 9 * -one_n[0]
        cy = one_pt[1] + 9 * -one_n[1]
        tx, ty = -one_n[1], one_n[0]
        o.append(f'<line x1="{cx - 8*tx:.1f}" y1="{cy - 8*ty:.1f}" '
                 f'x2="{cx + 8*tx:.1f}" y2="{cy + 8*ty:.1f}" '
                 'stroke="#33454e" stroke-width="1.9"/>')
        for pt, n, lbl in ((many_pt, many_n, "N" if cb == "N" else "1"),
                           (one_pt, one_n, "1" if cb == "N" else "N")):
            lx = pt[0] + 34 * -n[0] + 20 * -n[1]
            ly = pt[1] + 34 * -n[1] - 20 * -n[0] + 8
            o.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
                     'font-family="DejaVu Sans, sans-serif" font-size="'
                     f'{F_CARD}" font-weight="700" fill="#2f4049">{lbl}</text>')

    for eid, (x0, y0, x1, y1, name, attrs, nsize, nlines, hdr) in BOXES.items():
        o.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" '
                 'fill="#ffffff" stroke="#22323b" stroke-width="2"/>')
        o.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{hdr}" '
                 'fill="#eaf0f2" stroke="#22323b" stroke-width="2"/>')
        for i, ln in enumerate(nlines):
            o.append(f'<text x="{x0 + 12}" y="{y0 + 28 + i*(nsize+6)}" '
                     'font-family="DejaVu Sans, sans-serif" '
                     f'font-size="{nsize}" font-weight="700" '
                     f'fill="#15242c">{esc(ln)}</text>')
        y = y0 + hdr + 20
        for kind, attr in attrs:
            label = attr
            w = F_ATTR * 0.545 * len(label)
            if kind == "PK":
                o.append(f'<text x="{x0 + 14}" y="{y}" font-family="DejaVu Sans, sans-serif" '
                         f'font-size="{F_ATTR}" fill="#15242c">{esc(label)}</text>')
                o.append(f'<line x1="{x0 + 14}" y1="{y + 4}" x2="{x0 + 14 + w:.1f}" '
                         f'y2="{y + 4}" stroke="#15242c" stroke-width="1.2"/>')
            elif kind == "FK":
                o.append(f'<text x="{x0 + 14}" y="{y}" font-family="DejaVu Sans, sans-serif" '
                         f'font-size="{F_ATTR}" font-weight="600" fill="#15242c">'
                         f'{esc(label)}</text>')
            else:
                o.append(f'<text x="{x0 + 14}" y="{y}" font-family="DejaVu Sans, sans-serif" '
                         f'font-size="{F_ATTR}" fill="#37474f">{esc(label)}</text>')
            if kind:
                o.append(f'<text x="{x1 - 12}" y="{y}" text-anchor="end" '
                         'font-family="DejaVu Sans, sans-serif" font-size="15" '
                         f'fill="#6b7c85">{kind}</text>')
            y += ROW_LINE

    for i, ln in enumerate([
        "Every organ module writes into the same tables: one MODEL_REGISTRY row per "
        "organ, task and version.",
        "Findings, masks and reports are never overwritten, only versioned, so the "
        "audit trail stays complete.",
    ]):
        o.append(f'<text x="46" y="{H - 52 + i*26}" font-family="DejaVu Sans, '
                 f'sans-serif" font-size="{F_LEGEND}" fill="#4a5b63">{esc(ln)}</text>')
    o.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(o))


if __name__ == "__main__":
    bad = audit()
    print(f"defects={len(bad)} line-crossings={line_crossings()}")
    if bad:
        raise SystemExit("audit failed — figure not written")
    render("er_openmed.svg")
    print("wrote er_openmed.svg")
