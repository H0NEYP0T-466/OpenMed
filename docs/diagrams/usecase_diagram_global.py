#!/usr/bin/env python3
"""OpenMed SRS — GLOBAL use case diagram (Figure 5.6.1).

Scope: release 1.0 exactly as specified in docs/srs_openmed.tex — all organ
modules (10 organs plus the full-body atlas), all assistive modules (MedSAM-2
prompting, 3D viewer, clinical assistant, report engine) and all four user
classes. The radiologist's column carries the full reading workflow:
ingest -> classification -> segmentation -> MedSAM-2 refinement -> 3D review ->
assistant -> report -> sign-off.

Two columns: the radiologist's workflow on the left, the other three user
classes on the right. Every actor owns exactly one column, so no association
line has to cross another column.

Sizing is chosen for print: type is ~1.5% of the canvas width, which lands at
about 7pt when the figure is placed at 16.5cm in the SRS.

The audit is a gate, and the layout tunables are searched for, not guessed: if a
connector passes through an ellipse, rows sit too close, or anything leaves the
boundary, the run fails instead of writing a broken figure.

Pure stdlib. Run:  python3 usecase_diagram_global.py
Out:  usecase_openmed_global.svg
"""

import math
import sys

# --------------------------------------------------------------- content ---
# A_COL marks the left column; its x is a tunable the solver picks.
A_COL = -1
UC = [
    ("ingest",    A_COL, 160,  "Upload & Preprocess Scan"),
    ("classify",  A_COL, 305,  "Classify Disease Across Organs"),
    ("segment",   A_COL, 450,  "Segment Lesion"),
    ("refine",    A_COL, 595,  "Refine Mask with\nMedSAM-2 Prompts"),
    ("view3d",    A_COL, 740,  "View 3D Organ & Lesion"),
    ("assistant", A_COL, 885,  "Consult Clinical Assistant"),
    ("report",    A_COL, 1030, "Generate Clinical Report"),
    ("signoff",   A_COL, 1175, "Sign Off & Export PDF"),

    ("viewfinal", 1380, 300, "Review Finalized Report"),
    ("note",      1380, 450, "Add Consultation Note"),
    ("practice",  1380, 620, "Practice in Training Mode"),
    ("users",     1380, 790, "Manage Users &\nSystem Settings"),
    ("audit",     1380, 960, "Review Audit Trail"),
]

# aid, x, y, label — left actor x/y and the left column x are tunables
ACTORS = [
    ("radiologist",   -1, -1, "Radiologist /\nMedical Specialist"),
    ("referring",   1660, 375, "Referring Doctor"),
    ("resident",    1660, 620, "Radiology Resident /\nStudent"),
    ("admin",       1660, 945, "System Administrator"),
]

ASSOC = {
    "radiologist": ["ingest", "classify", "segment", "refine", "view3d",
                    "assistant", "report", "signoff"],
    "referring":   ["viewfinal", "note"],
    "resident":    ["practice"],
    "admin":       ["users", "audit"],
}

DEPS = [
    ("refine",  "segment", "extend"),
    ("signoff", "report",  "include"),
]

RX, RY = 186, 56
GAP_Y = 2 * RY + 14
BOUND = dict(x0=160, x1=1590, y0=100, y1=1270)
W, H = 1800, 1500
FONT_UC = 24
FONT_ACTOR = 23


# -------------------------------------------------------------- geometry ---
class Layout:
    """A concrete placement: left column x, radiologist x and y."""

    def __init__(self, col_a, rad_x, rad_y):
        self.col_a, self.rad_x, self.rad_y = col_a, rad_x, rad_y
        self.uc = [(u, col_a if x == A_COL else x, y, lab) for u, x, y, lab in UC]
        self.actors = [(a, rad_x if x < 0 else x, rad_y if y < 0 else y, lab)
                       for a, x, y, lab in ACTORS]

    def P(self, uid):
        for u, x, y, _ in self.uc:
            if u == uid:
                return x, y
        raise KeyError(uid)

    def A(self, aid):
        for a, x, y, _ in self.actors:
            if a == aid:
                return x, y
        raise KeyError(aid)

    def connectors(self):
        segs = []
        for aid, targets in ASSOC.items():
            ax, ay = self.A(aid)
            for uid in targets:
                tx, ty = self.P(uid)
                segs.append((uid, f"assoc {aid}->{uid}", (ax, ay),
                             edge_point(tx, ty, ax, ay)))
        for src, dst, kind in DEPS:
            sx, sy = self.P(src)
            dx, dy = self.P(dst)
            segs.append((dst, f"{kind} {src}->{dst}",
                         edge_point(sx, sy, dx, dy), edge_point(dx, dy, sx, sy)))
        return segs

    def defects(self):
        bad = []
        ids = [(u, x, y) for u, x, y, _ in self.uc]
        for skip, label, p1, p2 in self.connectors():
            for oid, ox, oy in ids:
                if oid == skip:
                    continue
                if seg_hits(p1, p2, ox, oy, RX - 4, RY - 4):
                    bad.append(f"{label} cuts through {oid}")
        for u, x, y, _ in self.uc:
            if (x - RX < BOUND["x0"] or x + RX > BOUND["x1"]
                    or y - RY < BOUND["y0"] or y + RY > BOUND["y1"]):
                bad.append(f"{u} leaves the system boundary")
        for cx in {self.col_a, 1380}:
            ys = sorted(y for _, x, y, _ in self.uc if x == cx)
            for i in range(len(ys) - 1):
                if ys[i + 1] - ys[i] < GAP_Y:
                    bad.append(f"column x={cx}: rows {ys[i]}/{ys[i+1]} too close")
        for aid, ax, ay, _ in self.actors:
            if BOUND["x0"] < ax < BOUND["x1"]:
                bad.append(f"actor {aid} sits inside the boundary")
        return bad

    def statistics(self):
        ids = [(u, x, y) for u, x, y, _ in self.uc]
        grazes, segs = 0, []
        for skip, label, p1, p2 in self.connectors():
            segs.append((p1, p2))
            for oid, ox, oy in ids:
                if oid != skip and seg_hits(p1, p2, ox, oy, RX + 16, RY + 16):
                    grazes += 1
        xr = sum(crosses(segs[i], segs[j])
                 for i in range(len(segs)) for j in range(i + 1, len(segs)))
        return grazes, xr


def seg_hits(p1, p2, cx, cy, rx, ry):
    fx, fy = p1[0] - cx, p1[1] - cy
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    a = (dx / rx) ** 2 + (dy / ry) ** 2
    if a == 0:
        return False
    b = 2.0 * ((fx * dx) / rx ** 2 + (fy * dy) / ry ** 2)
    c = (fx / rx) ** 2 + (fy / ry) ** 2 - 1.0
    disc = b * b - 4 * a * c
    if disc <= 0:
        return False
    r = math.sqrt(disc)
    t1, t2 = (-b - r) / (2 * a), (-b + r) / (2 * a)
    return max(t1, t2) > 0.03 and min(t1, t2) < 0.97


def edge_point(cx, cy, tx, ty):
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx + RX, cy
    s = 1.0 / math.hypot(dx / RX, dy / RY)
    return cx + dx * s, cy + dy * s


def crosses(s1, s2):
    (x1, y1), (x2, y2) = s1
    (x3, y3), (x4, y4) = s2
    d = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(d) < 1e-9:
        return False
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / d
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / d
    return 0.06 < t < 0.94 and 0.06 < u < 0.94


def solve():
    """Pick the left column x and radiologist position that audit clean."""
    best = None
    for col_a in range(720, 941, 10):
        for rad_x in (112, 126, 140):
            for rad_y in range(560, 791, 10):
                lay = Layout(col_a, rad_x, rad_y)
                if lay.defects():
                    continue
                g, xr = lay.statistics()
                key = (xr, g)
                if best is None or key < best[0]:
                    best = (key, col_a, rad_x, rad_y)
    return best


# ---------------------------------------------------------------- render ---
CHAR_W = 0.545          # DejaVu Sans average advance, in em


def wrap(label, font, max_width):
    """Word-wrap so no line is wider than the ellipse interior."""
    limit = max(8, int(max_width / (font * CHAR_W)))
    out = []
    for para in label.split("\n"):
        cur = ""
        for w in para.split():
            cand = (cur + " " + w).strip()
            if cur and len(cand) > limit:
                out.append(cur)
                cur = w
            else:
                cur = cand
        if cur:
            out.append(cur)
    return out


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def actor_glyph(x, y):
    return (f'<g stroke="#1f2d33" stroke-width="2.4" fill="none" stroke-linecap="round">'
            f'<circle cx="{x}" cy="{y-44}" r="13"/>'
            f'<line x1="{x}" y1="{y-31}" x2="{x}" y2="{y+6}"/>'
            f'<line x1="{x-23}" y1="{y-17}" x2="{x+23}" y2="{y-17}"/>'
            f'<line x1="{x}" y1="{y+6}" x2="{x-19}" y2="{y+40}"/>'
            f'<line x1="{x}" y1="{y+6}" x2="{x+19}" y2="{y+40}"/></g>')


def note(x, y, w, h, title, lines, fs=19):
    s = [f'<path d="M{x},{y} h{w-18} l18,18 v{h-18} h{-w} z" fill="#fbfaf0" '
         f'stroke="#b8ad74" stroke-width="1.2"/>',
         f'<path d="M{x+w-18},{y} v18 h18" fill="none" stroke="#b8ad74" stroke-width="1.2"/>',
         f'<text x="{x+12}" y="{y+24}" font-family="DejaVu Sans, sans-serif" '
         f'font-size="{fs+1}" font-weight="700" fill="#5d5326">{esc(title)}</text>']
    ty = y + 50
    for ln in lines:
        s.append(f'<text x="{x+12}" y="{ty}" font-family="DejaVu Sans, sans-serif" '
                 f'font-size="{fs}" fill="#4b431f">{esc(ln)}</text>')
        ty += 24
    return "".join(s)


def render(lay, path):
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           f'<text x="{BOUND["x0"]}" y="60" font-family="DejaVu Sans, sans-serif" '
           'font-size="34" font-weight="700" fill="#101820">OpenMed — Global Use Case '
           'Model, release 1.0</text>',
           f'<rect x="{BOUND["x0"]}" y="{BOUND["y0"]}" width="{BOUND["x1"]-BOUND["x0"]}" '
           f'height="{BOUND["y1"]-BOUND["y0"]}" fill="#ffffff" stroke="#263238" '
           'stroke-width="2.2"/>',
           f'<text x="{BOUND["x0"]+18}" y="{BOUND["y0"]+36}" '
           'font-family="DejaVu Sans, sans-serif" font-size="24" font-weight="700" '
           'fill="#263238">OpenMed Platform</text>']

    for _, label, p1, p2 in lay.connectors():
        if not label.startswith("assoc"):
            continue
        out.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" '
                   f'y2="{p2[1]:.1f}" stroke="#a3b3ba" stroke-width="1.15"/>')

    for _, label, p1, p2 in lay.connectors():
        if label.startswith("assoc"):
            continue
        kind = label.split()[0]
        col = "#7a5c3e" if kind == "include" else "#6b5b4a"
        out.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" '
                   f'y2="{p2[1]:.1f}" stroke="{col}" stroke-width="1.3" '
                   'stroke-dasharray="7,5"/>')
        ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        q2 = (p2[0] - 11 * math.cos(ang - 0.42), p2[1] - 11 * math.sin(ang - 0.42))
        q3 = (p2[0] - 11 * math.cos(ang + 0.42), p2[1] - 11 * math.sin(ang + 0.42))
        out.append(f'<polygon points="{p2[0]:.1f},{p2[1]:.1f} {q2[0]:.1f},{q2[1]:.1f} '
                   f'{q3[0]:.1f},{q3[1]:.1f}" fill="#ffffff" stroke="{col}" '
                   'stroke-width="1.3"/>')
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        nx, ny = -(p2[1] - p1[1]), (p2[0] - p1[0])
        n = math.hypot(nx, ny) or 1
        mx += nx / n * 20
        my += ny / n * 20 + 4
        out.append(f'<rect x="{mx-58:.1f}" y="{my-15:.1f}" width="116" height="24" '
                   'fill="#ffffff" opacity="0.95"/>')
        out.append(f'<text x="{mx:.1f}" y="{my+5:.1f}" text-anchor="middle" '
                   'font-family="DejaVu Sans, sans-serif" font-size="17" '
                   f'font-style="italic" fill="{col}">&lt;&lt;{kind}&gt;&gt;</text>')

    for uid, x, y, label in lay.uc:
        out.append(f'<ellipse cx="{x}" cy="{y}" rx="{RX}" ry="{RY}" fill="#ffffff" '
                   'stroke="#2b3a42" stroke-width="1.6"/>')
        ls = wrap(label, FONT_UC, 2 * RX - 26)
        base = y - (len(ls) - 1) * 14 + 8
        for i, ln in enumerate(ls):
            out.append(f'<text x="{x}" y="{base + i*29:.1f}" text-anchor="middle" '
                       'font-family="DejaVu Sans, sans-serif" '
                       f'font-size="{FONT_UC}" fill="#16202a">{esc(ln)}</text>')

    for aid, x, y, label in lay.actors:
        out.append(actor_glyph(x, y))
        ls = wrap(label, FONT_ACTOR, 240)
        for i, ln in enumerate(ls):
            out.append(f'<text x="{x}" y="{y + 82 + i*27}" text-anchor="middle" '
                       'font-family="DejaVu Sans, sans-serif" '
                       f'font-size="{FONT_ACTOR}" font-weight="600" '
                       f'fill="#1f2d33">{esc(ln)}</text>')

    out.append(note(160, 1300, 640, 160, "Organ modules", [
        "Brain · Lungs · Heart · Liver · Kidneys · Pancreas",
        "Breast · Skin · Eye/Retina · Blood/Vascular · Bone/Skeleton",
        "plus the full-body atlas. Each organ carries its own",
        "classifier and segmentation pipeline; organ pairs that need",
        "it run a two-model mixture of experts.",
    ]))
    out.append(note(840, 1300, 680, 160, "Rules the model keeps", [
        "Findings are assistive only; a licensed clinician signs",
        "off before export. Trainees cannot finalize records.",
        "Scans, masks and reports never leave the clinic network.",
        "One organ pipeline resident in RAM at a time (8 GB).",
        "Model weights load on demand and unload on organ switch.",
    ]))
    out.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(out))


if __name__ == "__main__":
    found = solve()
    if not found:
        raise SystemExit("no defect-free layout found")
    key, col_a, rad_x, rad_y = found
    lay = Layout(col_a, rad_x, rad_y)
    g, xr = lay.statistics()
    print(f"layout: col_a={col_a} rad=({rad_x},{rad_y}) "
          f"grazes={g} crossings={xr} defects={len(lay.defects())}")
    if lay.defects():
        raise SystemExit("audit failed — figure not written")
    render(lay, "usecase_openmed_global.svg")
    print("wrote usecase_openmed_global.svg")
