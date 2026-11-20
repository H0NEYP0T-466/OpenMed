#!/usr/bin/env python3
"""Generate the OpenMed SRS use case diagram (Figure 5.6.1) as SVG.

Pure stdlib. Deterministic layout, no external services.
Run:  python3 usecase_diagram.py  ->  usecase_openmed.svg
"""

import math

W, H = 1750, 1820

# ---------------------------------------------------------------- model ---
PACKAGES = [
    ("Case & Scan Intake", [
        ("ingest", "Create Case & Upload Scan"),
        ("review_scan", "Review Scan & Orientation"),
        ("reject_invalid", "Reject Invalid Scan"),
    ]),
    ("Diagnosis", [
        ("select_dept", "Select Organ Department"),
        ("analyze", "Run Organ Analysis"),
        ("preprocess", "Preprocess & Normalize Scan"),
        ("findings", "Review Diagnostic Findings"),
    ]),
    ("Segmentation", [
        ("refine_pts", "Refine with Point Prompts"),
        ("autoseg", "Run Automated Segmentation"),
        ("refine_box", "Refine with Box Prompt"),
        ("measure", "Measure Lesion Area & Volume"),
    ]),
    ("3D Visualization", [
        ("view3d", "View 3D Organ & Lesion"),
        ("atlas", "Explore Full-Body Atlas"),
        ("presets", "Apply Camera Preset"),
    ]),
    ("Reporting", [
        ("draft", "Generate & Edit Report"),
        ("signoff", "Sign Off & Export PDF"),
        ("viewfinal", "View Finalized Report"),
        ("consultnote", "Add Consultation Note"),
    ]),
    ("Decision Support", [
        ("assistant", "Consult Case Assistant"),
    ]),
    ("Case Management & Audit", [
        ("history", "Review Case History"),
        ("compare", "Compare Automated vs Adjusted"),
        ("audit", "Review Audit Trail"),
    ]),
    ("Administration", [
        ("users", "Manage Users & Roles"),
        ("ops", "Configure Storage & Monitor Health"),
    ]),
    ("Training Mode", [
        ("practice", "Practice on Teaching Case"),
        ("compare_model", "Compare Observation vs Model"),
    ]),
    ("Model Lifecycle", [
        ("load", "Load Organ Pipeline"),
        ("switch_ws", "Switch Organ Workspace"),
        ("unload", "Unload & Free Memory"),
    ]),
]

# id -> (package index, row index) computed below
UC = {}
for pi, (_, ucs) in enumerate(PACKAGES):
    for ri, (uid, label) in enumerate(ucs):
        UC[uid] = {"pkg": pi, "row": ri, "label": label}

COL_X = [230, 870]
COL_W = 600
ROW_Y = [140, 440, 740, 1040, 1340]
ROW_H = 280
PKG_COL = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
PKG_ROW = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]

for pi, (name, _) in enumerate(PACKAGES):
    px = COL_X[PKG_COL[pi]]
    py = ROW_Y[PKG_ROW[pi]]
    for ri, (uid, _) in enumerate(PACKAGES[pi][1]):
        UC[uid]["x"] = px + COL_W / 2
        UC[uid]["y"] = py + 70 + ri * 62

RX, RY = 160, 20

# ---------------------------------------------------------------- actors ---
ACTORS = [
    # id, label, side, y, color
    ("radiologist", "Radiologist / Medical Specialist", "left", 260, "#1565c0"),
    ("resident", "Radiology Resident / Student", "left", 900, "#2e7d32"),
    ("referring", "Referring Doctor", "right", 740, "#ef6c00"),
    ("admin", "Administrator", "right", 1150, "#6a1b9a"),
]
ACTOR_POS = {}
for aid, label, side, y, color in ACTORS:
    ACTOR_POS[aid] = {"x": 110 if side == "left" else 1640, "y": y,
                      "side": side, "color": color, "label": label}

ASSOC = {
    "radiologist": ["ingest", "review_scan", "reject_invalid", "select_dept",
                    "analyze", "findings", "autoseg", "refine_pts", "refine_box",
                    "measure", "view3d", "atlas", "presets", "draft", "signoff",
                    "assistant", "history", "compare", "switch_ws"],
    "resident": ["practice", "compare_model", "autoseg", "refine_pts",
                 "refine_box"],
    "referring": ["viewfinal", "consultnote", "findings", "view3d", "assistant"],
    "admin": ["users", "ops", "audit"],
}

# (source, target, stereotype)
DEPENDENCIES = [
    ("analyze", "preprocess", "include"),
    ("draft", "measure", "include", (470, 735)),
    ("signoff", "draft", "include"),
    ("view3d", "autoseg", "include"),
    ("refine_pts", "autoseg", "extend"),
    ("refine_box", "autoseg", "extend"),
    ("switch_ws", "load", "include"),
    ("switch_ws", "unload", "include"),
    ("practice", "compare_model", "include"),
]

GENERALIZATIONS = []

# ---------------------------------------------------------------- helpers --
def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def wrap(label, n=44):
    words, lines, cur = label.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines


def edge_point(cx, cy, tx, ty):
    """Point on ellipse (cx,cy,rx,ry) along direction to (tx,ty)."""
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx + RX, cy
    scale = 1.0 / math.hypot(dx / RX, dy / RY)
    return cx + dx * scale, cy + dy * scale


def line(x1, y1, x2, y2, color, width=1.4, dash=None, opacity=1.0):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}"{d} opacity="{opacity}"/>\n')


def arrowhead(x, y, angle, color, fill):
    a = 9
    p1 = (x, y)
    p2 = (x - a * math.cos(angle - 0.45), y - a * math.sin(angle - 0.45))
    p3 = (x - a * math.cos(angle + 0.45), y - a * math.sin(angle + 0.45))
    return (f'<polygon points="{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f} '
            f'{p3[0]:.1f},{p3[1]:.1f}" fill="{fill}" stroke="{color}" '
            f'stroke-width="1.2"/>\n')


def actor(ax, ay, label, color):
    out = [f'<g stroke="{color}" stroke-width="2" fill="none">\n',
           f'<circle cx="{ax}" cy="{ay - 34}" r="12"/>\n',
           f'<line x1="{ax}" y1="{ay - 22}" x2="{ax}" y2="{ay + 8}"/>\n',
           f'<line x1="{ax - 20}" y1="{ay - 10}" x2="{ax + 20}" y2="{ay - 10}"/>\n',
           f'<line x1="{ax}" y1="{ay + 8}" x2="{ax - 16}" y2="{ay + 36}"/>\n',
           f'<line x1="{ax}" y1="{ay + 8}" x2="{ax + 16}" y2="{ay + 36}"/>\n',
           '</g>\n']
    for i, ln in enumerate(wrap(label, 18)):
        out.append(f'<text x="{ax}" y="{ay + 56 + i * 15}" text-anchor="middle" '
                   f'font-family="DejaVu Sans, sans-serif" font-size="12" '
                   f'font-weight="600" fill="{color}" paint-order="stroke" '
                   f'stroke="#ffffff" stroke-width="3">{esc(ln)}</text>\n')
    return "".join(out)


def usecase(uid):
    u = UC[uid]
    x, y = u["x"], u["y"]
    out = [f'<ellipse cx="{x}" cy="{y}" rx="{RX}" ry="{RY}" fill="#ffffff" '
           f'stroke="#263238" stroke-width="1.4"/>\n']
    lines = wrap(u["label"], 34)
    start = y - (len(lines) - 1) * 6 + 3.5
    for i, ln in enumerate(lines):
        out.append(f'<text x="{x}" y="{start + i * 12:.1f}" text-anchor="middle" '
                   f'font-family="DejaVu Sans, sans-serif" font-size="10.5" '
                   f'fill="#1c1c1c">{esc(ln)}</text>\n')
    return "".join(out)


# ---------------------------------------------------------------- render ---
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}">\n',
       '<rect width="100%" height="100%" fill="#ffffff"/>\n',
       '<text x="875" y="36" text-anchor="middle" font-family="DejaVu Sans, sans-serif" '
       'font-size="20" font-weight="700" fill="#111">OpenMed — Multi-Organ Diagnostic '
       'Workstation, release 1.0</text>\n',
       '<rect x="200" y="80" width="1320" height="1560" fill="none" stroke="#263238" '
       'stroke-width="2"/>\n',
       '<text x="212" y="104" font-family="DejaVu Sans, sans-serif" font-size="14" '
       'font-weight="700" fill="#263238">OpenMed Platform</text>\n']

# package frames (headers are redrawn after the association lines)
headers = []
for pi, (name, ucs) in enumerate(PACKAGES):
    px = COL_X[PKG_COL[pi]]
    py = ROW_Y[PKG_ROW[pi]]
    svg.append(f'<rect x="{px}" y="{py}" width="{COL_W}" height="{ROW_H}" '
               f'fill="#f7f9fb" stroke="#78909c" stroke-width="1.2"/>\n')
    svg.append(f'<rect x="{px}" y="{py}" width="{COL_W}" height="26" fill="#e3eaf0" '
               f'stroke="#78909c" stroke-width="1.2"/>\n')
    headers.append((px + COL_W / 2, py + 18, name))

# associations (colored per actor)
for aid, uids in ASSOC.items():
    a = ACTOR_POS[aid]
    ax = a["x"] + (22 if a["side"] == "left" else -22)
    ay = a["y"] - 10
    for uid in uids:
        u = UC[uid]
        ex, ey = edge_point(u["x"], u["y"], ax, ay)
        svg.append(line(ax, ay, ex, ey, a["color"], 1.2, opacity=0.55))

# actor generalisation: Radiologist -> Clinician (hollow triangle at Clinician)
for child, parent in GENERALIZATIONS:
    c, p = ACTOR_POS[child], ACTOR_POS[parent]
    x1, y1 = c["x"], c["y"] - 52
    x2, y2 = p["x"], p["y"] + 42
    svg.append(line(x1, y1, x2, y2, "#546e7a", 1.6))
    ang = math.atan2(y1 - y2, x1 - x2)
    svg.append(arrowhead(x2, y2, ang, "#546e7a", "#ffffff"))

# include / extend (drawn under the use case ellipses)
def label_pos(x1, y1, x2, y2):
    """Pick a spot on the arrow that is clear of every ellipse and header band."""
    # Short vertical arrows (one row to the next): a centred label would cover
    # the whole arrow, so sit beside it instead.
    if math.hypot(x2 - x1, y2 - y1) < 90 and abs(y2 - y1) > abs(x2 - x1):
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        for off in (-92, 92, -120, 120):
            cx = mx + off
            if all(((cx - u["x"]) / (RX + 16)) ** 2
                   + ((my - u["y"]) / (RY + 12)) ** 2 >= 1 for u in UC.values()):
                return cx, my
    for t in (0.5, 0.62, 0.38, 0.72, 0.28, 0.8, 0.2):
        mx, my = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
        clear = True
        for u in UC.values():
            if ((mx - u["x"]) / (RX + 16)) ** 2 + ((my - u["y"]) / (RY + 12)) ** 2 < 1:
                clear = False
                break
        if clear:
            for pi, (_, _ucs) in enumerate(PACKAGES):
                px, py = COL_X[PKG_COL[pi]], ROW_Y[PKG_ROW[pi]]
                if px - 8 <= mx <= px + COL_W + 8 and py <= my <= py + 26:
                    clear = False
                    break
                # keep clear of the centred header word too
                if (py <= my <= py + 26
                        and abs(mx - (px + COL_W / 2)) < 90):
                    clear = False
                    break
        if clear:
            return mx, my
    return (x1 + x2) / 2, (y1 + y2) / 2


dep_labels = []
for dep in DEPENDENCIES:
    src, dst, kind = dep[:3]
    s, d = UC[src], UC[dst]
    x1, y1 = edge_point(s["x"], s["y"], d["x"], d["y"])
    x2, y2 = edge_point(d["x"], d["y"], s["x"], s["y"])
    color = "#455a64" if kind == "include" else "#8d6e63"
    svg.append(line(x1, y1, x2, y2, color, 1.3, dash="5,4"))
    ang = math.atan2(y2 - y1, x2 - x1)
    svg.append(arrowhead(x2, y2, ang, color, "#ffffff"))
    mx, my = dep[3] if len(dep) > 3 else label_pos(x1, y1, x2, y2)
    dep_labels.append((mx, my, kind, color))

# use cases (opaque, so association lines pass behind unrelated ellipses)
for _, ucs in PACKAGES:
    for uid, _ in ucs:
        svg.append(usecase(uid))

# dependency stereotypes on top
for mx, my, kind, color in dep_labels:
    svg.append(f'<rect x="{mx - 27}" y="{my - 6}" width="54" height="12" '
               f'fill="#ffffff" opacity="0.95"/>\n')
    svg.append(f'<text x="{mx}" y="{my + 3}" text-anchor="middle" '
               f'font-family="DejaVu Sans, sans-serif" font-size="8" '
               f'font-style="italic" fill="{color}">&lt;&lt;{kind}&gt;&gt;</text>\n')

# package header text on top of the lines
for hx, hy, name in headers:
    svg.append(f'<text x="{hx}" y="{hy}" text-anchor="middle" '
               f'font-family="DejaVu Sans, sans-serif" font-size="12.5" '
               f'font-weight="700" fill="#263238" paint-order="stroke" '
               f'stroke="#e3eaf0" stroke-width="4">{esc(name)}</text>\n')

# actors
for aid, label, side, y, color in ACTORS:
    svg.append(actor(ACTOR_POS[aid]["x"], y, label, color))

# resident constraint
_ry = ACTOR_POS["resident"]["y"]
for _i, _ln in enumerate(["{training mode:", "no official records, no sign-off}"]):
    svg.append(f'<text x="115" y="{_ry + 96 + _i * 12}" text-anchor="middle" '
               'font-family="DejaVu Sans, sans-serif" font-size="9.5" font-style="italic" '
               'paint-order="stroke" stroke="#ffffff" stroke-width="3" '
               f'fill="#2e7d32">{esc(_ln)}</text>\n')

# notes
def note(x, y, w, h, lines, title=None):
    out = [f'<path d="M{x},{y} h{w - 14} l14,14 v{h - 14} h{-w} z" fill="#fffde7" '
           f'stroke="#9e9d24" stroke-width="1.2"/>\n',
           f'<path d="M{x + w - 14},{y} v14 h14" fill="none" stroke="#9e9d24" '
           f'stroke-width="1.2"/>\n']
    ty = y + 18
    if title:
        out.append(f'<text x="{x + 10}" y="{ty}" font-family="DejaVu Sans, sans-serif" '
                   f'font-size="10.5" font-weight="700" fill="#33691e">{esc(title)}</text>\n')
        ty += 15
    for ln in lines:
        out.append(f'<text x="{x + 10}" y="{ty}" font-family="DejaVu Sans, sans-serif" '
                   f'font-size="10" fill="#33691e">{esc(ln)}</text>\n')
        ty += 13
    return "".join(out)


svg.append(note(220, 1680, 680, 104, [
    "Brain · Lungs · Heart · Kidneys · Eye/Retina · Blood/Vascular ·",
    "Liver · Bone/Skeleton · Breast · Skin · Pancreas · Intestines",
    "+ Full-Body Atlas. Each department plugs its own classification and",
    "segmentation pipeline into the same clinical workspace; some departments",
    "run a two-model mixture of experts.",
], title="Organ departments (Select Organ Department)"))

svg.append(note(920, 1680, 600, 104, [
    "AI findings are assistive only; licensed clinician sign-off is required before export.",
    "Residents and students cannot finalize official records.",
    "Patient data never leaves the local clinic network.",
    "Only one organ pipeline is resident in memory at a time.",
], title="Business rules"))

svg.append('</svg>\n')

with open("usecase_openmed.svg", "w", encoding="utf-8") as f:
    f.write("".join(svg))
print("wrote usecase_openmed.svg")
