#!/usr/bin/env python3
"""OpenMed SRS — System Architecture (Figure 5.6.3), platform-level view.

This is the GLOBAL architecture of the multi-organ platform, not a figure of the
two organs that happen to be implemented. Nothing in it is brain-specific or
lungs-specific: the brain and lungs appear only as two of the sixteen registered
organ modules, each an instance of the same contract.

Structure (top to bottom):

  client      organ selector, a workspace generated per organ, the prompt lab,
              the 3D atlas, the report editor + assistant
  transport   REST, the prompt WebSocket, multipart upload / PDF download
  api         FastAPI app; the model manager that keeps one organ resident
  organ       registry -> module contract -> one module per registered organ
  inference   the shared zoos every module calls (classifier, localiser,
              segmenter, frozen MedSAM-2) and the calibration sidecars
  data        checkpoints, dataset registry, scan cache, case store, audit trail

Audit gate: every label must measure inside its block, blocks must not overlap,
and no arrow segment may cross a block it does not start or end at.
Pure stdlib + Pillow for font metrics. Run: python3 architecture_diagram.py
"""

from PIL import ImageFont

REG = "/usr/share/fonts/TTF/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"

W, H = 1680, 1300
EDGE = 60
RIGHT = W - 40

# ------------------------------------------------------------------ model ---
# band: id, y0, y1, title, tint
BANDS = [
    ("client", 96, 268, "Client layer — browser, no plugins", "#eef3f5"),
    ("transport", 294, 356, "Transport", "#f3efe6"),
    ("api", 382, 500, "API service — FastAPI / Uvicorn, local host", "#eef3f5"),
    ("organ", 526, 820,
     "Organ layer — registry, one contract, one module per registered organ "
     "(16 registered, 8 shown)", "#eaf1ec"),
    ("infer", 846, 1080,
     "Shared inference layer — PyTorch, CPU only, one implementation for "
     "every organ", "#eef3f5"),
    ("data", 1106, 1224, "On-disk artifacts — nothing leaves the clinic",
     "#f3efe6"),
]

# block: id, x, y, w, h, title, lines
BLOCKS = [
    # ---- client -------------------------------------------------------
    ("selector", 60, 136, 290, 104, "Organ selector", [
        "16 organs, one grid",
        "availability read from registry",
        "task badges per organ",
    ]),
    ("workspace", 370, 136, 310, 104, "Diagnostic workspace", [
        "generated per organ",
        "slice scrubber · overlay",
        "calibrated dossier",
    ]),
    ("lab", 700, 136, 300, 104, "MedSAM-2 prompt lab", [
        "point or box prompt",
        "live mask refresh",
        "opacity · reset",
    ]),
    ("atlas", 1020, 136, 260, 104, "3D atlas — Three.js", [
        "WebGL 2.0 organ meshes",
        "lesion inside the mesh",
        "orbit · zoom · explode",
    ]),
    ("reportui", 1300, 136, 340, 104, "Report editor & assistant", [
        "editable draft · sign-off",
        "PDF export · assistant drawer",
        "grounded on this case only",
    ]),

    # ---- transport ----------------------------------------------------
    ("rest", 60, 300, 470, 40, "REST  /api/*  JSON, port 8000", []),
    ("wss", 550, 300, 430, 40, "WebSocket  /ws/segment/prompt", []),
    ("upload", 1000, 300, 640, 40, "multipart upload · PDF download", []),

    # ---- api ----------------------------------------------------------
    ("fastapi", 60, 402, 1580, 62, "FastAPI application", [
        "app.main · CORS origin allow-list · Pydantic response models · "
        "routers /api/organs · /api/lab · /api/reports · /api/cases",
    ]),

    # ---- organ layer --------------------------------------------------
    ("manager", 60, 546, 700, 84, "Model manager — lazy load / unload", [
        "zero weights on boot · one organ resident at a time",
        "unload + GC on organ switch · 8 GB ceiling",
    ]),
    ("registry", 780, 546, 860, 84, "Organ registry", [
        "one declarative manifest per organ: classes · tasks · checkpoints · "
        "thresholds",
        "16 organs mapped, each with dataset, licence and task list",
    ]),
    ("contract", 60, 652, 1580, 52,
     "Module contract — ingest → preprocess → classify → localize → "
     "segment → report", []),

    ("organ_brain", 60, 724, 186, 76, "Brain", ["cls + seg live"]),
    ("organ_lungs", 259, 724, 186, 76, "Lungs", ["cls live · seg TBD"]),
    ("organ_breast", 458, 724, 186, 76, "Breast", ["histo + mammo"]),
    ("organ_retina", 657, 724, 186, 76, "Retina", ["fundus + OCT"]),
    ("organ_blood", 856, 724, 186, 76, "Blood", ["microscopy"]),
    ("organ_skin", 1055, 724, 186, 76, "Skin", ["dermoscopy"]),
    ("organ_heart", 1254, 724, 186, 76, "Heart", ["echo + cine-MRI"]),
    ("organ_liver", 1453, 724, 186, 76, "Liver", ["CT + US"]),

    # ---- shared inference ---------------------------------------------
    ("bus", 60, 852, 1580, 44,
     "Every organ module calls the shared zoos below — no organ owns a model",
     []),
    ("clsfzoo", 60, 914, 390, 84, "Classifier zoo", [
        "one architecture per organ",
        "EfficientNetV2 · XRV · organ heads",
    ]),
    ("localizer", 470, 914, 350, 84, "Grad-CAM localiser", [
        "heat map → lesion box",
        "prompt for the segmenter",
    ]),
    ("segzoo", 840, 914, 390, 84, "Segmentation zoo", [
        "MedNeXt 2D · SegFormer-B0",
        "weights never shared across organs",
    ]),
    ("medsam", 1250, 914, 390, 84, "MedSAM-2 (frozen)", [
        "prompt encoder → mask decoder",
        "any organ, any modality",
    ]),
    ("calib", 60, 1018, 760, 56, "Calibration sidecars", [
        "temperature, thresholds, label space read at load",
    ]),
    ("runtime", 840, 1018, 800, 56, "Inference runtime", [
        "PyTorch CPU, threads pinned, zero GPU on the clinic machine",
    ]),

    # ---- data ---------------------------------------------------------
    ("ckpt", 60, 1136, 310, 76, "Checkpoints", [
        "per organ, per task",
        "metrics.json · train.log",
    ]),
    ("dset", 390, 1136, 330, 76, "Dataset registry", [
        "16 organs, counts, licences",
        "CC-0 / CC-BY only",
    ]),
    ("cache", 740, 1136, 280, 76, "Scan cache", [
        "sanitised uploads",
        "scan_id per case",
    ]),
    ("store", 1040, 1136, 310, 76, "Case & report store", [
        "drafts, signed reports",
        "versioned, immutable",
    ]),
    ("trail", 1370, 1136, 270, 76, "Audit trail", [
        "who read, edited, signed",
        "user · entity · time",
    ]),
]

CHIP_IDS = ["organ_brain", "organ_lungs", "organ_breast", "organ_retina",
            "organ_blood", "organ_skin", "organ_heart", "organ_liver"]

# arrows: from, to, label, style
#   v       straight down when the two boxes are aligned, otherwise an elbow
#   h       horizontal
#   elbow   down / across / down, crossing the gap between two bands
#   fan     straight down from the source's bottom edge at the target's x
#   lmargin up or down the left margin, then across
ARROWS = [
    ("selector", "rest", "scan + organ", "elbow"),
    ("workspace", "rest", "inference request", "elbow"),
    ("lab", "wss", "prompt coords", "elbow"),
    ("reportui", "upload", "PDF export", "elbow"),
    ("rest", "fastapi", "", "fan"),
    ("wss", "fastapi", "", "fan"),
    ("upload", "fastapi", "", "fan"),
    ("fastapi", "manager", "route to organ", "elbow"),
    ("manager", "registry", "", "h"),
    ("registry", "contract", "resolves module", "elbow"),
] + [(("contract"), c, "", "fan") for c in CHIP_IDS] + [
    (c, "bus", "", "fan") for c in CHIP_IDS
] + [
    ("bus", "clsfzoo", "", "fan"),
    ("bus", "localizer", "", "fan"),
    ("bus", "segzoo", "", "fan"),
    ("bus", "medsam", "", "fan"),
    ("ckpt", "manager", "weights", "lmargin"),
]

F_TITLE = 21
F_LINE = 17
F_BAND = 19
F_ARROW = 16


# --------------------------------------------------------------- geometry ---
def B(bid):
    for b in BLOCKS:
        if b[0] == bid:
            return b
    raise KeyError(bid)


def tx_width(text, size, bold=False):
    f = ImageFont.truetype(BOLD if bold else REG, size)
    return f.getlength(text)


def audit(verbose=True):
    bad = []
    for bid, x, y, w, h, title, lines in BLOCKS:
        if title:
            tw = tx_width(title, F_TITLE, True)
            if tw > w - 24:
                bad.append(f"{bid}: title overflows ({tw:.0f} > {w - 24})")
        for ln in lines:
            lw = tx_width(ln, F_LINE)
            if lw > w - 24:
                bad.append(f"{bid}: line overflows ({lw:.0f} > {w - 24}) "
                           f"'{ln[:28]}'")
        need = 30 + len(lines) * 23
        if need > h:
            bad.append(f"{bid}: text needs {need}px, block is {h}px")
    for i in range(len(BLOCKS)):
        for j in range(i + 1, len(BLOCKS)):
            a, b = BLOCKS[i], BLOCKS[j]
            if (a[1] < b[1] + b[3] and b[1] < a[1] + a[3]
                    and a[2] < b[2] + b[4] and b[2] < a[2] + a[4]):
                bad.append(f"{a[0]} overlaps {b[0]}")
    if verbose:
        for x in bad:
            print("  " + x)
    return bad


def arrow_geom(fr, to, style):
    ax, ay, aw, ah = B(fr)[1], B(fr)[2], B(fr)[3], B(fr)[4]
    bx, by, bw, bh = B(to)[1], B(to)[2], B(to)[3], B(to)[4]
    acx, bcx = ax + aw / 2, bx + bw / 2
    acy, bcy = ay + ah / 2, by + bh / 2

    if style == "fan":
        return [(bcx, ay + ah), (bcx, by)]

    if style == "h":
        if ax < bx:
            return [(ax + aw, acy), (bx, bcy)]
        return [(ax, acy), (bx + bw, bcy)]

    if style == "lmargin":
        mx = 48
        return [(ax, acy), (mx, acy), (mx, bcy), (bx, bcy)]

    if style in ("v", "elbow"):
        if abs(acx - bcx) < 6:
            if ay < by:
                return [(acx, ay + ah), (bcx, by)]
            return [(acx, ay), (bcx, by + bh)]
        mid = (ay + ah + by) / 2 if ay < by else (by + bh + ay) / 2
        if ay < by:
            return [(acx, ay + ah), (acx, mid), (bcx, mid), (bcx, by)]
        return [(acx, ay), (acx, mid), (bcx, mid), (bcx, by + bh)]

    raise ValueError(style)


def _seg_hits(p1, p2, b):
    x0, y0, x1, y1 = b[1] + 3, b[2] + 3, b[1] + b[3] - 3, b[2] + b[4] - 3
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
    return t0 < t1 - 0.02 and t1 > 0.05 and t0 < 0.95


def audit_arrows(verbose=True):
    bad = []
    for fr, to, _label, style in ARROWS:
        pts = arrow_geom(fr, to, style)
        for i in range(len(pts) - 1):
            for b in BLOCKS:
                if b[0] in (fr, to):
                    continue
                if _seg_hits(pts[i], pts[i + 1], b):
                    bad.append(f"arrow {fr}->{to} crosses {b[0]}")
    if verbose:
        for x in bad:
            print("  " + x)
    return bad


# ---------------------------------------------------------------- render ---
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(path):
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
         'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         '<path d="M 0 0 L 10 5 L 0 10 z" fill="#31434c"/></marker></defs>',
         '<rect width="100%" height="100%" fill="#ffffff"/>',
         f'<text x="{EDGE}" y="62" font-family="DejaVu Sans, sans-serif" '
         'font-size="34" font-weight="700" fill="#101820">OpenMed — '
         'System Architecture, multi-organ platform</text>']

    for bid, y0, y1, title, tint in BANDS:
        o.append(f'<rect x="40" y="{y0}" width="{RIGHT - 40}" '
                 f'height="{y1 - y0}" fill="{tint}" stroke="#c3d0d6" '
                 'stroke-width="1.2"/>')
        o.append(f'<text x="56" y="{y0 - 8}" font-family="DejaVu Sans, '
                 f'sans-serif" font-size="{F_BAND}" font-weight="700" '
                 f'fill="#4a5b63">{esc(title)}</text>')

    for fr, to, label, style in ARROWS:
        pts = arrow_geom(fr, to, style)
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        o.append(f'<path d="{d}" fill="none" stroke="#31434c" '
                 'stroke-width="1.7" marker-end="url(#ar)"/>')
        if label:
            p1, p2 = pts[0], pts[-1]
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            w = tx_width(label, F_ARROW)
            if abs(p1[0] - p2[0]) < abs(p1[1] - p2[1]):
                mx += 8
                my += 5
                anchor, rx = "start", mx - 3
            else:
                my -= 9
                anchor, rx = "middle", mx - w / 2 - 3
            o.append(f'<rect x="{rx:.1f}" y="{my - 14:.1f}" '
                     f'width="{w + 6:.1f}" height="19" fill="#ffffff" '
                     'opacity="0.92"/>')
            o.append(f'<text x="{mx:.1f}" y="{my:.1f}" '
                     f'text-anchor="{anchor}" font-family="DejaVu Sans, '
                     f'sans-serif" font-size="{F_ARROW}" fill="#3d4f58">'
                     f'{esc(label)}</text>')

    for bid, x, y, w, h, title, lines in BLOCKS:
        o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                 'fill="#ffffff" stroke="#22323b" stroke-width="1.8" rx="4"/>')
        o.append(f'<text x="{x + 12}" y="{y + 27}" font-family="DejaVu Sans, '
                 f'sans-serif" font-size="{F_TITLE}" font-weight="700" '
                 f'fill="#15242c">{esc(title)}</text>')
        yy = y + 27 + 23
        for ln in lines:
            o.append(f'<text x="{x + 12}" y="{yy}" font-family="DejaVu Sans, '
                     f'sans-serif" font-size="{F_LINE}" fill="#37474f">'
                     f'{esc(ln)}</text>')
            yy += 23

    o.append(f'<text x="{EDGE}" y="{H - 24}" font-family="DejaVu Sans, '
             'sans-serif" font-size="17" fill="#4a5b63">One clinic machine: '
             '4-core CPU, 8 GB RAM, no GPU. One architecture per organ, one '
             'contract for all of them. No cloud calls, no telemetry, patient '
             'pixels never leave the LAN.</text>')
    o.append('</svg>')
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(o))


if __name__ == "__main__":
    bad = audit() + audit_arrows()
    print(f"defects={len(bad)}")
    if bad:
        raise SystemExit("audit failed — figure not written")
    render("architecture_openmed.svg")
    print("wrote architecture_openmed.svg")
