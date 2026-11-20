#!/usr/bin/env python3
"""OpenMed SRS — Sequence Diagram (Figure 5.6.5).

Client-to-backend message order for one case: ingest, lazy organ load,
classification, automated segmentation, the interactive MedSAM-2 prompt loop,
and report export. Dashed arrows are returns.

Run: python3 sequence_diagram.py
"""

from svgkit import Canvas, build

c = Canvas(
    1500, 1560,
    "OpenMed — Sequence Diagram",
    subtitle="One case from upload to exported PDF, including the "
             "WebSocket refinement loop",
    footer="Only the selected organ is resident at any time; the manager "
           "unloads the previous organ before loading the next.",
)

c.band(240, 430, "1  ingest", "#f7f4ec")
c.band(450, 690, "2  resolve organ and lazy load", "#eef3f5")
c.band(710, 890, "3  classify", "#f7f4ec")
c.band(910, 1260, "4  segment, then refine interactively", "#eef3f5")
c.band(1280, 1400, "5  report", "#f7f4ec")

LL = [("browser", 160, "Browser client"),
      ("api", 400, "FastAPI"),
      ("mgr", 640, "Model manager"),
      ("organ", 880, "Organ module"),
      ("medsam", 1120, "MedSAM-2"),
      ("store", 1360, "Case store")]

for nid, x, title in LL:
    c.lifeline(nid, x, 200, 1450, title)

M = [
    (270, "browser", "api", "POST /api/cases (scan upload)", False),
    (320, "api", "store", "persist sanitised scan", False),
    (365, "store", "api", "scan_id", True),
    (410, "api", "browser", "202 Accepted, scan_id", True),

    (490, "browser", "api", "POST /organs/{organ}/warmup", False),
    (535, "api", "mgr", "ensure(organ)", False),
    (580, "mgr", "organ", "load checkpoint and label space", False),
    (625, "organ", "api", "ready, organ resident", True),
    (670, "api", "browser", "model-info, classes, calibration", True),

    (750, "browser", "api", "POST /organs/{organ}/classify", False),
    (795, "api", "organ", "classify(image)", False),
    (840, "organ", "api", "probabilities and Grad-CAM map", True),
    (875, "api", "browser", "ClassificationResponse", True),

    (950, "browser", "api", "POST /segment (automated)", False),
    (995, "api", "organ", "segment(volume)", False),
    (1040, "organ", "api", "mask per class", True),
    (1090, "browser", "api", "WS point or box prompt", False),
    (1135, "api", "medsam", "encode prompt, decode mask", False),
    (1180, "medsam", "api", "refined mask", True),
    (1225, "api", "browser", "mask update over the socket", True),

    (1300, "browser", "api", "POST /reports (accept draft)", False),
    (1345, "api", "store", "store draft, version it", False),
    (1385, "api", "browser", "PDF under the 2 s budget", True),
]

for y, fr, to, label, dashed in M:
    c.msg(fr, to, y, label, dashed=dashed)

build(c, "sequence_openmed.svg")
