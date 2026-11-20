#!/usr/bin/env python3
"""OpenMed SRS — Data Flow Diagram (Figure 5.6.9).

Two panels: the Level 0 context diagram, and the Level 1 decomposition of the
same system into the four processes the SRS names — ingest, classify, segment,
report — over the four stores the audit trail needs.

Notation: rectangle = external entity, circle = process, open-ended pair of
lines = data store, labelled arrow = data flow.

Run: python3 dfd_diagram.py
Out: dfd_openmed.svg
"""

from svgkit import Canvas, build

W, H = 1500, 1120
XS = [60, 440, 820, 1200]
PW = 240

c = Canvas(
    W, H,
    "OpenMed — Data Flow Diagram",
    subtitle="Level 0 context above, Level 1 decomposition below. The "
             "image stream is the through-line",
    footer="D3 (audit log) is appended by every process, 1.0 to 4.0; drawing "
           "those four flows would bury the figure, so its inbound flows are "
           "stated here instead. D2 holds one checkpoint and one label-space "
           "sidecar per organ and task.",
)

# ---------------------------------------------------------------- level 0 ---
c.band(150, 440, "Level 0 — the platform as one process")

c.box("e1", 60, 245, PW, 90, "Clinician",
      ["orders the study", "signs the report"])
p0 = c.box("p0", 450, 190, 600, 200, "0 · OpenMed platform",
           ["classification, segmentation, report,", "3D view, clinical Q&A"],
           shape="ellipse")
c.box("e2", 1200, 245, PW, 90, "Hospital records",
      ["signed reports only", "PACS archive"])

c.edge([(300, 290), (450, 290)], "study + metadata", side="up")
c.edge([(475, 330), (302, 330)], "findings, masks, draft", side="up")
c.edge([(1050, 290), (1198, 290)], "signed report", side="up")

c.note(60, 415,
       "Sixteen organ modules sit inside process 0. At this level they are one "
       "process because the clinician sees one workstation.", size=16)

# ---------------------------------------------------------------- level 1 ---
c.band(470, 1010, "Level 1 — ingest, classify, segment, report",
       tint="#f7f4ec")

c.box("e1b", 60, 500, PW, 80, "Clinician", ["uploads and reviews"])
c.box("e2b", 1200, 500, PW, 80, "Hospital records", ["receives signed PDF"])

p1 = c.box("p1", XS[0], 660, PW, 130, "1.0 Ingest",
           ["validate, normalise"], shape="ellipse")
p2 = c.box("p2", XS[1], 660, PW, 130, "2.0 Classify",
           ["GLN + XRV"], shape="ellipse")
p3 = c.box("p3", XS[2], 660, PW, 130, "3.0 Segment",
           ["auto + MedSAM-2"], shape="ellipse")
p4 = c.box("p4", XS[3], 660, PW, 130, "4.0 Report",
           ["draft, sign, export"], shape="ellipse")

c.box("d1", XS[0], 860, PW, 76, "D1", ["scan + sha256"], shape="store")
c.box("d2", XS[1], 860, PW, 76, "D2", ["checkpoints"], shape="store")
c.box("d3", XS[2], 860, PW, 76, "D3", ["audit log"], shape="store")
c.box("d4", XS[3], 860, PW, 76, "D4", ["report versions"], shape="store")

c.edge([(180, 580), (180, 658)], "study upload", side="r")
c.edge([(180, 792), (180, 858)], "scan + sha256", side="r")
c.edge([(300, 725), (438, 725)], "normalised volume", side="up")
c.edge([(610, 858), (610, 792)], "checkpoint + T", side="r")
c.edge([(680, 725), (818, 725)], "class + CAM box", side="up")
c.edge([(1060, 725), (1198, 725)], "mask, area, volume", side="up")
c.edge([(1340, 792), (1340, 858)], "draft report", side="r")
c.edge([(1340, 658), (1340, 582)], "signed PDF", side="l")

build(c, "dfd_openmed.svg")
