#!/usr/bin/env python3
"""OpenMed SRS — Component Diagram (Figure 5.6.6).

Tier-by-tier component view: presentation components, the FastAPI service
components, one domain module per organ, the model adapters every organ shares,
and the persistence components. Run: python3 component_diagram.py
"""

from svgkit import Canvas, build

c = Canvas(
    1500, 1020,
    "OpenMed — Component Diagram",
    subtitle="Five tiers; the domain tier instantiates one module per registered organ",
    footer="Adapters are shared code; weights and configuration are not. Each organ module owns "
           "its own checkpoint and manifest, and the calibration sidecar reads its temperature "
           "from the manifest beside that checkpoint.",
)

c.band(130, 262, "Presentation tier — React 19 and TypeScript, browser only")
c.band(292, 424, "Service tier — FastAPI application", "#f3efe6")
c.band(454, 586, "Domain tier — one module per registered organ")
c.band(616, 748, "Adapter tier — shared model plumbing", "#f3efe6")
c.band(778, 910, "Persistence tier — local disk, nothing remote")

XS = [60, 344, 628, 912, 1196]
W = 264

def row(y, items):
    for x, it in zip(XS, items):
        size = it[3] if len(it) > 3 else 20
        c.box(it[0], x, y, W, 84, it[1], it[2], f_title=size)

row(160, [
    ("ui_sel", "Organ selector", ["reads the registry"]),
    ("ui_ws", "Diagnostic workspace", ["slice view + overlay"], 18),
    ("ui_lab", "Prompt lab", ["point and box prompts"]),
    ("ui_atlas", "3D atlas viewer", ["Three.js canvas"]),
    ("ui_rep", "Report editor", ["draft and sign-off"]),
])

row(322, [
    ("sv_case", "Case router", ["upload, sanitise"]),
    ("sv_organ", "Organ router", ["classify, segment"]),
    ("sv_report", "Report router", ["draft, export"]),
    ("sv_ws", "WebSocket gateway", ["prompt stream"]),
    ("sv_cors", "CORS middleware", ["origin allow-list"]),
])

row(484, [
    ("dm_brain", "Brain module", ["classification + MedNeXt"]),
    ("dm_lungs", "Lungs module", ["classification, seg in build"]),
    ("dm_breast", "Breast module", ["histology + mammography"]),
    ("dm_more", "13 more modules", ["same contract, own weights"]),
    ("dm_reg", "Organ registry", ["manifests, availability"]),
])

row(646, [
    ("ad_cls", "Classifier wrapper", ["transform + forward"]),
    ("ad_seg", "Segmenter wrapper", ["volume to mask"]),
    ("ad_cam", "Grad-CAM", ["heat map to box"]),
    ("ad_sam", "MedSAM-2 adapter", ["prompt encode, decode"]),
    ("ad_cal", "Calibration sidecar", ["temperature, ECE"]),
])

c.box("ps_ckpt", 60, 808, 335, 84, "Checkpoint store", ["per organ, per task"])
c.box("ps_case", 410, 808, 335, 84, "Case store", ["scans, drafts, signed PDFs"])
c.box("ps_audit", 760, 808, 335, 84, "Audit log", ["append-only entries"])
c.box("ps_dset", 1110, 808, 335, 84, "Dataset registry", ["counts, licences, provenance"])

c.arrow("ui_ws", "sv_case", "upload", "elbow")
c.arrow("ui_lab", "sv_ws", "prompt", "elbow")
c.arrow("ui_rep", "sv_report", "export", "elbow")
c.arrow("sv_organ", "dm_reg", "resolve", "elbow")
c.arrow("dm_brain", "ad_cls", "uses", "elbow")
c.arrow("dm_brain", "ad_seg", "", "elbow")
c.arrow("dm_reg", "ps_dset", "provenance", "rmargin")
c.arrow("ad_sam", "ps_ckpt", "", "elbow")

build(c, "component_openmed.svg")
