#!/usr/bin/env python3
"""OpenMed SRS — Activity Diagram (Figure 5.6.4).

End-to-end case workflow: upload, validate, resolve the organ module, lazy-load
weights, classify, localise, segment, optionally refine through the MedSAM-2
prompt loop, then project, report, sign off and export.

Run: python3 activity_diagram.py
"""

from svgkit import Canvas, build

c = Canvas(
    1500, 2080,
    "OpenMed — Activity Diagram",
    subtitle="One case, ingestion to signed report, with the interactive "
             "refinement loop",
    footer="Every state change writes an audit entry. No step writes to a "
           "patient record without clinician sign-off.",
)

SPINE_X, SPINE_W = 100, 460

c.box("start", 285, 110, 90, 80, "", [], shape="ellipse")

c.box("upload", SPINE_X, 230, SPINE_W, 76, "1. Upload medical scan",
      ["DICOM, PNG, JPEG or NIfTI, single file or series"])
c.box("validate", SPINE_X, 340, SPINE_W, 76, "2. Validate and sanitise",
      ["format allow-list · strip EXIF and identifiers"])

c.box("d_format", 540, 430, 360, 96, "Format valid?", [], shape="diamond")
c.box("banner", 960, 430, 440, 76, "Reject with an amber banner",
      ["no inference runs · the user re-uploads"])

c.box("select", SPINE_X, 570, SPINE_W, 76, "3. Resolve the organ module",
      ["registry lookup by organ and modality"])
c.box("load", SPINE_X, 680, SPINE_W, 76, "4. Lazy-load organ weights",
      ["warmup call · any resident organ unloaded first"])
c.box("prep", SPINE_X, 790, SPINE_W, 76, "5. Preprocess to model spec",
      ["resize · normalise · channel order, per manifest"])
c.box("classify", SPINE_X, 900, SPINE_W, 76, "6. Classify",
      ["per-organ head · calibrated probabilities"])
c.box("cam", SPINE_X, 1010, SPINE_W, 76, "7. Localise",
      ["Grad-CAM heat map reduced to a lesion box"])
c.box("seg", SPINE_X, 1120, SPINE_W, 76, "8. Automated segmentation",
      ["organ-specific segmenter · mask per class"])

c.box("d_accept", 540, 1270, 360, 96, "Mask accepted?", [], shape="diamond")
c.box("refine", 960, 1400, 440, 76, "Refine with a prompt", [
    "point or box on the mask (MedSAM-2)",
    "mask refreshes over the WebSocket",
])

c.box("atlas", SPINE_X, 1500, SPINE_W, 76, "9. Project onto the 3D atlas",
      ["mask mapped to the organ mesh · orbit and explode"])
c.box("report", SPINE_X, 1610, SPINE_W, 76, "10. Draft the clinical report",
      ["findings · measurements · mandatory disclaimer"])
c.box("signoff", SPINE_X, 1720, SPINE_W, 76,
      "11. Clinician review and sign-off", [
          "edit, accept or discard before export"])
c.box("export", SPINE_X, 1830, SPINE_W, 76, "12. Export PDF, close the case",
      ["versioned record · audit entry written"])
c.box("end", 285, 1940, 90, 80, "", [], shape="ellipse")

c.arrow("start", "upload", "", "v")
c.arrow("upload", "validate", "", "v")
c.arrow("validate", "d_format", "", "elbow")
c.arrow("d_format", "banner", "no", "h")
c.arrow("banner", "upload", "re-upload", "ret", dashed=True)
c.arrow("d_format", "select", "yes", "elbow")
c.arrow("select", "load", "", "v")
c.arrow("load", "prep", "", "v")
c.arrow("prep", "classify", "", "v")
c.arrow("classify", "cam", "", "v")
c.arrow("cam", "seg", "", "v")
c.arrow("seg", "d_accept", "", "elbow")
c.arrow("d_accept", "refine", "no", "h")
c.arrow("refine", "d_accept", "re-check", "ret", dashed=True)
c.arrow("d_accept", "atlas", "yes", "elbow")
c.arrow("atlas", "report", "", "v")
c.arrow("report", "signoff", "", "v")
c.arrow("signoff", "export", "", "v")
c.arrow("export", "end", "", "v")

build(c, "activity_openmed.svg")
