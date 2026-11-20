#!/usr/bin/env python3
"""OpenMed SRS — Class Diagram (Figure 5.6.8).

Real names only. Every box below exists in the backend:

  app/organs/brain/classification/pipeline.py   BrainClassificationPipeline
  app/organs/lungs/classification/pipeline.py   DualLungPipeline / V2SPipeline /
                                                XRVPipeline
  app/organs/brain/segmentation/mednext/        MedNeXtSegmenter
  app/organs/brain/classification/label_space.py LabelSpace
  app/organs/brain/classification/router.py     ClassificationResponse /
                                                Top5Prediction
  app/organs/brain/segmentation/mednext/router.py AnalyzeResponse
  app/experimental/base.py                      LabAdapter, PointPrompt, SegPrompt
  app/experimental/medsam2/adapter.py           MedSAM2Adapter
  app/organs/*/classification/model.py          GradCAM

Run: python3 class_diagram.py
Out: class_openmed.svg
"""

from svgkit import Canvas, build

W, H = 1560, 1080
XS = [60, 430, 800, 1170]
BW = 330
ROWS = [150, 470, 790]

c = Canvas(
    W, H,
    "OpenMed — Class Diagram",
    subtitle="Backend pipelines, adapters and the response schemas the "
             "routers return",
    footer="Training-only modules (datasets, split planners, augmentation "
           "policy) are deliberately absent: they never ship inside the "
           "inference service.",
)

F_T, F_L = 16, 15


def uml(nid, col, row, title, attrs, methods, fl=F_L):
    h = max(96, 74 + 18 * (len(attrs) + len(methods)))
    return c.box(nid, XS[col], ROWS[row], BW, h, title, shape="uml",
                 attrs=attrs, methods=methods, f_title=F_T, f_line=fl)


def bottom(n):
    return n["y"] + n["h"]


def mid(n):
    return n["y"] + n["h"] / 2


# ------------------------------------------------------------------ row 1 ---
bcp = uml("bcp", 0, 0, "BrainClassificationPipeline",
          ["checkpoint_path: Path", "label_space: LabelSpace", "device: str"],
          ["load()", "predict(image)", "predict_with_gradcam(image)"])

v2s = uml("v2s", 1, 0, "V2SPipeline",
          ["checkpoint: Path", "temperature: float", "classes: list[str]"],
          ["load()", "predict(image)", "unload()"])

dlp = uml("dlp", 2, 0, "DualLungPipeline",
          ["v2s: V2SPipeline", "xrv: XRVPipeline"],
          ["load(which)", "classify(image)", "health()"])

xrv = uml("xrv", 3, 0, "XRVPipeline",
          ["cache_dir: Path", "weights: 'densenet121-nih'"],
          ["load()", "predict(image)", "unload()"])

# ------------------------------------------------------------------ row 2 ---
mednext = uml("mednext", 0, 1, "MedNeXtSegmenter",
              ["checkpoint_path: Path", "spec: MedNeXtSpec"],
              ["load()", "segment(volume)", "unload()"])

lspace = uml("lspace", 1, 1, "LabelSpace",
             ["classes: tuple[str, ...]", "temperature: float",
              "checkpoint_sha: str"],
             ["from_sidecar(path)", "calibrate(logits)"])

cam = uml("cam", 2, 1, "GradCAM",
          ["target_layer: nn.Module"],
          ["generate(tensor)", "overlay(image, cam)"])

lab = uml("lab", 3, 1, "LabAdapter",
          ["id: str  «abstract»", "title: str", "is_loaded: bool"],
          ["load()", "unload()", "predict(prompt)", "info()"])

# ------------------------------------------------------------------ row 3 ---
cres = uml("cres", 0, 2, "ClassificationResponse",
           ["predicted_class: str", "confidence: float",
            "top5: list[Top5Prediction]", "gradcam_ref: str | None"],
           ["model_dump()"])

ares = uml("ares", 1, 2, "AnalyzeResponse",
           ["segmentation: SegmentationResult | None"],
           ["model_dump()"], fl=14)

msam = uml("msam", 2, 2, "MedSAM2Adapter",
           ["model_id = 'medsam2'", "weights_path: Path", "device: str"],
           ["load()", "predict(PointPrompt | SegPrompt)", "unload()"])

# ---------------------------------------------------------------- relations ---
# classify, then segment the same volume
c.edge([(160, bottom(bcp)), (160, ROWS[1])],
       "segment after classify", dashed=True, side="r")

# the pipeline refuses to load a checkpoint whose sidecar names another space
c.edge([(300, bottom(bcp)), (300, 410), (595, 410), (595, ROWS[1])],
       "reads", side="up")

# DualLungPipeline owns both wrappers
c.edge([(XS[1] + BW, 300), (XS[2], 300)], head="dia")
c.edge([(XS[3], 300), (XS[2] + BW, 300)], head="dia")

# the adapter is the lab's contract
c.edge([(965, ROWS[2]), (965, 700), (1300, 700), (1300, bottom(lab))],
       "implements", side="up")

# one response extends the other
c.edge([(XS[1], mid(ares)), (XS[0] + BW, mid(cres))], head="tri")

# GradCAM is built per request against whichever classifier ran
c.edge([(XS[2], mid(cam)), (780, mid(cam)), (780, 130), (225, 130),
        (225, ROWS[0])],
       "target layer", dashed=True, side="up")

c.note(60, 1010,
       "Diamond is composition: DualLungPipeline constructs V2SPipeline and "
       "XRVPipeline and drops both on unload().", size=16)
c.note(60, 1036,
       "Hollow triangle is generalization: AnalyzeResponse is a "
       "ClassificationResponse plus one segmentation field; MedSAM2Adapter "
       "satisfies LabAdapter.", size=16)

build(c, "class_openmed.svg")
