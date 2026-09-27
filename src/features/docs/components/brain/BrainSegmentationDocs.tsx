import React from 'react'

export const BrainSegmentationDocs: React.FC = () => {
  return (
    <div className="task-block segmentation">
      <div className="task-header-strip">
        <div className="task-tag-group">
          <span className="task-type-tag seg">Task · Segmentation</span>
          <span className="task-name-text">Grad-CAM-Prompted Promptable Segmentation (LiteMedSAM)</span>
        </div>
        <span className="task-model-pill">LiteMedSAM · TinyViT-256 · 256×256 input</span>
      </div>

      <div className="boilerplate-card">
        <p className="boilerplate-copy">
          Segmentation runs only when the classifier returns a tumour class — a Normal prediction short-circuits it. The
          Grad-CAM activation map from that same forward pass is thresholded into a bounding box, and that box is the
          prompt handed to <strong>LiteMedSAM</strong> (TinyViT-256 image encoder + SAM prompt encoder + mask decoder) at a
          256×256 input resolution. The decoder returns a single binary tumour mask, which is composited over the source
          scan in the coral accent.
        </p>
        <div className="seg-targets-grid">
          <div className="seg-target-box">
            <span className="seg-badge wt">Box Prompt</span>
            <span className="seg-formula">argwhere(Grad-CAM ≥ τ) ± margin</span>
            <p className="seg-desc">
              Bounding box in 256² model space, derived from the classifier's own attention map — no manual click or
              annotation required.
            </p>
            <span className="seg-target-metric">Reported: box coordinates + decoder IoU estimate</span>
          </div>
          <div className="seg-target-box">
            <span className="seg-badge tc">Binary Tumour Mask</span>
            <span className="seg-formula">p(logit &gt; 0)</span>
            <p className="seg-desc">
              A single foreground/background mask, upscaled from the 64² low-resolution logits and cropped back to the
              original scan geometry.
            </p>
            <span className="seg-target-metric">Delivered as PNG + coral overlay, capped at 384px</span>
          </div>
          <div className="seg-target-box">
            <span className="seg-badge et">Prompt Constraint</span>
            <span className="seg-formula">boxes = … · masks = ∅</span>
            <p className="seg-desc">
              The released lite_medsam.pth weights were trained to segment from boxes. A dense mask prompt is accepted by
              the prompt encoder but returns an empty prediction, so the pipeline never sends one.
            </p>
            <span className="seg-target-metric">Model-reported: prompt_types = ["bounding_box"]</span>
          </div>
        </div>
      </div>
    </div>
  )
}
