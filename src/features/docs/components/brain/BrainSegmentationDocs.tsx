import React from 'react'

export const BrainSegmentationDocs: React.FC = () => {
  return (
    <div className="task-block segmentation">
      <div className="task-header-strip">
        <div className="task-tag-group">
          <span className="task-type-tag seg">Task · Segmentation</span>
          <span className="task-name-text">Promptable Segmentation (LiteMedSAM) · Heatmap-Prompt vs Raw Ablation</span>
        </div>
        <span className="task-model-pill">LiteMedSAM · TinyViT-256 · 256×256 input</span>
      </div>

      <div className="boilerplate-card">
        <p className="boilerplate-copy">
          Segmentation runs only when the classifier returns a tumour class — a Normal prediction short-circuits it in
          either prompt mode. Two modes exist at the API level. In <strong>heatmap box</strong> mode the classifier's{' '}
          <strong>Grad-CAM</strong> activation map is thresholded into a bounding box, and that box is the prompt handed
          to <strong>LiteMedSAM</strong> (TinyViT-256 image encoder + SAM prompt encoder + mask decoder) at a 256×256
          input resolution. In <strong>raw</strong> mode nothing is passed to the prompt encoder at all, so the mask
          decoder works from the image embedding alone — the unprompted arm of the ablation. In both modes the decoder
          returns a single binary mask, composited over the source scan in the coral accent, though unprompted it comes
          back empty on the released weights. A dense mask prompt is accepted by the prompt encoder but returns an empty
          prediction with those same weights, so it is never sent.
        </p>
        <p className="boilerplate-copy">
          <strong>The automated path is no longer wired into the workspace.</strong> It measured 0.3134 mean Dice
          against ground truth, because the CAM-derived box overlaps the lesion at only 0.123 IoU — the decoder was
          never the bottleneck, the prompt was. The Run button is therefore classification-only, and the prompt-mode
          toggle has been removed. Segmentation now lives in the{' '}
          <strong>Experimental Laboratory</strong>, where the clinician supplies the prompt and the same decoder reaches
          a median Dice of 0.917. The endpoints below are unchanged and still serve the automated path, ready for the
          retrained model. The engine is isolated at{' '}
          <code>backend/app/organs/brain/segmentation/experimental_lab/</code>.
        </p>
        <p className="boilerplate-copy">
          Two variants were trialled and <strong>reverted</strong>, and both are documented here because they are the
          kind of change that looks better on paper than on a scan. <strong>Grad-CAM++</strong> measured a tighter hot
          area (8.2% vs 10.2% of frame) but rendered a visibly fragmented map on real scans — scattered hot blobs rather
          than one coherent region over the lesion — so plain Grad-CAM remains the default
          (<code>OPENMED_CAM_METHOD=gradcam++</code> opts in). The <strong>adaptive percentile box</strong> measured a
          wash against the decoder's own IoU head but produced roughly 3× larger masks, and made the decoder segment well
          beyond the lesion; the fixed threshold and margin remain the default
          (<code>adaptive_box=True</code> opts in). A positive point at the peak activation is likewise
          <strong>off by default</strong>: it cost 0.051 mean IoU when sent unconditionally.
        </p>
        <div className="seg-targets-grid">
          <div className="seg-target-box">
            <span className="seg-badge wt">Heatmap Box Prompt · default</span>
            <span className="seg-formula">argwhere(Grad-CAM ≥ τ) ± margin</span>
            <p className="seg-desc">
              Bounding box in 256² model space, derived from the classifier's own attention map — no manual click or
              annotation required. The cut and the margin are fixed, which is the configuration this workspace was
              validated with.
            </p>
            <span className="seg-target-metric">
              Reported: prompt_mode=heatmap_box · box coordinates + decoder IoU estimate
            </span>
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
            <span className="seg-badge et">Raw · no prompt</span>
            <span className="seg-formula">points = ∅ · boxes = ∅ · masks = ∅</span>
            <p className="seg-desc">
              The unprompted arm: nothing reaches the prompt encoder, so the decoder segments from the image embedding
              alone. On the released weights this returns an <strong>empty mask</strong> — measured 0 foreground pixels
              versus 26,548 with the box prompt on the same scan. That is the finding: the box prompt is load-bearing,
              not an optimisation.
            </p>
            <span className="seg-target-metric">Reported: prompt_mode=raw · decoder IoU estimate (not meaningful when the mask is empty)</span>
          </div>
        </div>
      </div>
    </div>
  )
}
