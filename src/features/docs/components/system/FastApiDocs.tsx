import React from 'react'
import { Server } from 'lucide-react'
import { RomanSection } from '../../../../components/common/RomanSection'

export const FastApiDocs: React.FC = () => {
  return (
    <section id="section-api-infra" className="doc-section-wrapper">
      <RomanSection
        index={2}
        of={5}
        title="Infrastructura - Production FastAPI Inference Service"
        className="sp12"
      >
        <div className="doc-info-block">
          <div className="doc-block-header">
            <Server size={15} className="doc-block-icon" />
            <span className="doc-block-title">Backend Architecture &amp; Live REST API</span>
          </div>
          <p className="doc-block-copy">
            The inference backend runs on an asynchronous FastAPI server engineered to lazily instantiate model weights
            and run Grad-CAM convolutional attention generation on demand without unbounded memory retention. Both models
            are held as process-wide singletons behind an asyncio lock; inference runs on a threadpool so the event loop
            is never blocked. The client calls <code>/load-models</code> on entering the brain workspace and
            <code> /unload-models</code> on leaving, and every route below reports the device it actually resolved to —
            CUDA when available, otherwise CPU.
          </p>

          <div className="api-endpoints-grid">
            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method post">POST</span>
                <code className="endpoint-path">/api/brain/segment</code>
              </div>
              <p className="endpoint-desc">
                The end-to-end route the workspace uses. Accepts a multipart image upload (JPEG/PNG/WebP, ≤12 MB) plus a{' '}
                <code>use_heatmap_prompt</code> flag (default true). Classifies the scan, and — unless the prediction is
                Normal — runs LiteMedSAM: with the flag set, a bounding box is derived from the Grad-CAM heatmap and used
                as the prompt; with it cleared, no prompt is sent at all and the decoder runs from the image embedding
                alone. Returns the full classification payload plus the binary mask, the coral overlay, the{' '}
                <code>prompt_mode</code> that produced it, box coordinates and decoder IoU in heatmap mode, and per-stage
                timings.
              </p>
            </div>

            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method post">POST</span>
                <code className="endpoint-path">/api/brain/segment-click</code>
              </div>
              <p className="endpoint-desc">
                The assistive path. Takes the same multipart upload plus{' '}
                <code>click_x</code> / <code>click_y</code> (normalised 0–1) and an
                optional <code>box_size</code> (default 48px in 256² space). The
                click becomes a small box prompt and LiteMedSAM segments inside it
                — <strong>no classification runs</strong>, so the click responds
                without a classifier pass. Measured against ground truth on the
                held-out split: median Dice 0.917 at the default box size, against
                0.235 for the CAM-derived box. Because the released weights were
                trained on boxes, the click becomes a box rather than a bare point
                — a point alone scored 0.254.
              </p>
            </div>

            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method post">POST</span>
                <code className="endpoint-path">/api/brain/segment-box</code>
              </div>
              <p className="endpoint-desc">
                The same assistive fast path, but the clinician drags a box around
                the lesion rather than clicking. Takes <code>x1</code> /{' '}
                <code>y1</code> / <code>x2</code> / <code>y2</code> normalised
                0–1 in either corner order; corners are ordered and clamped to the
                frame, and a box under 16px in 256² space is refused with a 400.
                This is the prompt type the released weights were trained on, so a
                well-drawn box is the strongest prompt available. No
                classification runs.
              </p>
            </div>

            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method post">POST</span>
                <code className="endpoint-path">/api/brain/classify</code>
              </div>
              <p className="endpoint-desc">
                Classification only. Preprocesses the scan to the model's native 208×208 input, computes 9-class logits,
                extracts the top-5 differential ranking, maps the predicted WHO family onto 50 stereotactic MNI regions,
                and renders a base64 Grad-CAM activation overlay.
              </p>
            </div>

            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method get">GET</span>
                <code className="endpoint-path">/api/brain/health</code>
              </div>
              <p className="endpoint-desc">
                Classifier readiness probe. Reports whether the checkpoint is present on disk, whether the network is
                resident in memory, whether trained weights (rather than an untrained fallback) were loaded, and a
                human-readable detail string when something is missing.
              </p>
            </div>

            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method get">GET</span>
                <code className="endpoint-path">/api/brain/model-info</code>
              </div>
              <p className="endpoint-desc">
                Classifier metadata: architecture (EfficientNetV2-B2), the resolved timm tag, the 9-class name list,
                input resolution, which label-space source was used, and the recorded training metrics — validation and
                test accuracy, macro one-vs-rest AUC, and the train/val/test split sizes.
              </p>
            </div>

            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method post">POST</span>
                <code className="endpoint-path">/api/brain/load-models</code>
              </div>
              <p className="endpoint-desc">
                Eagerly instantiates the classifier and the segmenter into memory and returns the wall-clock load time.
                Called when the user enters the brain workspace so the first inference is not paying for cold weights.
              </p>
            </div>

            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method post">POST</span>
                <code className="endpoint-path">/api/brain/unload-models</code>
              </div>
              <p className="endpoint-desc">
                Releases both models, drops the singleton references, forces a garbage collection and empties the CUDA
                cache when a GPU is present. Called when the user leaves the brain workspace.
              </p>
            </div>

            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method get">GET</span>
                <code className="endpoint-path">/api/brain/seg-health</code>
              </div>
              <p className="endpoint-desc">
                Segmentation readiness probe. Reports both checkpoints' presence and both models' load state, and names
                the exact preparation command when the LiteMedSAM checkpoint is missing.
              </p>
            </div>

            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method get">GET</span>
                <code className="endpoint-path">/api/brain/seg-model-info</code>
              </div>
              <p className="endpoint-desc">
                Segmentation metadata: LiteMedSAM architecture (TinyViT-256 + SAM prompt encoder + mask decoder), its
                256×256 input resolution, the resolved device, and its declared prompt types — currently
                <code> ["bounding_box"]</code>, because the released weights segment from boxes only.
              </p>
            </div>
          </div>
        </div>
      </RomanSection>
    </section>
  )
}
