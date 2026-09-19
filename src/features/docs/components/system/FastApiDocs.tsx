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
            and run Grad-CAM convolutional attention generation on demand without unbounded memory retention.
          </p>

          <div className="api-endpoints-grid">
            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method post">POST</span>
                <code className="endpoint-path">/api/brain/classify</code>
              </div>
              <p className="endpoint-desc">
                Accepts multi-part image upload (JPEG/PNG). Preprocesses scan through resolution transforms, computes 39-class logits,
                extracts top-5 differential rankings, maps 38 stereotactic MNI regions, and renders base64 Grad-CAM activation heatmap.
              </p>
            </div>

            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method get">GET</span>
                <code className="endpoint-path">/api/brain/health</code>
              </div>
              <p className="endpoint-desc">
                Health monitor reporting server state, memory consumption, GPU device availability, and whether the checkpoint weights
                are loaded into VRAM.
              </p>
            </div>

            <div className="api-endpoint-card">
              <div className="endpoint-head">
                <span className="http-method get">GET</span>
                <code className="endpoint-path">/api/brain/model-info</code>
              </div>
              <p className="endpoint-desc">
                Returns static metadata: architecture name (EfficientNetV2-B2), class dictionary (39 labels), input resolution (512×512),
                and training dataset provenance.
              </p>
            </div>
          </div>
        </div>
      </RomanSection>
    </section>
  )
}
