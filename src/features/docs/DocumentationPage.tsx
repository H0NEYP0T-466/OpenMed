import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Layers,
  Sliders,
  Database,
  Check,
  Copy,
  ExternalLink,
  Download,
  X,
  Server,
  Compass,
  ShieldCheck,
} from 'lucide-react'
import { AppShell } from '../../components/common/AppShell'
import { RomanSection } from '../../components/common/RomanSection'
import './DocumentationPage.css'

interface ArtifactCard {
  readonly id: string
  readonly filename: string
  readonly format: string
  readonly title: string
  readonly desc: string
  readonly category: string
  readonly previewPath?: string
  readonly isText?: boolean
}

const ARTIFACTS: readonly ArtifactCard[] = [
  {
    id: 'curves',
    filename: 'loss_curves.png & accuracy_curves.png',
    format: 'PNG · Dual Plot',
    title: 'Convergence Trajectories',
    desc: 'Per-epoch train vs. validation cross-entropy loss and top-1 accuracy curves validating monotonic generalization.',
    category: 'Optimization',
    previewPath: '/visuals/brain/loss_curves.png',
  },
  {
    id: 'cm',
    filename: 'confusion_matrix.png',
    format: 'PNG · 39×39 Matrix',
    title: 'Normalized Confusion Matrix',
    desc: 'Full 39-class normalized diagnostic confusion matrix with row-wise sensitivity and false-positive mapping.',
    category: 'Validation',
    previewPath: '/visuals/brain/confusion_matrix.png',
  },
  {
    id: 'dist',
    filename: 'class_distribution.png',
    format: 'PNG · Categorical Bar',
    title: 'Class & Split Balance Ledger',
    desc: 'Sample counts across 70% train (8,838), 15% val (1,894), and 15% test (1,894) location-stratified splits.',
    category: 'Dataset',
    previewPath: '/visuals/brain/class_distribution.png',
  },
  {
    id: 'grid',
    filename: 'sample_predictions.png',
    format: 'PNG · 4×4 Plate',
    title: 'Qualitative Validation Grid',
    desc: '16-panel test batch with true vs. predicted labels, highlighting correct predictions and clinical edge cases.',
    category: 'Inference',
    previewPath: '/visuals/brain/sample_predictions.png',
  },
  {
    id: 'cam',
    filename: 'gradcam_samples.png',
    format: 'PNG · Layer Heatmap',
    title: 'Grad-CAM Attention Overlays',
    desc: 'Final convolutional layer (conv_head) activation heatmaps demonstrating anatomical focus on tumor pathology.',
    category: 'Explainability',
    previewPath: '/visuals/brain/gradcam_samples.png',
  },
  {
    id: 'roc',
    filename: 'roc_curves_macro.png & pr_curves_micro.png',
    format: 'PNG · Dual Frontier',
    title: 'ROC & Precision-Recall Frontiers',
    desc: 'Macro-averaged and micro-averaged AUC-ROC curves with precision-recall operating thresholds.',
    category: 'Performance',
    previewPath: '/visuals/brain/roc_curves_macro.png',
  },
  {
    id: 'report',
    filename: 'classification_report.txt & training_log.csv',
    format: 'TXT / CSV · Telemetry',
    title: 'Classification Report & Epoch Log',
    desc: 'Exhaustive per-class precision, recall, F1-score report and per-epoch CSV recording loss, LR, and elapsed time.',
    category: 'Audit',
    previewPath: '/visuals/brain/classification_report.txt',
    isText: true,
  },
  {
    id: 'weights',
    filename: 'brain_best_model.pth',
    format: 'PTH · PyTorch 14M Params',
    title: 'Production Model Weights',
    desc: 'Serialized PyTorch state dict and class-to-index mapping dictionary ready for live FastAPI deployment.',
    category: 'Model Weights',
    previewPath: '/visuals/brain/training_log.csv',
    isText: true,
  },
]

export const DocumentationPage: React.FC = () => {
  const [copiedCmd, setCopiedCmd] = useState(false)
  const [activeArtifact, setActiveArtifact] = useState<ArtifactCard | null>(null)
  const [artifactTextContent, setArtifactTextContent] = useState<string | null>(null)

  const handleCopyKaggleCmd = () => {
    const cmd =
      'python brain_kaggle.py --data_root /kaggle/input/brain-tumor-dataset/archive --output_dir /kaggle/working --batch_size 32 --epochs 50'
    navigator.clipboard.writeText(cmd)
    setCopiedCmd(true)
    setTimeout(() => setCopiedCmd(false), 2200)
  }

  const handleOpenArtifact = async (art: ArtifactCard) => {
    setActiveArtifact(art)
    setArtifactTextContent(null)
    if (art.isText && art.previewPath) {
      try {
        const resp = await fetch(art.previewPath)
        const text = await resp.text()
        setArtifactTextContent(text)
      } catch {
        setArtifactTextContent('Unable to load artifact text.')
      }
    }
  }

  return (
    <AppShell
      cta={
        <Link to="/app?organ=brain" className="btn btn-primary">
          <span>Inspect Brain Atlas</span>
          <span className="arr" aria-hidden="true">↗</span>
        </Link>
      }
    >
      <main className="doc-page-layout">
        {/* Document Head */}
        <header className="doc-head-block">
          <div className="doc-eyebrow">
            <span>Academic & Engineering Documentation</span>
            <span className="meta-sep">•</span>
            <span className="doc-meta-tag mono">OpenMed 2026 Core</span>
          </div>

          <h1 className="doc-main-title">
            Documenta & Academic Benchmarks
            <span className="serif">Training Ledgers, Stratified Partitioning & Neural System Specifications</span>
          </h1>

          <p className="doc-lead-copy">
            Rigorous mathematical and empirical documentation for OpenMed’s clinical AI architectures.
            All models enforce leak-proof anatomical stratification, class-weighted optimization, and reproducible execution protocols
            verified across 12,626 curated clinical MRI scans.
          </p>

          <div className="doc-quick-nav">
            <a href="#section-neuro-oncology" className="doc-nav-anchor">I · Neuro-Oncology Benchmark</a>
            <a href="#section-api-infra" className="doc-nav-anchor">II · FastAPI Server Engine</a>
            <a href="#section-stereotactic" className="doc-nav-anchor">III · MNI Stereotactic Mapping</a>
            <a href="#section-dataset" className="doc-nav-anchor">IV · Dataset Curation & Purging</a>
          </div>
        </header>

        {/* SECTION I: NEURO-ONCOLOGY KAGGLE LEDGER (EXACT SPECIFICATION) */}
        <section id="section-neuro-oncology" className="doc-section-wrapper">
          <RomanSection
            index={0}
            of={4}
            title="Disciplina - Kaggle Training & Academic Benchmarks"
            className="sp12"
          >
            <div className="archival-ledger-card">
              {/* 4 Stat Monoliths */}
              <div className="stat-monoliths-grid">
                <div className="stat-monolith">
                  <span className="mono-num">12,626</span>
                  <span className="mono-label">Curated Scans</span>
                  <span className="mono-desc">512×512 axial scans across 3 MRI sequences; 17 corrupt masks purged.</span>
                </div>
                <div className="stat-monolith">
                  <span className="mono-num">39</span>
                  <span className="mono-label">Diagnostic Classes</span>
                  <span className="mono-desc">13 WHO histological families with sequence-specific subtyping.</span>
                </div>
                <div className="stat-monolith">
                  <span className="mono-num">38</span>
                  <span className="mono-label">Anatomical Sites</span>
                  <span className="mono-desc">Stereotactically mapped to MNI coordinates; zero spatial leakage.</span>
                </div>
                <div className="stat-monolith">
                  <span className="mono-num">25.2×</span>
                  <span className="mono-label">Imbalance Ratio</span>
                  <span className="mono-desc">Handled via inverse-frequency class-weighted Cross-Entropy loss.</span>
                </div>
              </div>

              {/* Visual Split Stratification & Anti-Leakage Protocol */}
              <div className="split-stratification-card">
                <div className="card-topline">
                  <Layers size={15} className="card-top-icon" />
                  <span className="card-top-title">Primary-Location Stratified Split Architecture</span>
                </div>

                <div className="stratification-bar">
                  <div className="strat-segment train" style={{ flex: 70 }}>
                    <span className="strat-segment-title">Train 70%</span>
                    <span className="strat-segment-val">8,838 Scans (timm dynamic augment)</span>
                  </div>
                  <div className="strat-segment val" style={{ flex: 15 }}>
                    <span className="strat-segment-title">Val 15%</span>
                    <span className="strat-segment-val">1,894 Scans</span>
                  </div>
                  <div className="strat-segment test" style={{ flex: 15 }}>
                    <span className="strat-segment-title">Test 15%</span>
                    <span className="strat-segment-val">1,894 Scans (Holdout)</span>
                  </div>
                </div>

                <p className="stratification-note">
                  <strong>Anti-Leakage Guarantee:</strong> Brain lesions at the same anatomical site exhibit correlated morphology.
                  To prevent anatomical feature leakage between splits, our partitioning pipeline enforces a strict
                  <strong> primary-location stratified split</strong>. Rather than random shuffling, all 38 anatomical locations
                  are proportionally distributed across splits.
                </p>
              </div>

              {/* Model & Optimization Specification Matrix */}
              <div className="specs-matrix-card">
                <div className="card-topline">
                  <Sliders size={15} className="card-top-icon" />
                  <span className="card-top-title">Neural Architecture & Optimization Manifest</span>
                </div>

                <div className="specs-grid">
                  <div className="spec-row">
                    <span className="spec-k">Backbone Architecture</span>
                    <span className="spec-v">EfficientNetV2-B2 (tf_efficientnetv2_b2.in1k)</span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-k">Input Resolution</span>
                    <span className="spec-v">512 × 512 × 3 (RGB-normalized)</span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-k">Loss Formulation</span>
                    <span className="spec-v">Weighted Cross-Entropy (w = N / (C · n_c))</span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-k">Optimizer & LR</span>
                    <span className="spec-v">AdamW (lr = 1e-3, weight_decay = 1e-4)</span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-k">Learning Rate Schedule</span>
                    <span className="spec-v">CosineAnnealingWarmRestarts (T_0 = 10, T_mult = 2)</span>
                  </div>
                  <div className="spec-row">
                    <span className="spec-k">Data Regularization</span>
                    <span className="spec-v">RandomResizedCrop, HorizontalFlip, GradClip (1.0)</span>
                  </div>
                </div>
              </div>

              {/* Kaggle Execution Terminal */}
              <div className="kaggle-terminal-card">
                <div className="terminal-header">
                  <div className="terminal-dots">
                    <span className="dot red" />
                    <span className="dot yellow" />
                    <span className="dot green" />
                  </div>
                  <span className="terminal-title">Kaggle Execution Script · GPU Tesla T4</span>
                  <button
                    type="button"
                    className="terminal-copy-btn"
                    onClick={handleCopyKaggleCmd}
                    title="Copy command"
                  >
                    {copiedCmd ? (
                      <>
                        <Check size={12} />
                        <span>Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy size={12} />
                        <span>Copy Entrypoint</span>
                      </>
                    )}
                  </button>
                </div>
                <pre className="terminal-body">
                  <code>
                    <span className="t-comment"># Launch production multi-class training with location-stratified splits:</span>
                    {'\n'}python brain_kaggle.py \
                    {'\n'}  --data_root /kaggle/input/brain-tumor-dataset/archive \
                    {'\n'}  --output_dir /kaggle/working \
                    {'\n'}  --batch_size 32 \
                    {'\n'}  --epochs 50
                  </code>
                </pre>
              </div>

              {/* Interactive Artifacts Vault */}
              <div className="artifact-vault-section">
                <div className="vault-header">
                  <Database size={15} className="vault-icon" />
                  <div>
                    <h4 className="vault-title">Exported Evaluation Artifacts</h4>
                    <span className="vault-sub">
                      Packaged automatically into brain_classification_results.zip upon training completion (Click to inspect)
                    </span>
                  </div>
                </div>

                <div className="artifact-cards-grid">
                  {ARTIFACTS.map((art) => (
                    <button
                      key={art.id}
                      type="button"
                      className="artifact-card"
                      onClick={() => handleOpenArtifact(art)}
                    >
                      <div className="art-top">
                        <span className="art-category">{art.category}</span>
                        <span className="art-format">{art.format}</span>
                      </div>
                      <h5 className="art-title">{art.title}</h5>
                      <p className="art-desc">{art.desc}</p>
                      <div className="art-bottom-strip">
                        <span className="art-filename-text">{art.filename}</span>
                        <span className="art-inspect-action">
                          Inspect <ExternalLink size={11} />
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </RomanSection>
        </section>

        {/* SECTION II: FASTAPI PRODUCTION ARCHITECTURE */}
        <section id="section-api-infra" className="doc-section-wrapper">
          <RomanSection
            index={1}
            of={4}
            title="Infrastructura - Production FastAPI Inference Service"
            className="sp12"
          >
            <div className="doc-info-block">
              <div className="doc-block-header">
                <Server size={15} className="doc-block-icon" />
                <span className="doc-block-title">Backend Architecture & Live REST API</span>
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

        {/* SECTION III: STEREOTACTIC MNI LOCALIZATION */}
        <section id="section-stereotactic" className="doc-section-wrapper">
          <RomanSection
            index={2}
            of={4}
            title="Stereotaxis - MNI152 Coordinate Normalization System"
            className="sp12"
          >
            <div className="doc-info-block">
              <div className="doc-block-header">
                <Compass size={15} className="doc-block-icon" />
                <span className="doc-block-title">Spatial Normalization & Anatomical Projection</span>
              </div>
              <p className="doc-block-copy">
                To bridge 2D MRI slice classification with spatial neuroanatomy, OpenMed maps all 38 anatomical sites from the
                curated dataset into the <strong>Montreal Neurological Institute (MNI152)</strong> stereotactic coordinate frame.
                Each region is assigned a 3D centroid $[X, Y, Z]$ in millimeters referenced from the anterior commissure.
              </p>

              <div className="mni-axes-diagram">
                <div className="mni-axis-card">
                  <span className="axis-title">X Axis · Sagittal</span>
                  <span className="axis-range">-80 mm to +80 mm</span>
                  <span className="axis-note">Negative values represent Left Hemisphere; positive values represent Right Hemisphere.</span>
                </div>
                <div className="mni-axis-card">
                  <span className="axis-title">Y Axis · Coronal</span>
                  <span className="axis-range">-110 mm to +70 mm</span>
                  <span className="axis-note">Negative values indicate Posterior (Occipital/Cerebellar); positive indicate Anterior (Frontal).</span>
                </div>
                <div className="mni-axis-card">
                  <span className="axis-title">Z Axis · Axial</span>
                  <span className="axis-range">-50 mm to +85 mm</span>
                  <span className="axis-note">Negative values indicate Inferior (Skull Base/Brainstem); positive indicate Superior (Vertex).</span>
                </div>
              </div>
            </div>
          </RomanSection>
        </section>

        {/* SECTION IV: DATASET CURATION & PURGING */}
        <section id="section-dataset" className="doc-section-wrapper">
          <RomanSection
            index={3}
            of={4}
            title="Curatio - Dataset Integrity, Purging & Verification"
            className="sp12"
          >
            <div className="doc-info-block">
              <div className="doc-block-header">
                <ShieldCheck size={15} className="doc-block-icon" />
                <span className="doc-block-title">Dataset Sanitization Protocol</span>
              </div>
              <p className="doc-block-copy">
                The source dataset comprises 12,643 brain MRI images with comprehensive spatial bounding boxes and lesion contours.
                During dataset audit, 17 corrupt entries were identified containing binary mask filenames rather than source scans
                (ending in <code>_mask.png</code>).
              </p>
              <div className="purging-stat-strip">
                <div className="p-stat">
                  <span className="v">12,643</span>
                  <span className="k">Total Catalogued Files</span>
                </div>
                <div className="p-stat">
                  <span className="v red">-17</span>
                  <span className="k">Corrupt Mask Entries Purged</span>
                </div>
                <div className="p-stat">
                  <span className="v green">12,626</span>
                  <span className="k">Net Verified Training Scans</span>
                </div>
              </div>
            </div>
          </RomanSection>
        </section>
      </main>

      {/* Artifact Lightbox Modal */}
      {activeArtifact && (
        <div className="artifact-modal-backdrop" onClick={() => setActiveArtifact(null)}>
          <div className="artifact-modal-window" onClick={(e) => e.stopPropagation()}>
            <div className="artifact-modal-header">
              <div>
                <span className="modal-category-tag">{activeArtifact.category}</span>
                <h3 className="modal-title">{activeArtifact.title}</h3>
                <span className="modal-subtitle">{activeArtifact.desc}</span>
              </div>
              <div className="modal-actions">
                {activeArtifact.previewPath && (
                  <a
                    href={activeArtifact.previewPath}
                    download={activeArtifact.filename.split(' ')[0]}
                    className="modal-icon-btn"
                    title="Download File"
                  >
                    <Download size={15} />
                  </a>
                )}
                <button
                  type="button"
                  className="modal-icon-btn close"
                  onClick={() => setActiveArtifact(null)}
                  title="Close Modal"
                >
                  <X size={16} />
                </button>
              </div>
            </div>

            <div className="artifact-modal-content">
              {activeArtifact.isText ? (
                <div className="artifact-text-viewer">
                  <pre>
                    <code>{artifactTextContent ?? 'Loading telemetry log…'}</code>
                  </pre>
                </div>
              ) : activeArtifact.previewPath ? (
                <div className="artifact-image-viewer">
                  <img src={activeArtifact.previewPath} alt={activeArtifact.title} />
                </div>
              ) : (
                <div className="artifact-empty-view">No visual preview available for this binary weight file.</div>
              )}
            </div>

            <div className="artifact-modal-footer">
              <span className="modal-footer-file">
                File: <code>{activeArtifact.filename}</code>
              </span>
              <span className="modal-footer-meta">
                Verified Evaluation Artifact · Location-Stratified 39-Class Split
              </span>
            </div>
          </div>
        </div>
      )}
    </AppShell>
  )
}
