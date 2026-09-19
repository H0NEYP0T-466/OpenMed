import React, { useState } from 'react'
import {
  Layers,
  Sliders,
  Database,
  Check,
  Copy,
  ExternalLink,
  AlertTriangle,
  XCircle,
  CheckCircle2,
} from 'lucide-react'
import type { ArtifactCard } from '../../types'

interface BrainClassificationDocsProps {
  readonly onOpenArtifact: (artifact: ArtifactCard) => void
}

const ARTIFACTS: readonly ArtifactCard[] = [
  {
    id: 'dist',
    filename: 'class_distribution.png',
    format: 'PNG · Categorical Bar',
    title: 'Class & Split Balance Ledger',
    desc: 'Per-class sample counts for the 70/15/15 grouped split on a log axis, so the long tail of rare histologies stays legible next to the dominant classes.',
    category: '01 · Dataset',
    previewPath: '/visuals/brain/class_distribution.png',
  },
  {
    id: 'split_manifest',
    filename: 'split_manifest.csv',
    format: 'CSV · Provenance',
    title: 'Split Assignment Ledger',
    desc: 'Every image with its class and assigned split, so any reported metric can be traced back and audited for source-scan isolation.',
    category: '01 · Dataset',
    previewPath: '/visuals/brain/split_manifest.csv',
    isText: true,
  },
  {
    id: 'loss_curves',
    filename: 'loss_curves.png',
    format: 'PNG · Metric Plot',
    title: 'Training & Validation Loss',
    desc: 'Per-epoch class-weighted cross-entropy for both branches of the grouped split, under a one-cycle schedule stepped per minibatch.',
    category: '02 · Optimization',
    previewPath: '/visuals/brain/loss_curves.png',
  },
  {
    id: 'accuracy_curves',
    filename: 'accuracy_curves.png',
    format: 'PNG · Metric Plot',
    title: 'Top-1 Accuracy Trajectories',
    desc: 'Training versus validation accuracy per epoch. The gap between the two is the honest over-fitting signal once leakage is removed.',
    category: '03 · Optimization',
    previewPath: '/visuals/brain/accuracy_curves.png',
  },
  {
    id: 'cm',
    filename: 'confusion_matrix.png',
    format: 'PNG · Row-Normalised Matrix',
    title: 'Normalized Confusion Matrix',
    desc: 'Row-normalised matrix restricted to classes present in the evaluated split, avoiding all-zero rows that render as undefined cells.',
    category: '04 · Validation',
    previewPath: '/visuals/brain/confusion_matrix.png',
  },
  {
    id: 'roc',
    filename: 'roc_curves.png',
    format: 'PNG · Multi-Class Curves',
    title: 'One-vs-Rest ROC Frontiers',
    desc: 'Per-class ROC curves for the strongest classes with mean per-class and micro-averaged AUC reported in the figure title.',
    category: '05 · Performance',
    previewPath: '/visuals/brain/roc_curves.png',
  },
  {
    id: 'pr',
    filename: 'pr_curves.png',
    format: 'PNG · Precision Curves',
    title: 'Precision-Recall Frontiers',
    desc: 'Per-class precision-recall curves with average precision annotated, which is more informative than ROC under class imbalance.',
    category: '06 · Performance',
    previewPath: '/visuals/brain/pr_curves.png',
  },
  {
    id: 'grid',
    filename: 'sample_predictions.png',
    format: 'PNG · 4×4 Plate',
    title: 'Qualitative Validation Grid',
    desc: 'Sixteen held-out test images with true and predicted labels, captioned in red wherever the differential is wrong.',
    category: '07 · Inference',
    previewPath: '/visuals/brain/sample_predictions.png',
  },
  {
    id: 'cam',
    filename: 'gradcam_samples.png',
    format: 'PNG · Layer Heatmap',
    title: 'Grad-CAM Attention Overlays',
    desc: 'Activation maps for the final convolution stage on a fixed, seeded sample of the test split. Each panel is computed from its own image.',
    category: '08 · Explainability',
    previewPath: '/visuals/brain/gradcam_samples.png',
  },
  {
    id: 'report',
    filename: 'classification_report.txt & training_log.csv',
    format: 'TXT / CSV · Telemetry',
    title: 'Classification Report & Epoch Log',
    desc: 'Per-class precision, recall and F1 with the run provenance header — monitor, seed, split sizes and macro AUC — plus the per-epoch CSV.',
    category: '09 · Audit',
    previewPath: '/visuals/brain/classification_report.txt',
    isText: true,
  },
  {
    id: 'metrics',
    filename: 'metrics.json',
    format: 'JSON · Machine Readable',
    title: 'Machine-Readable Run Metrics',
    desc: 'Test accuracy, loss, macro one-vs-rest AUC, split sizes and the grouping rule, mirroring what the label file binds to the checkpoint.',
    category: '09 · Audit',
    previewPath: '/visuals/brain/metrics.json',
    isText: true,
  },
]

export const BrainClassificationDocs: React.FC<BrainClassificationDocsProps> = ({ onOpenArtifact }) => {
  const [copiedCmd, setCopiedCmd] = useState(false)

  const handleCopyKaggleCmd = () => {
    const cmd =
      'python brain_kaggle.py --data_root /kaggle/input/brain-tumor-dataset/archive --output_dir /kaggle/working --batch_size 32 --epochs 50 --seed 42'
    navigator.clipboard.writeText(cmd)
    setCopiedCmd(true)
    setTimeout(() => setCopiedCmd(false), 2200)
  }

  return (
    <div className="task-block classification">
      <div className="task-header-strip">
        <div className="task-tag-group">
          <span className="task-type-tag class">Task · Classification</span>
          <span className="task-name-text">42-Class Histological &amp; Sequence Differential</span>
        </div>
        <span className="task-model-pill">EfficientNetV2-B2 · 208×208 input · AdamW</span>
      </div>

      {/* Empirical Research Chronology */}
      <div className="chronology-wrapper">
        {/* Phase 1 */}
        <div className="chronology-item">
          <div className="chronology-marker">
            <span className="phase-num">01</span>
          </div>
          <div className="chronology-card">
            <div className="chronology-head">
              <span className="phase-badge done">Phase I · Production Baseline</span>
              <span className="phase-timestamp">Kaggle Tesla T4 · 50 Epochs · 51m 17s</span>
            </div>
            <h4 className="chronology-title">39-Class EfficientNetV2-B2 Baseline &amp; Stratified Holdout</h4>
            <p className="chronology-text">
              Trained an <strong>EfficientNetV2-B2</strong> backbone from ImageNet-1k weights on 12,626 curated cranial MRI scans across
              13 WHO histological families and 3 MRI sequences. To prevent anatomical morphological leakage between patient scans, we designed a
              strict <strong>primary-location stratified split</strong> across 38 MNI anatomical regions (70% train / 15% val / 15% holdout test).
            </p>
            <div className="chronology-metrics">
              <div className="c-metric">
                <span className="c-metric-v">96.99%</span>
                <span className="c-metric-k">Holdout Test Accuracy</span>
              </div>
              <div className="c-metric">
                <span className="c-metric-v">0.1604</span>
                <span className="c-metric-k">Test Cross-Entropy Loss</span>
              </div>
              <div className="c-metric">
                <span className="c-metric-v">96.88%</span>
                <span className="c-metric-k">Best Validation Accuracy</span>
              </div>
              <div className="c-metric">
                <span className="c-metric-v">1,894</span>
                <span className="c-metric-k">Unseen Test Samples</span>
              </div>
            </div>
          </div>
        </div>

        {/* Phase 2 */}
        <div className="chronology-item">
          <div className="chronology-marker warning">
            <span className="phase-num">02</span>
          </div>
          <div className="chronology-card warning-border">
            <div className="chronology-head">
              <span className="phase-badge warning">Phase II · Out-of-Distribution Stress Testing</span>
              <span className="phase-timestamp">External BTSC Dataset Validation</span>
            </div>
            <h4 className="chronology-title">Discovery of the Slice-Angle Confounder &amp; Missing Sellar Pathology</h4>
            <p className="chronology-text">
              When evaluating the 97% checkpoint against external cranial scans from the <code>btsc-unet-vit</code> benchmark,
              two material failure modes emerged:
            </p>

            <div className="anomaly-breakdown-grid">
              <div className="anomaly-box">
                <div className="anomaly-box-title">
                  <AlertTriangle size={15} className="text-warning" />
                  <span>1. The "No Tumor" Slice/Angle Confounder (Shortcut Learning)</span>
                </div>
                <p className="anomaly-copy">
                  <strong>Observed Anomaly:</strong> The model consistently predicted healthy, tumor-free MRI scans from the BTSC dataset as positive for brain tumors.
                </p>
                <p className="anomaly-copy">
                  <strong>Root Cause Diagnosis:</strong> In the initial 12.6k dataset, the "Normal" class images were confined to a narrow slice height and acquisition angle.
                  The convolutional feature extractors suffered from <em>shortcut learning</em>: rather than learning the absence of pathological tissue,
                  the network learned slice-level geometry as a surrogate for health. When presented with healthy slices at unfamiliar angles or skull-base levels,
                  it defaulted to whichever tumor class matched that slice geometry in training.
                </p>
              </div>

              <div className="anomaly-box">
                <div className="anomaly-box-title">
                  <AlertTriangle size={15} className="text-warning" />
                  <span>2. Complete Absence of Pituitary Adenomas</span>
                </div>
                <p className="anomaly-copy">
                  <strong>Observed Anomaly:</strong> Sellar masses (pituitary adenomas) from external scans were classified with false confidence as Meningiomas.
                </p>
                <p className="anomaly-copy">
                  <strong>Root Cause Diagnosis:</strong> Pituitary neoplasms were entirely absent from the initial 13-family taxonomy.
                  Because both Meningiomas and Pituitary adenomas enhance strongly on T1C+ scans in skull-base regions, the model suffered from Out-of-Distribution (OOD)
                  overconfidence, mapping sellar lesions into dural meningioma classes.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Phase 3 */}
        <div className="chronology-item">
          <div className="chronology-marker danger">
            <span className="phase-num">03</span>
          </div>
          <div className="chronology-card danger-border">
            <div className="chronology-head">
              <span className="phase-badge danger">Phase III · Architectural Trade-off Analysis</span>
              <span className="phase-timestamp">Rigorous Audit of Discarded Engineering Hypotheses</span>
            </div>
            <h4 className="chronology-title">Why Intermediate Fallback Cascades Were Formally Rejected</h4>
            <p className="chronology-text">
              Prior to committing to dataset unification and retraining, we rigorously audited three proposed dual-model shortcuts.
              All three were rejected on clinical and mathematical grounds:
            </p>

            <div className="decision-matrix-table-wrap">
              <table className="decision-matrix-table">
                <thead>
                  <tr>
                    <th>Proposed Route</th>
                    <th>Hypothesis</th>
                    <th>Observed Technical Pitfall</th>
                    <th>Engineering Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="route-name">
                      <strong>Pathway A</strong>
                      <span>Confidence Fallback (&lt;70%)</span>
                    </td>
                    <td>Run EfficientNet first; if confidence &lt; 70%, route scan to 4-class ViT.</td>
                    <td>
                      <strong>Softmax OOD Overconfidence:</strong> CNNs on unfamiliar slice angles output &gt;85% confidence for incorrect classes.
                      The fallback would rarely trigger, allowing false tumors to bypass review.
                    </td>
                    <td className="verdict-cell rejected">
                      <XCircle size={14} /> <span>REJECTED</span>
                    </td>
                  </tr>
                  <tr>
                    <td className="route-name">
                      <strong>Pathway B</strong>
                      <span>ViT 4-Class Gatekeeper</span>
                    </td>
                    <td>Run 4-class ViT first (Normal/Pituitary/Glioma/Meningioma); route tumors to EfficientNet.</td>
                    <td>
                      <strong>Taxonomic Truncation:</strong> ViT knows only 4 classes. Scans with Schwannoma, Medulloblastoma, Ependymoma,
                      or Germinoma get forced into Normal or Pituitary, creating fatal diagnostic false-negatives.
                    </td>
                    <td className="verdict-cell rejected">
                      <XCircle size={14} /> <span>REJECTED</span>
                    </td>
                  </tr>
                  <tr>
                    <td className="route-name">
                      <strong>Pathway C</strong>
                      <span>50-Epoch Checkpoint Fine-Tuning</span>
                    </td>
                    <td>Resume training from <code>brain_best_model.pth</code> with Pituitary and BTSC Normal added.</td>
                    <td>
                      <strong>Catastrophic Forgetting &amp; Head Norm Drift:</strong> The backbone already lost plasticity during the cosine schedule.
                      New classes with randomized weights face severe representation bias against 39 converged classes.
                    </td>
                    <td className="verdict-cell rejected">
                      <XCircle size={14} /> <span>REJECTED</span>
                    </td>
                  </tr>
                  <tr>
                    <td className="route-name highlight">
                      <strong>Pathway D</strong>
                      <span>Unified Dataset Retraining</span>
                    </td>
                    <td>Sequence-map BTSC Normal &amp; Pituitary; train EfficientNetV2-B2 from scratch on Kaggle T4.</td>
                    <td>
                      <strong>Clean Convergence:</strong> 51 minutes of GPU compute. All 41+ classes compete fairly from Epoch 1 with uniform
                      BatchNorm accumulation and angle-invariant data augmentations.
                    </td>
                    <td className="verdict-cell approved">
                      <CheckCircle2 size={14} /> <span>APPROVED</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Phase 4 */}
        <div className="chronology-item">
          <div className="chronology-marker active">
            <span className="phase-num">04</span>
          </div>
          <div className="chronology-card active-border">
            <div className="chronology-head">
              <span className="phase-badge active">Phase IV · Production Roadmap &amp; Execution Plan</span>
              <span className="phase-timestamp">Active Retraining Blueprint</span>
            </div>
            <h4 className="chronology-title">Sequence-Aware Dataset Merger &amp; 41-Class Retraining Blueprint</h4>
            <p className="chronology-text">
              The active engineering pathway focuses on unifying the datasets cleanly before launching the final Kaggle run:
            </p>

            <div className="roadmap-steps-grid">
              <div className="roadmap-step-card">
                <span className="step-idx">Step 01</span>
                <h5 className="step-title">MRI Sequence Auditing</h5>
                <p className="step-desc">
                  Audit BTSC "No Tumor" and "Pituitary" images to map each scan to its exact pulse sequence:
                  <strong> T1-weighted, T1 contrast-enhanced (T1C+), or T2-weighted</strong>.
                </p>
              </div>

              <div className="roadmap-step-card">
                <span className="step-idx">Step 02</span>
                <h5 className="step-title">Anatomical &amp; Patient-Level Partitioning</h5>
                <p className="step-desc">
                  Merge multi-angle normal scans into sequence-specific <code>Normal</code> buckets while strictly isolating patient IDs
                  to prevent inter-slice leakage across train/val/test splits.
                </p>
              </div>

              <div className="roadmap-step-card">
                <span className="step-idx">Step 03</span>
                <h5 className="step-title">Taxonomy Expansion (14 WHO Families)</h5>
                <p className="step-desc">
                  Introduce Pituitary adenomas as the 14th histological family mapped stereotactically to the Sellar/Suprasellar MNI region,
                  expanding the classification space to 41–42 classes.
                </p>
              </div>

              <div className="roadmap-step-card">
                <span className="step-idx">Step 04</span>
                <h5 className="step-title">Angle-Invariant Retraining</h5>
                <p className="step-desc">
                  Launch a clean 50-epoch run on Tesla T4 with expanded affine rotation (±15°), contrast jitter, and class-weighted
                  Cross-Entropy loss to permanently eliminate slice-angle bias.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Kaggle Execution Benchmarks & Artifacts Ledger */}
      <div className="archival-ledger-card">
        <div className="ledger-card-title-strip">
          <span className="ledger-tag">Production Benchmark Ledger</span>
          <span className="ledger-sub">Tesla T4 50-Epoch Kaggle Artifacts</span>
        </div>

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

        {/* Visual Split Stratification */}
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

        {/* Neural Architecture & Optimization Manifest */}
        <div className="specs-matrix-card">
          <div className="card-topline">
            <Sliders size={15} className="card-top-icon" />
            <span className="card-top-title">Neural Architecture &amp; Optimization Manifest</span>
          </div>

          <div className="specs-grid">
            <div className="spec-row">
              <span className="spec-k">Backbone Architecture</span>
              <span className="spec-v">EfficientNetV2-B2 (tf_efficientnetv2_b2.in1k)</span>
            </div>
            <div className="spec-row">
              <span className="spec-k">Input Resolution</span>
              <span className="spec-v">208 × 208 × 3, bicubic with crop_pct 0.89 (resolved from the timm model config)</span>
            </div>
            <div className="spec-row">
              <span className="spec-k">Loss Formulation</span>
              <span className="spec-v">Cross-Entropy, sqrt-smoothed inverse-frequency weights mean-normalised and clipped to [0.2, 5.0]</span>
            </div>
            <div className="spec-row">
              <span className="spec-k">Optimizer &amp; LR</span>
              <span className="spec-v">AdamW (lr = 1e-3, weight_decay = 1e-4)</span>
            </div>
            <div className="spec-row">
              <span className="spec-k">Learning Rate Schedule</span>
              <span className="spec-v">OneCycleLR stepped per minibatch (pct_start = 0.3), max_lr = 1e-3</span>
            </div>
            <div className="spec-row">
              <span className="spec-k">Split Protocol</span>
              <span className="spec-v">Grouped stratified — images sharing a source scan or byte-identical content stay in one split</span>
            </div>
            <div className="spec-row">
              <span className="spec-k">Checkpoint Monitor</span>
              <span className="spec-v">val_loss, with best-val-accuracy weights retained separately</span>
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
              <span className="t-comment"># Launch production multi-class training with grouped, leakage-free splits:</span>
              {'\n'}python brain_kaggle.py \
              {'\n'}  --data_root /kaggle/input/brain-tumor-dataset/archive \
              {'\n'}  --output_dir /kaggle/working \
              {'\n'}  --batch_size 32 \
              {'\n'}  --epochs 50 \
              {'\n'}  --seed 42
            </code>
          </pre>
        </div>

        {/* FIXED & CONSTRAINED 3-COLUMN ARTIFACT VAULT */}
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
                onClick={() => onOpenArtifact(art)}
              >
                <div className="art-card-thumb-wrap">
                  {art.isText ? (
                    <div className="art-text-thumb">
                      <Database size={22} className="art-text-thumb-icon" />
                      <span>Report &amp; Log · TXT / CSV</span>
                    </div>
                  ) : (
                    <img
                      src={art.previewPath}
                      alt={art.title}
                      className="art-thumb-img"
                      loading="lazy"
                    />
                  )}
                </div>
                <div className="art-card-body">
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
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
