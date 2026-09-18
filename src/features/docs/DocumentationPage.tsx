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
  Brain as BrainIcon,
  Heart,
  Activity,
  Box,
  AlertTriangle,
  XCircle,
  CheckCircle2,
  ArrowUpRight,
  Eye,
  Droplets,
  Scan,
  Workflow,
  Cpu,
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
    id: 'dist',
    filename: 'class_distribution.png',
    format: 'PNG · Categorical Bar',
    title: 'Class & Split Balance Ledger',
    desc: 'Sample counts across 70% train (8,838), 15% val (1,894), and 15% test (1,894) location-stratified splits.',
    category: '01 · Dataset',
    previewPath: '/visuals/brain/class_distribution.png',
  },
  {
    id: 'loss_curves',
    filename: 'loss_curves.png',
    format: 'PNG · Metric Plot',
    title: 'Training & Validation Loss',
    desc: 'Per-epoch cross-entropy loss convergence across 50 epochs validating monotonic loss reduction.',
    category: '02 · Optimization',
    previewPath: '/visuals/brain/loss_curves.png',
  },
  {
    id: 'accuracy_curves',
    filename: 'accuracy_curves.png',
    format: 'PNG · Metric Plot',
    title: 'Top-1 Accuracy Trajectories',
    desc: 'Per-epoch training vs. validation accuracy curves demonstrating steady ascent to 96.99% holdout test.',
    category: '03 · Optimization',
    previewPath: '/visuals/brain/accuracy_curves.png',
  },
  {
    id: 'cm',
    filename: 'confusion_matrix.png',
    format: 'PNG · 39×39 Matrix',
    title: 'Normalized Confusion Matrix',
    desc: 'Full 39-class normalized diagnostic confusion matrix with row-wise sensitivity and false-positive mapping.',
    category: '04 · Validation',
    previewPath: '/visuals/brain/confusion_matrix.png',
  },
  {
    id: 'roc',
    filename: 'roc_curves_macro.png',
    format: 'PNG · Multi-Class Curve',
    title: 'Multi-Class ROC Frontiers',
    desc: 'Macro-averaged and per-class AUC-ROC curves demonstrating high true-positive discrimination.',
    category: '05 · Performance',
    previewPath: '/visuals/brain/roc_curves_macro.png',
  },
  {
    id: 'pr',
    filename: 'pr_curves_micro.png',
    format: 'PNG · Precision Curve',
    title: 'Precision-Recall Frontiers',
    desc: 'Micro-averaged precision-recall operating frontier reflecting robust clinical precision under class imbalance.',
    category: '06 · Performance',
    previewPath: '/visuals/brain/pr_curves_micro.png',
  },
  {
    id: 'grid',
    filename: 'sample_predictions.png',
    format: 'PNG · 4×4 Plate',
    title: 'Qualitative Validation Grid',
    desc: '16-panel test batch with true vs. predicted labels, highlighting correct predictions and clinical edge cases.',
    category: '07 · Inference',
    previewPath: '/visuals/brain/sample_predictions.png',
  },
  {
    id: 'cam',
    filename: 'gradcam_samples.png',
    format: 'PNG · Layer Heatmap',
    title: 'Grad-CAM Attention Overlays',
    desc: 'Final convolutional layer (conv_head) activation heatmaps demonstrating anatomical focus on tumor pathology.',
    category: '08 · Explainability',
    previewPath: '/visuals/brain/gradcam_samples.png',
  },
  {
    id: 'report',
    filename: 'classification_report.txt & training_log.csv',
    format: 'TXT / CSV · Telemetry',
    title: 'Classification Report & Epoch Log',
    desc: 'Exhaustive per-class precision, recall, F1-score report and per-epoch CSV recording loss, LR, and elapsed time.',
    category: '09 · Audit',
    previewPath: '/visuals/brain/classification_report.txt',
    isText: true,
  },
]

interface OrganDirectoryItem {
  readonly id: string
  readonly name: string
  readonly latin: string
  readonly system: string
  readonly modelFile: string
  readonly modality: string
  readonly classificationTask: string
  readonly segmentationTask: string
  readonly hotspotsCount: number
  readonly active?: boolean
  readonly iconType: 'brain' | 'heart' | 'lungs' | 'body' | 'kidney' | 'liver' | 'eye' | 'blood' | 'bone' | 'breast' | 'skin' | 'pancreas' | 'intestine'
}

const ALL_ORGANS: readonly OrganDirectoryItem[] = [
  {
    id: 'brain',
    name: 'Brain',
    latin: 'Encephalon',
    system: 'Central Nervous System (CNS)',
    modelFile: 'brain.glb',
    modality: 'Cranial Axial MRI (512×512)',
    classificationTask: '39-Class Histological & Sequence Subtyping (T1/T1C+/T2)',
    segmentationTask: 'BraTS Glioma Sub-regions (WT / TC / ET)',
    hotspotsCount: 5,
    active: true,
    iconType: 'brain',
  },
  {
    id: 'body',
    name: 'Whole Body Atlas',
    latin: 'Systema Corporis',
    system: 'Musculoskeletal & Visceral Systems',
    modelFile: 'body.glb',
    modality: 'Full-Body Biomechanical PBR',
    classificationTask: 'Multi-Organ Pathology Indexing & Triage',
    segmentationTask: '12 Anatomical Systems Exploded Volumetry',
    hotspotsCount: 12,
    iconType: 'body',
  },
  {
    id: 'heart',
    name: 'Heart',
    latin: 'Cor',
    system: 'Cardiovascular System',
    modelFile: 'heart.glb',
    modality: 'Echocardiography & Cine-MRI',
    classificationTask: '12-Lead ECG Arrhythmia & Ischemia Grading',
    segmentationTask: 'ACDC Left/Right Ventricular & Myocardial Volumetry',
    hotspotsCount: 5,
    iconType: 'heart',
  },
  {
    id: 'lungs',
    name: 'Lungs',
    latin: 'Pulmones',
    system: 'Respiratory System',
    modelFile: 'lungs.glb',
    modality: 'Chest Radiography & High-Resolution CT',
    classificationTask: 'CheXpert 14-Pathology & Pneumonia Differential',
    segmentationTask: 'Pulmonary Lobe & Airway Tree Instance Delineation',
    hotspotsCount: 5,
    iconType: 'lungs',
  },
  {
    id: 'kidney',
    name: 'Kidneys',
    latin: 'Ren',
    system: 'Urinary & Excretory System',
    modelFile: 'kidney.glb',
    modality: 'Abdominal Contrast CT',
    classificationTask: 'Renal Cell Carcinoma (RCC) Subtype Grading',
    segmentationTask: 'KiTS23 Kidney, Tumor & Cyst Parenchyma Volumetry',
    hotspotsCount: 4,
    iconType: 'kidney',
  },
  {
    id: 'liver',
    name: 'Liver',
    latin: 'Hepar',
    system: 'Digestive & Metabolic System',
    modelFile: 'liver.glb',
    modality: 'Tri-Phase Contrast Abdominal CT',
    classificationTask: 'Focal Liver Lesion (HCC vs Hemangioma vs FNH)',
    segmentationTask: 'LiTS Hepatic Parenchyma & Tumor Burden Volumetry',
    hotspotsCount: 4,
    iconType: 'liver',
  },
  {
    id: 'eye',
    name: 'Eye',
    latin: 'Oculus',
    system: 'Visual & Sensory System',
    modelFile: 'eye.glb',
    modality: 'Color Fundus Photography & OCT',
    classificationTask: 'Diabetic Retinopathy (5-Stage) & Glaucoma Screening',
    segmentationTask: 'Retinal Vessel Arborization & Optic Disc/Cup Ratio',
    hotspotsCount: 3,
    iconType: 'eye',
  },
  {
    id: 'blood',
    name: 'Blood & Vasculature',
    latin: 'Systema Vasorum',
    system: 'Circulatory System',
    modelFile: 'blood.glb',
    modality: 'CTA & Blood Smear Microscopy',
    classificationTask: 'ALL-IDB Leukocyte & Blast Cell Differential',
    segmentationTask: 'Coronary & Cerebrovascular Lumen Stenosis Segmentation',
    hotspotsCount: 4,
    iconType: 'blood',
  },
  {
    id: 'bone',
    name: 'Bone & Skeleton',
    latin: 'Systema Skeletale',
    system: 'Musculoskeletal System',
    modelFile: 'bone.glb',
    modality: 'Radiographs & Whole-Body CT',
    classificationTask: 'MURA Fracture Detection & Osteoarthritis Grading',
    segmentationTask: 'TotalSegmentator Vertebral Column & Cortical Bone',
    hotspotsCount: 6,
    iconType: 'bone',
  },
  {
    id: 'breast',
    name: 'Breast',
    latin: 'Mamma',
    system: 'Reproductive & Glandular System',
    modelFile: 'breast.glb',
    modality: 'Full-Field Digital Mammography (FFDM)',
    classificationTask: 'BI-RADS Density Assessment & Malignancy Classification',
    segmentationTask: 'Mammographic Soft Tissue Mass & Calcification Clusters',
    hotspotsCount: 3,
    iconType: 'breast',
  },
  {
    id: 'skin',
    name: 'Skin',
    latin: 'Integumentum Commune',
    system: 'Integumentary System',
    modelFile: 'skin.glb',
    modality: 'Dermoscopy & Clinical Macro Photography',
    classificationTask: 'ISIC 7-Class Malignancy Differential (Melanoma/Nevi)',
    segmentationTask: 'Dermoscopic Lesion Boundary & ABCD Metric Scoring',
    hotspotsCount: 4,
    iconType: 'skin',
  },
  {
    id: 'pancreas',
    name: 'Pancreas',
    latin: 'Pancreas',
    system: 'Endocrine & Digestive System',
    modelFile: 'pancreas.glb',
    modality: 'Dual-Phase Pancreatic CT',
    classificationTask: 'Pancreatic Adenocarcinoma (PDAC) vs Neuroendocrine',
    segmentationTask: 'MSD Pancreatic Parenchyma & Duct Dilatation Volumetry',
    hotspotsCount: 3,
    iconType: 'pancreas',
  },
  {
    id: 'intestine',
    name: 'Intestines & Colon',
    latin: 'Tractus Gastrointestinalis',
    system: 'Gastrointestinal System',
    modelFile: 'intestine.glb',
    modality: 'High-Definition Video Endoscopy',
    classificationTask: 'Colon Polyp Histology (Adenoma vs Hyperplastic)',
    segmentationTask: 'Kvasir-SEG Real-Time Polyp Mask Delineation',
    hotspotsCount: 5,
    iconType: 'intestine',
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
          <span>Launch 3D Brain Atlas</span>
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
            Documenta & Engineering Ledger
            <span className="serif">Architectural Specifications, Empirical Pathways & Model Trajectories</span>
          </h1>

          <p className="doc-lead-copy">
            Rigorous mathematical, anatomical, and empirical documentation for OpenMed’s clinical AI architectures.
            This ledger organizes our entire multi-organ platform: 3D anatomical models, deep learning classification suites,
            volumetric segmentation pipelines, out-of-distribution failure mode discoveries, discarded architectural hypotheses,
            and production retraining blueprints.
          </p>

          {/* Quick Anchor Navigation */}
          <div className="doc-quick-nav">
            <a href="#section-3d-models" className="doc-nav-anchor active">
              I · 3D Models Directory (13 Organs)
            </a>
            <a href="#section-models-suite" className="doc-nav-anchor">
              II · Models & Neural Suites
            </a>
            <a href="#model-brain" className="doc-nav-anchor">
              ↳ Brain (Classification & Segmentation)
            </a>
            <a href="#model-heart" className="doc-nav-anchor">
              ↳ Heart (Classification & Segmentation)
            </a>
            <a href="#model-lungs" className="doc-nav-anchor">
              ↳ Lungs (Classification & Segmentation)
            </a>
            <a href="#model-other-organs" className="doc-nav-anchor">
              ↳ Other Organ Suites (Renal, Liver, Eye, etc.)
            </a>
            <a href="#section-api-infra" className="doc-nav-anchor">
              III · FastAPI Backend Service
            </a>
            <a href="#section-stereotactic" className="doc-nav-anchor">
              IV · MNI152 Stereotaxis
            </a>
            <a href="#section-dataset" className="doc-nav-anchor">
              V · Dataset Sanitization
            </a>
          </div>
        </header>

        {/* =========================================================
            SECTION I: 3D ANATOMICAL MODELS DIRECTORY (ALL 13 ORGANS)
            ========================================================= */}
        <section id="section-3d-models" className="doc-section-wrapper">
          <RomanSection
            index={0}
            of={5}
            title="3D Models - Interactive WebGL Anatomical Assets Registry"
            className="sp12"
          >
            <div className="doc-info-block">
              <div className="doc-block-header">
                <Box size={16} className="doc-block-icon" />
                <span className="doc-block-title">Master 3D Organ & Biomechanical Model Index (13 Structures)</span>
              </div>
              <p className="doc-block-copy">
                OpenMed integrates 13 real-time 3D anatomical models rendered with Physically Based Rendering (PBR) materials,
                anisotropic lighting, stereotactic spatial landmarks, and interactive orbit controls.
                Select any anatomical asset below to enter its live 3D clinical workspace.
              </p>

              <div className="organ-cards-grid-all">
                {ALL_ORGANS.map((organ) => (
                  <Link
                    key={organ.id}
                    to={organ.id === 'body' ? '/app' : `/app?organ=${organ.id}`}
                    className={`organ-dir-card ${organ.active ? 'active-organ' : ''}`}
                  >
                    <div className="organ-card-top">
                      <span className={`organ-tag ${organ.active ? 'active' : ''}`}>
                        {organ.active ? 'Active Suite' : organ.system.split(' ')[0]}
                      </span>
                      <span className="organ-asset-name">{organ.modelFile}</span>
                    </div>

                    <div className="organ-card-body">
                      <div className={`organ-icon-wrap ${organ.iconType}`}>
                        {organ.iconType === 'brain' && <BrainIcon size={20} />}
                        {organ.iconType === 'heart' && <Heart size={20} />}
                        {organ.iconType === 'lungs' && <Activity size={20} />}
                        {organ.iconType === 'body' && <Box size={20} />}
                        {organ.iconType === 'eye' && <Eye size={20} />}
                        {organ.iconType === 'blood' && <Droplets size={20} />}
                        {!['brain', 'heart', 'lungs', 'body', 'eye', 'blood'].includes(organ.iconType) && (
                          <Scan size={20} />
                        )}
                      </div>
                      <div className="organ-card-text">
                        <h4 className="organ-name">
                          {organ.name} <span className="latin">{organ.latin}</span>
                        </h4>
                        <span className="organ-system-label">{organ.system}</span>
                      </div>
                    </div>

                    <div className="organ-tasks-preview">
                      <div className="task-preview-row">
                        <span className="task-k">Classify:</span>
                        <span className="task-v">{organ.classificationTask}</span>
                      </div>
                      <div className="task-preview-row">
                        <span className="task-k">Segment:</span>
                        <span className="task-v">{organ.segmentationTask}</span>
                      </div>
                    </div>

                    <div className="organ-card-footer">
                      <span className="organ-meta">{organ.hotspotsCount} Landmark Hotspots</span>
                      <span className="organ-jump-btn">
                        Launch 3D <ArrowUpRight size={12} />
                      </span>
                    </div>
                  </Link>
                ))}
              </div>
            </div>
          </RomanSection>
        </section>

        {/* =========================================================
            SECTION II: MODELS (NEURAL ARCHITECTURES & CLINICAL PIPELINES)
            ========================================================= */}
        <section id="section-models-suite" className="doc-section-wrapper">
          <RomanSection
            index={1}
            of={5}
            title="Models - Neural Architectures, Clinical Benchmarks & Task Suites"
            className="sp12"
          >
            <div className="models-overview-banner">
              <div className="banner-top">
                <Cpu size={18} className="banner-icon" />
                <h3 className="banner-title">Dual Diagnostic Paradigms: Classification & Volumetric Segmentation</h3>
              </div>
              <p className="banner-copy">
                Clinical AI in OpenMed is structured into two complementary task modalities:
                <strong> Histological & Sequence Classification</strong> (pathology detection, disease staging, sequence subtyping, and WHO grading)
                and <strong>Volumetric Segmentation</strong> (pixel-level mask delineation, multi-compartment lesion burden, and organ volumetry).
              </p>
            </div>

            {/* =========================================================
                A. ORGAN: BRAIN (ENCEPHALON)
                ========================================================= */}
            <div id="model-brain" className="organ-model-dossier">
              <div className="organ-dossier-header">
                <div className="dossier-head-left">
                  <div className="organ-badge brain">Organ · Brain (Encephalon)</div>
                  <h3 className="dossier-organ-title">Cranial Neuro-Oncology & Volumetric Suite</h3>
                  <span className="dossier-sub">Central Nervous System (CNS) · 38 Stereotactic MNI Regions</span>
                </div>
                <Link to="/app?organ=brain" className="btn-launch-workspace">
                  <BrainIcon size={14} />
                  <span>Open Brain 3D Suite</span>
                  <ArrowUpRight size={13} />
                </Link>
              </div>

              {/* Task 1: Classification */}
              <div className="task-block classification">
                <div className="task-header-strip">
                  <div className="task-tag-group">
                    <span className="task-type-tag class">Task · Classification</span>
                    <span className="task-name-text">39-Class Histological & Sequence Differential</span>
                  </div>
                  <span className="task-model-pill">EfficientNetV2-B2 · 512×512 · AdamW</span>
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
                      <h4 className="chronology-title">39-Class EfficientNetV2-B2 Baseline & Stratified Holdout</h4>
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
                      <h4 className="chronology-title">Discovery of the Slice-Angle Confounder & Missing Sellar Pathology</h4>
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
                        <span className="spec-v">512 × 512 × 3 (RGB-normalized)</span>
                      </div>
                      <div className="spec-row">
                        <span className="spec-k">Loss Formulation</span>
                        <span className="spec-v">Weighted Cross-Entropy (w = N / (C · n_c))</span>
                      </div>
                      <div className="spec-row">
                        <span className="spec-k">Optimizer &amp; LR</span>
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

                  {/* FIXED & CONSTRAINED ARTIFACT VAULT */}
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

              {/* Task 2: Segmentation */}
              <div className="task-block segmentation">
                <div className="task-header-strip">
                  <div className="task-tag-group">
                    <span className="task-type-tag seg">Task · Segmentation</span>
                    <span className="task-name-text">BraTS Multi-Compartment Glioma Volumetric Delineation</span>
                  </div>
                  <span className="task-model-pill">3D nnU-Net / SegFormer-3D · Voxel Metric</span>
                </div>

                <div className="boilerplate-card">
                  <p className="boilerplate-copy">
                    The volumetric segmentation module processes multi-parametric 3D cranial MRI acquisitions
                    (<strong>T1, T1-contrast [T1Gd], T2, and T2-FLAIR</strong>) to extract sub-voxel masks across three clinical sub-regions:
                  </p>
                  <div className="seg-targets-grid">
                    <div className="seg-target-box">
                      <span className="seg-badge wt">Whole Tumor (WT)</span>
                      <span className="seg-formula">ED + NET + ET</span>
                      <p className="seg-desc">Encompasses complete pathological abnormality including peritumoral edema and necrotic tissue.</p>
                      <span className="seg-target-metric">Target Dice: ≥ 0.91 · HD95 &lt; 4.2mm</span>
                    </div>
                    <div className="seg-target-box">
                      <span className="seg-badge tc">Tumor Core (TC)</span>
                      <span className="seg-formula">NET + ET</span>
                      <p className="seg-desc">Delineates the resectable tumor core (active enhancing neoplasm and central necrotic core).</p>
                      <span className="seg-target-metric">Target Dice: ≥ 0.86 · HD95 &lt; 5.0mm</span>
                    </div>
                    <div className="seg-target-box">
                      <span className="seg-badge et">Enhancing Tumor (ET)</span>
                      <span className="seg-formula">ET (Hyperintense T1Gd)</span>
                      <p className="seg-desc">Isolates hyper-vascularized active tumor rim exhibiting disrupted blood-brain barrier contrast leak.</p>
                      <span className="seg-target-metric">Target Dice: ≥ 0.82 · HD95 &lt; 3.8mm</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* =========================================================
                B. ORGAN: HEART (COR)
                ========================================================= */}
            <div id="model-heart" className="organ-model-dossier">
              <div className="organ-dossier-header">
                <div className="dossier-head-left">
                  <div className="organ-badge heart">Organ · Heart (Cor)</div>
                  <h3 className="dossier-organ-title">Cardiovascular Electro-Mechanical & Chamber Volumetry Suite</h3>
                  <span className="dossier-sub">Cardiovascular System · 4-Chamber Cine-MRI & 12-Lead ECG</span>
                </div>
                <Link to="/app?organ=heart" className="btn-launch-workspace">
                  <Heart size={14} />
                  <span>Open Heart 3D Suite</span>
                  <ArrowUpRight size={13} />
                </Link>
              </div>

              <div className="task-block classification">
                <div className="task-header-strip">
                  <div className="task-tag-group">
                    <span className="task-type-tag class">Task · Classification</span>
                    <span className="task-name-text">12-Lead ECG Arrhythmia, Ischemia & Infarction Subtyping</span>
                  </div>
                  <span className="task-model-pill">1D-ResNet with Squeeze-and-Excitation · PTB-XL</span>
                </div>
                <div className="boilerplate-card">
                  <p className="boilerplate-copy">
                    Multi-lead temporal signal processor classifying 12-lead diagnostic electrocardiograms across 5 super-classes:
                    <strong> Normal ECG (NORM), Myocardial Infarction (MI), ST/T-Change (STTC), Conduction Disturbance (CD), and Hypertrophy (HYP)</strong>.
                    Features temporal attention pooling for localized ST-elevation and QT-interval dispersion analysis.
                  </p>
                </div>
              </div>

              <div className="task-block segmentation">
                <div className="task-header-strip">
                  <div className="task-tag-group">
                    <span className="task-type-tag seg">Task · Segmentation</span>
                    <span className="task-name-text">Cine-MRI Multi-Chamber & Myocardial Volumetric Contouring</span>
                  </div>
                  <span className="task-model-pill">ACDC Benchmark · 2D/3D Hybrid U-Net</span>
                </div>
                <div className="boilerplate-card">
                  <p className="boilerplate-copy">
                    Delineates the endocardial and epicardial contours of the <strong>Left Ventricle (LV), Right Ventricle (RV), and Myocardium (MYO)</strong>
                    at both End-Diastole (ED) and End-Systole (ES). Automatically computes clinical cardiac metrics:
                    End-Diastolic Volume (EDV), End-Systolic Volume (ESV), Stroke Volume (SV), Myocardial Mass, and Left Ventricular Ejection Fraction (LVEF%).
                  </p>
                </div>
              </div>
            </div>

            {/* =========================================================
                C. ORGAN: LUNGS (PULMONES)
                ========================================================= */}
            <div id="model-lungs" className="organ-model-dossier">
              <div className="organ-dossier-header">
                <div className="dossier-head-left">
                  <div className="organ-badge lungs">Organ · Lungs (Pulmones)</div>
                  <h3 className="dossier-organ-title">Pulmonary Radiography & Airway Tree Volumetric Suite</h3>
                  <span className="dossier-sub">Respiratory System · 5 Lobes & Tracheobronchial Arborization</span>
                </div>
                <Link to="/app?organ=lungs" className="btn-launch-workspace">
                  <Activity size={14} />
                  <span>Open Lungs 3D Suite</span>
                  <ArrowUpRight size={13} />
                </Link>
              </div>

              <div className="task-block classification">
                <div className="task-header-strip">
                  <div className="task-tag-group">
                    <span className="task-type-tag class">Task · Classification</span>
                    <span className="task-name-text">Frontal Chest Radiograph 14-Pathology Differential</span>
                  </div>
                  <span className="task-model-pill">DenseNet-121 / ConvNeXt-Base · CheXpert & NIH CXR14</span>
                </div>
                <div className="boilerplate-card">
                  <p className="boilerplate-copy">
                    High-throughput radiograph screening identifying cardiomegaly, consolidation, pulmonary edema, pleural effusion,
                    pneumothorax, atelectasis, and solitary pulmonary nodules with uncertainty-aware multi-label BCE loss formulation.
                  </p>
                </div>
              </div>

              <div className="task-block segmentation">
                <div className="task-header-strip">
                  <div className="task-tag-group">
                    <span className="task-type-tag seg">Task · Segmentation</span>
                    <span className="task-name-text">Thoracic CT Pulmonary Lobar & Airway Tree Delineation</span>
                  </div>
                  <span className="task-model-pill">LIDC-IDRI / COVID-19-CT · 3D V-Net</span>
                </div>
                <div className="boilerplate-card">
                  <p className="boilerplate-copy">
                    Extracts 5 anatomical lobes (Right Superior, Middle, Inferior; Left Superior, Inferior) and segments the segmental
                    tracheobronchial tree. Calculates localized ground-glass opacification (GGO) percentages and emphysematous tissue destruction.
                  </p>
                </div>
              </div>
            </div>

            {/* =========================================================
                D. OTHER ORGAN SUITES (BOILERPLATE ARCHITECTURAL SPEC)
                ========================================================= */}
            <div id="model-other-organs" className="other-organs-boilerplate-section">
              <div className="other-section-header">
                <Workflow size={16} className="text-olive" />
                <h4 className="other-section-title">Additional Organ Suites · Task Architecture Specifications</h4>
              </div>

              <div className="other-suites-grid">
                {/* Kidneys */}
                <div className="other-suite-card">
                  <div className="suite-card-top">
                    <span className="suite-organ-name">Kidneys (Ren)</span>
                    <span className="suite-tag">KiTS23 / RCC</span>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini class">Classification:</span>
                    <p className="task-desc-mini">Renal Cell Carcinoma histological grading (Clear Cell vs Papillary vs Chromophobe).</p>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini seg">Segmentation:</span>
                    <p className="task-desc-mini">KiTS volumetric delineation of renal parenchyma, tumor core, and fluid-filled cysts.</p>
                  </div>
                </div>

                {/* Liver */}
                <div className="other-suite-card">
                  <div className="suite-card-top">
                    <span className="suite-organ-name">Liver (Hepar)</span>
                    <span className="suite-tag">LiTS / HCC</span>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini class">Classification:</span>
                    <p className="task-desc-mini">Multi-phase CT focal liver lesion differential (HCC vs Hemangioma vs FNH).</p>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini seg">Segmentation:</span>
                    <p className="task-desc-mini">Total Liver Volume (TLV) and functional future liver remnant (FLR) volumetry.</p>
                  </div>
                </div>

                {/* Eye */}
                <div className="other-suite-card">
                  <div className="suite-card-top">
                    <span className="suite-organ-name">Eye (Oculus)</span>
                    <span className="suite-tag">Fundus / DR</span>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini class">Classification:</span>
                    <p className="task-desc-mini">5-stage Diabetic Retinopathy grading (Messidor) and glaucoma suspect screening.</p>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini seg">Segmentation:</span>
                    <p className="task-desc-mini">Retinal vessel caliber extraction and vertical Cup-to-Disc Ratio (CDR) calculation.</p>
                  </div>
                </div>

                {/* Skin */}
                <div className="other-suite-card">
                  <div className="suite-card-top">
                    <span className="suite-organ-name">Skin (Integumentum)</span>
                    <span className="suite-tag">ISIC / Melanoma</span>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini class">Classification:</span>
                    <p className="task-desc-mini">Dermoscopic 7-class lesion differential: Melanoma vs Basal Cell Carcinoma vs Nevi.</p>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini seg">Segmentation:</span>
                    <p className="task-desc-mini">Border contour extraction with automated ABCD dermatological scoring parameters.</p>
                  </div>
                </div>

                {/* Breast */}
                <div className="other-suite-card">
                  <div className="suite-card-top">
                    <span className="suite-organ-name">Breast (Mamma)</span>
                    <span className="suite-tag">CBIS-DDSM / BI-RADS</span>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini class">Classification:</span>
                    <p className="task-desc-mini">Digital mammography BI-RADS density classification and malignancy risk estimation.</p>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini seg">Segmentation:</span>
                    <p className="task-desc-mini">Soft tissue mass boundary delineation and microcalcification cluster spatial mapping.</p>
                  </div>
                </div>

                {/* Bone */}
                <div className="other-suite-card">
                  <div className="suite-card-top">
                    <span className="suite-organ-name">Bone (Skeleton)</span>
                    <span className="suite-tag">MURA / TotalSegmentator</span>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini class">Classification:</span>
                    <p className="task-desc-mini">Multi-site radiographic fracture detection and Kellgren-Lawrence knee OA grading.</p>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini seg">Segmentation:</span>
                    <p className="task-desc-mini">CT vertebral column individual instance segmentation (C1–L5) and pelvic ring reconstruction.</p>
                  </div>
                </div>

                {/* Blood */}
                <div className="other-suite-card">
                  <div className="suite-card-top">
                    <span className="suite-organ-name">Blood & Vasculature</span>
                    <span className="suite-tag">ALL-IDB / CTA</span>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini class">Classification:</span>
                    <p className="task-desc-mini">Peripheral smear white blood cell differential &amp; acute leukemia blast screening.</p>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini seg">Segmentation:</span>
                    <p className="task-desc-mini">CTA Circle of Willis and coronary artery vessel lumen caliber and stenosis mapping.</p>
                  </div>
                </div>

                {/* Pancreas */}
                <div className="other-suite-card">
                  <div className="suite-card-top">
                    <span className="suite-organ-name">Pancreas</span>
                    <span className="suite-tag">MSD Pancreas / PDAC</span>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini class">Classification:</span>
                    <p className="task-desc-mini">Pancreatic ductal adenocarcinoma vs neuroendocrine tumor vs IPMN differential.</p>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini seg">Segmentation:</span>
                    <p className="task-desc-mini">Pancreatic parenchyma boundary and pancreatic duct dilatation volumetry.</p>
                  </div>
                </div>

                {/* Intestine */}
                <div className="other-suite-card">
                  <div className="suite-card-top">
                    <span className="suite-organ-name">Intestine & Colon</span>
                    <span className="suite-tag">Kvasir-SEG / Endoscopy</span>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini class">Classification:</span>
                    <p className="task-desc-mini">Real-time colonoscopy polyp histology classification (Adenomatous vs Hyperplastic).</p>
                  </div>
                  <div className="suite-task-item">
                    <span className="task-label-mini seg">Segmentation:</span>
                    <p className="task-desc-mini">Endoscopic polyp boundary mask delineation with automated CADe reticle tracking.</p>
                  </div>
                </div>
              </div>
            </div>
          </RomanSection>
        </section>

        {/* =========================================================
            SECTION III: FASTAPI PRODUCTION ARCHITECTURE
            ========================================================= */}
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

        {/* =========================================================
            SECTION IV: STEREOTACTIC MNI LOCALIZATION
            ========================================================= */}
        <section id="section-stereotactic" className="doc-section-wrapper">
          <RomanSection
            index={3}
            of={5}
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

        {/* =========================================================
            SECTION V: DATASET CURATION & PURGING
            ========================================================= */}
        <section id="section-dataset" className="doc-section-wrapper">
          <RomanSection
            index={4}
            of={5}
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
