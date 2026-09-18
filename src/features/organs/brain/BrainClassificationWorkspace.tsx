import React, { useState, useRef, useEffect } from 'react'
import {
  UploadCloud,
  Sparkles,
  Copy,
  Check,
  Layers,
  Activity,
  Brain as BrainIcon,
  ShieldCheck,
  Database,
  Sliders,
  X,
  ExternalLink,
  Download,
} from 'lucide-react'
import { RomanSection } from '../../../components/common/RomanSection'
import { BrainRegionViewer } from './BrainRegionViewer'
import { classifyBrainImage, getBrainHealth } from './brainApi'
import type { BrainClassificationResult } from './brainTypes'
import { TUMOR_DATA } from './tumorData'
import type { OrganMetadata, Hotspot } from '../../../types/organ'
import './BrainClassificationWorkspace.css'

export interface BrainClassificationWorkspaceProps {
  readonly organ?: OrganMetadata
  readonly activeHotspot?: Hotspot | null
  readonly onSelectHotspot?: (hotspot: Hotspot | null) => void
}

interface SpecimenPreset {
  readonly plate: string
  readonly name: string
  readonly path: string
  readonly seq: string
  readonly subtitle: string
  readonly site: string
}

const SPECIMEN_PRESETS: readonly SpecimenPreset[] = [
  {
    plate: 'Pl. A',
    name: 'Meningioma',
    path: '/samples/brain/Meningioma_T1Cplus.jpg',
    seq: 'T1C+',
    subtitle: 'Extra-axial Dural Mass',
    site: 'Anterior Fossa / Frontal',
  },
  {
    plate: 'Pl. B',
    name: 'Glioblastoma',
    path: '/samples/brain/Glioblastoma_T1Cplus.jpg',
    seq: 'T1C+',
    subtitle: 'High-Grade Infiltrative Glioma',
    site: 'Occipital / Ventricle',
  },
  {
    plate: 'Pl. C',
    name: 'Astrocytoma',
    path: '/samples/brain/Astrocytoma_T1.jpg',
    seq: 'T1',
    subtitle: 'Diffuse Astrocytic Neoplasm',
    site: 'Temporal White Matter',
  },
  {
    plate: 'Pl. D',
    name: 'Medulloblastoma',
    path: '/samples/brain/Medulloblastoma_T1.jpg',
    seq: 'T1',
    subtitle: 'Posterior Fossa Embryonal',
    site: 'Cerebellar Vermis / Brainstem',
  },
]

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

type DossierViewMode = 'all' | 'diagnostic' | 'benchmarks' | 'anatomical'

const romanOfSpot = (i: number): string =>
  ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'][i] ?? String(i + 1)

export const BrainClassificationWorkspace: React.FC<BrainClassificationWorkspaceProps> = ({
  organ,
  activeHotspot,
  onSelectHotspot,
}) => {
  const [viewMode, setViewMode] = useState<DossierViewMode>('all')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<BrainClassificationResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [apiReady, setApiReady] = useState<boolean>(false)
  const [copiedCmd, setCopiedCmd] = useState(false)
  const [activeArtifact, setActiveArtifact] = useState<ArtifactCard | null>(null)
  const [artifactTextContent, setArtifactTextContent] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    getBrainHealth()
      .then(() => setApiReady(true))
      .catch(() => setApiReady(false))
  }, [])

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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0]
    if (selected) {
      if (!selected.type.startsWith('image/')) {
        setError('Selected file is not an image (JPEG, PNG).')
        return
      }
      setFile(selected)
      setPreviewUrl(URL.createObjectURL(selected))
      setSelectedPreset(null)
      setResult(null)
      setError(null)
    }
  }

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    const dropped = e.dataTransfer.files?.[0]
    if (dropped) {
      if (!dropped.type.startsWith('image/')) {
        setError('Dropped file is not an image (JPEG, PNG).')
        return
      }
      setFile(dropped)
      setPreviewUrl(URL.createObjectURL(dropped))
      setSelectedPreset(null)
      setResult(null)
      setError(null)
    }
  }

  const handleLoadPreset = async (preset: SpecimenPreset) => {
    setError(null)
    setResult(null)
    setSelectedPreset(preset.plate)
    try {
      const resp = await fetch(preset.path)
      const blob = await resp.blob()
      const sampleFile = new File([blob], `${preset.name}_${preset.seq}.jpg`, {
        type: 'image/jpeg',
      })
      setFile(sampleFile)
      setPreviewUrl(preset.path)
    } catch {
      setError(`Unable to load preset specimen at ${preset.path}`)
    }
  }

  const handleAnalyze = async () => {
    if (!file) return

    setIsAnalyzing(true)
    setError(null)
    try {
      const res = await classifyBrainImage(file)
      setResult(res)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Inference failed.'
      setError(msg)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleCopyKaggleCmd = () => {
    const cmd =
      'python brain_kaggle.py --data_root /kaggle/input/brain-tumor-dataset/archive --output_dir /kaggle/working --batch_size 32 --epochs 50'
    navigator.clipboard.writeText(cmd)
    setCopiedCmd(true)
    setTimeout(() => setCopiedCmd(false), 2200)
  }

  const tumorInfo = result?.tumor_type ? TUMOR_DATA[result.tumor_type] : null
  const profile = organ?.clinicalProfile

  // Total sections count depends on whether inference results are loaded and view mode
  const totalSections = viewMode === 'all' ? (result ? 10 : 6) : viewMode === 'anatomical' ? 4 : viewMode === 'diagnostic' ? (result ? 5 : 1) : 1

  return (
    <article className="organ-essay brain-master-dossier">
      {/* Dossier Editorial View Controller */}
      <nav className="dossier-view-nav" aria-label="Dossier Section Filter">
        <div className="dossier-nav-title">
          <BrainIcon size={14} className="dossier-nav-icon" />
          <span>Neuro-Oncology Atlas & Neural Suite</span>
        </div>
        <div className="dossier-view-pills">
          <button
            type="button"
            className={`dossier-pill ${viewMode === 'all' ? 'active' : ''}`}
            onClick={() => setViewMode('all')}
          >
            Complete Dossier
          </button>
          <button
            type="button"
            className={`dossier-pill ${viewMode === 'diagnostic' ? 'active' : ''}`}
            onClick={() => setViewMode('diagnostic')}
          >
            Diagnostic Suite {result && <span className="pill-dot" />}
          </button>
          <button
            type="button"
            className={`dossier-pill ${viewMode === 'benchmarks' ? 'active' : ''}`}
            onClick={() => setViewMode('benchmarks')}
          >
            Academic Benchmarks
          </button>
          <button
            type="button"
            className={`dossier-pill ${viewMode === 'anatomical' ? 'active' : ''}`}
            onClick={() => setViewMode('anatomical')}
          >
            Anatomical Monograph
          </button>
        </div>
      </nav>

      <div className="dossier-grid">
        {/* =========================================================
            PART A: ANATOMICAL MONOGRAPH (I - IV)
            ========================================================= */}
        {(viewMode === 'all' || viewMode === 'anatomical') && profile && (
          <>
            {/* I - Observatio */}
            <RomanSection index={0} of={totalSections} title="Observatio - Physiology" className="sp7">
              <p className="sec-copy">{profile.physiology}</p>
              {profile.medicalNote && <p className="sec-copy note">{profile.medicalNote}</p>}
            </RomanSection>

            {/* II - Constantia */}
            <RomanSection index={1} of={totalSections} title="Constantia - Physiological Fact" className="sp5">
              <figure className="pull-quote">
                <span className="pq-mark" aria-hidden="true">“</span>
                <p>{profile.dailyFact}</p>
                <figcaption>Nº Dies - {profile.system}</figcaption>
              </figure>
            </RomanSection>

            {/* III - Landmarks */}
            {organ && organ.hotspots.length > 0 && (
              <RomanSection
                index={2}
                of={totalSections}
                title={`Landmarks - ${organ.hotspots.length} Anatomical Sites`}
                className="sp12"
              >
                <p className="sec-hint">Select a landmark below to focus camera on the 3D anatomical plate above.</p>
                <div className="lm-list">
                  {organ.hotspots.map((spot, i) => {
                    const isSelected = activeHotspot?.id === spot.id
                    return (
                      <button
                        key={spot.id}
                        type="button"
                        className={`lm-item ${isSelected ? 'selected' : ''}`}
                        onClick={() => onSelectHotspot?.(isSelected ? null : spot)}
                        aria-pressed={isSelected}
                      >
                        <span className="lm-idx">{romanOfSpot(i)}</span>
                        <span className="lm-swatch" style={{ backgroundColor: spot.color }} />
                        <span className="lm-body">
                          <span className="lm-name">
                            {spot.label}
                            <span className="latin">{spot.latinTerm}</span>
                          </span>
                          <span className="lm-detail">{spot.detail}</span>
                        </span>
                      </button>
                    )
                  })}
                </div>
              </RomanSection>
            )}

            {/* IV - Vasa & Pathologiae */}
            <RomanSection
              index={3}
              of={totalSections}
              title="Vasa & Pathologiae - Cerebrovascular Circulation & Differential Spectrum"
              className="sp12"
            >
              <div className="vasa-layout-grid">
                <div className="vasa-card">
                  <div className="vasa-card-head">
                    <Activity size={15} className="vasa-icon" />
                    <span className="vasa-card-title">Cerebrovascular Perfusion (Circle of Willis)</span>
                  </div>
                  <p className="vasa-card-copy">{profile.bloodSupply}</p>
                  <div className="vasa-meta-tags">
                    <span className="vasa-tag">Internal Carotid Arterial Axis</span>
                    <span className="vasa-tag">Vertebrobasilar Perfusion Territory</span>
                    <span className="vasa-tag">Blood-Brain Barrier (BBB) Dynamics</span>
                  </div>
                </div>

                <div className="vasa-card">
                  <div className="vasa-card-head">
                    <ShieldCheck size={15} className="vasa-icon" />
                    <span className="vasa-card-title">Differential Pathology Spectrum</span>
                  </div>
                  <div className="pathology-matrix">
                    {profile.commonConditions.map((cond) => (
                      <div key={cond} className="pathology-pill">
                        <span className="pathology-bullet" />
                        <span className="pathology-label">{cond}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </RomanSection>
          </>
        )}

        {/* =========================================================
            PART B: DIAGNOSTIC SUITE & RADIOLOGICAL INGESTION (V - VII)
            ========================================================= */}
        {(viewMode === 'all' || viewMode === 'diagnostic') && (
          <>
            {/* V - Scan Ingestion & Specimen Lightbox */}
            <RomanSection
              index={viewMode === 'all' ? 4 : 0}
              of={totalSections}
              title="Ingestio - MRI Scan Acquisition & Specimen Cassettes"
              className="sp12"
            >
              {/* Telemetry Header */}
              <div className="ingest-telemetry-strip">
                <div className="ingest-spec-group">
                  <span className="ingest-label">Modality:</span>
                  <span className="ingest-val">Cranial MRI (Axial)</span>
                  <span className="ingest-sep">•</span>
                  <span className="ingest-label">Matrix:</span>
                  <span className="ingest-val">512 × 512 px</span>
                  <span className="ingest-sep">•</span>
                  <span className="ingest-label">Sequences:</span>
                  <span className="ingest-val">T1, T1C+, T2</span>
                </div>

                <div className="ingest-status-badge">
                  {apiReady ? (
                    <span className="status-pill online">
                      <span className="status-dot" /> Live Neural Engine (Port 8016)
                    </span>
                  ) : (
                    <span className="status-pill offline" title="FastAPI server offline. Running on verified local specimen cache.">
                      <span className="status-dot" /> Demonstration Archive (Local Cache)
                    </span>
                  )}
                </div>
              </div>

              {/* Tactile Specimen Cassette Rack */}
              <div className="specimen-cassette-rack">
                <div className="rack-header">
                  <span className="rack-title">Curated Specimen Cassettes</span>
                  <span className="rack-sub">Click a clinical case plate to mount onto the radiology lightbox</span>
                </div>
                <div className="cassette-grid">
                  {SPECIMEN_PRESETS.map((preset) => {
                    const isSelected = selectedPreset === preset.plate
                    return (
                      <button
                        key={preset.plate}
                        type="button"
                        className={`cassette-card ${isSelected ? 'active' : ''}`}
                        onClick={() => handleLoadPreset(preset)}
                      >
                        <div className="cassette-thumbnail-box">
                          <img src={preset.path} alt={preset.name} className="cassette-thumbnail" />
                          <span className="cassette-seq-badge">{preset.seq}</span>
                        </div>
                        <div className="cassette-info">
                          <div className="cassette-meta">
                            <span className="cassette-plate">{preset.plate}</span>
                            <span className="cassette-site">{preset.site}</span>
                          </div>
                          <div className="cassette-name">{preset.name}</div>
                          <div className="cassette-sub">{preset.subtitle}</div>
                        </div>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Main Radiological Lightbox */}
              <div
                className={`radiology-lightbox ${previewUrl ? 'has-scan' : ''} ${isAnalyzing ? 'analyzing' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={handleDragOver}
                onDrop={handleDrop}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  className="hidden-input"
                  accept="image/jpeg,image/png"
                />

                {/* Reticle marks */}
                <div className="reticle corner-tl" aria-hidden="true" />
                <div className="reticle corner-tr" aria-hidden="true" />
                <div className="reticle corner-bl" aria-hidden="true" />
                <div className="reticle corner-br" aria-hidden="true" />

                {!previewUrl ? (
                  <div className="lightbox-prompt">
                    <div className="prompt-reticle-circle">
                      <UploadCloud size={30} className="prompt-icon" />
                    </div>
                    <span className="prompt-headline">Drop Brain MRI Scan or Browse File</span>
                    <span className="prompt-meta">
                      Accepts 512×512 Axial T1, T1C+, or T2 DICOM / JPEG series • Or select a Specimen Cassette above
                    </span>
                    <span className="prompt-btn">Browse Local Drive</span>
                  </div>
                ) : (
                  <div className="lightbox-mounted-view">
                    <div className="film-viewport">
                      <img src={previewUrl} alt="Mounted MRI Scan" className="film-image" />
                      <div className="film-overlay-hud">
                        <span className="hud-tag top-left">AXIAL T1/T2</span>
                        <span className="hud-tag top-right">FOV 240mm</span>
                        <span className="hud-tag btm-left">{file?.name ?? 'Specimen Scan'}</span>
                        <span className="hud-tag btm-right">{file ? `${(file.size / 1024).toFixed(1)} KB` : '512×512'}</span>
                      </div>
                    </div>

                    <div className="film-action-bar">
                      <div className="film-meta-copy">
                        <span className="meta-title">{file?.name ?? 'Loaded Specimen'}</span>
                        <span className="meta-sub">Input tensor resolved to 512×512 · Model ready for 39-class evaluation</span>
                      </div>

                      <div className="film-actions">
                        <button
                          type="button"
                          className="btn btn-primary btn-evaluate"
                          onClick={(e) => {
                            e.stopPropagation()
                            handleAnalyze()
                          }}
                          disabled={isAnalyzing}
                        >
                          <span>
                            <Sparkles size={15} />
                            {isAnalyzing ? 'Evaluating 39 Classes…' : 'Run EfficientNetV2-B2 Differential'}
                          </span>
                          <span className="arr" aria-hidden="true">↗</span>
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {error && <div className="lightbox-error-banner">{error}</div>}
            </RomanSection>

            {/* Diagnostic Inference Results (When Evaluated) */}
            {result && (
              <>
                {/* VI - Diagnostic Finding */}
                <RomanSection
                  index={viewMode === 'all' ? 5 : 1}
                  of={totalSections}
                  title="Diagnostica - 39-Class Histological Finding"
                  className="sp7"
                >
                  <div className="diagnostic-readout-card">
                    <div className="diag-header-block">
                      <span className="diag-eyebrow">Automated Differential Finding</span>
                      <h3 className="diag-headline">{result.predicted_class}</h3>
                    </div>

                    <div className="diag-meta-strip">
                      <div className="meta-chip">
                        <span className="k">Histological Family</span>
                        <span className="v">{result.tumor_type}</span>
                      </div>
                      <div className="meta-chip">
                        <span className="k">MRI Sequence</span>
                        <span className="v">{result.sequence}</span>
                      </div>
                      <div className="meta-chip">
                        <span className="k">Backbone</span>
                        <span className="v">EfficientNetV2-B2</span>
                      </div>
                    </div>

                    <div className="confidence-gauge">
                      <div className="gauge-label">
                        <span>Classification Confidence</span>
                        <b>{(result.confidence * 100).toFixed(1)}%</b>
                      </div>
                      <div className="gauge-track">
                        <div
                          className="gauge-fill"
                          style={{ width: `${Math.min(result.confidence * 100, 100)}%` }}
                        />
                      </div>
                    </div>

                    <div className="differential-ranking">
                      <span className="diff-header">Differential Ranking (Top 5 of 39 Classes)</span>
                      <div className="diff-rows-container">
                        {result.top5.map((pred, i) => (
                          <div key={pred.class} className="diff-row">
                            <div className="diff-text">
                              <span className="diff-name">{i + 1}. {pred.class}</span>
                              <span className="diff-val">{(pred.confidence * 100).toFixed(1)}%</span>
                            </div>
                            <div className="diff-track">
                              <div
                                className="diff-fill"
                                style={{
                                  width: `${Math.min(pred.confidence * 100, 100)}%`,
                                  backgroundColor: i === 0 ? 'var(--accent)' : 'var(--ink-mute)',
                                }}
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </RomanSection>

                {/* VII - Attentio / Grad-CAM */}
                <RomanSection
                  index={viewMode === 'all' ? 6 : 2}
                  of={totalSections}
                  title="Attentio - Grad-CAM Feature Activation"
                  className="sp5"
                >
                  <div className="gradcam-dual-plate">
                    {previewUrl && (
                      <div className="scan-frame">
                        <img src={previewUrl} alt="Source MRI Scan" />
                        <span className="scan-tag">Source MRI</span>
                      </div>
                    )}
                    <div className="scan-frame">
                      {result.gradcam_base64 ? (
                        <img src={result.gradcam_base64} alt="Grad-CAM Activation" />
                      ) : (
                        <div className="simulated-cam-plate">
                          {previewUrl && <img src={previewUrl} alt="Base" className="underlay-img" />}
                          <div className="cam-glow-layer" />
                        </div>
                      )}
                      <span className="scan-tag highlight">CAM (conv_head)</span>
                    </div>
                  </div>
                </RomanSection>

                {/* VIII - Locus Anatomical Site */}
                <RomanSection
                  index={viewMode === 'all' ? 7 : 3}
                  of={totalSections}
                  title="Locus - Anatomical Localization & Stereotactic Centroid"
                  className="sp7"
                >
                  <BrainRegionViewer locations={result.locations_3d} />
                </RomanSection>

                {/* IX - Monograph */}
                <RomanSection
                  index={viewMode === 'all' ? 8 : 4}
                  of={totalSections}
                  title="Monograph - Clinical Tumor Dossier"
                  className="sp5"
                >
                  <div className="monograph-box">
                    {tumorInfo ? (
                      <>
                        <h4 className="monograph-title">{tumorInfo.name}</h4>
                        <p className="monograph-copy">{tumorInfo.description}</p>

                        <div className="monograph-grid">
                          <div className="monograph-cell">
                            <span className="mono-cell-k">WHO Grade</span>
                            <span className="mono-cell-v">{tumorInfo.grading}</span>
                          </div>
                          <div className="monograph-cell">
                            <span className="mono-cell-k">Typical Sites</span>
                            <span className="mono-cell-v">{tumorInfo.commonLocations.slice(0, 2).join(', ')}</span>
                          </div>
                          <div className="monograph-cell full-span">
                            <span className="mono-cell-k">MRI Signal Characteristics</span>
                            <span className="mono-cell-v">{tumorInfo.mriCharacteristics}</span>
                          </div>
                          <div className="monograph-cell full-span">
                            <span className="mono-cell-k">Clinical Prognosis</span>
                            <span className="mono-cell-v">{tumorInfo.prognosis}</span>
                          </div>
                        </div>
                      </>
                    ) : (
                      <p className="monograph-copy">No specific clinical monograph catalogued for this classification class.</p>
                    )}
                  </div>
                </RomanSection>
              </>
            )}
          </>
        )}

        {/* =========================================================
            PART C: ACADEMIC BENCHMARKS & KAGGLE TRAINING (DISCIPLINA)
            ========================================================= */}
        {(viewMode === 'all' || viewMode === 'benchmarks') && (
          <RomanSection
            index={viewMode === 'all' ? (result ? 9 : 5) : 0}
            of={totalSections}
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
                    <span className="vault-sub">Packaged automatically into brain_classification_results.zip upon training completion</span>
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
        )}
      </div>

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

      {/* Unified Dossier Colophon Footer */}
      <footer className="dossier-foot">
        <span>
          Asset - brain.glb · Central Nervous System (CNS) · OpenMed Neuro-Oncology Suite
        </span>
        <span className="fin">fin.</span>
      </footer>
    </article>
  )
}
