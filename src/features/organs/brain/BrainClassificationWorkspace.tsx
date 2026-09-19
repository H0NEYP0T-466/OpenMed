import React, { useState, useRef, useEffect } from 'react'
import { Link } from 'react-router-dom'
import {
  UploadCloud,
  Sparkles,
  Activity,
  Brain as BrainIcon,
  ShieldCheck,
  ArrowUpRight,
} from 'lucide-react'
import { RomanSection } from '../../../components/common/RomanSection'
import { BrainRegionViewer } from './BrainRegionViewer'
import { classifyBrainImage, getBrainHealth, getBrainModelInfo } from './brainApi'
import type { BrainClassificationResult, BrainModelInfo } from './brainTypes'
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

type DossierViewMode = 'all' | 'diagnostic' | 'anatomical'

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
  const [modelInfo, setModelInfo] = useState<BrainModelInfo | null>(null)
  const [serviceDetail, setServiceDetail] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    let cancelled = false

    getBrainHealth()
      .then((health) => {
        if (cancelled) return
        setApiReady(health.status === 'ok')
        setServiceDetail(health.detail ?? null)
      })
      .catch(() => {
        if (!cancelled) setApiReady(false)
      })

    getBrainModelInfo()
      .then((info) => {
        if (!cancelled) setModelInfo(info)
      })
      .catch(() => {
        /* model copy falls back to static defaults below */
      })

    return () => {
      cancelled = true
    }
  }, [])

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

  const tumorInfo = result?.tumor_type ? TUMOR_DATA[result.tumor_type] : null
  const profile = organ?.clinicalProfile

  const classCount = modelInfo?.num_classes ?? null
  const classLabel = classCount === null ? 'multi-class' : `${classCount}-class`
  const classCountLabel = classCount === null ? 'classes' : `${classCount} classes`
  const inputLabel = modelInfo?.input_size ?? 'model-native'
  const isSimulated = result?.simulated === true

  // Total sections count depends on whether inference results are loaded and view mode
  const totalSections =
    viewMode === 'all'
      ? (result ? 9 : 5)
      : viewMode === 'anatomical'
        ? 4
        : (result ? 5 : 1)

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
            className={`dossier-pill ${viewMode === 'anatomical' ? 'active' : ''}`}
            onClick={() => setViewMode('anatomical')}
          >
            Anatomical Monograph
          </button>
          <Link
            to="/docs"
            className="dossier-pill dossier-pill-link"
            title="Inspect complete training ledger, Kaggle execution manifest, and evaluation artifacts"
          >
            <span>Academic Benchmarks & Ledger</span>
            <ArrowUpRight size={12} className="pill-ext-icon" />
          </Link>
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
                        <span className="meta-sub">Input tensor resolved to {inputLabel} · Model ready for {classLabel} evaluation</span>
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
                            {isAnalyzing ? `Evaluating ${classCountLabel}…` : 'Run EfficientNetV2-B2 Differential'}
                          </span>
                          <span className="arr" aria-hidden="true">↗</span>
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {!apiReady && serviceDetail && (
                <div className="service-unavailable-banner" role="status">
                  <span className="service-unavailable-title">Analysis service unavailable</span>
                  <span className="service-unavailable-copy">{serviceDetail}</span>
                </div>
              )}

              {error && <div className="lightbox-error-banner">{error}</div>}
            </RomanSection>

            {/* Diagnostic Inference Results (When Evaluated) */}
            {result && (
              <>
                {/* VI - Diagnostic Finding */}
                <RomanSection
                  index={viewMode === 'all' ? 5 : 1}
                  of={totalSections}
                  title={`Diagnostica - ${classLabel} Histological Finding`}
                  className="sp7"
                >
                  <div className="diagnostic-readout-card">
                    <div className="diag-header-block">
                      <span className="diag-eyebrow">Automated Differential Finding</span>
                      <h3 className="diag-headline">{result.predicted_class}</h3>
                      {isSimulated && (
                        <p className="simulated-result-flag" role="status">
                          Simulated specimen preset. The analysis service was unreachable, so this
                          differential was not computed from the uploaded image.
                        </p>
                      )}
                      {result.model_trained === false && !isSimulated && (
                        <p className="simulated-result-flag" role="status">
                          Trained checkpoint not loaded — output is not clinically meaningful.
                        </p>
                      )}
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
                      <span className="diff-header">Differential Ranking (Top 5 of {classCount ?? result.top5.length})</span>
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
                      <span className="scan-tag highlight">
                        {result.gradcam_base64
                          ? `CAM (${result.explainability.layer})`
                          : 'Illustrative overlay — no activation computed'}
                      </span>
                    </div>
                  </div>
                </RomanSection>

                {/* VIII - Locus Anatomical Site */}
                <RomanSection
                  index={viewMode === 'all' ? 7 : 3}
                  of={totalSections}
                  title="Locus - Typical Presentation Sites & Approximate Centroid"
                  className="sp7"
                >
                  <BrainRegionViewer
                    locations={result.locations_3d}
                    basis={result.localization_basis}
                    tumourType={result.tumor_type}
                  />
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

      </div>

      {/* Academic Documentation Callout Banner */}
      <div className="docs-callout-banner">
        <div className="docs-callout-copy">
          <span className="docs-callout-eyebrow">Academic Benchmarks &amp; System Architecture</span>
          <p className="docs-callout-text">
            Looking for grouped-split training logs, convergence trajectories, a {classLabel} confusion matrix, MNI stereotactic mapping, or FastAPI service specs?
          </p>
        </div>
        <Link to="/docs" className="docs-callout-btn">
          <span>Explore Documentation</span>
          <ArrowUpRight size={14} />
        </Link>
      </div>

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
