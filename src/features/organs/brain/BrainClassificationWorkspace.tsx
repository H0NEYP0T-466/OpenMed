import React, { useState, useRef, useEffect } from 'react'
import { UploadCloud, Sparkles } from 'lucide-react'
import { RomanSection } from '../../../components/common/RomanSection'
import { BrainRegionViewer } from './BrainRegionViewer'
import { classifyBrainImage, getBrainHealth } from './brainApi'
import type { BrainClassificationResult } from './brainTypes'
import { TUMOR_DATA } from './tumorData'
import './BrainClassificationWorkspace.css'

interface SpecimenPreset {
  readonly plate: string
  readonly name: string
  readonly path: string
  readonly seq: string
}

const SPECIMEN_PRESETS: readonly SpecimenPreset[] = [
  {
    plate: 'Pl. A',
    name: 'Meningioma',
    path: '/samples/brain/Meningioma_T1Cplus.jpg',
    seq: 'T1C+',
  },
  {
    plate: 'Pl. B',
    name: 'Glioblastoma',
    path: '/samples/brain/Glioblastoma_T1Cplus.jpg',
    seq: 'T1C+',
  },
  {
    plate: 'Pl. C',
    name: 'Astrocytoma',
    path: '/samples/brain/Astrocytoma_T1.jpg',
    seq: 'T1',
  },
  {
    plate: 'Pl. D',
    name: 'Medulloblastoma',
    path: '/samples/brain/Medulloblastoma_T1.jpg',
    seq: 'T1',
  },
]

export const BrainClassificationWorkspace: React.FC = () => {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<BrainClassificationResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [apiReady, setApiReady] = useState<boolean>(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    getBrainHealth()
      .then(() => setApiReady(true))
      .catch(() => setApiReady(false))
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

  return (
    <article className="organ-essay brain-classification-essay">
      {/* Specimen Presets Bar */}
      <div className="specimen-presets-bar">
        <span className="specimen-lead-tag">Specimen Cases</span>
        {SPECIMEN_PRESETS.map((preset) => (
          <button
            key={preset.plate}
            type="button"
            className={`specimen-chip ${selectedPreset === preset.plate ? 'active' : ''}`}
            onClick={() => handleLoadPreset(preset)}
          >
            <span>{preset.plate} · {preset.name}</span>
            <span className="seq-badge">{preset.seq}</span>
          </button>
        ))}
      </div>

      <div className="dossier-grid">
        {/* VII — Scan Ingestion & Analysis */}
        <RomanSection index={6} of={9} title="Ingestio — MRI Scan Acquisition" className="sp12">
          {!apiReady && (
            <div className="error-message">
              FastAPI backend service is offline at <code>http://localhost:8016</code>. Run <code>python -m app.main</code> to enable live AI inference.
            </div>
          )}

          <div
            className={`scan-dropzone ${isAnalyzing ? 'is-loading' : ''}`}
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

            {!previewUrl ? (
              <div className="dropzone-prompt">
                <UploadCloud size={28} className="dropzone-icon" />
                <span className="dropzone-title">Drop Brain MRI Scan or Browse File</span>
                <span className="dropzone-sub">512×512 Matrix · T1, T1C+, or T2-Weighted Series</span>
              </div>
            ) : (
              <div className="scan-preview-flow">
                <img src={previewUrl} alt="Acquired MRI Scan" className="scan-preview-img" />
                <span className="scan-preview-caption">
                  {file?.name} · {file ? `${(file.size / 1024).toFixed(1)} KB` : ''}
                </span>

                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={(e) => {
                    e.stopPropagation()
                    handleAnalyze()
                  }}
                  disabled={isAnalyzing || !apiReady}
                >
                  <span>
                    <Sparkles size={14} />
                    {isAnalyzing ? 'Evaluating 39 Classes…' : 'Classify Scan (EfficientNetV2-B2)'}
                  </span>
                  <span className="arr" aria-hidden="true">↗</span>
                </button>
              </div>
            )}
          </div>

          {error && <div className="error-message">{error}</div>}
        </RomanSection>

        {/* Diagnostic Results & Visual Attention Map */}
        {result && (
          <>
            {/* VIII — Diagnostic Findings & Grad-CAM */}
            <RomanSection index={7} of={9} title="Diagnosis — 39-Class Neuro-Oncology Finding" className="sp7">
              <div className="diagnostic-readout-card">
                <div>
                  <div className="diag-eyebrow">Automated Differential Finding</div>
                  <h3 className="diag-headline">{result.predicted_class}</h3>
                </div>

                <div className="diag-meta-strip">
                  <span>Subtype: <b>{result.tumor_type}</b></span>
                  <span>Modality: <b>MRI {result.sequence}</b></span>
                  <span>Architecture: <b>EfficientNetV2-B2</b></span>
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
                  <span className="diff-header">Differential Ranking (Top 5 of 39)</span>
                  {result.top5.map((pred, i) => (
                    <div key={pred.class} className="diff-row">
                      <div className="diff-text">
                        <span>{i + 1}. {pred.class}</span>
                        <span>{(pred.confidence * 100).toFixed(1)}%</span>
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
            </RomanSection>

            <RomanSection index={7} of={9} title="Attentio — Grad-CAM Feature Map" className="sp5">
              <div className="gradcam-dual-plate">
                {previewUrl && (
                  <div className="scan-frame">
                    <img src={previewUrl} alt="Source MRI Scan" />
                    <span className="scan-tag">Source MRI</span>
                  </div>
                )}
                {result.gradcam_base64 && (
                  <div className="scan-frame">
                    <img src={result.gradcam_base64} alt="Grad-CAM Activation" />
                    <span className="scan-tag">CAM (conv_head)</span>
                  </div>
                )}
              </div>
            </RomanSection>

            {/* Spatial Localization & Clinical Monograph */}
            <RomanSection index={8} of={9} title="Locus 3D — Spatial Coordinate Localization" className="sp7">
              <BrainRegionViewer locations={result.locations_3d} />
            </RomanSection>

            <RomanSection index={8} of={9} title="Monograph — Clinical Guidance" className="sp5">
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
                      <div className="monograph-cell" style={{ gridColumn: '1 / -1' }}>
                        <span className="mono-cell-k">MRI Signal Characteristics</span>
                        <span className="mono-cell-v">{tumorInfo.mriCharacteristics}</span>
                      </div>
                      <div className="monograph-cell" style={{ gridColumn: '1 / -1' }}>
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

        {/* Academic Model & Kaggle Training Ledger */}
        <RomanSection index={8} of={9} title="Disciplina — Kaggle Training & Academic Benchmarks" className="sp12">
          <div className="archival-ledger-card">
            <div className="ledger-stats-strip">
              <div className="ledger-stat">
                <span className="num">12,626</span>
                <span className="lbl">Curated Scans</span>
              </div>
              <div className="ledger-stat">
                <span className="num">39</span>
                <span className="lbl">Classes</span>
              </div>
              <div className="ledger-stat">
                <span className="num">38</span>
                <span className="lbl">Anatomical Sites</span>
              </div>
              <div className="ledger-stat">
                <span className="num">25.2×</span>
                <span className="lbl">Imbalance Ratio</span>
              </div>
            </div>

            <p className="ledger-prose">
              The dataset contains 12,626 verified 512×512 brain MRI scans (17 corrupt <code>_mask.png</code> entries excluded).
              To prevent anatomical feature leakage between splits, training enforces a <strong>primary-location stratified split</strong> (70% train, 15% validation, 15% held-out test).
              Class imbalance between Meningioma T1C+ (1,057 scans) and Ganglioglioma T1 (42 scans) is addressed via inverse-frequency weighted Cross-Entropy Loss:
            </p>

            <div className="ledger-code-slab">
              # Kaggle Execution Entrypoint:
              python brain_kaggle.py --data_root /kaggle/input/brain-tumor-dataset/archive --output_dir /kaggle/working --batch_size 32 --epochs 50
            </div>

            <div>
              <span className="specimen-lead-tag">Exported Evaluation Artifacts (in brain_classification_results.zip):</span>
              <div className="artifact-checklist">
                <span className="artifact-item"><i className="check-dot" /> loss_curves.png & accuracy_curves.png</span>
                <span className="artifact-item"><i className="check-dot" /> confusion_matrix.png (39×39 Normalized Heatmap)</span>
                <span className="artifact-item"><i className="check-dot" /> class_distribution.png (Splits breakdown)</span>
                <span className="artifact-item"><i className="check-dot" /> sample_predictions.png (4×4 Inference Grid)</span>
                <span className="artifact-item"><i className="check-dot" /> gradcam_samples.png (Layer Activation Overlays)</span>
                <span className="artifact-item"><i className="check-dot" /> roc_curves_macro.png & pr_curves_micro.png</span>
                <span className="artifact-item"><i className="check-dot" /> classification_report.txt & training_log.csv</span>
                <span className="artifact-item"><i className="check-dot" /> brain_best_model.pth (Trained Weights Checkpoint)</span>
              </div>
            </div>
          </div>
        </RomanSection>
      </div>
    </article>
  )
}
