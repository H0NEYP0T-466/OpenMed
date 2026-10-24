import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { DiagnosticThinkingHUD, ThinkingOrbOverlay } from '../../../components/common/ThinkingOrb'
import { getXrvStatus, loadXrv, predictXrv, unloadXrv } from '../torchxrayvisionApi'
import type { XrvResult, XrvStatus } from '../torchxrayvisionApi'

interface ChestPlate {
  readonly id: string
  readonly label: string
}

/**
 * One plate per class folder in `backend/datasets/Lungs/classification/`,
 * copied into `public/samples/chest/` so the picker exercises every label.
 */
const PLATES: readonly ChestPlate[] = [
  { id: 'atelectasis', label: 'Atelectasis' },
  { id: 'cardiomegaly', label: 'Cardiomegaly' },
  { id: 'consolidation', label: 'Consolidation' },
  { id: 'edema', label: 'Edema' },
  { id: 'effusion', label: 'Effusion' },
  { id: 'emphysema', label: 'Emphysema' },
  { id: 'fibrosis', label: 'Fibrosis' },
  { id: 'hernia', label: 'Hernia' },
  { id: 'infiltration', label: 'Infiltration' },
  { id: 'mass', label: 'Mass' },
  { id: 'nodule', label: 'Nodule' },
  { id: 'pleural_thickening', label: 'Pleural Thickening' },
  { id: 'pneumonia', label: 'Pneumonia' },
  { id: 'pneumothorax', label: 'Pneumothorax' },
  { id: 'no_finding', label: 'No Finding' },
  { id: 'multi_label', label: 'Multi-Label' },
]

const platePath = (id: string): string => `/samples/chest/${id}.png`

export const TorchXRayVisionTab: React.FC = () => {
  const [status, setStatus] = useState<XrvStatus | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [result, setResult] = useState<XrvResult | null>(null)
  const [isRunning, setIsRunning] = useState(false)
  const [isLoadingModel, setIsLoadingModel] = useState(false)
  const [opacity, setOpacity] = useState(55)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Model lifecycle: load on tab select, release on leave.
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        // First visit downloads 28 MB of weights — the full-screen orb is the
        // model-loading indicator; inference uses the diagnostic HUD instead.
        if (!cancelled) setIsLoadingModel(true)
        const state = await loadXrv()
        if (!cancelled) setStatus(state)
      } catch (cause) {
        if (!cancelled) {
          setError(cause instanceof Error ? cause.message : 'Model unavailable.')
          void getXrvStatus().then((s) => !cancelled && setStatus(s)).catch(() => undefined)
        }
      } finally {
        if (!cancelled) setIsLoadingModel(false)
      }
    })()
    return () => {
      cancelled = true
      void unloadXrv()
    }
  }, [])

  // Blob URLs hold the file in memory until revoked — release the previous
  // one whenever the preview changes, and the last one on unmount.
  useEffect(() => () => {
    if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  const mountFile = useCallback((candidate: File) => {
    if (!candidate.type.startsWith('image/')) {
      setError('Selected file is not an image (JPEG, PNG).')
      return
    }
    setError(null)
    setResult(null)
    setFile(candidate)
    setPreviewUrl(URL.createObjectURL(candidate))
  }, [])

  const pickPlate = useCallback(async (plate: ChestPlate) => {
    setError(null)
    setSelected(plate.id)
    try {
      const response = await fetch(platePath(plate.id))
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const blob = await response.blob()
      setFile(new File([blob], `${plate.id}.png`, { type: 'image/png' }))
      setPreviewUrl(platePath(plate.id))
      setResult(null)
    } catch {
      setError(`Unable to load ${platePath(plate.id)}`)
    }
  }, [])

  const run = useCallback(async () => {
    if (!file || isRunning) return
    setIsRunning(true)
    setError(null)
    try {
      setResult(await predictXrv(file))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Inference failed.')
    } finally {
      setIsRunning(false)
    }
  }, [file, isRunning])

  const ranked = useMemo(
    () => (result ? [...result.predictions].sort((a, b) => b.probability - a.probability) : []),
    [result],
  )

  const heatmapSrc = result ? `data:image/png;base64,${result.gradcam_base64}` : null
  const notInstalled = status !== null && !status.installed

  return (
    <div className="xrv">
      <ThinkingOrbOverlay isVisible={isLoadingModel} label="Loading TorchXRayVision…" size={210} />

      <DiagnosticThinkingHUD
        isVisible={isRunning}
        classCountLabel="18 findings"
        task="classification"
      />

      <header className="xrv-head">
        <div>
          <h2 className="xrv-title">TorchXRayVision · DenseNet-121</h2>
          <p className="xrv-sub">
            Multi-dataset clinical pretraining across NIH, CheXpert, and MIMIC — 18-pathology space.
          </p>
        </div>
        <ul className="xrv-chips">
          <li className={status?.installed ? 'ok' : 'warn'}>
            {status?.installed === false ? 'xrv not installed' : 'xrv ready'}
          </li>
          <li className={status?.loaded ? 'ok' : ''}>
            {status?.loaded ? `loaded ${status.load_ms ? `${Math.round(status.load_ms)} ms` : ''}` : 'idle'}
          </li>
          <li className={status?.weights_present ? 'ok' : ''}>
            {status?.weights_present ? 'weights cached' : 'weights on demand'}
          </li>
        </ul>
      </header>

      {error && (
        <div className="xrv-error" role="alert">
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)} aria-label="Dismiss">✕</button>
        </div>
      )}

      <div className="xrv-grid">
        <section className="xrv-panel">
          <div className="xrv-panel-head">
            <span className="xrv-kicker">Plate</span>
            <span className="xrv-hint mono">{PLATES.length} labelled radiographs</span>
          </div>

          <div className="xrv-plates">
            {PLATES.map((plate) => (
              <button
                key={plate.id}
                type="button"
                className={`xrv-plate ${selected === plate.id ? 'is-selected' : ''}`}
                onClick={() => void pickPlate(plate)}
                title={plate.label}
              >
                <img src={platePath(plate.id)} alt={plate.label} loading="lazy" decoding="async" />
                <span>{plate.label}</span>
              </button>
            ))}
          </div>

          <div className="xrv-upload">
            <label className="xrv-upload-btn">
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                onChange={(event) => {
                  const picked = event.target.files?.[0]
                  if (picked) {
                    setSelected(null)
                    mountFile(picked)
                  }
                  event.target.value = ''
                }}
              />
              <span>Upload a scan</span>
            </label>
            {notInstalled && (
              <span className="xrv-upload-note">
                Install <code>torchxrayvision</code> and restart the API to run.
              </span>
            )}
          </div>

          <div className="xrv-viewer">
            <div className="xrv-frame">
              {previewUrl ? (
                <>
                  <img src={previewUrl} alt="Chest radiograph" className="xrv-base" />
                  {heatmapSrc && (
                    <img
                      src={heatmapSrc}
                      alt="Grad-CAM saliency"
                      className="xrv-heat"
                      style={{ opacity: opacity / 100 }}
                    />
                  )}
                  <div className="xrv-hud mono">
                    <span>{result ? 'GRAD-CAM' : 'SOURCE'}</span>
                    <span>224 px</span>
                    <span>{file?.name ?? 'scan'}</span>
                  </div>
                </>
              ) : (
                <p className="xrv-empty">Pick a plate or upload a radiograph.</p>
              )}
            </div>

            <label className="xrv-slider">
              <span className="mono">Saliency</span>
              <input
                type="range"
                min={0}
                max={100}
                value={opacity}
                onChange={(event) => setOpacity(Number(event.target.value))}
                disabled={!heatmapSrc}
              />
              <span className="mono xrv-slider-value">{opacity}%</span>
            </label>

            <div className="xrv-actions">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => void run()}
                disabled={!file || isRunning || notInstalled}
              >
                {isRunning ? 'Scoring…' : 'Run inference'}
              </button>
              <button
                type="button"
                className="btn lab-ghost"
                onClick={() => {
                  setFile(null)
                  setPreviewUrl(null)
                  setResult(null)
                  setSelected(null)
                }}
                disabled={!previewUrl}
              >
                Clear
              </button>
            </div>
          </div>
        </section>

        <section className="xrv-panel xrv-panel-results">
          <div className="xrv-panel-head">
            <span className="xrv-kicker">Findings</span>
            <span className="xrv-hint mono">
              {result ? `${result.inference_ms.toFixed(0)} ms · ${result.weights}` : 'awaiting run'}
            </span>
          </div>

          {result ? (
            <>
              <div className="xrv-lead">
                <p className="xrv-lead-label mono">Leading finding</p>
                <p className="xrv-lead-name">{result.top_pathology}</p>
                <p className="xrv-lead-value mono">{(result.confidence * 100).toFixed(1)}%</p>
              </div>

              <ul className="xrv-bars">
                {ranked.map((entry) => (
                  <li key={entry.pathology} className={entry.probability >= 0.5 ? 'is-positive' : ''}>
                    <span className="xrv-bar-label">{entry.pathology}</span>
                    <span className="xrv-bar-track">
                      <span style={{ width: `${Math.max(entry.probability * 100, 0.5)}%` }} />
                    </span>
                    <span className="xrv-bar-value mono">{(entry.probability * 100).toFixed(1)}</span>
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <div className="xrv-awaiting">
              <p>Nothing scored yet.</p>
              <p className="mono">Run inference to fill 18 probabilities.</p>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

export default TorchXRayVisionTab
