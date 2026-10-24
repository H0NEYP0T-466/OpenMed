import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ThinkingOrbOverlay } from '../../../components/common/ThinkingOrb'
import {
  NIH_FINDINGS,
  classifyLungScan,
  getLungHealth,
  getLungModelInfo,
} from './lungsApi'
import type { LungClassificationResult, LungHealthStatus, LungModelInfo } from './lungTypes'
import './LungsClassificationWorkspace.css'

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

const percent = (value: number): string => `${(value * 100).toFixed(1)}%`

const formatFinding = (finding: string): string => finding.replace(/_/g, ' ')

export const LungsClassificationWorkspace: React.FC = () => {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<LungClassificationResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [health, setHealth] = useState<LungHealthStatus | null>(null)
  const [modelInfo, setModelInfo] = useState<LungModelInfo | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const objectUrlRef = useRef<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void getLungHealth()
      .then((status) => {
        if (!cancelled) setHealth(status)
      })
      .catch(() => undefined)
    void getLungModelInfo()
      .then((info) => {
        if (!cancelled) setModelInfo(info)
      })
      .catch(() => undefined)
    return () => {
      cancelled = true
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    }
  }, [])

  // Lock page scroll while inference runs, restore on exit.
  useEffect(() => {
    if (!isAnalyzing) return
    const previous = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = previous
    }
  }, [isAnalyzing])

  const mountFile = useCallback((candidate: File) => {
    if (!ALLOWED_TYPES.includes(candidate.type)) {
      setError('Selected file is not a JPEG, PNG or WebP image.')
      return
    }
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    const url = URL.createObjectURL(candidate)
    objectUrlRef.current = url
    setPreviewUrl(url)
    setFile(candidate)
    setResult(null)
    setError(null)
    setShowHeatmap(false)
  }, [])

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const selected = event.target.files?.[0]
      if (selected) mountFile(selected)
      event.target.value = ''
    },
    [mountFile],
  )

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault()
      const dropped = event.dataTransfer.files?.[0]
      if (dropped) mountFile(dropped)
    },
    [mountFile],
  )

  const reset = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current)
      objectUrlRef.current = null
    }
    setPreviewUrl(null)
    setFile(null)
    setResult(null)
    setError(null)
    setShowHeatmap(false)
  }, [])

  const run = useCallback(async () => {
    if (!file || isAnalyzing) return
    setIsAnalyzing(true)
    setError(null)
    try {
      const payload = await classifyLungScan(file)
      setResult(payload)
      setShowHeatmap(true)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Inference failed.')
    } finally {
      setIsAnalyzing(false)
    }
  }, [file, isAnalyzing])

  const ranked = useMemo(() => {
    if (!result) return []
    return NIH_FINDINGS.map((finding, index) => ({
      finding,
      probability: result.probabilities[index] ?? 0,
    })).sort((a, b) => b.probability - a.probability)
  }, [result])

  const heatmapSrc = result ? `data:image/png;base64,${result.gradcam_base64}` : null

  return (
    <section className="lungs-workspace">
      {isAnalyzing && (
        <ThinkingOrbOverlay isVisible label="Evaluating 14 findings…" size={210} />
      )}

      <header className="lw-head">
        <div>
          <p className="lw-eyebrow">Thoracic radiograph · multi-label</p>
          <h2 className="lw-title">Z-Net · ChestX-ray14</h2>
          <p className="lw-sub">
            {modelInfo?.paper ?? 'Ahmad et al., Diagnostics 2023'} · 224 px · 14 findings
          </p>
        </div>
        <div className="lw-status">
          <span className={`lw-dot ${health?.model_loaded ? 'is-on' : ''}`} aria-hidden="true" />
          <span>{health?.model_loaded ? 'Model in memory' : 'Model idle · loads on entry'}</span>
        </div>
      </header>

      {error && (
        <div className="lw-error" role="alert">
          <span>{error}</span>
          <button type="button" className="lw-error-dismiss" onClick={() => setError(null)}>
            ✕
          </button>
        </div>
      )}

      {!previewUrl ? (
        <div
          className="lw-prompt"
          onDragOver={(event) => event.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') fileInputRef.current?.click()
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            className="hidden-input"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleFileChange}
          />
          <p className="lw-prompt-title">Drop Chest Radiograph or Browse File</p>
          <p className="lw-prompt-note">
            JPEG · PNG · WebP — 12 MB max. Inference runs locally on the Z-Net
            checkpoint; nothing is uploaded anywhere else.
          </p>
          <span className="lw-browse">Browse Local Drive</span>
        </div>
      ) : (
        <div className="lw-mounted">
          <div className="lw-plate">
            <div className="lw-plate-frame">
              <img
                src={showHeatmap && heatmapSrc ? heatmapSrc : previewUrl}
                alt={showHeatmap ? 'Grad-CAM overlay' : 'Uploaded chest radiograph'}
                className="lw-image"
              />
              <div className="lw-hud">
                <span>{showHeatmap ? 'GRAD-CAM' : 'SOURCE'}</span>
                <span>224 px</span>
                <span>{file?.name ?? 'scan'}</span>
              </div>
            </div>
            <div className="lw-controls">
              <button
                type="button"
                className="btn btn-primary lw-run"
                onClick={() => void run()}
                disabled={isAnalyzing}
              >
                {isAnalyzing ? 'Evaluating 14 findings…' : 'Run Z-Net Differential'}
              </button>
              <button
                type="button"
                className="btn lw-ghost"
                onClick={() => setShowHeatmap((on) => !on)}
                disabled={!result}
              >
                {showHeatmap ? 'Show source' : 'Show Grad-CAM'}
              </button>
              <button type="button" className="btn lw-ghost" onClick={reset}>
                Change image
              </button>
            </div>
          </div>

          {result && (
            <div className="lw-result">
              <div className="lw-headline">
                <p className="lw-eyebrow">Leading finding</p>
                <p className="lw-finding">{formatFinding(result.top_finding)}</p>
                <p className="lw-confidence">{percent(result.confidence)}</p>
                <p className="lw-meta">
                  {result.inference_ms.toFixed(0)} ms · {result.model_tag}
                  {result.bbox ? ` · box ${result.bbox.join(', ')}` : ''}
                </p>
              </div>

              <div className="lw-top5">
                <p className="lw-eyebrow">Differential — top five</p>
                <ol>
                  {result.top5.map((entry) => (
                    <li key={entry.finding}>
                      <span>{formatFinding(entry.finding)}</span>
                      <span className="lw-bar">
                        <span style={{ width: `${entry.probability * 100}%` }} />
                      </span>
                      <span className="lw-value">{percent(entry.probability)}</span>
                    </li>
                  ))}
                </ol>
              </div>

              <div className="lw-all">
                <p className="lw-eyebrow">All 14 labels</p>
                <ul>
                  {ranked.map((entry) => (
                    <li key={entry.finding}>
                      <span>{formatFinding(entry.finding)}</span>
                      <span className="lw-value">{percent(entry.probability)}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  )
}

export default LungsClassificationWorkspace
