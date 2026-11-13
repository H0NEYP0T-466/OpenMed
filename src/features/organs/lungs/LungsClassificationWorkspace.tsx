import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  V2S_CLASS_ORDER,
  XRV_THRESHOLD,
  classifyLungScan,
  fetchLungXrvCam,
  getLungHealth,
  getLungModelInfo,
} from './lungsApi'
import { DiagnosticThinkingHUD } from '../../../components/common/ThinkingOrb'
import type { LungRequestProgress } from './lungsApi'
import type {
  LungClassifyResponse,
  LungHealthStatus,
  LungModelInfo,
} from './lungTypes'
import { LUNG_SPECIMEN_PRESETS } from './lungSpecimenPresets'
import type { LungSpecimenPreset } from './lungSpecimenPresets'
import './LungsClassificationWorkspace.css'

/** Mirrors `MAX_UPLOAD_BYTES` in the router. Rejected here so a 40 MB phone
 *  photo never leaves the browser. */
const MAX_BYTES = 12 * 1024 * 1024

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

const EXTENSION_TYPES: Record<string, string> = {
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  png: 'image/png',
  webp: 'image/webp',
}

/**
 * The four stages the backend walks in one request. The first and last are
 * measured from the transfer; the two model stages in between are inferred
 * from the previous run's `inference_ms` because a single HTTP response carries
 * no intermediate signal.
 */
const STAGES = ['Preprocessing', 'LungV2S', 'XRV', 'Grad-CAM'] as const

type ViewMode = 'original' | 'overlay' | 'heatmap'

const VIEW_MODES: ReadonlyArray<{ id: ViewMode; label: string }> = [
  { id: 'original', label: 'Original' },
  { id: 'overlay', label: 'Overlay' },
  { id: 'heatmap', label: 'Heatmap only' },
]

/** Must match `overlay_cam_on_image(..., alpha=0.42)` in the serving pipeline. */
const OVERLAY_ALPHA = 0.42

const DEFAULT_V2S_MS = 1200

const percent = (value: number): string => `${(value * 100).toFixed(1)}%`

const formatLabel = (label: string): string => label.replace(/_/g, ' ')

const formatElapsed = (ms: number): string => {
  const totalSeconds = ms / 1000
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds - minutes * 60
  return `${minutes}:${seconds.toFixed(1).padStart(4, '0')}`
}

const loadImage = (src: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Image could not be decoded.'))
    image.src = src
  })

/** Below this the residual is resampling noise, not signal. */
const HEATMAP_FLOOR = 8

/**
 * Recover the heatmap-only layer from the blended overlay.
 *
 * The server ships one image: `0.42 * heatmap + 0.58 * scan`, clipped to
 * [0, 255]. The scan itself is the file the user mounted, so the blend can be
 * inverted — `heatmap = (overlay - 0.58 * scan) / 0.42`. Where the blend
 * saturated a channel the recovered value pins at 255, which is the closest
 * the clipped data allows. The floor drops the residue left by the browser
 * scaling the scan slightly differently than Pillow did.
 */
const reconstructHeatmap = async (
  overlaySrc: string,
  baseSrc: string,
  alpha: number,
): Promise<string | null> => {
  const [overlay, base] = await Promise.all([loadImage(overlaySrc), loadImage(baseSrc)])
  const width = overlay.naturalWidth || overlay.width
  const height = overlay.naturalHeight || overlay.height
  if (!width || !height) return null

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d')
  if (!context) return null

  context.drawImage(base, 0, 0, width, height)
  const basePixels = context.getImageData(0, 0, width, height).data

  context.clearRect(0, 0, width, height)
  context.drawImage(overlay, 0, 0, width, height)
  const overlayPixels = context.getImageData(0, 0, width, height).data

  const out = context.createImageData(width, height)
  const baseWeight = 1 - alpha
  for (let index = 0; index < out.data.length; index += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      const raw = (overlayPixels[index + channel] - baseWeight * basePixels[index + channel]) / alpha
      out.data[index + channel] = raw < HEATMAP_FLOOR ? 0 : raw > 255 ? 255 : raw
    }
    out.data[index + 3] = 255
  }

  context.putImageData(out, 0, 0)
  return canvas.toDataURL('image/png')
}

/** Accept a file whose MIME type is missing but whose extension is known. */
const normaliseFile = (candidate: File): File | null => {
  if (ALLOWED_TYPES.includes(candidate.type)) return candidate
  if (candidate.type !== '') return null

  const extension = candidate.name.split('.').pop()?.toLowerCase() ?? ''
  const inferred = EXTENSION_TYPES[extension]
  if (!inferred) return null
  return new File([candidate], candidate.name, { type: inferred })
}

export const LungsClassificationWorkspace: React.FC = () => {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [selectedPlate, setSelectedPlate] = useState<string | null>(null)

  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [stage, setStage] = useState(0)
  const [stageFraction, setStageFraction] = useState<number | null>(0)
  const [elapsedMs, setElapsedMs] = useState(0)

  const [result, setResult] = useState<LungClassifyResponse | null>(null)
  const [viewMode, setViewMode] = useState<ViewMode>('original')
  const [heatmapOnlySrc, setHeatmapOnlySrc] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const [health, setHealth] = useState<LungHealthStatus | null>(null)
  const [modelInfo, setModelInfo] = useState<LungModelInfo | null>(null)

  const [detailOpen, setDetailOpen] = useState(false)
  // The 18 scores ship with the main classify response. Only the XRV Grad-CAM
  // is fetched on demand, through the lightweight /xrv-cam endpoint.
  const [xrvCam, setXrvCam] = useState<string | null>(null)
  const [camLoading, setCamLoading] = useState(false)
  const [camError, setCamError] = useState<string | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const objectUrlRef = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const stageTimerRef = useRef<number | null>(null)
  const v2sEstimateRef = useRef<number>(DEFAULT_V2S_MS)

  // Model residency is owned by OrganDashboard: entering the lungs section
  // warms both models, leaving releases them. The workspace only reports state.
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
      abortRef.current?.abort()
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    }
  }, [])

  const mountFile = useCallback((candidate: File) => {
    const normalised = normaliseFile(candidate)
    if (!normalised) {
      setError('Only JPEG, PNG or WebP radiographs are accepted.')
      return
    }
    if (normalised.size > MAX_BYTES) {
      setError(
        `That image is ${(normalised.size / (1024 * 1024)).toFixed(1)} MB. The limit is 12 MB.`,
      )
      return
    }

    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
    const url = URL.createObjectURL(normalised)
    objectUrlRef.current = url

    setPreviewUrl(url)
    setFile(normalised)
    setResult(null)
    setViewMode('original')
    setHeatmapOnlySrc(null)
    setDetailOpen(false)
    setXrvCam(null)
    setCamError(null)
    setError(null)
  }, [])

  const loadPreset = useCallback(
    async (preset: LungSpecimenPreset) => {
      if (isAnalyzing) return
      try {
        const response = await fetch(preset.path)
        if (!response.ok) throw new Error(`${response.status}`)
        const blob = await response.blob()
        mountFile(
          new File([blob], `${preset.name}.png`, { type: blob.type || 'image/png' }),
        )
        setSelectedPlate(preset.plate)
      } catch {
        setError(`Unable to load the ${preset.name} plate.`)
      }
    },
    [isAnalyzing, mountFile],
  )

  const handleFileChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const selected = event.target.files?.[0]
      if (selected) {
        mountFile(selected)
        setSelectedPlate(null)
      }
      event.target.value = ''
    },
    [mountFile],
  )

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault()
      if (isAnalyzing) return
      const dropped = event.dataTransfer.files?.[0]
      if (dropped) {
        mountFile(dropped)
        setSelectedPlate(null)
      }
    },
    [isAnalyzing, mountFile],
  )

  // Clipboard paste anywhere on the section — screenshots of a PACS viewer are
  // the most common way a scan arrives.
  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      if (isAnalyzing) return
      const items = event.clipboardData?.items
      if (!items) return
      for (let index = 0; index < items.length; index += 1) {
        const item = items[index]
        if (item.kind !== 'file') continue
        const pasted = item.getAsFile()
        if (!pasted) continue
        event.preventDefault()
        mountFile(pasted)
        setSelectedPlate(null)
        return
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [isAnalyzing, mountFile])

  const clear = useCallback(() => {
    abortRef.current?.abort()
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current)
      objectUrlRef.current = null
    }
    setPreviewUrl(null)
    setFile(null)
    setResult(null)
    setViewMode('original')
    setHeatmapOnlySrc(null)
    setDetailOpen(false)
    setXrvCam(null)
    setCamError(null)
    setError(null)
    setSelectedPlate(null)
  }, [])

  const run = useCallback(async () => {
    if (!file || isAnalyzing) return

    const controller = new AbortController()
    abortRef.current = controller
    const startedAt = performance.now()

    setResult(null)
    setDetailOpen(false)
    setXrvCam(null)
    setCamError(null)
    setHeatmapOnlySrc(null)
    setViewMode('original')
    setError(null)
    setIsAnalyzing(true)
    setStage(0)
    setStageFraction(0)
    setElapsedMs(0)

    const ticker = window.setInterval(
      () => setElapsedMs(performance.now() - startedAt),
      100,
    )
    let processingSeen = false

    const onProgress = (progress: LungRequestProgress) => {
      if (progress.phase === 'upload') {
        setStage(0)
        setStageFraction(progress.fraction)
        return
      }
      if (progress.phase === 'processing') {
        if (processingSeen) return
        processingSeen = true
        setStage(1)
        setStageFraction(null)
        // Both models run inside one response, so the V2S -> XRV hand-off is
        // timed from the last measured V2S inference rather than observed.
        stageTimerRef.current = window.setTimeout(
          () => setStage(2),
          Math.max(400, v2sEstimateRef.current),
        )
        return
      }
      setStage(3)
      setStageFraction(progress.fraction)
    }

    try {
      const payload = await classifyLungScan(file, {
        onProgress,
        signal: controller.signal,
      })
      setStage(3)
      setStageFraction(1)
      setResult(payload)
      if (payload.v2s) v2sEstimateRef.current = payload.v2s.inference_ms
      setViewMode(payload.v2s ? 'overlay' : 'original')
    } catch (cause) {
      if (controller.signal.aborted) return
      setError(cause instanceof Error ? cause.message : 'Inference failed.')
    } finally {
      window.clearInterval(ticker)
      if (stageTimerRef.current !== null) {
        window.clearTimeout(stageTimerRef.current)
        stageTimerRef.current = null
      }
      setIsAnalyzing(false)
      abortRef.current = null
    }
  }, [file, isAnalyzing])

  const loadCam = useCallback(async () => {
    if (!file || camLoading || xrvCam) return
    setCamLoading(true)
    setCamError(null)
    try {
      setXrvCam(await fetchLungXrvCam(file))
    } catch (cause) {
      setCamError(cause instanceof Error ? cause.message : 'XRV Grad-CAM failed.')
    } finally {
      setCamLoading(false)
    }
  }, [file, camLoading, xrvCam])

  const toggleDetail = useCallback(() => {
    const next = !detailOpen
    setDetailOpen(next)
    if (next && !xrvCam && !camLoading && !camError) void loadCam()
  }, [detailOpen, xrvCam, camLoading, camError, loadCam])

  const v2s = result?.v2s ?? null
  const xrv = result?.xrv ?? null
  const overlaySrc = v2s?.gradcam_base64
    ? `data:image/png;base64,${v2s.gradcam_base64}`
    : null

  // The heatmap-only layer is derived, not fetched: it is rebuilt whenever a
  // new overlay arrives. `heatmapOnlySrc` is cleared by the code that clears
  // the result, so this effect only ever fills it in.
  useEffect(() => {
    if (!overlaySrc || !previewUrl) return
    let cancelled = false
    void reconstructHeatmap(overlaySrc, previewUrl, OVERLAY_ALPHA)
      .then((src) => {
        if (!cancelled) setHeatmapOnlySrc(src)
      })
      .catch(() => {
        if (!cancelled) setHeatmapOnlySrc(null)
      })
    return () => {
      cancelled = true
    }
  }, [overlaySrc, previewUrl])

  const classBars = useMemo(() => {
    if (!v2s) return []
    const names: readonly string[] =
      v2s.class_names.length === v2s.probabilities.length
        ? v2s.class_names
        : V2S_CLASS_ORDER
    return names.map((name, index) => ({
      name,
      probability: v2s.probabilities[index] ?? 0,
      isTop: name === v2s.top_class,
    }))
  }, [v2s])

  // The panel's contract is "at or above the threshold only". The service
  // already filters, but the cut is applied here too so a stale response can
  // never leak a sub-threshold label into the panel.
  const xrvFindings = useMemo(
    () => (xrv ? xrv.findings.filter((finding) => finding.probability >= xrv.threshold) : []),
    [xrv],
  )

  const bbox = useMemo(() => {
    if (!v2s?.bbox || v2s.bbox.length < 4) return null
    const size = v2s.infer_size || v2s.gradcam_grid[0] || 0
    if (!size) return null
    const [x1, y1, x2, y2] = v2s.bbox
    return {
      left: (x1 / size) * 100,
      top: (y1 / size) * 100,
      width: ((x2 - x1) / size) * 100,
      height: ((y2 - y1) / size) * 100,
    }
  }, [v2s])

  const displaySrc =
    viewMode === 'original'
      ? previewUrl
      : viewMode === 'heatmap'
        ? (heatmapOnlySrc ?? overlaySrc ?? previewUrl)
        : (overlaySrc ?? previewUrl)

  const viewLabel =
    viewMode === 'original'
      ? 'Source'
      : viewMode === 'overlay'
        ? 'Grad-CAM'
        : 'Heatmap only'

  const stageState = (index: number): 'done' | 'active' | 'pending' =>
    index < stage ? 'done' : index === stage ? 'active' : 'pending'

  return (
    <section className="lungs-workspace">
      {/* Inference loading. Learning #51: weights => ThinkingOrbOverlay (no
          card, owned by OrganDashboard); analysing a scan => this carded HUD. */}
      <DiagnosticThinkingHUD
        isVisible={isAnalyzing}
        classCountLabel="4 classes + 18 XRV labels"
        task="classification"
      />

      <input
        ref={fileInputRef}
        type="file"
        className="hidden-input"
        accept="image/jpeg,image/png,image/webp"
        onChange={handleFileChange}
      />

      <header className="lw-head">
        <div>
          <p className="lw-eyebrow">
            Thoracic radiograph · COVID · Lung Opacity · Normal · Viral Pneumonia
          </p>
          <h2 className="lw-title">LungV2S · four-class differential</h2>
          <p className="lw-sub">
            {modelInfo?.v2s.model_tag ?? 'tf_efficientnetv2_s.in21k_ft_in1k'} ·{' '}
            {modelInfo?.v2s.input_size ?? 300} px · adjunct XRV-18
          </p>
        </div>
        <div className="lw-status">
          {(['v2s', 'xrv'] as const).map((key) => {
            const entry = health?.[key]
            const label = entry
              ? !entry.checkpoint_present
                ? 'not trained'
                : entry.loaded
                  ? 'in memory'
                  : 'idle'
              : 'checking…'
            return (
              <span key={key} className="lw-model-state">
                <span
                  className={`lw-dot ${entry?.checkpoint_present ? 'is-on' : ''}`}
                  aria-hidden="true"
                />
                <span className="lw-model-name">{key === 'v2s' ? 'LungV2S' : 'XRV'}</span>
                <span>{label}</span>
              </span>
            )
          })}
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

      <div className="lung-rack">
        <div className="lung-rack-head">
          <span className="lung-rack-title">Curated Specimen Plates</span>
          <span className="lung-rack-sub">
            Sixteen labelled radiographs · click to mount
          </span>
        </div>
        <div className="lung-rack-strip">
          {LUNG_SPECIMEN_PRESETS.map((preset) => (
            <button
              key={preset.plate}
              type="button"
              className={`lung-cassette ${selectedPlate === preset.plate ? 'is-active' : ''}`}
              onClick={() => void loadPreset(preset)}
              disabled={isAnalyzing}
              title={`${preset.name} — ${preset.subtitle}`}
            >
              <span className="lung-cassette-thumb">
                <img src={preset.path} alt={preset.name} loading="lazy" decoding="async" />
              </span>
              <span className="lung-cassette-body">
                <span className="lung-cassette-plate">{preset.plate}</span>
                <span className="lung-cassette-name">{preset.name}</span>
                <span className="lung-cassette-site">{preset.site}</span>
              </span>
            </button>
          ))}
        </div>
      </div>

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
          <p className="lw-prompt-title">Drop Chest Radiograph or Browse File</p>
          <p className="lw-prompt-note">
            JPEG · PNG · WebP — 12 MB max. Drag a file, click to browse, or paste a
            screenshot straight from the clipboard. Inference runs locally; nothing
            leaves the machine.
          </p>
          <span className="lw-browse">Browse Local Drive</span>
        </div>
      ) : (
        <div className="lw-mounted">
          <div className="lw-plate">
            <div className="lw-plate-frame">
              <div className="lw-plate-media">
                <img
                  src={displaySrc ?? previewUrl ?? ''}
                  alt={viewLabel}
                  className="lw-image"
                  decoding="async"
                />
                {bbox && (
                  <span
                    className="lw-bbox"
                    style={{
                      left: `${bbox.left}%`,
                      top: `${bbox.top}%`,
                      width: `${bbox.width}%`,
                      height: `${bbox.height}%`,
                    }}
                  >
                    <span className="lw-bbox-tag">CAM ROI</span>
                  </span>
                )}
                <span className="lw-corner is-tl" aria-hidden="true" />
                <span className="lw-corner is-tr" aria-hidden="true" />
                <span className="lw-corner is-bl" aria-hidden="true" />
                <span className="lw-corner is-br" aria-hidden="true" />
              </div>
              <div className="lw-hud">
                <span>{viewLabel}</span>
                {v2s && <span>{v2s.infer_size} px</span>}
                <span className="lw-hud-name">{file?.name ?? 'scan'}</span>
              </div>
            </div>

            <div className="lw-view-toggle" role="group" aria-label="Heatmap view">
              {VIEW_MODES.map((mode) => (
                <button
                  key={mode.id}
                  type="button"
                  className={`lw-view-btn ${viewMode === mode.id ? 'is-active' : ''}`}
                  onClick={() => setViewMode(mode.id)}
                  disabled={!result || (mode.id !== 'original' && !overlaySrc)}
                  aria-pressed={viewMode === mode.id}
                >
                  {mode.label}
                </button>
              ))}
            </div>

            <div className="lw-controls">
              <button
                type="button"
                className="btn btn-primary lw-run"
                onClick={() => void run()}
                disabled={!file || isAnalyzing}
              >
                {isAnalyzing ? 'Analysing…' : 'Run Lung Analysis'}
              </button>
              <button
                type="button"
                className="btn lw-ghost"
                onClick={() => fileInputRef.current?.click()}
                disabled={isAnalyzing}
              >
                Replace
              </button>
              <button
                type="button"
                className="btn lw-ghost"
                onClick={clear}
                disabled={isAnalyzing}
              >
                Clear
              </button>
            </div>
          </div>

          <div className="lw-panel">
            {isAnalyzing ? (
              <div className="lw-stage-panel">
                <p className="lw-eyebrow">
                  Analysing · {formatElapsed(elapsedMs)}
                </p>
                <ol className="lw-stage-list">
                  {STAGES.map((label, index) => {
                    const state = stageState(index)
                    const fraction =
                      state === 'done' ? 1 : state === 'active' ? stageFraction : 0
                    return (
                      <li key={label} className={`lw-stage is-${state}`}>
                        <span className="lw-stage-index">
                          {String(index + 1).padStart(2, '0')}
                        </span>
                        <span className="lw-stage-body">
                          <span className="lw-stage-label">{label}</span>
                          <span className="lw-stage-track">
                            <span
                              className={`lw-stage-fill ${fraction === null ? 'is-indeterminate' : ''}`}
                              style={
                                fraction === null
                                  ? undefined
                                  : { width: `${Math.round(fraction * 100)}%` }
                              }
                            />
                          </span>
                        </span>
                        <span className="lw-stage-state">
                          {state === 'done'
                            ? 'done'
                            : state === 'active'
                              ? fraction === null
                                ? 'running'
                                : `${Math.round(fraction * 100)}%`
                              : 'queued'}
                        </span>
                      </li>
                    )
                  })}
                </ol>
              </div>
            ) : result ? (
              <div className="lw-result">
                {v2s ? (
                  <div className="lw-primary">
                    <div className="lw-headline">
                      <p className="lw-eyebrow">Primary class</p>
                      <p className="lw-finding">{formatLabel(v2s.top_class)}</p>
                      <p className="lw-confidence">
                        {percent(v2s.confidence)}
                        <span className="lw-confidence-note"> confidence</span>
                      </p>
                      <p className="lw-meta">
                        {v2s.model_tag} · {v2s.infer_size} px ·{' '}
                        {v2s.inference_ms.toFixed(0)} ms
                        {bbox ? ' · CAM region marked' : ''}
                      </p>
                    </div>

                    <div className="lw-classes">
                      <p className="lw-eyebrow">Class probabilities</p>
                      <ul>
                        {classBars.map((entry) => (
                          <li key={entry.name} className={entry.isTop ? 'is-top' : undefined}>
                            <span className="lw-class-name">{formatLabel(entry.name)}</span>
                            <span className="lw-bar">
                              <span style={{ width: `${entry.probability * 100}%` }} />
                            </span>
                            <span className="lw-value">{percent(entry.probability)}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ) : (
                  <div className="lw-model-error">
                    <p className="lw-eyebrow">LungV2S unavailable</p>
                    <p className="lw-model-error-body">
                      {result.errors.v2s ?? 'The four-class model returned no result.'}
                    </p>
                  </div>
                )}

                {xrv && xrvFindings.length > 0 && (
                  <div className="lw-secondary">
                    <p className="lw-eyebrow">
                      XRV-18 · findings at or above {percent(xrv.threshold)}
                    </p>
                    <ul>
                      {xrvFindings.map((finding) => (
                        <li key={finding.finding}>
                          <span>{formatLabel(finding.finding)}</span>
                          <span className="lw-bar">
                            <span style={{ width: `${finding.probability * 100}%` }} />
                          </span>
                          <span className="lw-value">{percent(finding.probability)}</span>
                        </li>
                      ))}
                    </ul>
                    <p className="lw-meta">
                      {xrvFindings.length} of 18 labels · {xrv.model_tag} ·{' '}
                      {xrv.inference_ms.toFixed(0)} ms
                    </p>
                  </div>
                )}

                {!xrv && result.errors.xrv && (
                  <div className="lw-model-error">
                    <p className="lw-eyebrow">XRV unavailable</p>
                    <p className="lw-model-error-body">{result.errors.xrv}</p>
                  </div>
                )}

                {xrv && (
                  <section className="lw-detail" aria-label="Detailed XRV scores">
                    <button
                      type="button"
                      className="lw-disclosure"
                      aria-expanded={detailOpen}
                      onClick={toggleDetail}
                    >
                      <span className="lw-disclosure-copy">
                        <span className="lw-disclosure-title">Detailed XRV scores</span>
                        <span className="lw-disclosure-hint">
                          {xrv.scores?.length ?? 18} labels · highest first
                        </span>
                      </span>
                      <span className="lw-disclosure-mark" aria-hidden="true">
                        {detailOpen ? '−' : '+'}
                      </span>
                    </button>

                    {detailOpen && (
                      <div className="lw-detail-body">
                        <div className="lw-detail-head">
                          <span>All pathology labels</span>
                          <span>{xrv.model_tag}</span>
                        </div>

                        {xrv.scores && xrv.scores.length > 0 ? (
                          <ol className="lw-score-list">
                            {xrv.scores.map((score, index) => (
                              <li
                                key={score.finding}
                                className={score.probability >= XRV_THRESHOLD ? 'is-positive' : ''}
                              >
                                <span className="lw-score-rank">
                                  {String(index + 1).padStart(2, '0')}
                                </span>
                                <span className="lw-score-name">
                                  {formatLabel(score.finding)}
                                </span>
                                <span className="lw-score-value">
                                  {percent(score.probability)}
                                </span>
                                <span className="lw-bar">
                                  <span style={{ width: `${score.probability * 100}%` }} />
                                </span>
                              </li>
                            ))}
                          </ol>
                        ) : (
                          <p className="lw-detail-note">This run returned no XRV scores.</p>
                        )}

                        {camError && (
                          <div className="lw-detail-fail">
                            <p className="lw-model-error-body">{camError}</p>
                            <button
                              type="button"
                              className="btn lw-ghost"
                              onClick={() => void loadCam()}
                            >
                              Retry
                            </button>
                          </div>
                        )}

                        {xrvCam && (
                          <figure className="lw-detail-cam">
                            <img
                              src={`data:image/png;base64,${xrvCam}`}
                              alt="XRV Grad-CAM"
                              decoding="async"
                            />
                            <figcaption>XRV Grad-CAM · highest-scoring label</figcaption>
                          </figure>
                        )}
                        {camLoading && !xrvCam && (
                          <p className="lw-detail-note">Generating the XRV activation map…</p>
                        )}
                      </div>
                    )}
                  </section>
                )}
              </div>
            ) : (
              <div className="lw-idle">
                <p className="lw-eyebrow">Ready</p>
                <p className="lw-idle-body">
                  Run the analysis to get the LungV2S class, the four-class
                  probability spread, and the XRV findings above {percent(XRV_THRESHOLD)}.
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  )
}

export default LungsClassificationWorkspace
