import React, { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import {
  LungSegError,
  getLungSegHealth,
  getLungSegModelInfo,
  segmentLungScan,
} from './lungsSegApi'
import type {
  LungSegHealth,
  LungSegModelInfoResponse,
  LungSegResponse,
} from './lungSegTypes'
import { LUNG_SEG_TEMPLATES } from './lungSegTemplates'
import type { LungSegTemplate } from './lungSegTemplates'
import { DiagnosticThinkingHUD } from '../../../components/common/ThinkingOrb'
import './LungsSegmentationPanel.css'

/** Mirrors the router's upload cap. Rejected here so a huge file never leaves the browser. */
const MAX_BYTES = 12 * 1024 * 1024

const ALLOWED_TYPES = ['image/jpeg', 'image/png']

const EXTENSION_TYPES: Record<string, string> = {
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  png: 'image/png',
}

const NOT_RECORDED = 'Not recorded'

/** Data colours. Cool steel for the lung field; the opacity layer reads `--accent`. */
const LUNG_RGB: readonly [number, number, number] = [74, 122, 156]
const ACCENT_FALLBACK_RGB: readonly [number, number, number] = [237, 111, 92]

const DEFAULT_LUNG_ALPHA = 0.34
const DEFAULT_OPACITY_ALPHA = 0.62

/** Largest tolerated gap between the upload's aspect ratio and the mask's. */
const ASPECT_TOLERANCE = 0.015

type Rgb = readonly [number, number, number]

/** Accept a file whose MIME type is missing but whose extension is known. */
const normaliseFile = (candidate: File): File | null => {
  if (ALLOWED_TYPES.includes(candidate.type)) return candidate
  if (candidate.type !== '') return null
  const extension = candidate.name.split('.').pop()?.toLowerCase() ?? ''
  const inferred = EXTENSION_TYPES[extension]
  if (!inferred) return null
  return new File([candidate], candidate.name, { type: inferred })
}

const loadImage = (src: string): Promise<HTMLImageElement> =>
  new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('A mask image could not be decoded.'))
    image.src = src
  })

/** Paint a binary PNG mask (white = foreground) into `canvas` as a solid colour on transparency. */
const paintMask = async (canvas: HTMLCanvasElement, src: string, rgb: Rgb): Promise<void> => {
  const image = await loadImage(src)
  const width = image.naturalWidth
  const height = image.naturalHeight
  if (!width || !height) throw new Error('A mask image has no pixels.')

  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) throw new Error('A 2D canvas is not available in this browser.')

  context.clearRect(0, 0, width, height)
  context.drawImage(image, 0, 0)
  const pixels = context.getImageData(0, 0, width, height)
  const data = pixels.data
  for (let index = 0; index < data.length; index += 4) {
    const foreground = data[index] >= 128
    data[index] = rgb[0]
    data[index + 1] = rgb[1]
    data[index + 2] = rgb[2]
    data[index + 3] = foreground ? 255 : 0
  }
  context.putImageData(pixels, 0, 0)
}

/** Resolve the Atelier Zero accent from the stylesheet so the data colour never drifts from it. */
const readAccentRgb = (): Rgb => {
  const raw = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()
  const match = /^#([0-9a-f]{6})$/i.exec(raw)
  if (!match) return ACCENT_FALLBACK_RGB
  const value = parseInt(match[1], 16)
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255]
}

const isFiniteNumber = (value: number | null | undefined): value is number =>
  typeof value === 'number' && Number.isFinite(value)

const fmtInt = (value: number | null | undefined, suffix = ''): string =>
  isFiniteNumber(value) ? `${Math.round(value).toLocaleString()}${suffix}` : NOT_RECORDED

const fmtDice = (value: number | null | undefined): string =>
  isFiniteNumber(value) ? value.toFixed(3) : NOT_RECORDED

const fmtProbability = fmtDice

const fmtPct = (value: number | null | undefined, digits = 1): string => {
  if (!isFiniteNumber(value)) return NOT_RECORDED
  const pct = value * 100
  const floor = 10 ** -digits
  if (pct > 0 && pct < floor) return `<${floor.toFixed(digits)}%`
  return `${pct.toFixed(digits)}%`
}

const fmtText = (value: string | number | null | undefined): string =>
  value === null || value === undefined || value === '' ? NOT_RECORDED : String(value)

const describeError = (error: unknown): string =>
  error instanceof Error ? error.message : 'Unknown error.'

const isAbortError = (error: unknown): boolean =>
  error instanceof DOMException && error.name === 'AbortError'

interface ResidencyView {
  readonly tone: 'idle' | 'ready' | 'warm' | 'warn'
  readonly label: string
}

const residencyView = (health: LungSegHealth | null, healthError: string | null): ResidencyView => {
  if (healthError) return { tone: 'warn', label: 'Service unreachable' }
  if (!health) return { tone: 'idle', label: 'Checking service' }
  if (health.status === 'unavailable') return { tone: 'warn', label: 'Checkpoint unavailable' }
  if (!health.model_loaded) return { tone: 'warm', label: 'Loads on first use' }
  return { tone: 'ready', label: 'Weights resident' }
}

interface FigureProps {
  readonly label: string
  readonly value: string
}

const Figure: React.FC<FigureProps> = ({ label, value }) => (
  <div className="lsp-figure">
    <dt>{label}</dt>
    <dd>{value}</dd>
  </div>
)

export const LungsSegmentationPanel: React.FC = () => {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [inputProblem, setInputProblem] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const [isSegmenting, setIsSegmenting] = useState(false)
  const [result, setResult] = useState<LungSegResponse | null>(null)
  const [requestError, setRequestError] = useState<LungSegError | null>(null)
  const [maskError, setMaskError] = useState<string | null>(null)

  const [health, setHealth] = useState<LungSegHealth | null>(null)
  const [healthError, setHealthError] = useState<string | null>(null)
  const [modelInfo, setModelInfo] = useState<LungSegModelInfoResponse | null>(null)
  const [infoError, setInfoError] = useState<string | null>(null)

  const [showLung, setShowLung] = useState(true)
  const [showOpacity, setShowOpacity] = useState(true)
  const [lungAlpha, setLungAlpha] = useState(DEFAULT_LUNG_ALPHA)
  const [opacityAlpha, setOpacityAlpha] = useState(DEFAULT_OPACITY_ALPHA)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const objectUrlRef = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const mountedRef = useRef(true)
  const lungCanvasRef = useRef<HTMLCanvasElement>(null)
  const opacityCanvasRef = useRef<HTMLCanvasElement>(null)

  const uid = useId()
  const lungToggleId = `${uid}-lung-toggle`
  const lungAlphaId = `${uid}-lung-alpha`
  const opacityToggleId = `${uid}-opacity-toggle`
  const opacityAlphaId = `${uid}-opacity-alpha`
  const titleId = `${uid}-title`

  const refreshHealth = useCallback(async () => {
    try {
      const next = await getLungSegHealth()
      if (!mountedRef.current) return
      setHealth(next)
      setHealthError(null)
    } catch (error) {
      if (!mountedRef.current || isAbortError(error)) return
      setHealthError(describeError(error))
    }
  }, [])

  const refreshInfo = useCallback(async () => {
    try {
      const next = await getLungSegModelInfo()
      if (!mountedRef.current) return
      setModelInfo(next)
      setInfoError(null)
    } catch (error) {
      if (!mountedRef.current || isAbortError(error)) return
      setInfoError(describeError(error))
    }
  }, [])

  // Residency is owned by OrganDashboard (warm on entry, unload on exit). The
  // panel only reads it, so what it shows is what the server reports.
  useEffect(() => {
    mountedRef.current = true
    const controller = new AbortController()

    getLungSegHealth(controller.signal)
      .then((next) => {
        if (!controller.signal.aborted) setHealth(next)
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted && !isAbortError(error)) setHealthError(describeError(error))
      })

    getLungSegModelInfo(controller.signal)
      .then((next) => {
        if (!controller.signal.aborted) setModelInfo(next)
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted && !isAbortError(error)) setInfoError(describeError(error))
      })

    return () => {
      controller.abort()
      mountedRef.current = false
      abortRef.current?.abort()
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current)
        objectUrlRef.current = null
      }
    }
  }, [])

  // Tint the PNG masks client-side. Cancelled on a new result so a slow decode
  // can never paint over a newer image.
  useEffect(() => {
    if (!result) return undefined
    let cancelled = false

    const jobs: Promise<void>[] = []
    const lungCanvas = lungCanvasRef.current
    if (lungCanvas) jobs.push(paintMask(lungCanvas, result.lung_mask_png, LUNG_RGB))
    const opacityCanvas = opacityCanvasRef.current
    if (opacityCanvas && result.opacity_mask_png) {
      jobs.push(paintMask(opacityCanvas, result.opacity_mask_png, readAccentRgb()))
    }

    void Promise.all(jobs).catch((error: unknown) => {
      if (!cancelled) setMaskError(describeError(error))
    })

    return () => {
      cancelled = true
    }
  }, [result])

  const runSegmentation = useCallback(
    async (target: File) => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      setResult(null)
      setRequestError(null)
      setMaskError(null)
      setIsSegmenting(true)

      try {
        const response = await segmentLungScan(target, { signal: controller.signal })
        if (controller.signal.aborted) return
        setResult(response)
      } catch (error) {
        if (controller.signal.aborted || isAbortError(error)) return
        setRequestError(
          error instanceof LungSegError
            ? error
            : new LungSegError('unknown_error', describeError(error), 0),
        )
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null
          setIsSegmenting(false)
          if (mountedRef.current) {
            void refreshHealth()
            if (!modelInfo?.available) void refreshInfo()
          }
        }
      }
    },
    [modelInfo, refreshHealth, refreshInfo],
  )

  const mountFile = useCallback(
    (candidate: File) => {
      const normalised = normaliseFile(candidate)
      if (!normalised) {
        const kind = candidate.type === '' ? 'with no recognised extension' : `(${candidate.type})`
        setInputProblem(
          `That file ${kind} was not sent. COVID opacity segmentation accepts JPEG or PNG images only.`,
        )
        return
      }
      if (normalised.size > MAX_BYTES) {
        setInputProblem(
          `That image is ${(normalised.size / (1024 * 1024)).toFixed(1)} MB and was not sent. The limit is 12 MB.`,
        )
        return
      }

      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current)
      const url = URL.createObjectURL(normalised)
      objectUrlRef.current = url

      setPreviewUrl(url)
      setFile(normalised)
      setInputProblem(null)
      void runSegmentation(normalised)
    },
    [runSegmentation],
  )

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
      setIsDragging(false)
      const dropped = event.dataTransfer.files?.[0]
      if (dropped) mountFile(dropped)
    },
    [mountFile],
  )

  const mountTemplate = useCallback(
    async (preset: LungSegTemplate) => {
      try {
        const response = await fetch(preset.path)
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const blob = await response.blob()
        const extension = preset.path.endsWith('.png') ? 'png' : 'jpg'
        mountFile(
          new File([blob], `${preset.name}.${extension}`, { type: blob.type || 'image/png' }),
        )
      } catch (error) {
        setInputProblem(`That template could not be loaded: ${describeError(error)}`)
      }
    },
    [mountFile],
  )

  const clear = useCallback(() => {
    abortRef.current?.abort()
    abortRef.current = null
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current)
      objectUrlRef.current = null
    }
    setPreviewUrl(null)
    setFile(null)
    setResult(null)
    setRequestError(null)
    setMaskError(null)
    setInputProblem(null)
    setIsSegmenting(false)
  }, [])

  const residency = residencyView(health, healthError)
  const modelDetails = modelInfo?.available ? modelInfo.info : null
  const opacityDrawn = Boolean(result?.opacity_available && result.opacity_mask_png)
  const scopeNotice = result?.scope_notice ?? modelDetails?.scope_notice ?? null

  const aspectMismatch = useMemo(() => {
    if (!result) return false
    const { image_size: image, mask_size: mask } = result
    if (!image.width || !image.height || !mask.width || !mask.height) return false
    const imageRatio = image.width / image.height
    const maskRatio = mask.width / mask.height
    return Math.abs(imageRatio - maskRatio) / imageRatio > ASPECT_TOLERANCE
  }, [result])

  const stats = result?.stats ?? null

  return (
    <section className="lsp" aria-labelledby={titleId}>
      <header className="lsp-head">
        <div>
          <p className="lsp-eyebrow">Segmentation</p>
          <h2 className="lsp-title" id={titleId}>
            COVID opacity segmentation
          </h2>
          <p className="lsp-sub">
            Finds the lung field on a chest X-ray, then looks for COVID-type opacities inside it.
          </p>
        </div>
        <div className="lsp-residency">
          <span className={`lsp-dot is-${residency.tone}`} aria-hidden="true" />
          <span>{residency.label}</span>
        </div>
      </header>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,.jpg,.jpeg,.png"
        className="lsp-file-input"
        onChange={handleFileChange}
        tabIndex={-1}
        aria-hidden="true"
      />

      {/* Residency, written out. Exactly one of these applies at a time. */}
      <div className="lsp-service" aria-live="polite">
        {healthError ? (
          <p className="lsp-notice is-warn">
            <strong>The segmentation service did not answer.</strong> Reading the service status
            failed: {healthError}. Check that the lungs backend is running, then try again.{' '}
            <button type="button" className="lsp-link" onClick={() => void refreshHealth()}>
              Check again
            </button>
          </p>
        ) : !health ? (
          <p className="lsp-notice">
            Checking whether the segmentation checkpoint is present and loaded.
          </p>
        ) : health.status === 'unavailable' ? (
          <p className="lsp-notice is-warn">
            <strong>Segmentation checkpoint unavailable.</strong>{' '}
            {health.detail ?? 'The server reported the checkpoint as unavailable and gave no detail.'}{' '}
            <button type="button" className="lsp-link" onClick={() => void refreshHealth()}>
              Check again
            </button>
          </p>
        ) : !health.model_loaded ? (
          <p className="lsp-notice">
            The checkpoint is present but the weights are not in memory yet. They load on first
            use, so the first image takes longer than later ones.
          </p>
        ) : (
          <p className="lsp-notice">
            The weights are resident in server memory
            {modelDetails?.device ? ` on ${modelDetails.device}` : ''}. They are released when you
            leave the lungs section.
          </p>
        )}
      </div>

      {inputProblem ? (
        <p className="lsp-notice is-warn" role="alert">
          {inputProblem}
        </p>
      ) : null}

      {/* One plate per population, so the module can be exercised without a file
          on hand. The NIH plates say NIH on the plate because the segmenter was
          never trained on that corpus. */}
      <div className="lsp-rack" role="group" aria-label="Template radiographs">
        <p className="lsp-rack-note">
          Or start from a plate. The three COVID-QU-Ex plates are held-out test images the model
          never saw; the two NIH plates were never in the training corpus at all.
        </p>
        <ul className="lsp-rack-list">
          {LUNG_SEG_TEMPLATES.map((preset) => (
            <li key={preset.path}>
              <button
                type="button"
                className="lsp-plate"
                onClick={() => void mountTemplate(preset)}
                disabled={isSegmenting}
                title={preset.expectation}
              >
                <span className="lsp-plate-thumb">
                  <img src={preset.path} alt="" loading="lazy" />
                  {preset.source === 'NIH' ? <span className="lsp-plate-badge">NIH</span> : null}
                </span>
                <span className="lsp-plate-meta">
                  <span className="lsp-plate-name">{preset.name}</span>
                  <span className="lsp-plate-detail">{preset.detail}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="lsp-body">
        <div
          className={`lsp-stage${isDragging ? ' is-dragging' : ''}`}
          onDragOver={(event) => {
            event.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          {file && previewUrl ? (
            <figure className="lsp-figure-frame">
              <div className="lsp-viewer">
                <img className="lsp-xray" src={previewUrl} alt={`Chest X-ray: ${file.name}`} />
                {result ? (
                  <>
                    <canvas
                      ref={lungCanvasRef}
                      className="lsp-layer"
                      style={{ opacity: showLung ? lungAlpha : 0 }}
                      aria-hidden="true"
                    />
                    {opacityDrawn ? (
                      <canvas
                        ref={opacityCanvasRef}
                        className="lsp-layer"
                        style={{ opacity: showOpacity ? opacityAlpha : 0 }}
                        aria-hidden="true"
                      />
                    ) : null}
                  </>
                ) : null}
              </div>
              <figcaption className="lsp-caption">
                <span className="lsp-file-name">{file.name}</span>
                <span className="lsp-caption-actions">
                  <button
                    type="button"
                    className="lsp-ghost"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Choose another image
                  </button>
                  <button type="button" className="lsp-ghost" onClick={clear}>
                    Remove
                  </button>
                </span>
              </figcaption>
            </figure>
          ) : (
            <div className="lsp-prompt">
              <p className="lsp-prompt-title">No image chosen yet</p>
              <p className="lsp-prompt-note">
                Drop a chest X-ray here, or choose one. JPEG or PNG, up to 12 MB. Nothing is drawn
                until you pick an image, and it is sent only to the lungs service you are
                connected to.
              </p>
              <button
                type="button"
                className="btn btn-primary lsp-choose"
                onClick={() => fileInputRef.current?.click()}
              >
                Choose image
              </button>
            </div>
          )}
        </div>

        <div className="lsp-side">
          <div className="lsp-status" aria-live="polite" aria-atomic="true">
            {isSegmenting ? (
              <div className="lsp-notice is-busy">
                <p>
                  <strong>Segmenting {file?.name ?? 'the image'}.</strong> The server finds the
                  lung field first, then searches only inside it for COVID-type opacities. The
                  overlay appears here when it finishes.
                </p>
                <span className="lsp-bar" aria-hidden="true" />
              </div>
            ) : requestError ? (
              <div className="lsp-notice is-warn" role="alert">
                <p>
                  <strong>Segmentation request failed.</strong>
                </p>
                <p className="lsp-error-line">
                  <span className="lsp-code">{requestError.code}</span>
                  {requestError.status > 0 ? ` (HTTP ${requestError.status})` : ' (no response)'}
                </p>
                <p>{requestError.message}</p>
                {file ? (
                  <button
                    type="button"
                    className="lsp-ghost"
                    onClick={() => void runSegmentation(file)}
                  >
                    Try again
                  </button>
                ) : null}
              </div>
            ) : result ? (
              <p className="lsp-notice">
                Done in {Math.round(result.inference_ms).toLocaleString()} ms by {result.model_tag}.
                Mask resolution {result.mask_size.width} by {result.mask_size.height} px; the
                upload was {result.image_size.width} by {result.image_size.height} px.
              </p>
            ) : file ? (
              <p className="lsp-notice">
                Image ready. Segmentation has not produced a result for it yet.
              </p>
            ) : (
              <p className="lsp-notice">
                Choose an image to see the lung field and any COVID-type opacities drawn on it.
              </p>
            )}
          </div>

          {result && !result.opacity_available ? (
            <p className="lsp-notice is-warn">
              <strong>COVID opacity layer unavailable.</strong>{' '}
              {result.opacity_unavailable_reason ?? 'The server gave no reason.'} The lung field
              layer is still shown.
            </p>
          ) : null}

          {maskError ? (
            <p className="lsp-notice is-warn" role="alert">
              The masks arrived but could not be drawn: {maskError}
            </p>
          ) : null}

          {aspectMismatch ? (
            <p className="lsp-notice">
              The mask and the upload have different aspect ratios, so the overlay may not line up
              exactly with the X-ray.
            </p>
          ) : null}

          <fieldset className="lsp-layers">
            <legend className="lsp-label">Layers</legend>

            <div className="lsp-layer-row">
              <div className="lsp-layer-head">
                <input
                  id={lungToggleId}
                  type="checkbox"
                  checked={showLung}
                  onChange={(event) => setShowLung(event.target.checked)}
                />
                <label htmlFor={lungToggleId}>Show lung field</label>
              </div>
              <div className="lsp-slider">
                <label htmlFor={lungAlphaId}>Lung field strength</label>
                <input
                  id={lungAlphaId}
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={Math.round(lungAlpha * 100)}
                  onChange={(event) => setLungAlpha(Number(event.target.value) / 100)}
                  disabled={!showLung}
                />
                <output htmlFor={lungAlphaId}>{Math.round(lungAlpha * 100)}%</output>
              </div>
            </div>

            <div className="lsp-layer-row">
              <div className="lsp-layer-head">
                <input
                  id={opacityToggleId}
                  type="checkbox"
                  checked={showOpacity}
                  onChange={(event) => setShowOpacity(event.target.checked)}
                  disabled={Boolean(result) && !opacityDrawn}
                />
                <label htmlFor={opacityToggleId}>Show COVID-type opacity</label>
              </div>
              <div className="lsp-slider">
                <label htmlFor={opacityAlphaId}>COVID-type opacity strength</label>
                <input
                  id={opacityAlphaId}
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={Math.round(opacityAlpha * 100)}
                  onChange={(event) => setOpacityAlpha(Number(event.target.value) / 100)}
                  disabled={!showOpacity || (Boolean(result) && !opacityDrawn)}
                />
                <output htmlFor={opacityAlphaId}>{Math.round(opacityAlpha * 100)}%</output>
              </div>
            </div>
          </fieldset>

          <div className="lsp-legend">
            <p className="lsp-label">Legend</p>
            <ul>
              <li>
                <span className="lsp-swatch is-lung" aria-hidden="true" />
                <span>Lung field</span>
              </li>
              <li>
                <span className="lsp-swatch is-opacity" aria-hidden="true" />
                <span>COVID-type opacity (clipped to the lung field)</span>
              </li>
            </ul>
            <p className="lsp-legend-note">
              The opacity layer only delineates COVID-type opacities. It does not delineate other
              pneumonia, for example bacterial consolidation.
            </p>
          </div>
        </div>
      </div>

      {stats ? (
        <section className="lsp-section" aria-label="Measurements for this image">
          <h3 className="lsp-section-title">This image</h3>
          <p className="lsp-muted">
            Areas are counted in mask pixels at {result?.mask_size.width} by{' '}
            {result?.mask_size.height}, not in the original upload&apos;s pixels.
          </p>
          <dl className="lsp-figures is-wide">
            <Figure label="Lung field area" value={fmtInt(stats.lung_area_px, ' px')} />
            <Figure label="Lung field, share of frame" value={fmtPct(stats.lung_fraction_of_frame)} />
            <Figure label="Lung field components" value={fmtInt(stats.lung_components)} />
            <Figure label="COVID-type opacity area" value={fmtInt(stats.opacity_area_px, ' px')} />
            <Figure
              label="COVID-type opacity, share of lung field"
              value={fmtPct(stats.opacity_fraction_of_lung, 2)}
            />
            <Figure label="COVID-type opacity components" value={fmtInt(stats.opacity_components)} />
            <Figure
              label="Opacity probability, peak"
              value={fmtProbability(stats.opacity_peak_probability)}
            />
            <Figure
              label="Opacity probability, mean"
              value={fmtProbability(stats.opacity_mean_probability)}
            />
          </dl>
          {stats.lung_area_px === 0 ? (
            <p className="lsp-notice is-warn">
              No lung field was found in this image, so there is nothing to clip an opacity to.
            </p>
          ) : null}
          {stats.crop_fallback ? (
            <p className="lsp-notice">
              No lung was found to crop to, so the whole frame was given to the opacity model
              instead. Treat the opacity layer with extra caution.
            </p>
          ) : null}
        </section>
      ) : null}

      <section className="lsp-section" aria-label="Model documentation">
        <h3 className="lsp-section-title">Model card</h3>
        <p className="lsp-muted">
          Architecture, training corpus, the post-processing search and the held-out numbers are
          documented in the Documentation section, under Lungs and then Segmentation.{' '}
          {modelDetails?.checkpoint?.name
            ? `This deployment serves ${fmtText(modelDetails.checkpoint.name)}.`
            : ''}
        </p>
        {infoError ? (
          <p className="lsp-notice is-warn">
            The checkpoint sidecar could not be read: {infoError}.{' '}
            <button type="button" className="lsp-link" onClick={() => void refreshInfo()}>
              Try again
            </button>
          </p>
        ) : null}
      </section>

      {scopeNotice ? <p className="lsp-scope">{scopeNotice}</p> : null}

      <DiagnosticThinkingHUD isVisible={isSegmenting} task="segmentation" />
    </section>
  )
}

export default LungsSegmentationPanel
