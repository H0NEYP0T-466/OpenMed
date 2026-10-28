import React, { useCallback, useEffect, useRef, useState } from 'react'
import { DiagnosticThinkingHUD } from '../../components/common/ThinkingOrb'
import { predictLabMask } from './labApi'
import type { LabBox, LabMaskResult, LabPoint } from './labApi'

/**
 * Shared promptable-segmentation surface.
 *
 * SAM2 composes three prompt types and so does this component:
 *   box     — drag a rectangle
 *   points  — click foreground, shift-click background, any number
 *   mask    — upload a mask, or promote the last result and correct it
 *
 * Loading is always the project's ThinkingOrb: inline while weights load, the
 * full HUD while an inference runs. No bespoke spinners anywhere.
 */

type Mode = 'box' | 'points' | 'mask'

interface Props {
  readonly modelId: string
  readonly file: File | null
  readonly previewUrl: string | null
  readonly disabledReason?: string | null
  readonly loadingModel?: boolean
}

interface Point {
  x: number
  y: number
}

const MIN_BOX_PX = 8
const DATA_PREFIX = 'data:image/png;base64,'

const asDataUrl = (base64: string): string =>
  base64.startsWith('data:') ? base64 : `${DATA_PREFIX}${base64}`

export const PromptableSegmentation: React.FC<Props> = ({
  modelId,
  file,
  previewUrl,
  disabledReason,
  loadingModel = false,
}) => {
  const imageRef = useRef<HTMLImageElement>(null)
  const maskInputRef = useRef<HTMLInputElement>(null)
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null)
  const [mode, setMode] = useState<Mode>('box')
  const [start, setStart] = useState<Point | null>(null)
  const [current, setCurrent] = useState<Point | null>(null)
  const [points, setPoints] = useState<LabPoint[]>([])
  const [maskFile, setMaskFile] = useState<File | null>(null)
  const [maskPreview, setMaskPreview] = useState<string | null>(null)
  const [result, setResult] = useState<LabMaskResult | null>(null)
  const [showMask, setShowMask] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    setStart(null)
    setCurrent(null)
    setPoints([])
    setMaskFile(null)
    setMaskPreview(null)
    setResult(null)
    setError(null)
    setNatural(null)
  }, [previewUrl])

  const ensureNatural = useCallback((): { w: number; h: number } | null => {
    if (natural) return natural
    const element = imageRef.current
    if (element?.complete && element.naturalWidth > 0) {
      const size = { w: element.naturalWidth, h: element.naturalHeight }
      setNatural(size)
      return size
    }
    return null
  }, [natural])

  const toImageSpace = useCallback((event: React.PointerEvent<HTMLImageElement>): Point => {
    const element = imageRef.current
    const size = natural ?? { w: element?.naturalWidth ?? 0, h: element?.naturalHeight ?? 0 }
    if (!element || !size.w) return { x: 0, y: 0 }
    const rect = element.getBoundingClientRect()
    return {
      x: Math.min(Math.max((event.clientX - rect.left) * (size.w / rect.width), 0), size.w),
      y: Math.min(Math.max((event.clientY - rect.top) * (size.h / rect.height), 0), size.h),
    }
  }, [natural])

  const run = useCallback(
    async (prompt: { box?: LabBox | null; points?: LabPoint[]; maskFile?: File | null }) => {
      if (!file) return
      setBusy(true)
      setError(null)
      try {
        const payload = await predictLabMask(modelId, file, prompt)
        setResult(payload)
        setShowMask(false)
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : 'Inference failed.')
      } finally {
        setBusy(false)
      }
    },
    [file, modelId],
  )

  const onPointerDown = useCallback((event: React.PointerEvent<HTMLImageElement>) => {
    if (disabledReason || loadingModel) return
    if (!ensureNatural()) return

    // Mask mode takes no clicks: the uploaded mask *is* the prompt. Mixing a
    // stray click into it silently changes what the user asked for.
    if (mode === 'mask') return

    if (mode === 'points') {
      const point = toImageSpace(event)
      setPoints((previous) => [...previous, { x: point.x, y: point.y, positive: !event.shiftKey }])
      setResult(null)
      setError(null)
      return
    }

    try {
      event.currentTarget.setPointerCapture(event.pointerId)
    } catch {
      /* pointer not capturable — dragging still works without it */
    }
    const point = toImageSpace(event)
    setStart(point)
    setCurrent(point)
    setResult(null)
    setError(null)
  }, [disabledReason, loadingModel, ensureNatural, mode, toImageSpace])

  const onPointerMove = useCallback((event: React.PointerEvent<HTMLImageElement>) => {
    if (!start) return
    setCurrent(toImageSpace(event))
  }, [start, toImageSpace])

  const onPointerUp = useCallback(() => {
    if (!start || !current) return
    const box: LabBox = {
      x1: Math.min(start.x, current.x),
      y1: Math.min(start.y, current.y),
      x2: Math.max(start.x, current.x),
      y2: Math.max(start.y, current.y),
    }
    setStart(null)
    setCurrent(null)
    if (box.x2 - box.x1 < MIN_BOX_PX || box.y2 - box.y1 < MIN_BOX_PX) {
      setError('Box too small — drag across the lesion.')
      return
    }
    void run({ box })
  }, [start, current, run])

  const runCurrent = useCallback(() => {
    if (mode === 'mask') {
      if (!maskFile) {
        setError('Choose a mask PNG first, or promote a result with “Use as mask”.')
        return
      }
      void run({ maskFile, points })
      return
    }
    if (points.length === 0) {
      setError('Click at least one point on the lesion. Shift-click marks background.')
      return
    }
    void run({ points })
  }, [mode, maskFile, points, run])

  const promoteResult = useCallback(() => {
    if (!result) return
    const binary = atob(result.mask_base64)
    const bytes = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
    setMaskFile(new File([bytes], 'promoted_mask.png', { type: 'image/png' }))
    setMaskPreview(asDataUrl(result.mask_base64))
    setMode('mask')
    setPoints([])
    setResult(null)
    setError(null)
  }, [result])

  const onMaskChosen = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const chosen = event.target.files?.[0]
    event.target.value = ''
    if (!chosen) return
    setMaskFile(chosen)
    setMaskPreview(URL.createObjectURL(chosen))
    setMode('mask')
    setResult(null)
    setError(null)
  }, [])

  const clearPoints = useCallback(() => {
    setPoints([])
    setResult(null)
    setError(null)
  }, [])

  const clearMask = useCallback(() => {
    setMaskFile(null)
    setMaskPreview(null)
    setResult(null)
    setError(null)
  }, [])

  const boxRect = start && current && natural
    ? {
        left: `${(Math.min(start.x, current.x) / natural.w) * 100}%`,
        top: `${(Math.min(start.y, current.y) / natural.h) * 100}%`,
        width: `${(Math.abs(current.x - start.x) / natural.w) * 100}%`,
        height: `${(Math.abs(current.y - start.y) / natural.h) * 100}%`,
      }
    : null

  if (!file || !previewUrl) {
    return (
      <div className="lab-idle">
        {loadingModel ? 'Loading weights…' : 'Pick a specimen above, or upload a scan, to begin.'}
      </div>
    )
  }

  return (
    <div className="lab-seg">
      <DiagnosticThinkingHUD isVisible={busy} task="segmentation" />

      <div className="lab-seg-stage">
        <div className="lab-seg-frame">
          <img
            ref={imageRef}
            src={result && !showMask ? asDataUrl(result.overlay_base64) : previewUrl}
            alt="Prompt image"
            className={`lab-seg-image ${mode === 'mask' ? 'is-inert' : ''}`}
            onLoad={(event) => {
              const img = event.currentTarget
              setNatural({ w: img.naturalWidth, h: img.naturalHeight })
            }}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
            draggable={false}
          />
          {boxRect && <span className="lab-seg-box" style={boxRect} />}
          {natural && points.map((point, index) => (
            <span
              key={`${point.x}-${point.y}-${index}`}
              className={`lab-seg-point ${point.positive ? 'is-positive' : 'is-negative'}`}
              style={{ left: `${(point.x / natural.w) * 100}%`, top: `${(point.y / natural.h) * 100}%` }}
            />
          ))}
        </div>

        <div className="lab-seg-toolbar">
          <div className="lab-seg-modes" role="tablist">
            {(['box', 'points', 'mask'] as Mode[]).map((value) => (
              <button
                key={value}
                type="button"
                role="tab"
                aria-selected={mode === value}
                className={mode === value ? 'is-active' : ''}
                onClick={() => {
                  setMode(value)
                  if (value !== 'points') setPoints([])
                  setResult(null)
                  setError(null)
                }}
              >
                {value === 'box' ? 'Box' : value === 'points' ? 'Points' : 'Mask'}
              </button>
            ))}
          </div>

          <span className="lab-seg-hint">
            {mode === 'box' && 'Drag a rectangle around the lesion.'}
            {mode === 'points' && 'Click the lesion · shift-click background · then Run.'}
            {mode === 'mask' && (maskFile ? 'Mask loaded · press Run. The image is not clickable here.' : 'Choose a mask PNG to start from.')}
          </span>
        </div>
      </div>

      <aside className="lab-seg-side">
        <div className="lab-seg-panel">
          <h4 className="lab-seg-panel-title">Prompt</h4>

          {mode === 'box' && (
            <p className="lab-seg-count">
              Drag on the image — the box is the prompt. It runs on release.
            </p>
          )}

          {mode === 'mask' && (
            <div className="lab-seg-mask-row">
              <button type="button" className="lab-ghost" onClick={() => maskInputRef.current?.click()}>
                Choose mask…
              </button>
              {maskPreview && (
                <span className="lab-seg-mask-thumb" title="mask prompt">
                  <img src={maskPreview} alt="mask prompt" />
                </span>
              )}
              {maskFile && (
                <button type="button" className="lab-ghost lab-ghost--quiet" onClick={clearMask}>
                  Clear
                </button>
              )}
              <input
                ref={maskInputRef}
                type="file"
                accept="image/png"
                className="hidden-input"
                onChange={onMaskChosen}
              />
            </div>
          )}

          {mode === 'points' && (
            <p className="lab-seg-count">
              {points.length} point{points.length === 1 ? '' : 's'}
              {points.some((p) => !p.positive) && ` · ${points.filter((p) => !p.positive).length} negative`}
            </p>
          )}

          {mode !== 'box' && (
            <div className="lab-seg-actions">
              <button type="button" className="lab-primary" onClick={runCurrent} disabled={busy}>
                Run
              </button>
              <button
                type="button"
                className="lab-ghost"
                onClick={clearPoints}
                disabled={busy || points.length === 0}
              >
                Clear points
              </button>
            </div>
          )}

          {disabledReason && <p className="lab-upload-error">{disabledReason}</p>}
          {error && <p className="lab-upload-error">{error}</p>}
        </div>

        {result && (
          <div className="lab-seg-panel">
            <h4 className="lab-seg-panel-title">Result</h4>
            <dl className="lab-seg-stats">
              <div>
                <dt>Score</dt>
                <dd>{result.score.toFixed(3)}</dd>
              </div>
              <div>
                <dt>Mask area</dt>
                <dd>{(result.mask_fraction * 100).toFixed(1)}%</dd>
              </div>
              <div>
                <dt>Inference</dt>
                <dd>{result.inference_ms.toFixed(0)} ms</dd>
              </div>
              <div>
                <dt>Prompt</dt>
                <dd>{result.prompt_kinds.join(' + ') || '—'}</dd>
              </div>
            </dl>

            <div className="lab-seg-actions">
              <button type="button" className="lab-ghost" onClick={() => setShowMask((v) => !v)}>
                {showMask ? 'Show overlay' : 'Show mask'}
              </button>
              <button type="button" className="lab-ghost" onClick={promoteResult} disabled={busy}>
                Use as mask
              </button>
            </div>
            <p className="lab-seg-footnote">
              “Use as mask” promotes this result into a mask prompt — add clicks to correct it, then run.
            </p>
          </div>
        )}
      </aside>
    </div>
  )
}

export default PromptableSegmentation
