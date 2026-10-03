import React, { useEffect, useRef, useState } from 'react'
import { DiagnosticThinkingHUD } from '../../../components/common/ThinkingOrb'
import { segmentByBox, segmentByClick } from './brainApi'
import type { ClickSegmentationResult } from './brainTypes'
import './assistive.css'

export interface AssistiveSegmentationProps {
  readonly file: File
  /** Object URL or served path for the scan being marked up. */
  readonly previewUrl: string
}

/**
 * The assistive segmentation module: the clinician marks the lesion and
 * LiteMedSAM segments inside that mark.
 *
 * Two ways to mark it, both one gesture:
 *
 * - **Click** — the click becomes a 48px box centred on it. Measured against
 *   ground truth: mean Dice 0.795, median 0.917.
 * - **Draw** — drag a box around the lesion; the box is used as drawn. This is
 *   the prompt type the released weights were trained on, so it is the
 *   strongest prompt available.
 *
 * Neither path runs the classifier, so a mark responds without a classification
 * pass. The model is loaded lazily by the backend on first use, which makes the
 * very first mark slow and the rest quick.
 */
export const AssistiveSegmentation: React.FC<AssistiveSegmentationProps> = ({
  file,
  previewUrl,
}) => {
  const [promptResult, setPromptResult] = useState<ClickSegmentationResult | null>(null)
  const [isBusy, setIsBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<'click' | 'draw'>('click')
  /**
   * Draw start in a ref, not state: pointermove fires faster than React
   * commits, so a handler reading state can drop moves.
   */
  const drawStartRef = useRef<{ x: number; y: number } | null>(null)
  const [drawRect, setDrawRect] = useState<
    { x1: number; y1: number; x2: number; y2: number } | null
  >(null)

  // A new scan invalidates everything computed from the previous one.
  useEffect(() => {
    setPromptResult(null)
    setError(null)
    setDrawRect(null)
    drawStartRef.current = null
  }, [previewUrl])

  const normPos = (e: React.PointerEvent<HTMLImageElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    return {
      x: Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width)),
      y: Math.min(1, Math.max(0, (e.clientY - rect.top) / rect.height)),
    }
  }

  const handleClickSegment = async (e: React.MouseEvent<HTMLImageElement>) => {
    if (isBusy || mode !== 'click') return
    const rect = e.currentTarget.getBoundingClientRect()
    const clickX = (e.clientX - rect.left) / rect.width
    const clickY = (e.clientY - rect.top) / rect.height
    if (clickX < 0 || clickX > 1 || clickY < 0 || clickY > 1) return

    setIsBusy(true)
    setError(null)
    try {
      setPromptResult(await segmentByClick(file, { clickX, clickY }))
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Click segmentation failed.')
    } finally {
      setIsBusy(false)
    }
  }

  const submitDrawnBox = async (rect: { x1: number; y1: number; x2: number; y2: number }) => {
    if (isBusy) return
    setIsBusy(true)
    setError(null)
    try {
      setPromptResult(await segmentByBox(file, rect))
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Box segmentation failed.')
    } finally {
      setIsBusy(false)
    }
  }

  const handleDrawStart = (e: React.PointerEvent<HTMLImageElement>) => {
    if (mode !== 'draw' || isBusy) return
    // Also stop the browser's own image drag from starting.
    e.preventDefault()
    e.currentTarget.setPointerCapture(e.pointerId)
    drawStartRef.current = normPos(e)
    const p = drawStartRef.current
    setDrawRect({ x1: p.x, y1: p.y, x2: p.x, y2: p.y })
  }

  const handleDrawMove = (e: React.PointerEvent<HTMLImageElement>) => {
    const start = drawStartRef.current
    if (!start) return
    const p = normPos(e)
    setDrawRect({ x1: start.x, y1: start.y, x2: p.x, y2: p.y })
  }

  const handleDrawEnd = (e: React.PointerEvent<HTMLImageElement>) => {
    const start = drawStartRef.current
    if (!start) return
    drawStartRef.current = null
    const p = normPos(e)
    const rect = { x1: start.x, y1: start.y, x2: p.x, y2: p.y }
    setDrawRect(rect)
    // Ignore accidental clicks in draw mode — a real drag has some extent.
    if (Math.abs(rect.x2 - rect.x1) < 0.03 || Math.abs(rect.y2 - rect.y1) < 0.03) {
      setDrawRect(null)
      return
    }
    void submitDrawnBox(rect)
  }

  return (
    <div className="assist-block">
      {/* Same full-screen orb as the classifier run, with elapsed time. */}
      <DiagnosticThinkingHUD isVisible={isBusy} task="segmentation" classCountLabel="LiteMedSAM" />

      <div className="assist-head">
        <div className="assist-copy">
          <span className="assist-title">Assistive Segmentation</span>
          <span className="assist-sub">
            {mode === 'click'
              ? 'Click the suspicious region — the click becomes the prompt.'
              : 'Drag a box around the suspicious region — the box itself is the prompt.'}{' '}
            Runs without a classification pass.
          </span>
        </div>
        <div className="prompt-mode-switch" role="group" aria-label="Prompt mode">
          <button
            type="button"
            className={mode === 'click' ? 'active' : ''}
            onClick={() => {
              setMode('click')
              drawStartRef.current = null
              setDrawRect(null)
            }}
          >
            Click
          </button>
          <button
            type="button"
            className={mode === 'draw' ? 'active' : ''}
            onClick={() => setMode('draw')}
          >
            Draw box
          </button>
        </div>
      </div>

      <div className="assist-plates">
        <div className={`assist-frame ${isBusy ? 'busy' : ''}`}>
          <img
            src={previewUrl}
            alt="Mark a region to segment"
            className="assist-target"
            draggable={false}
            style={{ touchAction: 'none' }}
            onClick={(e) => {
              e.stopPropagation()
              void handleClickSegment(e)
            }}
            onPointerDown={handleDrawStart}
            onPointerMove={handleDrawMove}
            onPointerUp={handleDrawEnd}
          />
          {drawRect && (
            <div
              className="draw-rect"
              style={{
                left: `${Math.min(drawRect.x1, drawRect.x2) * 100}%`,
                top: `${Math.min(drawRect.y1, drawRect.y2) * 100}%`,
                width: `${Math.abs(drawRect.x2 - drawRect.x1) * 100}%`,
                height: `${Math.abs(drawRect.y2 - drawRect.y1) * 100}%`,
              }}
            />
          )}
          {promptResult?.prompt_mode === 'click_box' && promptResult.click && (
            <span
              className="click-marker"
              style={{
                left: `${(promptResult.click[0] / 255) * 100}%`,
                top: `${(promptResult.click[1] / 255) * 100}%`,
              }}
            />
          )}
          <span className="assist-tag">
            {mode === 'click'
              ? promptResult?.prompt_mode === 'click_box'
                ? 'Clicked region'
                : 'Click the lesion'
              : drawRect
                ? 'Drawn box'
                : 'Drag to draw a box'}
          </span>
        </div>

        <div className="assist-frame">
          {promptResult?.seg_mask_base64 ? (
            <img src={promptResult.seg_mask_base64} alt="Segmentation mask" loading="lazy" />
          ) : (
            <div className="assist-empty">
              {mode === 'click' ? 'Click the scan' : 'Drag a box'} to produce a mask
            </div>
          )}
          <span className="assist-tag highlight">
            {mode === 'click' ? 'Click box' : 'Drawn box'} · mask
          </span>
        </div>

        <div className="assist-frame">
          {promptResult?.seg_overlay_base64 ? (
            <img src={promptResult.seg_overlay_base64} alt="Segmentation overlay" loading="lazy" />
          ) : (
            <div className="assist-empty">No overlay yet</div>
          )}
          <span className="assist-tag highlight">
            {mode === 'click' ? 'Click box' : 'Drawn box'} · overlay
          </span>
        </div>
      </div>

      {error && <div className="assist-error">{error}</div>}

      {promptResult && (
        <div className="assist-chips">
          <span className="assist-chip is-accent">
            <span className="k">Prompt mode</span>
            <span className="v">
              {promptResult.prompt_mode === 'click_box' ? 'Click box' : 'Drawn box'}
            </span>
          </span>
          {promptResult.prompt_mode === 'click_box' && promptResult.click && (
            <span className="assist-chip">
              <span className="k">Click (256²)</span>
              <span className="v">{promptResult.click.map((c) => c.toFixed(0)).join(', ')}</span>
            </span>
          )}
          <span className="assist-chip">
            <span className="k">Box</span>
            <span className="v">
              {`${Math.round(promptResult.box_coords[2] - promptResult.box_coords[0])} × ${Math.round(
                promptResult.box_coords[3] - promptResult.box_coords[1],
              )} px`}
            </span>
          </span>
          <span className="assist-chip">
            <span className="k">Predicted IoU</span>
            <span className="v">{promptResult.iou_pred?.toFixed(3) ?? '—'}</span>
          </span>
          <span className="assist-chip">
            <span className="k">Mask foreground</span>
            <span className="v">
              {promptResult.mask_foreground_px == null
                ? '—'
                : `${promptResult.mask_foreground_px.toLocaleString()} px`}
            </span>
          </span>
          <span className="assist-chip">
            <span className="k">Response</span>
            <span className="v">{promptResult.total_ms?.toFixed(0)} ms</span>
          </span>
        </div>
      )}
    </div>
  )
}
