/**
 * DiagnosticThinkingHUD — full-screen "thinking" overlay shown while a scan is
 * being analysed.
 *
 * Renders the ThinkingOrb avatar in the coral accent (#ed6f5c):
 *   - starts with Connecting (neural constellation)
 *   - crossfades to Solving (voxel cube lattice)
 *   - simple phase text plus a three-step progress track underneath
 */

import React, { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Activity } from 'lucide-react'
import { MODE_FRAMES, resolvePreset, type OrbFrame } from 'thinking-orbs/engine'
import './DiagnosticThinkingHUD.css'

export interface DiagnosticThinkingHUDProps {
  /**
   * Drives the overlay in and out. The component stays mounted so the exit
   * transition can play — it renders nothing once fully closed, and the orb
   * animation stops as soon as it starts closing.
   */
  readonly isVisible: boolean
  readonly classCountLabel?: string
  /**
   * Which pipeline is running. Drives the phase wording and the step track so
   * the HUD never claims work that is not happening — the classifier run has a
   * Grad-CAM stage, the assistive segmentation run does not.
   */
  readonly task?: 'classification' | 'segmentation'
}

/** Must match the opacity transition on `.diagnostic-hud-fullscreen-backdrop`. */
const FADE_MS = 320

type OrbPhase = 'connecting' | 'transition' | 'solving'

// Coral accent palette
const CORAL_R = 237
const CORAL_G = 111
const CORAL_B = 92

// Luminous coral highlight for front signal pulses
const HIGHLIGHT_R = 247
const HIGHLIGHT_G = 142
const HIGHLIGHT_B = 122

// Sequence timing (seconds)
const DURATION_CONNECTING = 2.6
const DURATION_TRANSITION = 0.8
const DURATION_SOLVING = 2.6
const CYCLE_TOTAL = DURATION_CONNECTING + DURATION_TRANSITION + DURATION_SOLVING + DURATION_TRANSITION // 6.8s

// Colour strings cached per (palette, alpha bucket); drawing grouped by
// colour — one path per bucket, one fill. Same treatment as the overlay:
// per-dot fills and string allocation were the stutter.
const ALPHA_BUCKETS = 16
const colourCache = new Map<number, string>()

function coralFor(isHighlight: boolean, alpha: number): string {
  const bucket = Math.max(1, Math.min(ALPHA_BUCKETS, Math.round(alpha * ALPHA_BUCKETS)))
  const key = (isHighlight ? 1000 : 0) + bucket
  let value = colourCache.get(key)
  if (value === undefined) {
    const a = (bucket / ALPHA_BUCKETS).toFixed(3)
    value = isHighlight ? `rgba(${HIGHLIGHT_R}, ${HIGHLIGHT_G}, ${HIGHLIGHT_B}, ${a})`
                        : `rgba(${CORAL_R}, ${CORAL_G}, ${CORAL_B}, ${a})`
    colourCache.set(key, value)
  }
  return value
}

function drawOrbFrame(
  ctx: CanvasRenderingContext2D,
  frame: OrbFrame,
  alphaWeight: number,
  scaleFactor: number,
  size: number
) {
  if (alphaWeight <= 0.002) return

  ctx.save()
  const cx = size / 2
  const cy = size / 2

  if (scaleFactor !== 1) {
    ctx.translate(cx, cy)
    ctx.scale(scaleFactor, scaleFactor)
    ctx.translate(-cx, -cy)
  }

  // Lines: bucket by (colour, rounded width), stroke each bucket once.
  if (frame.lines && frame.lines.length > 0) {
    const lineBuckets = new Map<string, number[][]>()
    for (const line of frame.lines) {
      const lineAlpha = (line.a ?? 1) * alphaWeight * 0.75
      if (lineAlpha < 0.02) continue
      const width = Math.round(Math.max(0.4, line.w || 0.6) * 2) / 2
      const key = `${coralFor(false, lineAlpha)}|${width}`
      const bucket = lineBuckets.get(key)
      if (bucket) bucket.push([line.x1, line.y1, line.x2, line.y2])
      else lineBuckets.set(key, [[line.x1, line.y1, line.x2, line.y2]])
    }
    for (const [key, segments] of lineBuckets) {
      const [colour, widthText] = key.split('|')
      ctx.strokeStyle = colour
      ctx.lineWidth = Number(widthText)
      ctx.beginPath()
      for (const [x1, y1, x2, y2] of segments) {
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
      }
      ctx.stroke()
    }
  }

  // Dots: one path per colour bucket, one fill.
  if (frame.dots && frame.dots.length > 0) {
    const dotBuckets = new Map<string, number[][]>()
    for (const dot of frame.dots) {
      const baseAlpha = dot.a ?? 1
      if (baseAlpha < 0.02) continue
      const zNorm = Math.min(1, Math.max(0, (dot.z + 1) / 2))
      const dotAlpha = Math.min(1, Math.max(0.2, baseAlpha * (0.35 + 0.65 * zNorm))) * alphaWeight
      const isHighlight = (dot.white ?? 0) > 0.62 || zNorm > 0.85
      const key = coralFor(isHighlight, dotAlpha)
      const entry = [dot.x, dot.y, Math.max(0.4, dot.r)]
      const bucket = dotBuckets.get(key)
      if (bucket) bucket.push(entry)
      else dotBuckets.set(key, [entry])
    }
    for (const [colour, circles] of dotBuckets) {
      ctx.fillStyle = colour
      ctx.beginPath()
      for (const [x, y, r] of circles) {
        ctx.moveTo(x + r, y)
        ctx.arc(x, y, r, 0, Math.PI * 2)
      }
      ctx.fill()
    }
  }

  ctx.restore()
}

export const DiagnosticThinkingHUD: React.FC<DiagnosticThinkingHUDProps> = ({
  isVisible,
  classCountLabel = '4 classes',
  task = 'classification',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const animFrameRef = useRef<number>(0)
  const startTimeRef = useRef<number>(0)

  const [elapsedSec, setElapsedSec] = useState<number>(0)
  const [activePhase, setActivePhase] = useState<OrbPhase>('connecting')
  const [stage, setStage] = useState<'open' | 'closing' | 'hidden'>(
    isVisible ? 'open' : 'hidden',
  )

  // Mirrors of state the render loop reads, so it can skip redundant updates.
  const phaseRef = useRef<OrbPhase>('connecting')
  const openedRef = useRef(isVisible)

  // Rendered orb diameter in CSS px. The engine's preset table only ships
  // 64 / 32 / 20 tuning entries, but the geometry functions take a free
  // `size`, so we keep the 64 preset and render the canvas large — the orb
  // radius scales linearly with `size`, which keeps it crisp instead of
  // upscaling a 64px bitmap.
  const size = 176

  // Open / close lifecycle. The overlay fades out instead of vanishing.
  //
  // Holding a node on screen for an exit transition is the one thing that
  // cannot be derived at render time — "stay mounted for another 320ms after
  // the prop flips" is inherently post-render state. Hence the setState here.
  useEffect(() => {
    if (isVisible) {
      openedRef.current = true
      // eslint-disable-next-line react/set-state-in-effect
      setStage('open')
      return
    }
    if (!openedRef.current) return

    setStage('closing')
    const timer = window.setTimeout(() => {
      openedRef.current = false
      setStage('hidden')
    }, FADE_MS)
    return () => window.clearTimeout(timer)
  }, [isVisible])

  // Orb animation. Only runs while open — it stops the instant the overlay
  // begins closing, so the fade-out costs nothing.
  useEffect(() => {
    if (stage !== 'open') return

    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = Math.min(1.25, typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1)
    canvas.width = Math.round(size * dpr)
    canvas.height = Math.round(size * dpr)

    const pConnecting = resolvePreset('connecting', 64)
    const pSolving = resolvePreset('solving', 64)
    const fnConnecting = MODE_FRAMES[pConnecting.mode]
    const fnSolving = MODE_FRAMES[pSolving.mode]

    startTimeRef.current = performance.now()
    let isRunning = true
    let lastPaint = 0
    const FRAME_MS = 1000 / 30  // ambient indicator; the React tick stays at 10 Hz

    // The canvas redraws at 30 fps, but React must NOT re-render every
    // frame. The timer only shows tenths, so state is pushed at 10 Hz, and
    // the phase only when it actually changes. Re-rendering this whole card
    // at 60 fps was what made the animation stutter.
    let lastTick = -1

    const render = (now: number) => {
      if (!isRunning) return
      animFrameRef.current = requestAnimationFrame(render)
      if (now - lastPaint < FRAME_MS) return
      lastPaint = now

      const elapsed = (now - startTimeRef.current) / 1000

      const tick = Math.floor(elapsed * 10)
      if (tick !== lastTick) {
        lastTick = tick
        setElapsedSec(tick / 10)
      }

      const cycleTime = elapsed % CYCLE_TOTAL

      let weightConnecting = 1
      let weightSolving = 0
      let scaleConnecting = 1
      let scaleSolving = 0.92
      let phase: OrbPhase = 'connecting'

      if (cycleTime < DURATION_CONNECTING) {
        weightConnecting = 1
        weightSolving = 0
        scaleConnecting = 1
        scaleSolving = 0.92
        phase = 'connecting'
      } else if (cycleTime < DURATION_CONNECTING + DURATION_TRANSITION) {
        const p = (cycleTime - DURATION_CONNECTING) / DURATION_TRANSITION
        const ease = p * p * (3 - 2 * p)
        weightConnecting = 1 - ease
        weightSolving = ease
        scaleConnecting = 1 + 0.08 * ease
        scaleSolving = 0.92 + 0.08 * ease
        phase = 'transition'
      } else if (cycleTime < DURATION_CONNECTING + DURATION_TRANSITION + DURATION_SOLVING) {
        weightConnecting = 0
        weightSolving = 1
        scaleConnecting = 0.92
        scaleSolving = 1
        phase = 'solving'
      } else {
        const p =
          (cycleTime - (DURATION_CONNECTING + DURATION_TRANSITION + DURATION_SOLVING)) /
          DURATION_TRANSITION
        const ease = p * p * (3 - 2 * p)
        weightSolving = 1 - ease
        weightConnecting = ease
        scaleSolving = 1 + 0.08 * ease
        scaleConnecting = 0.92 + 0.08 * ease
        phase = 'transition'
      }

      // Only touch React state when the phase actually flips.
      if (phase !== phaseRef.current) {
        phaseRef.current = phase
        setActivePhase(phase)
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, size, size)

      if (weightConnecting > 0.001) {
        const frameConn = fnConnecting(size, elapsed * pConnecting.speed, pConnecting.opts)
        drawOrbFrame(ctx, frameConn, weightConnecting, scaleConnecting, size)
      }

      if (weightSolving > 0.001) {
        const frameSolv = fnSolving(size, elapsed * pSolving.speed, pSolving.opts)
        drawOrbFrame(ctx, frameSolv, weightSolving, scaleSolving, size)
      }

    }

    animFrameRef.current = requestAnimationFrame(render)

    return () => {
      isRunning = false
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
      }
    }
  }, [stage, size])

  // Formatting elapsed time mm:ss.d
  const mins = Math.floor(elapsedSec / 60)
  const secs = (elapsedSec % 60).toFixed(1)
  const timeFormatted = `${String(mins).padStart(2, '0')}:${secs.padStart(4, '0')}s`

  // Phase headline — plain language, no jargon, and only the stages that this
  // task actually runs.
  const isSeg = task === 'segmentation'
  let phaseTitle = 'Analyzing scan'
  let phaseSubtitle = isSeg ? 'Reading the marked region' : 'Classifying the MRI'
  if (activePhase === 'transition') {
    phaseTitle = 'Analyzing scan'
    phaseSubtitle = isSeg ? 'Encoding the prompt' : 'Preparing segmentation'
  } else if (activePhase === 'solving') {
    phaseTitle = isSeg ? 'Segmenting the mark' : 'Segmenting tumour'
    phaseSubtitle = isSeg ? 'Tracing the boundary inside it' : 'Tracing the tumour boundary'
  }

  if (stage === 'hidden') return null
  if (typeof document === 'undefined') return null

  const modalContent = (
    <div
      className={`diagnostic-hud-fullscreen-backdrop ${stage === 'open' ? 'is-open' : 'is-closing'}`}
      role="dialog"
      aria-modal="true"
      aria-label="Analyzing scan"
    >
      {/* Overlay card */}
      <div className="diagnostic-hud-big-card">
        {/* Header — elapsed time only */}
        <div className="hud-card-header">
          <div className="hud-card-status">
            <span className="status-dot" aria-hidden="true" />
            <span className="status-label">Processing</span>
          </div>
          <div className="hud-card-timer">
            <Activity size={12} className="timer-pulse-icon" />
            <span className="timer-value">{timeFormatted}</span>
          </div>
        </div>

        {/* Center Orb Stage */}
        <div className="hud-orb-center-stage">
          <div className="hud-orb-ambient-glow" aria-hidden="true" />
          <canvas
            ref={canvasRef}
            className="hud-orb-canvas"
            style={{ width: `${size}px`, height: `${size}px` }}
          />
        </div>

        {/* Dynamic Phase Text */}
        <div className="hud-phase-meta">
          <div className="hud-phase-title">{phaseTitle}</div>
          <div className="hud-phase-sub">{phaseSubtitle}</div>
        </div>

        {/* Progress Track */}
        <div className="hud-stage-tracker">
          <div className={`tracker-step ${elapsedSec >= 0 ? 'active' : ''}`}>
            <span className="step-num">01</span>
            <span className="step-label">Reading scan</span>
          </div>
          <div className="tracker-divider" />
          <div className={`tracker-step ${elapsedSec >= 1.5 ? 'active' : ''}`}>
            <span className="step-num">02</span>
            <span className="step-label">{isSeg ? 'Prompt' : 'Grad-CAM'}</span>
          </div>
          <div className="tracker-divider" />
          <div className={`tracker-step ${elapsedSec >= 3.0 ? 'active' : ''}`}>
            <span className="step-num">03</span>
            <span className="step-label">Segmentation</span>
          </div>
        </div>

        {/* Footer Meta */}
        <div className="hud-card-footer">
          <span className="footer-target">
            {isSeg ? 'RUNNING' : 'EVALUATING'}: <strong>{classCountLabel}</strong>
          </span>
        </div>
      </div>
    </div>
  )

  return createPortal(modalContent, document.body)
}
