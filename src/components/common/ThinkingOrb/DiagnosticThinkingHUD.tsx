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
  classCountLabel?: string
}

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

  // Draw connecting synaptic lines in coral orange
  if (frame.lines && frame.lines.length > 0) {
    for (const line of frame.lines) {
      const lineAlpha = (line.a ?? 1) * alphaWeight * 0.75
      if (lineAlpha < 0.02) continue

      ctx.strokeStyle = `rgba(${CORAL_R}, ${CORAL_G}, ${CORAL_B}, ${lineAlpha})`
      ctx.lineWidth = Math.max(0.4, line.w || 0.6)
      ctx.beginPath()
      ctx.moveTo(line.x1, line.y1)
      ctx.lineTo(line.x2, line.y2)
      ctx.stroke()
    }
  }

  // Draw dots in coral orange (NO white)
  if (frame.dots && frame.dots.length > 0) {
    for (const dot of frame.dots) {
      const baseAlpha = dot.a ?? 1
      if (baseAlpha < 0.02) continue

      const zNorm = Math.min(1, Math.max(0, (dot.z + 1) / 2))
      const dotAlpha = Math.min(1, Math.max(0.2, baseAlpha * (0.35 + 0.65 * zNorm))) * alphaWeight

      const isHighlight = (dot.white ?? 0) > 0.62 || zNorm > 0.85
      const fillR = isHighlight ? HIGHLIGHT_R : CORAL_R
      const fillG = isHighlight ? HIGHLIGHT_G : CORAL_G
      const fillB = isHighlight ? HIGHLIGHT_B : CORAL_B

      ctx.fillStyle = `rgba(${fillR}, ${fillG}, ${fillB}, ${dotAlpha})`
      ctx.beginPath()
      ctx.arc(dot.x, dot.y, Math.max(0.4, dot.r), 0, Math.PI * 2)
      ctx.fill()
    }
  }

  ctx.restore()
}

export const DiagnosticThinkingHUD: React.FC<DiagnosticThinkingHUDProps> = ({
  classCountLabel = '4 classes',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const animFrameRef = useRef<number>(0)
  const startTimeRef = useRef<number>(0)

  const [elapsedSec, setElapsedSec] = useState<number>(0)
  const [activePhase, setActivePhase] = useState<'connecting' | 'transition' | 'solving'>('connecting')

  // Rendered orb diameter in CSS px. The engine's preset table only ships
  // 64 / 32 / 20 tuning entries, but the geometry functions take a free
  // `size`, so we keep the 64 preset and render the canvas large — the orb
  // radius scales linearly with `size`, which keeps it crisp instead of
  // upscaling a 64px bitmap.
  const size = 176

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = Math.min(2, typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1)
    canvas.width = Math.round(size * dpr)
    canvas.height = Math.round(size * dpr)

    const pConnecting = resolvePreset('connecting', 64)
    const pSolving = resolvePreset('solving', 64)
    const fnConnecting = MODE_FRAMES[pConnecting.mode]
    const fnSolving = MODE_FRAMES[pSolving.mode]

    startTimeRef.current = performance.now()
    let isRunning = true

    const render = (now: number) => {
      if (!isRunning) return

      const elapsed = (now - startTimeRef.current) / 1000
      setElapsedSec(elapsed)
      const cycleTime = elapsed % CYCLE_TOTAL

      let weightConnecting = 1
      let weightSolving = 0
      let scaleConnecting = 1
      let scaleSolving = 0.92

      if (cycleTime < DURATION_CONNECTING) {
        weightConnecting = 1
        weightSolving = 0
        scaleConnecting = 1
        scaleSolving = 0.92
        setActivePhase('connecting')
      } else if (cycleTime < DURATION_CONNECTING + DURATION_TRANSITION) {
        const p = (cycleTime - DURATION_CONNECTING) / DURATION_TRANSITION
        const ease = p * p * (3 - 2 * p)
        weightConnecting = 1 - ease
        weightSolving = ease
        scaleConnecting = 1 + 0.08 * ease
        scaleSolving = 0.92 + 0.08 * ease
        setActivePhase('transition')
      } else if (cycleTime < DURATION_CONNECTING + DURATION_TRANSITION + DURATION_SOLVING) {
        weightConnecting = 0
        weightSolving = 1
        scaleConnecting = 0.92
        scaleSolving = 1
        setActivePhase('solving')
      } else {
        const p =
          (cycleTime - (DURATION_CONNECTING + DURATION_TRANSITION + DURATION_SOLVING)) /
          DURATION_TRANSITION
        const ease = p * p * (3 - 2 * p)
        weightSolving = 1 - ease
        weightConnecting = ease
        scaleSolving = 1 + 0.08 * ease
        scaleConnecting = 0.92 + 0.08 * ease
        setActivePhase('transition')
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

      animFrameRef.current = requestAnimationFrame(render)
    }

    animFrameRef.current = requestAnimationFrame(render)

    return () => {
      isRunning = false
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current)
      }
    }
  }, [size])

  // Formatting elapsed time mm:ss.d
  const mins = Math.floor(elapsedSec / 60)
  const secs = (elapsedSec % 60).toFixed(1)
  const timeFormatted = `${String(mins).padStart(2, '0')}:${secs.padStart(4, '0')}s`

  // Phase headline — plain language, no jargon
  let phaseTitle = 'Analyzing scan'
  let phaseSubtitle = 'Classifying the MRI'
  if (activePhase === 'transition') {
    phaseTitle = 'Analyzing scan'
    phaseSubtitle = 'Preparing segmentation'
  } else if (activePhase === 'solving') {
    phaseTitle = 'Segmenting tumour'
    phaseSubtitle = 'Tracing the tumour boundary'
  }

  const modalContent = (
    <div
      className="diagnostic-hud-fullscreen-backdrop"
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
            <span className="step-label">Grad-CAM</span>
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
            EVALUATING: <strong>{classCountLabel}</strong>
          </span>
        </div>
      </div>
    </div>
  )

  if (typeof document === 'undefined') return null
  return createPortal(modalContent, document.body)
}
