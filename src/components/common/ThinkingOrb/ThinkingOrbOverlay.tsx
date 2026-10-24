/**
 * ThinkingOrbOverlay — unclickable blurred loading overlay.
 *
 * Full-screen frosted backdrop that makes the application unclickable while
 * rendering the ThinkingOrb in the coral accent (#ed6f5c). Cycles between two
 * states:
 *   1. Connecting (neural constellation wiring itself)
 *   2. Solving (voxel lattice permuting)
 *
 * Borderless and cardless: just the orb on the blurred screen.
 */

import React, { useEffect, useRef } from 'react'
import { MODE_FRAMES, resolvePreset, type OrbFrame } from 'thinking-orbs/engine'
import './ThinkingOrbOverlay.css'

export interface ThinkingOrbOverlayProps {
  isVisible: boolean
  label?: string
  /** Rendered orb diameter in CSS px. */
  size?: number
}

// Coral accent palette
const CORAL_R = 237
const CORAL_G = 111
const CORAL_B = 92

// Subtle highlight for active signal pulses
const HIGHLIGHT_R = 247
const HIGHLIGHT_G = 142
const HIGHLIGHT_B = 122

// Timing configuration for the sequence
const DURATION_CONNECTING = 2.6 // seconds
const DURATION_TRANSITION = 0.8 // seconds
const DURATION_SOLVING = 2.6 // seconds
const CYCLE_TOTAL = DURATION_CONNECTING + DURATION_TRANSITION + DURATION_SOLVING + DURATION_TRANSITION // 6.8s

// Colour strings are cached per (palette, alpha bucket) so a frame allocates
// nothing. Drawing is grouped by colour: one path per bucket, one fill.
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

  // Lines: bucket by (colour, rounded width) and stroke each bucket once.
  if (frame.lines && frame.lines.length > 0) {
    const lineBuckets = new Map<string, number[][]>()
    for (const line of frame.lines) {
      const lineAlpha = (line.a ?? 1) * alphaWeight * 0.72
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

  // Dots: one path per colour bucket, one fill. Radii stay per-dot.
  if (frame.dots && frame.dots.length > 0) {
    const dotBuckets = new Map<string, number[][]>()
    for (const dot of frame.dots) {
      const baseAlpha = dot.a ?? 1
      if (baseAlpha < 0.02) continue
      const zNorm = Math.min(1, Math.max(0, (dot.z + 1) / 2))
      const dotAlpha = Math.min(1, Math.max(0.18, baseAlpha * (0.35 + 0.65 * zNorm))) * alphaWeight
      const isHighlight = (dot.white ?? 0) > 0.62 || zNorm > 0.85
      const key = coralFor(isHighlight, dotAlpha)
      const bucket = dotBuckets.get(key)
      const entry = [dot.x, dot.y, Math.max(0.4, dot.r)]
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

export const ThinkingOrbOverlay: React.FC<ThinkingOrbOverlayProps> = ({
  isVisible,
  label,
  size = 220,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const animFrameRef = useRef<number>(0)
  const startTimeRef = useRef<number>(0)

  useEffect(() => {
    if (!isVisible) return

    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // A particle field gains nothing from 2x pixels; 1.5 halves the fill cost.
    const dpr = Math.min(1.25, typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1)
    canvas.width = Math.round(size * dpr)
    canvas.height = Math.round(size * dpr)

    // Pre-resolve engine presets for 64-avatar scale
    const pConnecting = resolvePreset('connecting', 64)
    const pSolving = resolvePreset('solving', 64)
    const fnConnecting = MODE_FRAMES[pConnecting.mode]
    const fnSolving = MODE_FRAMES[pSolving.mode]

    startTimeRef.current = performance.now()
    let isRunning = true
    let lastPaint = 0
    const FRAME_MS = 1000 / 24  // ambient indicator: 24 fps is indistinguishable

    const render = (now: number) => {
      if (!isRunning) return
      animFrameRef.current = requestAnimationFrame(render)
      if (now - lastPaint < FRAME_MS) return
      lastPaint = now

      const elapsedSec = (now - startTimeRef.current) / 1000
      const cycleTime = elapsedSec % CYCLE_TOTAL

      // Calculate weights and transition scales
      let weightConnecting = 1
      let weightSolving = 0
      let scaleConnecting = 1
      let scaleSolving = 0.92

      if (cycleTime < DURATION_CONNECTING) {
        // Stage 1: Connecting active
        weightConnecting = 1
        weightSolving = 0
        scaleConnecting = 1
        scaleSolving = 0.92
      } else if (cycleTime < DURATION_CONNECTING + DURATION_TRANSITION) {
        // Stage 2: Transition from Connecting -> Solving
        const p = (cycleTime - DURATION_CONNECTING) / DURATION_TRANSITION
        const ease = p * p * (3 - 2 * p) // cubic ease in-out
        weightConnecting = 1 - ease
        weightSolving = ease
        scaleConnecting = 1 + 0.08 * ease
        scaleSolving = 0.92 + 0.08 * ease
      } else if (cycleTime < DURATION_CONNECTING + DURATION_TRANSITION + DURATION_SOLVING) {
        // Stage 3: Solving active
        weightConnecting = 0
        weightSolving = 1
        scaleConnecting = 0.92
        scaleSolving = 1
      } else {
        // Stage 4: Transition from Solving -> Connecting
        const p =
          (cycleTime - (DURATION_CONNECTING + DURATION_TRANSITION + DURATION_SOLVING)) /
          DURATION_TRANSITION
        const ease = p * p * (3 - 2 * p)
        weightSolving = 1 - ease
        weightConnecting = ease
        scaleSolving = 1 + 0.08 * ease
        scaleConnecting = 0.92 + 0.08 * ease
      }

      // Clear canvas with high-DPI transform
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, size, size)

      // 1. Draw Connecting frame if visible
      if (weightConnecting > 0.001) {
        const frameConn = fnConnecting(size, elapsedSec * pConnecting.speed, pConnecting.opts)
        drawOrbFrame(ctx, frameConn, weightConnecting, scaleConnecting, size)
      }

      // 2. Draw Solving frame if visible
      if (weightSolving > 0.001) {
        const frameSolv = fnSolving(size, elapsedSec * pSolving.speed, pSolving.opts)
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
  }, [isVisible, size])

  if (!isVisible) return null

  return (
    <div
      className="thinking-orb-overlay"
      role="status"
      aria-live="polite"
      aria-label="Processing diagnostic evaluation"
    >
      <div className="thinking-orb-stage">
        {/* Ambient coral radiance (pure light, zero borders/card) */}
        <div
          className="thinking-orb-ambient-glow"
          aria-hidden="true"
          style={{ width: size * 2.2, height: size * 2.2 }}
        />

        {/* Pure 3D Canvas */}
        <canvas
          ref={canvasRef}
          className="thinking-orb-canvas"
          style={{ width: `${size}px`, height: `${size}px` }}
        />

        {/* Minimal unboxed caption if provided */}
        {label && <span className="thinking-orb-caption">{label}</span>}
      </div>
    </div>
  )
}
