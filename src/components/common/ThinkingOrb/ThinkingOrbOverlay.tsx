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

  // Draw connecting synaptic lines
  if (frame.lines && frame.lines.length > 0) {
    for (const line of frame.lines) {
      const lineAlpha = (line.a ?? 1) * alphaWeight * 0.72
      if (lineAlpha < 0.02) continue

      ctx.strokeStyle = `rgba(${CORAL_R}, ${CORAL_G}, ${CORAL_B}, ${lineAlpha})`
      ctx.lineWidth = Math.max(0.4, line.w || 0.6)
      ctx.beginPath()
      ctx.moveTo(line.x1, line.y1)
      ctx.lineTo(line.x2, line.y2)
      ctx.stroke()
    }
  }

  // Draw dots with depth shading in pure coral-orange
  if (frame.dots && frame.dots.length > 0) {
    for (const dot of frame.dots) {
      const baseAlpha = dot.a ?? 1
      if (baseAlpha < 0.02) continue

      // Normalized depth z: [-1, 1] -> [0, 1]
      const zNorm = Math.min(1, Math.max(0, (dot.z + 1) / 2))
      const dotAlpha = Math.min(1, Math.max(0.18, baseAlpha * (0.35 + 0.65 * zNorm))) * alphaWeight

      // Highlight/signal points get luminous coral, standard dots get pure #ed6f5c
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

export const ThinkingOrbOverlay: React.FC<ThinkingOrbOverlayProps> = ({
  isVisible,
  label,
  size = 176,
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

    const dpr = Math.min(2, typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1)
    canvas.width = Math.round(size * dpr)
    canvas.height = Math.round(size * dpr)

    // Pre-resolve engine presets for 64-avatar scale
    const pConnecting = resolvePreset('connecting', 64)
    const pSolving = resolvePreset('solving', 64)
    const fnConnecting = MODE_FRAMES[pConnecting.mode]
    const fnSolving = MODE_FRAMES[pSolving.mode]

    startTimeRef.current = performance.now()
    let isRunning = true

    const render = (now: number) => {
      if (!isRunning) return

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

      animFrameRef.current = requestAnimationFrame(render)
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
        <div className="thinking-orb-ambient-glow" aria-hidden="true" />

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
