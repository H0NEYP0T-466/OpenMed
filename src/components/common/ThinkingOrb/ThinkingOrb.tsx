/**
 * ThinkingOrb — Canvas-based 3D particle orb animation.
 *
 * Faithfully ports the geometry engines from thinking-orbs (connecting / web and
 * solving / rubik) with the OpenMed coral accent (#ed6f5c) and high-DPI canvas
 * rendering.
 *
 * Supported modes:
 * - 'connecting': 3D synaptic neural constellation with proximity edges and signal pulses
 * - 'solving'   : 3D rotating Rubik's/voxel lattice with permuting slice rotations
 * - 'hybrid'    : Blended composite mixing the connecting constellation with the solving core
 */

import React, { useEffect, useRef } from 'react'
import './ThinkingOrb.css'

export type OrbMode = 'connecting' | 'solving' | 'hybrid'
export type OrbTheme = 'dark' | 'light' | 'coral'

export interface ThinkingOrbProps {
  mode?: OrbMode
  size?: number // e.g. 64 for avatar, 20 for inline
  theme?: OrbTheme
  speed?: number
  className?: string
  ariaLabel?: string
}

// ── Math & Geometry Primitives ─────────────────────────────────────────────

function project3D(theta: number, phi: number, cx: number, cy: number, scale: number) {
  const sinPhi = Math.sin(phi)
  const cosPhi = Math.cos(phi)
  const sinTheta = Math.sin(theta)
  const cosTheta = Math.cos(theta)
  return (x: number, y: number, z: number): [number, number, number] => {
    const rx = x * cosTheta + z * sinTheta
    const rz = -x * sinTheta + z * cosTheta
    const ry = y * cosPhi - rz * sinPhi
    const rDepth = y * sinPhi + rz * cosPhi
    return [cx + rx * scale, cy - ry * scale, rDepth]
  }
}

// Fibonacci sphere lattice
function fibSphere(i: number, total: number): [number, number, number] {
  const phi = Math.PI * (3 - Math.sqrt(5))
  const y = 1 - 2 * (i + 0.5) / total
  const radius = Math.sqrt(Math.max(0, 1 - y * y))
  const theta = i * phi
  return [radius * Math.cos(theta), y, radius * Math.sin(theta)]
}

// Pseudo-random noise & smooth interpolation
function hash21(x: number, y: number): number {
  const val = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453
  return val - Math.floor(val)
}

function noise2D(x: number, y: number): number {
  const ix = Math.floor(x)
  const iy = Math.floor(y)
  let fx = x - ix
  let fy = y - iy
  fx = fx * fx * (3 - 2 * fx)
  fy = fy * fy * (3 - 2 * fy)
  const a = hash21(ix, iy)
  const b = hash21(ix + 1, iy)
  const c = hash21(ix, iy + 1)
  const d = hash21(ix + 1, iy + 1)
  return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy
}

// Rubik slice rotation moves generator
interface Move {
  axis: number
  lo: number
  hi: number
  ang: number
}

function buildRubikMoves(count: number): Move[] {
  const moves: Move[] = []
  for (let i = 0; i < count; i++) {
    const axis = Math.min(2, Math.floor(hash21(i, 2.3) * 3))
    const slice = -1 + 0.5 * Math.min(3, Math.floor(hash21(i, 5.9) * 4))
    const dir = hash21(i, 7.7) < 0.5 ? 1 : -1
    moves.push({ axis, lo: slice, hi: slice + 0.5, ang: (dir * Math.PI) / 2 })
  }
  return moves
}

function rubikTimeline(t: number, count: number, stepSec: number, pauseSec: number) {
  const cycle = 2 * count * stepSec + pauseSec
  const s = t % cycle
  const amounts = new Array(count).fill(0)
  let active = -1

  if (s < 2 * count * stepSec) {
    const idx = Math.floor(s / stepSec)
    const frac = (s - idx * stepSec) / stepSec
    const ease = 1 - Math.pow(1 - Math.min(1, frac / 0.7), 3)

    if (idx < count) {
      for (let r = 0; r < idx; r++) amounts[r] = 1
      amounts[idx] = ease
      active = idx
    } else {
      const rev = 2 * count - 1 - idx
      for (let r = 0; r < rev; r++) amounts[r] = 1
      amounts[rev] = 1 - ease
      active = rev
    }
  }
  return { amounts, active }
}

function applyRubikTransforms(
  pos: [number, number, number],
  moves: Move[],
  timeline: { amounts: number[]; active: number }
): [number, number, number, boolean] {
  let [x, y, z] = pos
  let isActive = false

  for (let m = 0; m < moves.length; m++) {
    if (timeline.amounts[m] <= 0) continue
    const mv = moves[m]
    const coord = mv.axis === 0 ? x : mv.axis === 1 ? y : z
    if (coord < mv.lo || coord >= mv.hi) continue

    if (m === timeline.active) isActive = true

    const ang = mv.ang * timeline.amounts[m]
    const cosA = Math.cos(ang)
    const sinA = Math.sin(ang)

    if (mv.axis === 0) {
      const ny = y * cosA - z * sinA
      z = y * sinA + z * cosA
      y = ny
    } else if (mv.axis === 1) {
      const nx = x * cosA + z * sinA
      z = -x * sinA + z * cosA
      x = nx
    } else {
      const nx = x * cosA - y * sinA
      y = x * sinA + y * cosA
      x = nx
    }
  }
  return [x, y, z, isActive]
}

// ── Particle & Line Elements ───────────────────────────────────────────────

interface Dot {
  x: number
  y: number
  z: number
  r: number
  alpha: number
  coral?: boolean
}

interface Line {
  x1: number
  y1: number
  x2: number
  y2: number
  w: number
  alpha: number
  coral?: boolean
}

// ── Model 1: Connecting (Web) ──────────────────────────────────────────────

function computeConnecting(size: number, time: number): { dots: Dot[]; lines: Line[] } {
  const cx = size / 2
  const cy = size / 2
  const scale = size * 0.4
  const proj = project3D(time * 0.45, 0.32, cx, cy, scale)

  const nodeCount = 28
  const threshold = 0.72
  const nodes: [number, number, number][] = []

  for (let i = 0; i < nodeCount; i++) {
    const base = fibSphere(i, nodeCount)
    const wanderX = base[0] + 0.3 * (noise2D(i * 0.31 + 9, time * 0.24) - 0.5) * 2
    const wanderY = base[1] + 0.3 * (noise2D(i * 0.53 + 27, time * 0.21) - 0.5) * 2
    const wanderZ = base[2] + 0.3 * (noise2D(i * 0.77 + 55, time * 0.27) - 0.5) * 2
    const len = Math.sqrt(wanderX * wanderX + wanderY * wanderY + wanderZ * wanderZ) || 1
    nodes.push([wanderX / len, wanderY / len, wanderZ / len])
  }

  const lines: Line[] = []
  const dots: Dot[] = []

  // Connect neighbors within distance threshold
  for (let i = 0; i < nodeCount; i++) {
    for (let j = i + 1; j < nodeCount; j++) {
      const dx = nodes[i][0] - nodes[j][0]
      const dy = nodes[i][1] - nodes[j][1]
      const dz = nodes[i][2] - nodes[j][2]
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (dist >= threshold) continue

      const [x1, y1, z1] = proj(nodes[i][0], nodes[i][1], nodes[i][2])
      const [x2, y2, z2] = proj(nodes[j][0], nodes[j][1], nodes[j][2])
      const depth = ((z1 + z2) / 2 + 1) / 2

      lines.push({
        x1,
        y1,
        x2,
        y2,
        w: Math.max(0.6, (size / 64) * 0.9),
        alpha: (1 - dist / threshold) * (0.2 + 0.65 * depth),
      })
    }
  }

  // Draw node points
  for (let i = 0; i < nodeCount; i++) {
    const [x, y, z] = proj(nodes[i][0], nodes[i][1], nodes[i][2])
    const depth = (z + 1) / 2
    const pulse = 1 + 0.25 * Math.sin(time * 2.8 + i * 2.7)
    dots.push({
      x,
      y,
      z,
      r: Math.max(0.7, (1.2 + 1.8 * depth) * pulse * (size / 64)),
      alpha: 0.35 + 0.65 * depth,
    })
  }

  // Signal pulses traveling between nodes (coral accent)
  const signalCount = 5
  for (let i = 0; i < signalCount; i++) {
    const tCycle = time * 0.8 + i * 7.31
    const p = Math.floor(tCycle)
    const frac = tCycle - p
    const idxA = Math.floor(hash21(p, i * 3.1 + 1.7) * nodeCount)
    const idxB = Math.floor(hash21(p, i * 5.7 + 4.2) * nodeCount)
    if (idxA === idxB) continue

    const nx = nodes[idxA][0] + (nodes[idxB][0] - nodes[idxA][0]) * frac
    const ny = nodes[idxA][1] + (nodes[idxB][1] - nodes[idxA][1]) * frac
    const nz = nodes[idxA][2] + (nodes[idxB][2] - nodes[idxA][2]) * frac
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz) || 1

    const [sx, sy, sz] = proj(nx / len, ny / len, nz / len)
    const depth = (sz + 1) / 2
    dots.push({
      x: sx,
      y: sy,
      z: sz,
      r: (2.4 + 2.0 * depth) * (size / 64),
      alpha: 0.6 + 0.4 * depth,
      coral: true,
    })
  }

  dots.sort((a, b) => a.z - b.z)
  return { dots, lines }
}

// ── Model 2: Solving (Rubik's / Voxel Sphere) ──────────────────────────────

function computeSolving(size: number, time: number): { dots: Dot[]; lines: Line[] } {
  const cx = size / 2
  const cy = size / 2
  const scale = size * 0.41
  const proj = project3D(time * 0.85, 0.35 + 0.1 * Math.sin(time * 1.1), cx, cy, scale)

  const moveCount = 14
  const moves = buildRubikMoves(moveCount)
  const timeline = rubikTimeline(time, moveCount, 0.42, 1.2)

  const dots: Dot[] = []
  const latRings = size >= 48 ? 14 : 9
  const lonDensity = size >= 48 ? 36 : 20

  for (let w = 0; w <= latRings; w++) {
    const lat = -Math.PI / 2 + (w / latRings) * Math.PI
    const cosLat = Math.cos(lat)
    const sinLat = Math.sin(lat)
    const countOnRing = Math.max(1, Math.round(Math.abs(cosLat) * lonDensity))

    for (let d = 0; d < countOnRing; d++) {
      const lon = (d / countOnRing) * 2 * Math.PI
      const p3D: [number, number, number] = [
        cosLat * Math.cos(lon),
        sinLat,
        cosLat * Math.sin(lon),
      ]

      const [tx, ty, tz, isActive] = applyRubikTransforms(p3D, moves, timeline)
      const [px, py, pz] = proj(tx, ty, tz)
      const depth = (pz + 1) / 2

      dots.push({
        x: px,
        y: py,
        z: pz,
        r: Math.max(0.6, (0.8 + 1.6 * depth + (isActive ? 0.9 : 0)) * (size / 64)),
        alpha: 0.25 + 0.75 * depth,
        coral: isActive,
      })
    }
  }

  dots.sort((a, b) => a.z - b.z)
  return { dots, lines: [] }
}

// ── Model 3: Hybrid (Connecting Constellation + Solving Core) ───────────────

function computeHybrid(size: number, time: number): { dots: Dot[]; lines: Line[] } {
  const conn = computeConnecting(size, time)
  const solv = computeSolving(size * 0.68, time * 1.15)

  // Scale and offset solving dots into the center of the constellation
  const offset = (size - size * 0.68) / 2
  const scaledSolvDots = solv.dots.map((d) => ({
    ...d,
    x: d.x + offset * 0.68,
    y: d.y + offset * 0.68,
    alpha: d.alpha * 0.85,
  }))

  const allDots = [...conn.dots, ...scaledSolvDots]
  allDots.sort((a, b) => a.z - b.z)

  return {
    dots: allDots,
    lines: conn.lines,
  }
}

// ── Component ──────────────────────────────────────────────────────────────

export const ThinkingOrb: React.FC<ThinkingOrbProps> = ({
  mode = 'hybrid',
  size = 64,
  theme = 'dark',
  speed = 1.0,
  className = '',
  ariaLabel = 'Thinking orb loading indicator',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const animFrameRef = useRef<number | null>(null)
  const startTimeRef = useRef<number>(performance.now())

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = typeof window !== 'undefined' ? Math.min(window.devicePixelRatio || 1, 2) : 1
    canvas.width = size * dpr
    canvas.height = size * dpr

    let running = true

    const render = (now: number) => {
      if (!running) return

      const elapsed = ((now - startTimeRef.current) / 1000) * speed
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.save()
      ctx.scale(dpr, dpr)

      let frameData: { dots: Dot[]; lines: Line[] }
      if (mode === 'connecting') {
        frameData = computeConnecting(size, elapsed)
      } else if (mode === 'solving') {
        frameData = computeSolving(size, elapsed)
      } else {
        frameData = computeHybrid(size, elapsed)
      }

      // Palette resolution — coral accent (#ed6f5c)
      const coralR = 237
      const coralG = 111
      const coralB = 92
      const highlightR = 247
      const highlightG = 142
      const highlightB = 122

      // Draw lines in coral
      for (const line of frameData.lines) {
        ctx.beginPath()
        ctx.moveTo(line.x1, line.y1)
        ctx.lineTo(line.x2, line.y2)
        ctx.lineWidth = line.w
        const alpha = line.coral ? line.alpha : line.alpha * 0.7
        ctx.strokeStyle = `rgba(${coralR}, ${coralG}, ${coralB}, ${alpha})`
        ctx.stroke()
      }

      // Draw dots in coral
      for (const dot of frameData.dots) {
        ctx.beginPath()
        ctx.arc(dot.x, dot.y, dot.r, 0, Math.PI * 2)
        const isHighlight = dot.coral || dot.z > 0.5
        const r = isHighlight ? highlightR : coralR
        const g = isHighlight ? highlightG : coralG
        const b = isHighlight ? highlightB : coralB
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${dot.alpha})`
        ctx.fill()
      }

      ctx.restore()
      animFrameRef.current = requestAnimationFrame(render)
    }

    animFrameRef.current = requestAnimationFrame(render)

    return () => {
      running = false
      if (animFrameRef.current !== null) {
        cancelAnimationFrame(animFrameRef.current)
      }
    }
  }, [mode, size, theme, speed])

  return (
    <div
      className={`thinking-orb-wrap thinking-orb-${mode} thinking-orb-${theme} ${className}`}
      style={{ width: size, height: size }}
      role="status"
      aria-label={ariaLabel}
    >
      <canvas
        ref={canvasRef}
        className="thinking-orb-canvas"
        style={{ width: size, height: size }}
      />
    </div>
  )
}

export default ThinkingOrb
