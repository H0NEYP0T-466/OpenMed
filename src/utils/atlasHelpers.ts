import type { Part } from '../types/atlas'

export interface LayoutCell {
  x: number
  y: number
  width: number
  height: number
}

/**
 * Packs visible source meshes into an explosion layout.
 */
export function createExplosionLayout(parts: readonly Part[], aspect = 1): {
  cells: Map<string, LayoutCell>
  width: number
  height: number
} {
  const cards = parts.map((p) => ({
    id: p.id,
    system: p.system,
    width: Math.max(0.035, p.bounds[1][0] - p.bounds[0][0]) + 0.04,
    height: Math.max(0.035, p.bounds[1][1] - p.bounds[0][1]) + 0.04,
  }))

  const area = cards.reduce((n, c) => n + c.width * c.height, 0)
  const maxWidth = Math.max(0.3, ...cards.map((c) => c.width))
  const targetWidth = Math.max(
    maxWidth,
    Math.sqrt(area * Math.max(0.5, Math.min(1.5, aspect))) * 1.18
  )

  cards.sort((a, b) => b.height - a.height || a.id.localeCompare(b.id))

  const cells = new Map<string, LayoutCell>()
  let x = 0
  let y = 0
  let row = 0
  let usedWidth = 0

  for (const c of cards) {
    if (x > 0 && x + c.width > targetWidth) {
      x = 0
      y += row
      row = 0
    }
    cells.set(c.id, {
      x: x + c.width / 2,
      y: -y - c.height / 2,
      width: c.width,
      height: c.height,
    })
    x += c.width
    usedWidth = Math.max(usedWidth, x)
    row = Math.max(row, c.height)
  }

  const height = y + row
  cells.forEach((c) => {
    c.x -= usedWidth / 2
    c.y += height / 2
  })

  return { cells, width: usedWidth, height }
}

/**
 * Distinguishes a single click/tap from camera drag, pan, or pinch.
 */
export class PointerTap {
  private active = new Map<number, { x: number; y: number; threshold: number }>()
  private blocked = false

  down(id: number, x: number, y: number, threshold: number): void {
    if (this.active.size === 0) this.blocked = false
    this.active.set(id, { x, y, threshold })
    if (this.active.size > 1) this.blocked = true
  }

  move(id: number, x: number, y: number): void {
    const start = this.active.get(id)
    if (start && Math.hypot(x - start.x, y - start.y) > start.threshold) {
      this.blocked = true
    }
  }

  up(id: number, x: number, y: number): boolean {
    this.move(id, x, y)
    const tap = this.active.has(id) && this.active.size === 1 && !this.blocked
    this.active.delete(id)
    return tap
  }

  cancel(id: number): void {
    this.active.delete(id)
    this.blocked = true
  }
}

/**
 * Decodes compressed or raw binary chunk responses.
 */
export async function decodeModelResponse(
  response: Response,
  expectedBytes: number,
  compressed: boolean
): Promise<ArrayBuffer> {
  if (!response.ok) throw new Error('An anatomy file could not be loaded.')
  const payload = await response.arrayBuffer()
  const signature = new Uint8Array(payload, 0, Math.min(2, payload.byteLength))
  const isGzip = compressed && signature[0] === 0x1f && signature[1] === 0x8b

  let buffer: ArrayBuffer
  if (isGzip && typeof DecompressionStream !== 'undefined') {
    buffer = await new Response(
      new Blob([payload]).stream().pipeThrough(new DecompressionStream('gzip'))
    ).arrayBuffer()
  } else {
    buffer = payload
  }

  if (expectedBytes > 0 && buffer.byteLength !== expectedBytes) {
    console.warn(
      `Atlas chunk byte mismatch: expected ${expectedBytes}, got ${buffer.byteLength}`
    )
  }

  return buffer
}
