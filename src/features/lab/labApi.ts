/**
 * Experimental Laboratory API client.
 *
 * One client for every model: the backend exposes the same five endpoints per
 * model id, so adding a model to the registry needs no change here.
 */

const BASE: string =
  import.meta.env.VITE_LAB_API_BASE ?? 'http://localhost:8016/api/experimental'

export interface LabModelCard {
  readonly id: string
  readonly title: string
  readonly description: string
  readonly input_size: number
  readonly available: boolean
  readonly unavailable_reason: string | null
  readonly loaded: boolean
}

export interface LabModelsResponse {
  readonly models: LabModelCard[]
  readonly loaded: string[]
}

export interface LabMaskResult {
  readonly model: string
  readonly score: number
  readonly inference_ms: number
  readonly image_size: number[]
  readonly mask_base64: string
  readonly overlay_base64: string
  readonly prompt_kinds: string[]
  readonly mask_pixels: number
  readonly mask_fraction: number
  readonly extra: Record<string, unknown>
}

export interface LabBox {
  readonly x1: number
  readonly y1: number
  readonly x2: number
  readonly y2: number
}

/** One click: image pixels plus foreground(1)/background(0). */
export interface LabPoint {
  readonly x: number
  readonly y: number
  readonly positive: boolean
}

/** Any combination — SAM2 composes box, points and mask in one call. */
export interface LabPrompt {
  readonly box?: LabBox | null
  readonly points?: readonly LabPoint[]
  readonly maskFile?: File | null
}

const json = async <T,>(response: Response): Promise<T> => {
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return (await response.json()) as T
}

export const listLabModels = async (): Promise<LabModelsResponse> =>
  // no-store: availability changes when weights are dropped in, and a cached
  // "not installed" would outlive the fix.
  json<LabModelsResponse>(await fetch(`${BASE}/models`, { cache: 'no-store' }))

export const loadLabModel = async (modelId: string): Promise<{ load_ms: number }> =>
  json<{ load_ms: number }>(await fetch(`${BASE}/${modelId}/load`, { method: 'POST' }))

export const unloadLabModel = async (modelId: string): Promise<void> => {
  await fetch(`${BASE}/${modelId}/unload`, { method: 'POST' })
}

export const predictLabMask = async (
  modelId: string,
  file: File,
  prompt: LabPrompt,
): Promise<LabMaskResult> => {
  const body = new FormData()
  body.append('file', file)
  if (prompt.box) {
    body.append('x1', String(Math.round(prompt.box.x1)))
    body.append('y1', String(Math.round(prompt.box.y1)))
    body.append('x2', String(Math.round(prompt.box.x2)))
    body.append('y2', String(Math.round(prompt.box.y2)))
  }
  if (prompt.points && prompt.points.length > 0) {
    body.append(
      'points',
      JSON.stringify(prompt.points.map((p) => [Math.round(p.x), Math.round(p.y), p.positive ? 1 : 0])),
    )
  }
  if (prompt.maskFile) {
    body.append('mask', prompt.maskFile)
  }
  return json<LabMaskResult>(await fetch(`${BASE}/${modelId}/predict`, { method: 'POST', body }))
}
