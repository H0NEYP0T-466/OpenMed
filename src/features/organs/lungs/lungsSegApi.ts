/**
 * Client for the lung segmentation endpoints under `/api/lungs`.
 *
 * The base URL is re-declared rather than imported from `lungsApi.ts`, because
 * that module imports the warm / unload helpers from here, so a shared import
 * would form a cycle.
 */

import type {
  LungSegHealth,
  LungSegModelInfoResponse,
  LungSegResidencyResult,
  LungSegResponse,
  LungSegUnloadResponse,
  LungSegWarmupResponse,
} from './lungSegTypes'

const API_BASE_URL: string =
  import.meta.env.VITE_LUNGS_API_BASE ?? 'http://localhost:8016/api/lungs'

/** Codes the server sends, plus the three the client raises itself. */
export type LungSegErrorKind =
  | 'checkpoint_missing'
  | 'checkpoint_unusable'
  | 'invalid_image'
  | 'unsupported_media'
  | 'payload_too_large'
  | 'inference_failed'
  | 'network_error'
  | 'malformed_response'
  | 'http_error'

export class LungSegError extends Error {
  /** Server `detail.code` when present; otherwise a client-side kind. */
  readonly code: LungSegErrorKind | (string & {})
  /** HTTP status, or `0` when no response arrived. */
  readonly status: number

  constructor(code: LungSegErrorKind | (string & {}), message: string, status: number) {
    super(message)
    this.name = 'LungSegError'
    this.code = code
    this.status = status
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null

/** FastAPI `{detail: {code, message}}`; a plain-string `detail` is tolerated. */
const toSegError = async (response: Response): Promise<LungSegError> => {
  const fallback = `HTTP ${response.status}`
  let body: unknown = null
  try {
    body = await response.json()
  } catch {
    /* non-JSON error body */
  }
  const detail = isRecord(body) ? body.detail : undefined
  if (typeof detail === 'string' && detail.trim() !== '') {
    return new LungSegError('http_error', detail, response.status)
  }
  if (isRecord(detail)) {
    const code = typeof detail.code === 'string' ? detail.code : 'http_error'
    const message =
      typeof detail.message === 'string' && detail.message.trim() !== ''
        ? detail.message
        : fallback
    return new LungSegError(code, message, response.status)
  }
  return new LungSegError('http_error', fallback, response.status)
}

const isAbort = (error: unknown): boolean =>
  error instanceof DOMException && error.name === 'AbortError'

export interface SegmentLungScanOptions {
  readonly signal?: AbortSignal
}

/**
 * POST /api/lungs/segment. Aborting rejects with the browser's `AbortError`
 * (not a `LungSegError`) so callers can tell a cancel from a failure.
 */
export const segmentLungScan = async (
  file: File,
  options: SegmentLungScanOptions = {},
): Promise<LungSegResponse> => {
  const body = new FormData()
  body.append('file', file)

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/segment`, {
      method: 'POST',
      body,
      signal: options.signal,
    })
  } catch (error) {
    if (isAbort(error)) throw error
    throw new LungSegError('network_error', 'Could not reach the lung service.', 0)
  }

  if (!response.ok) throw await toSegError(response)

  try {
    return (await response.json()) as LungSegResponse
  } catch (error) {
    if (isAbort(error)) throw error
    throw new LungSegError(
      'malformed_response',
      'The lung service returned a response that could not be read.',
      response.status,
    )
  }
}

const getJson = async <T>(path: string, label: string, signal?: AbortSignal): Promise<T> => {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/${path}`, { signal })
  } catch (error) {
    if (isAbort(error)) throw error
    throw new LungSegError('network_error', `${label}: could not reach the lung service.`, 0)
  }
  if (!response.ok) throw await toSegError(response)
  try {
    return (await response.json()) as T
  } catch {
    throw new LungSegError(
      'malformed_response',
      `${label}: the response could not be read.`,
      response.status,
    )
  }
}

export const getLungSegHealth = (signal?: AbortSignal): Promise<LungSegHealth> =>
  getJson<LungSegHealth>('seg-health', 'Segmentation health', signal)

export const getLungSegModelInfo = (signal?: AbortSignal): Promise<LungSegModelInfoResponse> =>
  getJson<LungSegModelInfoResponse>('seg-model-info', 'Segmentation model info', signal)

const describeUnknown = (error: unknown): string =>
  error instanceof Error ? error.message : 'Unknown error.'

/**
 * Best-effort: load the segmenter into backend memory. Never throws and never
 * reports on the classifier; a missing checkpoint is a normal `ok: false`.
 */
export const warmLungSegmenter = async (): Promise<LungSegResidencyResult> => {
  try {
    const response = await fetch(`${API_BASE_URL}/seg-warmup`, { method: 'POST' })
    if (!response.ok) return { ok: false, detail: `HTTP ${response.status}` }
    const payload = (await response.json()) as LungSegWarmupResponse
    return {
      ok: payload.status === 'ok' && payload.model_loaded,
      detail: payload.detail ?? null,
    }
  } catch (error) {
    return { ok: false, detail: describeUnknown(error) }
  }
}

/** Best-effort: release the segmenter. Never throws. */
export const unloadLungSegmenter = async (): Promise<LungSegResidencyResult> => {
  try {
    const response = await fetch(`${API_BASE_URL}/seg-unload`, { method: 'POST' })
    if (!response.ok) return { ok: false, detail: `HTTP ${response.status}` }
    const payload = (await response.json()) as LungSegUnloadResponse
    return { ok: payload.status === 'ok' && payload.unloaded, detail: payload.message ?? null }
  } catch (error) {
    return { ok: false, detail: describeUnknown(error) }
  }
}
