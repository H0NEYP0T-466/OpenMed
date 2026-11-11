/**
 * Client for `/api/lungs`.
 *
 * `classifyLungScan` uses XMLHttpRequest rather than fetch for one reason: the
 * workspace shows a staged progress rail, and XHR is the only browser API that
 * reports how much of the upload has been sent and how much of the response has
 * arrived. Those are the two stages that can be measured; the model stages
 * between them are inferred by the caller from the previous run's timings.
 */

import type {
  LungClassifyResponse,
  LungHealthStatus,
  LungModelInfo,
  LungModelSelector,
  LungUnloadResponse,
  LungWarmupResponse,
} from './lungTypes'

const API_BASE_URL: string =
  import.meta.env.VITE_LUNGS_API_BASE ?? 'http://localhost:8016/api/lungs'

/** Fixed order of the LungV2S head. The server echoes this in `class_names`. */
export const V2S_CLASS_ORDER: readonly string[] = [
  'COVID',
  'Lung_Opacity',
  'Normal',
  'Viral Pneumonia',
]

/** Operating point the XRV panel filters on. */
export const XRV_THRESHOLD = 0.5

export type LungRequestPhase = 'upload' | 'processing' | 'download'

export interface LungRequestProgress {
  readonly phase: LungRequestPhase
  /** `null` when the browser cannot compute a fraction (chunked response). */
  readonly fraction: number | null
  readonly loadedBytes: number
  readonly totalBytes: number
}

export interface ClassifyLungScanOptions {
  /** Second, on-demand call: adds all 18 XRV scores and the XRV Grad-CAM. */
  readonly includeXrvDetail?: boolean
  readonly onProgress?: (progress: LungRequestProgress) => void
  readonly signal?: AbortSignal
}

const describeFailure = (status: number, body: string): string => {
  try {
    const payload = JSON.parse(body) as { detail?: unknown }
    const detail = payload.detail
    if (typeof detail === 'string') return detail
    if (detail && typeof detail === 'object') {
      const record = detail as { message?: unknown; v2s?: unknown; xrv?: unknown }
      const parts: string[] = []
      if (typeof record.message === 'string') parts.push(record.message)
      if (typeof record.v2s === 'string') parts.push(`LungV2S: ${record.v2s}`)
      if (typeof record.xrv === 'string') parts.push(`XRV: ${record.xrv}`)
      if (parts.length > 0) return parts.join(' · ')
    }
  } catch {
    /* non-JSON error body */
  }
  return `HTTP ${status}`
}

/** Run both lung models — POST /api/lungs/classify. */
export const classifyLungScan = (
  file: File,
  options: ClassifyLungScanOptions = {},
): Promise<LungClassifyResponse> => {
  const { includeXrvDetail = false, onProgress, signal } = options

  return new Promise<LungClassifyResponse>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'))
      return
    }

    const query = includeXrvDetail ? '?include_xrv_detail=true' : ''
    const form = new FormData()
    form.append('file', file)

    const xhr = new XMLHttpRequest()
    let settled = false
    let uploadFinished = false

    const detach = () => {
      if (signal) signal.removeEventListener('abort', handleAbort)
    }
    const finish = (action: () => void) => {
      if (settled) return
      settled = true
      detach()
      action()
    }
    function handleAbort() {
      xhr.abort()
    }

    xhr.open('POST', `${API_BASE_URL}/classify${query}`, true)
    xhr.responseType = 'text'
    // Both models plus two Grad-CAM passes on a cold CPU box; the socket stays
    // open well past a default timeout.
    xhr.timeout = 180_000

    if (signal) signal.addEventListener('abort', handleAbort, { once: true })

    xhr.upload.onprogress = (event) => {
      if (!onProgress) return
      const fraction = event.lengthComputable && event.total > 0 ? event.loaded / event.total : null
      onProgress({ phase: 'upload', fraction, loadedBytes: event.loaded, totalBytes: event.total })
    }

    // `onloadend` fires after the body is handed off whether or not a single
    // progress event was ever reported, so it — not `onload` — is what moves the
    // rail past preprocessing.
    const markProcessing = () => {
      if (uploadFinished) return
      uploadFinished = true
      onProgress?.({
        phase: 'processing',
        fraction: null,
        loadedBytes: 0,
        totalBytes: 0,
      })
    }

    xhr.upload.onload = markProcessing
    xhr.upload.onloadend = markProcessing

    xhr.onprogress = (event) => {
      if (!onProgress) return
      const fraction = event.lengthComputable && event.total > 0 ? event.loaded / event.total : null
      onProgress({
        phase: 'download',
        fraction,
        loadedBytes: event.loaded,
        totalBytes: event.total,
      })
    }

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          finish(() => resolve(JSON.parse(xhr.responseText) as LungClassifyResponse))
        } catch {
          finish(() => reject(new Error('The lung service returned a malformed response.')))
        }
        return
      }
      finish(() => reject(new Error(describeFailure(xhr.status, xhr.responseText))))
    }

    xhr.onerror = () =>
      finish(() => reject(new Error('Could not reach the lung service.')))
    xhr.ontimeout = () =>
      finish(() => reject(new Error('The lung service timed out.')))
    xhr.onabort = () => finish(() => reject(new DOMException('Aborted', 'AbortError')))

    xhr.send(form)
  })
}

export const getLungHealth = async (): Promise<LungHealthStatus> => {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) throw new Error(`Health check failed (HTTP ${response.status})`)
  return (await response.json()) as LungHealthStatus
}

export const getLungModelInfo = async (): Promise<LungModelInfo> => {
  const response = await fetch(`${API_BASE_URL}/model-info`)
  if (!response.ok) throw new Error(`Model info failed (HTTP ${response.status})`)
  return (await response.json()) as LungModelInfo
}

/**
 * Load the lung models into backend memory.
 *
 * Called by the organ dashboard when the lungs section opens, not by the
 * workspace — residency belongs to the section, so entering and leaving is the
 * only thing that moves weights in and out of memory.
 */
export const warmLungModels = async (
  model: LungModelSelector = 'all',
): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/warmup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    })
    if (!response.ok) return false
    const payload = (await response.json()) as LungWarmupResponse
    return payload.loaded.v2s || payload.loaded.xrv
  } catch {
    return false
  }
}

/** Release the lung models — called when the lungs section closes. */
export const unloadLungModels = async (
  model: LungModelSelector = 'all',
): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/unload`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model }),
    })
    if (!response.ok) return false
    const payload = (await response.json()) as LungUnloadResponse
    return payload.unloaded.v2s || payload.unloaded.xrv
  } catch {
    return false
  }
}
