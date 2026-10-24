const API_BASE_URL: string =
  import.meta.env.VITE_XRV_API_BASE ?? 'http://localhost:8016/api/experimental/torchxrayvision'

export interface XrvStatus {
  readonly installed: boolean
  readonly loaded: boolean
  readonly weights_present: boolean
  readonly pathologies: string[]
  readonly loaded_at: number | null
  readonly load_ms: number | null
}

export interface XrvPrediction {
  readonly pathology: string
  readonly probability: number
}

export interface XrvResult {
  readonly predictions: XrvPrediction[]
  readonly top_pathology: string
  readonly confidence: number
  readonly gradcam_base64: string
  readonly inference_ms: number
  readonly weights: string
}

const describe = async (response: Response): Promise<string> => {
  try {
    const payload = (await response.json()) as { detail?: string }
    return payload.detail ?? `HTTP ${response.status}`
  } catch {
    return `HTTP ${response.status}`
  }
}

export const getXrvStatus = async (): Promise<XrvStatus> => {
  const response = await fetch(`${API_BASE_URL}/status`)
  if (!response.ok) throw new Error(await describe(response))
  return (await response.json()) as XrvStatus
}

/** Load the model — fired when the tab is selected. */
export const loadXrv = async (): Promise<XrvStatus> => {
  const response = await fetch(`${API_BASE_URL}/load`, { method: 'POST' })
  if (!response.ok) throw new Error(await describe(response))
  return (await response.json()) as XrvStatus
}

/** Release it — fired from the tab's unmount cleanup. */
export const unloadXrv = async (): Promise<XrvStatus | null> => {
  try {
    const response = await fetch(`${API_BASE_URL}/unload`, { method: 'POST' })
    if (!response.ok) return null
    return (await response.json()) as XrvStatus
  } catch {
    return null
  }
}

export const predictXrv = async (file: File): Promise<XrvResult> => {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch(`${API_BASE_URL}/predict`, { method: 'POST', body: form })
  if (!response.ok) throw new Error(await describe(response))
  return (await response.json()) as XrvResult
}
