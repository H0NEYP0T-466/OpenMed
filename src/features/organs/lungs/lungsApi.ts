import type { LungClassificationResult, LungHealthStatus, LungModelInfo } from './lungTypes'

const API_BASE_URL: string =
  import.meta.env.VITE_LUNGS_API_BASE ?? 'http://localhost:8016/api/lungs'

export const NIH_FINDINGS = [
  'Atelectasis',
  'Cardiomegaly',
  'Effusion',
  'Infiltration',
  'Mass',
  'Nodule',
  'Pneumonia',
  'Pneumothorax',
  'Consolidation',
  'Edema',
  'Emphysema',
  'Fibrosis',
  'Pleural_Thickening',
  'Hernia',
] as const

/** Run Z-Net on one chest radiograph — POST /api/lungs/classify. */
export const classifyLungScan = async (file: File): Promise<LungClassificationResult> => {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch(`${API_BASE_URL}/classify`, { method: 'POST', body: form })
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const payload = (await response.json()) as { detail?: string }
      detail = payload.detail ?? detail
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail)
  }
  return (await response.json()) as LungClassificationResult
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

/** Load Z-Net into backend memory — called when the lungs section opens. */
export const warmLungModels = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/warmup`, { method: 'POST' })
    return response.ok
  } catch {
    return false
  }
}

/** Release Z-Net — called when the lungs section closes. */
export const unloadLungModels = async (): Promise<boolean> => {
  try {
    const response = await fetch(`${API_BASE_URL}/unload`, { method: 'POST' })
    return response.ok
  } catch {
    return false
  }
}
