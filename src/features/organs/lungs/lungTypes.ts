export interface LungTop5Entry {
  readonly finding: string
  readonly probability: number
  readonly rank: number
}

export interface LungClassificationResult {
  readonly top_finding: string
  readonly confidence: number
  readonly probabilities: number[]
  readonly top5: LungTop5Entry[]
  readonly gradcam_base64: string
  readonly gradcam_grid: number[]
  readonly bbox: number[] | null
  readonly inference_ms: number
  readonly model_tag: string
}

export interface LungHealthStatus {
  readonly status: 'ok' | 'unavailable'
  readonly model_loaded: boolean
  readonly checkpoint_present: boolean
  readonly detail: string | null
}

export interface LungModelInfo {
  readonly model_tag: string
  readonly findings: string[]
  readonly paper: string
  readonly input_size: number
  readonly checkpoint: {
    readonly present: boolean
    readonly path: string
    readonly size_mb?: number
    readonly modified?: string
  }
}
