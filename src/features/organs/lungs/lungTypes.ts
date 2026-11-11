/**
 * Response shapes for `/api/lungs`.
 *
 * These mirror `backend/app/organs/lungs/classification/pipeline.py` and
 * `router.py` exactly, including the nullable fields. A `null` model means that
 * model failed for this request while the other one succeeded — it is not an
 * empty result, and the two must not be collapsed into one another.
 */

/** Model A — LungV2S, four-class single-label softmax. */
export interface LungV2SResult {
  readonly top_class: string
  readonly confidence: number
  /** One entry per class, in `class_names` order. */
  readonly probabilities: number[]
  readonly class_names: string[]
  /** PNG, base64 (no data-URL prefix): the Grad-CAM blended over the scan. */
  readonly gradcam_base64: string
  /** `[width, height]` of the rendered Grad-CAM canvas, in infer-size pixels. */
  readonly gradcam_grid: number[]
  /** `[x1, y1, x2, y2]` in infer-size pixels, or `null` when the CAM is empty. */
  readonly bbox: number[] | null
  readonly inference_ms: number
  readonly model_tag: string
  readonly infer_size: number
}

export interface LungXrvFinding {
  readonly finding: string
  readonly probability: number
  readonly rank: number
}

export interface LungXrvScore {
  readonly finding: string
  readonly probability: number
}

/** Model B — stock XRV DenseNet-121, 18-label multilabel sigmoid. */
export interface LungXrvResult {
  /** Only findings at or above `threshold`, sorted by probability descending. */
  readonly findings: LungXrvFinding[]
  readonly threshold: number
  readonly count_above_threshold: number
  /** All 18 scores, only present when the request asked for detail. */
  readonly scores: LungXrvScore[] | null
  /** XRV Grad-CAM PNG base64, only present when the request asked for detail. */
  readonly gradcam_base64: string | null
  readonly inference_ms: number
  readonly model_tag: string
}

/** `POST /classify` */
export interface LungClassifyResponse {
  readonly v2s: LungV2SResult | null
  readonly xrv: LungXrvResult | null
  readonly errors: {
    readonly v2s: string | null
    readonly xrv: string | null
  }
  readonly total_ms: number
}

export interface LungModelHealth {
  readonly loaded: boolean
  readonly checkpoint_present: boolean
  readonly detail: string | null
}

/** `GET /health` */
export interface LungHealthStatus {
  readonly status: 'ok' | 'degraded' | 'unavailable'
  readonly v2s: LungModelHealth
  readonly xrv: LungModelHealth
}

export interface LungV2SCheckpointInfo {
  readonly path: string
  readonly sidecar: string
  readonly present: boolean
  readonly sidecar_present: boolean
  readonly loaded: boolean
  readonly epoch: number | null
  readonly kind: string | null
  readonly metrics: Record<string, unknown> | null
  readonly infer_size: number
  readonly size_mb?: number
  readonly modified?: string
}

export interface LungXrvCheckpointInfo {
  readonly path: string
  readonly present: boolean
  readonly loaded: boolean
  readonly weights: string
  readonly pretrained: boolean
  readonly epoch: number | null
  readonly kind: string
  readonly input_size: number
  readonly size_mb?: number
  readonly modified?: string
}

/** `GET /model-info` */
export interface LungModelInfo {
  readonly v2s: {
    readonly model_tag: string
    readonly classes: string[]
    readonly input_size: number
    readonly checkpoint: LungV2SCheckpointInfo
  }
  readonly xrv: {
    readonly model_tag: string
    readonly classes: string[]
    readonly input_size: number
    readonly checkpoint: LungXrvCheckpointInfo
  }
}

export type LungModelSelector = 'v2s' | 'xrv' | 'all'

/** `POST /warmup` */
export interface LungWarmupResponse {
  readonly status: 'ok' | 'partial'
  readonly requested: LungModelSelector
  readonly loaded: { readonly v2s: boolean; readonly xrv: boolean }
  readonly errors: { readonly v2s: string | null; readonly xrv: string | null }
  readonly device: string
}

/** `POST /unload` */
export interface LungUnloadResponse {
  readonly status: 'ok'
  readonly requested: LungModelSelector
  readonly unloaded: { readonly v2s: boolean; readonly xrv: boolean }
  readonly message: string
}
