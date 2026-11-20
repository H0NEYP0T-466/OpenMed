/**
 * Types for the lung-field + COVID opacity segmentation endpoints under
 * `/api/lungs` (`/segment`, `/seg-health`, `/seg-model-info`, `/seg-warmup`,
 * `/seg-unload`). They mirror the HTTP contract field for field.
 */

export type LungSegErrorCode =
  | 'checkpoint_missing'
  | 'checkpoint_unusable'
  | 'invalid_image'
  | 'unsupported_media'
  | 'payload_too_large'
  | 'inference_failed'

export interface LungSegSize {
  readonly width: number
  readonly height: number
}

export interface LungSegLegendEntry {
  readonly value: number
  readonly name: string
}

export interface LungSegPostprocessLung {
  readonly threshold: number
  readonly min_area_frac: number
  readonly keep_largest_k: number
  readonly fill_holes: boolean
  readonly tta_hflip: boolean
}

export interface LungSegPostprocessOpacity {
  readonly threshold: number
  readonly min_area_frac: number
  readonly tta_hflip: boolean
}

export interface LungSegPostprocessInfo {
  readonly lung: LungSegPostprocessLung
  readonly opacity: LungSegPostprocessOpacity | null
}

export interface LungSegStats {
  readonly lung_area_px: number
  readonly lung_fraction_of_frame: number
  readonly lung_components: number
  readonly lung_bbox: number[] | null
  readonly opacity_area_px: number | null
  readonly opacity_fraction_of_lung: number | null
  readonly opacity_components: number | null
  readonly opacity_peak_probability: number | null
  readonly opacity_mean_probability: number | null
  /** `[x0, y0, x1, y1]` in `mask_size` pixels: the lung crop given to the opacity head. */
  readonly crop_box: number[] | null
  /** True when no lung was found and the whole frame was used. */
  readonly crop_fallback: boolean
}

export interface LungSegResponse {
  readonly task: string
  readonly model_tag: string
  readonly opacity_available: boolean
  readonly opacity_unavailable_reason: string | null
  /** Dimensions of the original upload. */
  readonly image_size: LungSegSize
  /** Dimensions of the three PNGs below. */
  readonly mask_size: LungSegSize
  /** `data:image/png;base64,...` white (255) = lung field. */
  readonly lung_mask_png: string
  /** Same encoding, already clipped to the lung field. */
  readonly opacity_mask_png: string | null
  /** Pixel VALUES 0/1/2 (0 background, 1 lung field only, 2 opacity). Not for display. */
  readonly label_map_png: string
  readonly legend: LungSegLegendEntry[]
  readonly stats: LungSegStats
  readonly postprocess: LungSegPostprocessInfo
  readonly inference_ms: number
  readonly scope_notice: string
}

export interface LungSegSummary {
  readonly n: number
  readonly n_finite?: number
  readonly mean: number | null
  readonly median: number | null
  readonly std: number | null
  readonly p5: number | null
  readonly p95: number | null
}

export interface LungSegByGroup {
  readonly n: number
  readonly dice: LungSegSummary
}

export interface LungSegLungFieldMetrics {
  readonly dice: LungSegSummary
  readonly iou: LungSegSummary
  readonly hd95_px: LungSegSummary
  readonly by_class: Record<string, LungSegByGroup>
}

export interface LungSegOpacityPositives {
  readonly n: number
  readonly dice: LungSegSummary
  readonly iou: LungSegSummary
  readonly hd95_px: LungSegSummary
  readonly by_size: Record<string, LungSegByGroup>
}

export interface LungSegOpacityNormal {
  readonly n: number
  readonly n_with_prediction: number
  readonly false_positive_rate: number | null
  readonly mean_predicted_fraction_of_lung: number | null
}

export interface LungSegOpacityNonCovid {
  readonly n: number
  readonly any_opacity_rate: number | null
  readonly mean_predicted_fraction_of_lung: number | null
  readonly note: string
}

export interface LungSegOpacityMetrics {
  readonly positives: LungSegOpacityPositives
  readonly normal: LungSegOpacityNormal
  readonly non_covid: LungSegOpacityNonCovid
}

export interface LungSegMetrics {
  readonly split: string
  readonly n_images: number
  readonly note: string
  readonly lung_field: LungSegLungFieldMetrics
  readonly opacity: LungSegOpacityMetrics | null
}

export interface LungSegTrainingInfo {
  readonly epoch: number | null
  readonly weights: string | null
  readonly stage: string | null
  readonly seed: number | null
  readonly split_counts: Record<string, number> | null
}

export interface LungSegCheckpointInfo {
  readonly name: string
  readonly size_mb: number | null
  readonly modified: string | null
}

export interface LungSegInfo {
  readonly task: string
  readonly model_tag: string
  readonly parameters: number
  readonly input_size: string
  readonly loaded: boolean
  readonly device: string | null
  readonly heads: string[]
  readonly heads_trained: string[]
  readonly preprocess_version: number
  readonly postprocess: LungSegPostprocessInfo
  readonly metrics: {
    readonly test: LungSegMetrics | null
    readonly val: LungSegMetrics | null
  }
  readonly training: LungSegTrainingInfo
  readonly checkpoint: LungSegCheckpointInfo
  readonly scope_notice: string
}

export interface LungSegModelInfoResponse {
  readonly available: boolean
  readonly detail: string | null
  /** `{}` when unavailable. */
  readonly info: Partial<LungSegInfo>
}

export interface LungSegHealth {
  readonly status: 'ok' | 'unavailable'
  readonly model_loaded: boolean
  readonly checkpoint_present: boolean
  readonly detail: string | null
}

export interface LungSegWarmupResponse extends LungSegHealth {
  readonly device: string | null
}

export interface LungSegUnloadResponse {
  readonly status: 'ok'
  readonly unloaded: boolean
  readonly message: string
}

/** Outcome of a best-effort residency call. Never thrown. */
export interface LungSegResidencyResult {
  readonly ok: boolean
  /** Server detail or a client-side reason; `null` on a clean success. */
  readonly detail: string | null
}
