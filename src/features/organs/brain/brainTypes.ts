/**
 * Types for the brain classification API.
 * Must match the FastAPI response schema from router.py.
 */

export type InferenceStatus = 'ok' | 'unavailable'

export interface Top5Prediction {
  readonly class: string
  readonly confidence: number
}

export interface BrainRegion3D {
  readonly name: string
  readonly display_name: string
  /** Approximate MNI centroid in millimetres. */
  readonly coordinates_3d: [number, number, number]
  readonly color: string
  readonly lobe: string
  readonly description: string
  /** Typical-site prior scaled by classifier confidence, not a lesion measurement. */
  readonly probability: number
  readonly rank: number
  readonly basis: string
}

export interface Explainability {
  readonly method: string
  readonly layer: string
  readonly interpretation: string
}

export interface BrainClassificationResult {
  readonly predicted_class: string
  readonly confidence: number
  readonly tumor_type: string
  readonly sequence: string
  readonly top5: readonly Top5Prediction[]
  readonly probabilities: readonly number[]
  readonly gradcam_base64: string
  readonly gradcam_grid: readonly [number, number]
  readonly explainability: Explainability
  readonly locations_3d: readonly BrainRegion3D[]
  readonly localization_basis: string
  readonly model_trained: boolean
  readonly label_space_source: string
  readonly inference_ms: number
  /** Present only when the backend was unreachable and a preset was simulated. */
  readonly simulated?: boolean
}

export interface DatasetSummary {
  readonly samples: number | null
  readonly tumour_types: number | null
  readonly sequences: readonly string[]
}

export interface BrainModelInfo {
  readonly model_name: string
  readonly model_tag: string
  readonly declared_model_tag: string
  readonly architecture_matches_declaration: boolean
  readonly num_classes: number
  readonly class_names: readonly string[]
  readonly input_size: string
  readonly task: string
  readonly dataset: DatasetSummary
  readonly label_space_source: string
  readonly trained_weights_loaded: boolean
  readonly metrics: Readonly<Record<string, unknown>>
}

export interface BrainHealthStatus {
  readonly status: InferenceStatus
  readonly model_loaded: boolean
  readonly trained_weights: boolean
  readonly checkpoint_present: boolean
  readonly detail?: string
  readonly warnings?: readonly string[]
}
