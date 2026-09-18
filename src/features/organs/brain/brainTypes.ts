/**
 * Types for the brain classification API.
 * Must match the FastAPI response schema from router.py.
 */

export interface Top5Prediction {
  readonly class: string
  readonly confidence: number
}

export interface BrainRegion3D {
  readonly name: string
  readonly display_name: string
  readonly coordinates_3d: [number, number, number]
  readonly color: string
  readonly lobe: string
  readonly description: string
  readonly probability: number
}

export interface BrainClassificationResult {
  readonly predicted_class: string
  readonly confidence: number
  readonly tumor_type: string
  readonly sequence: string
  readonly top5: Top5Prediction[]
  readonly probabilities: number[]
  readonly gradcam_base64: string
  readonly locations_3d: BrainRegion3D[]
}

export interface BrainModelInfo {
  readonly model_name: string
  readonly model_tag: string
  readonly num_classes: number
  readonly class_names: string[]
  readonly input_size: string
  readonly task: string
  readonly dataset: string
}

export interface BrainHealthStatus {
  readonly status: string
  readonly model_loaded: boolean
}
