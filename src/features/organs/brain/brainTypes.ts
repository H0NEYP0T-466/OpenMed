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
  /** Not returned by the /segment endpoint; kept for /classify compatibility. */
  readonly probabilities?: readonly number[]
  readonly gradcam_base64: string
  readonly gradcam_grid: readonly [number, number]
  readonly explainability: Explainability
  readonly locations_3d: readonly BrainRegion3D[]
  readonly localization_basis: string
  readonly model_trained: boolean
  readonly label_space_source: string
  readonly inference_ms: number
}

/**
 * Which prompt was handed to LiteMedSAM for the segmentation pass.
 *
 * - `heatmap_box` — a bounding box derived from the Grad-CAM heatmap. This is
 *   the path the released LiteMedSAM weights were trained for.
 * - `raw` — no prompt at all; the decoder runs from the image embedding alone.
 *   The unprompted arm of the ablation.
 */
export type SegPromptMode = 'heatmap_box' | 'raw'

/**
 * Response from POST /api/brain/segment.
 *
 * A superset of {@link BrainClassificationResult}: every classification field
 * is present, plus the LiteMedSAM segmentation result. When the classifier
 * predicts "Normal", ``segmentation_performed`` is ``false`` and no mask is
 * produced — the pipeline only segments when a tumour is detected.
 */
export interface BrainSegmentationResult extends BrainClassificationResult {
  readonly segmentation_performed: boolean
  /** Why segmentation did not run (e.g. classified as Normal). */
  readonly segmentation_skipped_reason?: string

  /** Which prompt produced the mask below. */
  readonly prompt_mode?: SegPromptMode

  /** Mask payload, present whenever a mask was produced, in either mode. */
  readonly seg_mask_base64?: string
  readonly seg_overlay_base64?: string
  readonly iou_pred?: number
  /**
   * Foreground pixel count of the returned mask. `0` means the model ran and
   * found nothing — distinct from `undefined`, which means no mask was produced
   * at all. An unprompted run commonly returns 0.
   */
  readonly mask_foreground_px?: number

  /** Box-prompt specifics — only meaningful when prompt_mode is `heatmap_box`. */
  readonly box_prompt_used?: boolean
  readonly box_coords?: number[]

  readonly segmentation_input_size?: string
  readonly original_size?: string
  readonly total_ms?: number
}

/**
 * Response from POST /api/brain/segment-click — the assistive path.
 *
 * The clinician clicks the suspicious region; the backend turns that click into
 * a small box and segments inside it. No classification runs on this path, so
 * there are no classifier fields here.
 */
export interface ClickSegmentationResult {
  readonly segmentation_performed: boolean
  readonly prompt_mode: 'click_box' | 'drawn_box'
  /** The click, in 256×256 prompt space. Present for click prompts only. */
  readonly click?: readonly [number, number]
  /** The box built around the click, or the box as drawn. */
  readonly box_coords: readonly number[]
  readonly seg_mask_base64?: string
  readonly seg_overlay_base64?: string
  readonly iou_pred?: number
  readonly mask_foreground_px?: number
  readonly segmentation_input_size?: string
  readonly original_size?: string
  readonly total_ms?: number
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
