import type {
  BrainHealthStatus,
  BrainModelInfo,
  BrainClassificationResult,
  BrainSegmentationResult,
  ClickSegmentationResult,
} from './brainTypes'

const API_BASE_URL: string =
  import.meta.env.VITE_BRAIN_API_BASE ?? 'http://localhost:8016/api/brain'

/**
 * Options for {@link analyzeBrainImage}.
 */
export interface AnalyzeBrainImageOptions {
  /**
   * When `true` (the default) the Grad-CAM heatmap is reduced to a bounding box
   * and handed to LiteMedSAM as its prompt. When `false` the model receives no
   * prompt at all and segments from the image embedding alone.
   *
   * Classification is unaffected either way, and a "Normal" prediction still
   * skips segmentation entirely.
   */
  readonly useHeatmapPrompt?: boolean
}

/**
 * Classify an MRI scan — POST /api/brain/classify.
 *
 * This is what the Run button calls. Segmentation was taken out of this flow
 * when the automated heatmap-prompt path was retired: against ground truth it
 * scored 0.31 mean Dice, because a CAM-derived box overlaps the lesion at only
 * 0.123 IoU. Segmentation now lives in the Experimental Laboratory, where the
 * clinician supplies the prompt and the same decoder reaches 0.92 median Dice.
 *
 * ``/api/brain/segment`` is untouched and still runs classification plus
 * LiteMedSAM; it is kept for the retrained model.
 *
 * Every result is computed from the uploaded image by the backend. There is no
 * cached or simulated fallback: if the service is unreachable this throws, so
 * the UI can never show an invented finding.
 */
export const classifyBrainImage = async (file: File): Promise<BrainClassificationResult> => {
  const formData = new FormData()
  formData.append('file', file)

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/classify`, { method: 'POST', body: formData })
  } catch (error) {
    throw new Error(
      `The analysis service is unreachable at ${API_BASE_URL}. Start it with ` +
        '`python -m app.main` from backend/, then run the analysis again. ' +
        'No result is shown without a real inference run.',
      { cause: error },
    )
  }

  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    const message =
      detail && typeof detail.detail === 'string'
        ? detail.detail
        : `Classification failed (HTTP ${response.status}).`
    throw new Error(message)
  }

  return (await response.json()) as BrainClassificationResult
}

/**
 * Classify an MRI scan *and* segment any detected tumour in one round trip.
 *
 * POST /api/brain/segment runs the full pipeline: EfficientNetV2-B2
 * classification with Grad-CAM, and if the prediction is any tumour type
 * (not "Normal"), LiteMedSAM segmentation. The segmentation prompt depends on
 * `options.useHeatmapPrompt`: a bounding box derived from the Grad-CAM heatmap
 * when enabled, or no prompt at all when disabled. Either way the released
 * LiteMedSAM weights never receive a dense mask prompt.
 *
 * No longer called by the workspace — kept for the retrained segmentation
 * model. Use {@link classifyBrainImage} for the classification-only run.
 */
export const analyzeBrainImage = async (
  file: File,
  options: AnalyzeBrainImageOptions = {},
): Promise<BrainSegmentationResult> => {
  const useHeatmapPrompt = options.useHeatmapPrompt ?? true

  const formData = new FormData()
  formData.append('file', file)
  formData.append('use_heatmap_prompt', String(useHeatmapPrompt))

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/segment`, { method: 'POST', body: formData })
  } catch (error) {
    throw new Error(
      `The analysis service is unreachable at ${API_BASE_URL}. Start it with ` +
        '`python -m app.main` from backend/, then run the analysis again. ' +
        'No result is shown without a real inference run.',
      { cause: error },
    )
  }

  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    const message =
      detail && typeof detail.detail === 'string'
        ? detail.detail
        : `Analysis failed (HTTP ${response.status}).`
    throw new Error(message)
  }

  return (await response.json()) as BrainSegmentationResult
}

export interface SegmentByClickOptions {
  /** Click position, normalised 0–1 against the displayed image. */
  readonly clickX: number
  readonly clickY: number
  /** Side of the box built around the click, in 256² prompt space. */
  readonly boxSize?: number
}

/**
 * Segment from a single click on the suspicious region — the assistive path.
 *
 * The backend turns the click into a small box centred on it and runs
 * LiteMedSAM once, with no classification pass, which is what keeps the click
 * feeling immediate. Measured against ground truth on the held-out split, the
 * 48px click box reached a median Dice of 0.888 against 0.254 for the same
 * click sent as a bare point — the released weights respond to boxes, not
 * points.
 */
export const segmentByClick = async (
  file: File,
  { clickX, clickY, boxSize = 48 }: SegmentByClickOptions,
): Promise<ClickSegmentationResult> => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('click_x', String(clickX))
  formData.append('click_y', String(clickY))
  formData.append('box_size', String(boxSize))

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/segment-click`, {
      method: 'POST',
      body: formData,
    })
  } catch (error) {
    throw new Error(
      `The analysis service is unreachable at ${API_BASE_URL}. ` +
        'Start it with `python -m app.main` from backend/, then click again.',
      { cause: error },
    )
  }

  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    const message =
      detail && typeof detail.detail === 'string'
        ? detail.detail
        : `Click segmentation failed (HTTP ${response.status}).`
    throw new Error(message)
  }

  return (await response.json()) as ClickSegmentationResult
}

export interface SegmentByBoxOptions {
  /** Box corners, normalised 0–1 against the displayed image. Either corner order. */
  readonly x1: number
  readonly y1: number
  readonly x2: number
  readonly y2: number
}

/**
 * Segment inside a box the clinician drew on the scan.
 *
 * The strongest assistive prompt available — the box is the prompt type the
 * released weights were trained on, so a well-drawn one scores near the oracle
 * (0.89 Dice measured against ground truth for a box roughly matching the
 * lesion). Same no-classification fast path as the click.
 */
export const segmentByBox = async (
  file: File,
  { x1, y1, x2, y2 }: SegmentByBoxOptions,
): Promise<ClickSegmentationResult> => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('x1', String(x1))
  formData.append('y1', String(y1))
  formData.append('x2', String(x2))
  formData.append('y2', String(y2))

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/segment-box`, {
      method: 'POST',
      body: formData,
    })
  } catch (error) {
    throw new Error(
      `The analysis service is unreachable at ${API_BASE_URL}. ` +
        'Start it with `python -m app.main` from backend/, then draw again.',
      { cause: error },
    )
  }

  if (!response.ok) {
    const detail = await response.json().catch(() => null)
    const message =
      detail && typeof detail.detail === 'string'
        ? detail.detail
        : `Box segmentation failed (HTTP ${response.status}).`
    throw new Error(message)
  }

  return (await response.json()) as ClickSegmentationResult
}

export const getBrainModelInfo = async (): Promise<BrainModelInfo> => {  const response = await fetch(`${API_BASE_URL}/model-info`)
  if (!response.ok) {
    throw new Error(`Failed to fetch model info (HTTP ${response.status})`)
  }
  return (await response.json()) as BrainModelInfo
}

export const getBrainHealth = async (): Promise<BrainHealthStatus> => {
  const response = await fetch(`${API_BASE_URL}/health`)
  if (!response.ok) {
    throw new Error(`Health check failed (HTTP ${response.status})`)
  }
  return (await response.json()) as BrainHealthStatus
}
