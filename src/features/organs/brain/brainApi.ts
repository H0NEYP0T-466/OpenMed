import type {
  BrainHealthStatus,
  BrainModelInfo,
  BrainSegmentationResult,
} from './brainTypes'

const API_BASE_URL: string =
  import.meta.env.VITE_BRAIN_API_BASE ?? 'http://localhost:8016/api/brain'

/**
 * Classify an MRI scan *and* segment any detected tumour in one round trip.
 *
 * POST /api/brain/segment runs the full pipeline: EfficientNetV2-B2
 * classification with Grad-CAM, and if the prediction is any tumour type
 * (not "Normal"), LiteMedSAM segmentation prompted by the bounding box
 * derived from the Grad-CAM heatmap. The released LiteMedSAM weights are
 * box-prompt-only, so no dense mask prompt is sent.
 *
 * Every result returned here is computed from the uploaded image by the
 * backend. There is no cached or simulated fallback: if the service is
 * unreachable this throws, so the UI can never show an invented finding.
 */
export const analyzeBrainImage = async (
  file: File,
): Promise<BrainSegmentationResult> => {
  const formData = new FormData()
  formData.append('file', file)

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

export const getBrainModelInfo = async (): Promise<BrainModelInfo> => {
  const response = await fetch(`${API_BASE_URL}/model-info`)
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
