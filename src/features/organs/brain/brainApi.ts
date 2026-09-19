import type {
  BrainClassificationResult,
  BrainHealthStatus,
  BrainModelInfo,
  BrainRegion3D,
  Top5Prediction,
} from './brainTypes'

const API_BASE_URL: string =
  import.meta.env.VITE_BRAIN_API_BASE ?? 'http://localhost:8016/api/brain'

interface PresetSpec {
  readonly key: string
  readonly predictedClass: string
  readonly confidence: number
  readonly differential: readonly [string, number][]
  readonly regions: readonly Omit<BrainRegion3D, 'rank' | 'basis'>[]
}

const PRESET_SPECS: readonly PresetSpec[] = [
  {
    key: 'meningioma',
    predictedClass: 'Meningioma T1C+',
    confidence: 0.948,
    differential: [
      ['Meningioma T1C+', 0.948],
      ['Schwannoma T1C+', 0.026],
      ['Glioblastoma T1C+', 0.014],
      ['Astrocytoma T1C+', 0.007],
      ['Hemangiopericytoma T1C+', 0.005],
    ],
    regions: [
      {
        name: 'convexity',
        display_name: 'Cerebral Convexity',
        coordinates_3d: [27.2, 0.0, 28.6],
        color: '#c58696',
        lobe: 'Meningeal',
        description: 'Outer brain surface beneath the calvarium — meningioma site.',
        probability: 0.948,
      },
      {
        name: 'falx',
        display_name: 'Falx Cerebri',
        coordinates_3d: [0.0, 0.0, 26.0],
        color: '#c58696',
        lobe: 'Meningeal',
        description: 'Midline dural fold separating cerebral hemispheres.',
        probability: 0.682,
      },
    ],
  },
  {
    key: 'glioblastoma',
    predictedClass: 'Glioblastoma T1C+',
    confidence: 0.923,
    differential: [
      ['Glioblastoma T1C+', 0.923],
      ['Astrocytoma T1C+', 0.045],
      ['Oligodendroglioma T1C+', 0.018],
      ['Meningioma T1C+', 0.009],
      ['Medulloblastoma T1', 0.005],
    ],
    regions: [
      {
        name: 'frontal',
        display_name: 'Frontal Lobe',
        coordinates_3d: [0.0, 45.5, 15.6],
        color: '#ee7c6a',
        lobe: 'Frontal',
        description: 'Anterior cerebral cortex — executive function, motor planning.',
        probability: 0.923,
      },
      {
        name: 'corpus callosum',
        display_name: 'Corpus Callosum',
        coordinates_3d: [0.0, 0.0, 15.6],
        color: '#b2bec3',
        lobe: 'White Matter',
        description: 'Major commissure connecting the two hemispheres.',
        probability: 0.664,
      },
    ],
  },
  {
    key: 'astrocytoma',
    predictedClass: 'Astrocytoma T1',
    confidence: 0.894,
    differential: [
      ['Astrocytoma T1', 0.894],
      ['Oligodendroglioma T1', 0.052],
      ['Glioblastoma T1', 0.031],
      ['Ganglioglioma T1', 0.014],
      ['Dysembryoplastic Neuroepithelial Tumor T1', 0.009],
    ],
    regions: [
      {
        name: 'temporal',
        display_name: 'Temporal Lobe',
        coordinates_3d: [47.6, -7.0, -10.4],
        color: '#6393d8',
        lobe: 'Temporal',
        description: 'Lateral cortex — auditory processing, hippocampus, memory.',
        probability: 0.894,
      },
      {
        name: 'insular',
        display_name: 'Insular Cortex',
        coordinates_3d: [37.4, 3.5, 0.0],
        color: '#fab1a0',
        lobe: 'Insular',
        description: 'Deep cortex beneath the Sylvian fissure — glioma hotspot.',
        probability: 0.644,
      },
    ],
  },
  {
    key: 'medulloblastoma',
    predictedClass: 'Medulloblastoma T1',
    confidence: 0.951,
    differential: [
      ['Medulloblastoma T1', 0.951],
      ['Ependymoma - Subependymoma T1', 0.027],
      ['Astrocytoma T1', 0.012],
      ['Neurocytoma T1', 0.006],
      ['Germinoma T1', 0.004],
    ],
    regions: [
      {
        name: 'cerebellum',
        display_name: 'Cerebellum',
        coordinates_3d: [0.0, -49.0, -23.4],
        color: '#1abc9c',
        lobe: 'Posterior Fossa',
        description: 'Motor coordination, procedural learning, balance.',
        probability: 0.951,
      },
      {
        name: 'fourth ventricle',
        display_name: 'Fourth Ventricle',
        coordinates_3d: [0.0, -31.5, -18.2],
        color: '#81ecec',
        lobe: 'Posterior Fossa',
        description: 'CSF cavity between brainstem and cerebellum.',
        probability: 0.685,
      },
    ],
  },
]

const LOCALIZATION_BASIS =
  'Typical presentation sites for the predicted tumour type, weighted by the ' +
  "classifier's confidence. This is a population prior, not a measurement of " +
  "where this patient's lesion sits."

function buildPreset(spec: PresetSpec): BrainClassificationResult {
  const top5: Top5Prediction[] = spec.differential.map(([name, confidence]) => ({
    class: name,
    confidence,
  }))

  return {
    predicted_class: spec.predictedClass,
    confidence: spec.confidence,
    tumor_type: spec.predictedClass.replace(/ (T1C\+|T1|T2)$/, ''),
    sequence: (spec.predictedClass.match(/(T1C\+|T1|T2)$/) ?? ['Unknown'])[0],
    top5,
    probabilities: [],
    gradcam_base64: '',
    gradcam_grid: [7, 7],
    explainability: {
      method: 'gradcam',
      layer: 'Conv2d',
      interpretation:
        'Simulated specimen preset. No activation map was computed because the backend was unreachable.',
    },
    locations_3d: spec.regions.map((region, index) => ({
      ...region,
      rank: index + 1,
      basis: 'tumour_type_prior',
    })),
    localization_basis: LOCALIZATION_BASIS,
    model_trained: true,
    label_space_source: 'preset',
    inference_ms: 0,
    simulated: true,
  }
}

const SIMULATED_PRESETS: Readonly<Record<string, BrainClassificationResult>> = Object.fromEntries(
  PRESET_SPECS.map((spec) => [spec.key, buildPreset(spec)]),
)

function isNetworkFailure(error: unknown): boolean {
  if (!(error instanceof TypeError)) return false
  const message = error instanceof Error ? error.message : ''
  return /fetch|network|load/i.test(message)
}

function matchPreset(fileName: string): BrainClassificationResult | null {
  const lowered = fileName.toLowerCase()
  for (const [key, preset] of Object.entries(SIMULATED_PRESETS)) {
    if (lowered.includes(key)) return preset
  }
  return null
}

export const classifyBrainImage = async (
  file: File,
): Promise<BrainClassificationResult> => {
  const formData = new FormData()
  formData.append('file', file)

  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}/classify`, { method: 'POST', body: formData })
  } catch (error) {
    if (isNetworkFailure(error)) {
      const preset = matchPreset(file.name)
      if (preset) return preset
      throw new Error(
        `The analysis service is offline at ${API_BASE_URL}. Start it with ` +
          '`python -m app.main` from backend/, or use a labelled specimen preset. ' +
          'Uploaded images are never answered from cached results.',
      )
    }
    throw error
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

export const isSimulatedResult = (
  result: BrainClassificationResult | null,
): boolean => result?.simulated === true
