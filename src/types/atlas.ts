export type SystemId =
  | 'skeletal'
  | 'muscular'
  | 'arterial'
  | 'venous'
  | 'nervous'
  | 'digestive'
  | 'respiratory'
  | 'urinary'
  | 'reproductive'
  | 'lymphatic'
  | 'endocrine'
  | 'integumentary'
  | 'connective'
  | 'sensory'
  | 'cardiac'

export interface SystemDefinition {
  readonly id: SystemId
  readonly name: string
  readonly color: string
  readonly description: string
}

export const ATLAS_SYSTEMS: readonly SystemDefinition[] = [
  {
    id: 'skeletal',
    name: 'Skeleton',
    color: '#e2d9ba',
    description: 'Bones form the supporting framework of the body, protect vital organs, and store metabolic minerals while producing blood cells in bone marrow.',
  },
  {
    id: 'muscular',
    name: 'Muscles',
    color: '#a85b50',
    description: 'Skeletal muscles contract to pull on tendons, generating biomechanical force, stabilizing posture, and maintaining thermal body heat.',
  },
  {
    id: 'cardiac',
    name: 'Heart',
    color: '#b96760',
    description: 'Four-chamber muscular pump with specialized valve networks coordinating systemic and pulmonary circulation.',
  },
  {
    id: 'arterial',
    name: 'Arteries',
    color: '#c05245',
    description: 'High-pressure vascular conduits transporting oxygen-rich blood from the left ventricle outward to systemic capillary beds.',
  },
  {
    id: 'venous',
    name: 'Veins',
    color: '#527c9f',
    description: 'Capacitance network returning deoxygenated blood and metabolic metabolites back toward the right atrium.',
  },
  {
    id: 'nervous',
    name: 'Nervous System',
    color: '#d8b565',
    description: 'Brain, spinal cord, and peripheral axon tracts processing sensory inputs, conscious thought, and autonomic regulation.',
  },
  {
    id: 'respiratory',
    name: 'Respiratory',
    color: '#b98991',
    description: 'Tracheobronchial arborization and pulmonary lobes executing rapid alveolar-capillary oxygen/carbon dioxide diffusion.',
  },
  {
    id: 'digestive',
    name: 'Digestive',
    color: '#b8916b',
    description: 'Gastrointestinal tract and accessory metabolic organs (liver, pancreas) converting nutrients and managing systemic metabolism.',
  },
  {
    id: 'urinary',
    name: 'Urinary',
    color: '#b47961',
    description: 'Bilateral kidneys, ureters, and bladder filtering nitrogenous waste and balancing plasma electrolytes and systemic blood pressure.',
  },
  {
    id: 'lymphatic',
    name: 'Lymphatic',
    color: '#879f7c',
    description: 'Lymph nodes, vessels, and spleen maintaining interstitial fluid balance and executing adaptive immune surveillance.',
  },
  {
    id: 'endocrine',
    name: 'Endocrine',
    color: '#c5a09a',
    description: 'Ductless glandular organs releasing systemic regulatory hormones into circulation to govern metabolic setpoints.',
  },
  {
    id: 'integumentary',
    name: 'Body Surface / Skin',
    color: '#ba9b7d',
    description: 'Cutaneous boundary serving as protective microbial barrier, thermoregulatory radiator, and tactile sensory surface.',
  },
  {
    id: 'sensory',
    name: 'Sensory Organs',
    color: '#b0c8ce',
    description: 'Specialized sensory structures (eyes, optic nerves, auditory apparatus) transducing external physical stimuli into neural action potentials.',
  },
  {
    id: 'connective',
    name: 'Connective Tissue',
    color: '#aec3bb',
    description: 'Cartilage, fascia, and articular ligaments distributing mechanical stress and stabilizing musculoskeletal joints.',
  },
  {
    id: 'reproductive',
    name: 'Reproductive',
    color: '#bda098',
    description: 'Structures supporting gametogenesis, reproductive endocrine signaling, and hormonal maturation.',
  },
]

export interface Part {
  readonly id: string
  readonly name: string
  readonly conceptId: string
  readonly system: SystemId
  readonly chunk: number
  readonly positions: number
  readonly normals: number
  readonly indices: number
  readonly vertexCount: number
  readonly indexCount: number
  readonly bounds: readonly [readonly number[], readonly number[]]
}

export interface Concept {
  readonly id: string
  readonly name: string
  readonly elements: readonly string[]
}

export interface AtlasChunk {
  readonly url: string
  readonly bytes: number
  readonly gzip?: string
  readonly gzipBytes?: number
}

export interface AtlasData {
  readonly version: string
  readonly sex?: 'male' | 'female'
  readonly source?: string
  readonly scope?: string
  readonly parts: readonly Part[]
  readonly concepts: readonly Concept[]
  readonly chunks: readonly AtlasChunk[]
  readonly triangles: number
}

export type AtlasViewAngle = 'three-quarter' | 'front' | 'back' | 'side'

export interface AtlasSceneState {
  readonly explode: number
  readonly visibleSystems: readonly SystemId[]
  readonly selectedPartId: string | null
  readonly isolate: boolean
  readonly viewAngle: AtlasViewAngle
  readonly autoRotate: boolean
}

export const DEFAULT_ATLAS_SYSTEMS: readonly SystemId[] = [
  'cardiac',
  'sensory',
  'skeletal',
  'muscular',
  'arterial',
  'venous',
  'nervous',
  'respiratory',
  'digestive',
  'urinary',
  'lymphatic',
  'endocrine',
  'reproductive',
  'connective',
  'integumentary',
]
