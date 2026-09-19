export interface ArtifactCard {
  readonly id: string
  readonly filename: string
  readonly format: string
  readonly title: string
  readonly desc: string
  readonly category: string
  readonly previewPath?: string
  readonly isText?: boolean
}

export interface OrganDirectoryItem {
  readonly id: string
  readonly name: string
  readonly latin: string
  readonly system: string
  readonly modelFile: string
  readonly modality: string
  readonly classificationTask: string
  readonly segmentationTask: string
  readonly hotspotsCount: number
  readonly active?: boolean
  readonly iconType:
    | 'brain'
    | 'heart'
    | 'lungs'
    | 'body'
    | 'kidney'
    | 'liver'
    | 'eye'
    | 'blood'
    | 'bone'
    | 'breast'
    | 'skin'
    | 'pancreas'
    | 'intestine'
}
