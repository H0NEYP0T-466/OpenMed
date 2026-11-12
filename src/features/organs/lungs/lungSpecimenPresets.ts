/**
 * The lung specimen rack — one radiograph per class in the 4-class LungV2S
 * label space, so every class can be exercised without hunting for a file.
 *
 * Each plate is a real image from the COVID-19 Radiography Database, chosen
 * from the corpus the model was trained on. The subtitle names the source, not
 * a diagnosis — these are annotations, and the card should not claim more than
 * the label does.
 *
 * The four classes are, in head order: COVID, Lung_Opacity, Normal,
 * Viral Pneumonia (see `V2S_CLASS_ORDER` in `lungsApi.ts`).
 */

export interface LungSpecimenPreset {
  readonly plate: string
  readonly name: string
  readonly path: string
  readonly subtitle: string
  readonly site: string
}

const SOURCE = 'COVID-19 Radiography Database'

export const LUNG_SPECIMEN_PRESETS: readonly LungSpecimenPreset[] = [
  {
    plate: 'Pl. A',
    name: 'COVID',
    path: '/samples/lungs/covid.png',
    subtitle: SOURCE,
    site: 'Bilateral / peripheral',
  },
  {
    plate: 'Pl. B',
    name: 'Lung Opacity',
    path: '/samples/lungs/lung_opacity.png',
    subtitle: SOURCE,
    site: 'Focal / multifocal opacity',
  },
  {
    plate: 'Pl. C',
    name: 'Normal',
    path: '/samples/lungs/normal.png',
    subtitle: SOURCE,
    site: 'Clear lung fields',
  },
  {
    plate: 'Pl. D',
    name: 'Viral Pneumonia',
    path: '/samples/lungs/viral_pneumonia.png',
    subtitle: SOURCE,
    site: 'Bilateral / interstitial',
  },
]

export default LUNG_SPECIMEN_PRESETS
