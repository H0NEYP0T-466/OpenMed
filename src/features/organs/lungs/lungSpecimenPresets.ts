/**
 * The lung specimen rack — one radiograph per class in the 4-class LungV2S
 * label space, plus the two NIH plates, so every class can be exercised without
 * hunting for a file.
 *
 * The first four are real images from the COVID-19 Radiography Database, chosen
 * from the corpus the model was trained on. Each one was run through the live
 * classifier and returns its own class as the top label, so a plate never
 * promises an answer the model will not give.
 *
 * The last two are NIH ChestX-ray14 images, and they belong here rather than in
 * the segmentation rack because XRV's DenseNet was trained on that corpus: the
 * 18-label head actually names findings on these. LungV2S has never seen NIH, so
 * its 4-class read on them is a domain-shift observation, not a score. Both
 * plates were picked after a live run — every NIH sample in `samples/chest` was
 * classified and these are the two whose findings the models return correctly:
 *
 *   pneumonia.png      XRV: Lung Opacity 0.74, Pneumonia 0.65   v2s: Lung_Opacity 0.94
 *   consolidation.png  XRV: Lung Opacity 0.79, Pneumonia 0.65   v2s: Lung_Opacity 0.97
 *
 * (infiltration.png, the previous NIH plate, scored 0 findings above threshold and
 * read Normal on LungV2S — it was dropped rather than shipped as a wrong answer.)
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
const NIH = 'NIH ChestX-ray14'

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
  {
    plate: 'Pl. E',
    name: 'Pneumonia',
    path: '/samples/chest/pneumonia.png',
    subtitle: NIH,
    site: 'NIH · XRV Pneumonia 0.65',
  },
  {
    plate: 'Pl. F',
    name: 'Consolidation',
    path: '/samples/chest/consolidation.png',
    subtitle: NIH,
    site: 'NIH · XRV Opacity 0.79',
  },
]

export default LUNG_SPECIMEN_PRESETS
