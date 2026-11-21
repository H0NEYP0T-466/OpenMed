/**
 * Segmentation plates — radiographs to exercise the COVID opacity segmenter
 * without hunting for a file.
 *
 * All three are held-out test images from the corpus the module was trained on,
 * and they are in the split the model never saw: checked against
 * `split_manifest.csv`, all three carry `split = test`. They are the three
 * populations the module has to behave differently on, so each plate states what
 * the honest expectation is rather than what a diagnosis is.
 *
 * Nothing else belongs here. The segmenter was trained on COVID-QU-Ex only, so a
 * plate from any other corpus is out of distribution and reads as a mystery
 * rather than a demonstration; NIH ChestX-ray14 plates live in the
 * classification rack, where the XRV suite was actually trained on them.
 */

export interface LungSegTemplate {
  readonly plate: string
  readonly name: string
  readonly path: string
  readonly detail: string
  readonly expectation: string
}

export const LUNG_SEG_TEMPLATES: readonly LungSegTemplate[] = [
  {
    plate: 'Pl. A',
    name: 'COVID-19',
    path: '/samples/lungs/seg/covid.png',
    detail: 'Held-out test split',
    expectation: 'Lung field, plus a COVID-type opacity inside it',
  },
  {
    plate: 'Pl. B',
    name: 'Normal',
    path: '/samples/lungs/seg/normal.png',
    detail: 'Held-out test split',
    expectation: 'Lung field, no opacity drawn',
  },
  {
    plate: 'Pl. C',
    name: 'Non-COVID',
    path: '/samples/lungs/seg/noncovid.png',
    detail: 'Held-out test split',
    expectation: 'Lung field; opacity behaviour here is expected, not a score',
  },
]
