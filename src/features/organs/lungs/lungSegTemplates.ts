/**
 * Segmentation plates — radiographs to exercise the COVID opacity segmenter
 * without hunting for a file.
 *
 * The first three are held-out test images from the corpus the module was
 * trained on, and they are in the split the model never saw: checked against
 * `split_manifest.csv`, all three carry `split = test`. They are the three
 * populations the module has to behave differently on, so each plate states what
 * the honest expectation is rather than what a diagnosis is.
 *
 * The last two come from the NIH corpus (the fourteen ChestX-ray14 findings the
 * XRV suite serves). The segmenter was never trained on that corpus, so they are
 * deliberately out of distribution — the point of showing them is that the panel
 * says so on the plate instead of quietly implying these are training examples.
 */

export interface LungSegTemplate {
  readonly plate: string
  readonly name: string
  readonly path: string
  readonly source: 'COVID-QU-Ex' | 'NIH'
  readonly detail: string
  readonly expectation: string
}

export const LUNG_SEG_TEMPLATES: readonly LungSegTemplate[] = [
  {
    plate: 'Pl. A',
    name: 'COVID-19',
    path: '/samples/lungs/seg/covid.png',
    source: 'COVID-QU-Ex',
    detail: 'Held-out test split',
    expectation: 'Lung field, plus a COVID-type opacity inside it',
  },
  {
    plate: 'Pl. B',
    name: 'Normal',
    path: '/samples/lungs/seg/normal.png',
    source: 'COVID-QU-Ex',
    detail: 'Held-out test split',
    expectation: 'Lung field, no opacity drawn',
  },
  {
    plate: 'Pl. C',
    name: 'Non-COVID',
    path: '/samples/lungs/seg/noncovid.png',
    source: 'COVID-QU-Ex',
    detail: 'Held-out test split',
    expectation: 'Lung field; opacity behaviour here is expected, not a score',
  },
  {
    plate: 'Pl. D',
    name: 'Pneumonia',
    path: '/samples/chest/pneumonia.png',
    source: 'NIH',
    detail: 'Not in the training corpus',
    expectation: 'Out of distribution: the module was not trained on this corpus',
  },
  {
    plate: 'Pl. E',
    name: 'Lung Opacity',
    path: '/samples/chest/infiltration.png',
    source: 'NIH',
    detail: 'Not in the training corpus',
    expectation: 'Out of distribution: the module was not trained on this corpus',
  },
]
