/**
 * The specimen rack — one scan per class in the classifier's 9-class label
 * space, so every class can be exercised from a cassette.
 *
 * Shared by the brain workspace and the Experimental Laboratory. Each plate is
 * a real MRI pulled from `backend/datasets/brain/<class>/`; the subtitle and
 * site come from that series' own clinical filename, not from invented values.
 */
export interface SpecimenPreset {
  readonly plate: string
  readonly name: string
  readonly path: string
  readonly subtitle: string
  readonly site: string
}

export const SPECIMEN_PRESETS: readonly SpecimenPreset[] = [
  {
    plate: 'Pl. A',
    name: 'Germ Cell Tumors',
    path: '/samples/brain/GermCellTumors.jpg',
    subtitle: 'Pineal Germinoma',
    site: 'Pineal Region / Ventricle',
  },
  {
    plate: 'Pl. B',
    name: 'Gliomas',
    path: '/samples/brain/Gliomas.jpg',
    subtitle: 'Cystic Glioblastoma',
    site: 'Occipital / Ventricle',
  },
  {
    plate: 'Pl. C',
    name: 'Medulloblastoma',
    path: '/samples/brain/Medulloblastoma.jpg',
    subtitle: 'Desmoplastic Medulloblastoma',
    site: 'Posterior Fossa / Cerebellum',
  },
  {
    plate: 'Pl. D',
    name: 'Meningothelial Tumors',
    path: '/samples/brain/MeningothelialTumors.jpg',
    subtitle: 'Angiomatous Meningioma',
    site: 'Anterior Cranial Fossa / Frontal',
  },
  {
    plate: 'Pl. E',
    name: 'Mesenchymal (Non-Meningothelial)',
    path: '/samples/brain/Mesenchymal.jpg',
    subtitle: 'Dural Solitary Fibrous Tumor',
    site: 'Parafalcine / Falx',
  },
  {
    plate: 'Pl. F',
    name: 'Mixed Neuronal & Neuronal-Glial',
    path: '/samples/brain/MixedNeuronal.jpg',
    subtitle: 'Central Neurocytoma',
    site: 'Intraventricular / Ventricle',
  },
  {
    plate: 'Pl. G',
    name: 'Normal',
    path: '/samples/brain/Normal.jpg',
    subtitle: 'No Tumour Detected',
    site: 'Whole Brain',
  },
  {
    plate: 'Pl. H',
    name: 'Pituitary',
    path: '/samples/brain/Pituitary.jpg',
    subtitle: 'Pituitary Tumour',
    site: 'Sellar Region',
  },
  {
    plate: 'Pl. I',
    name: 'Schwannoma',
    path: '/samples/brain/Schwannoma.jpg',
    subtitle: 'Acoustic Schwannoma',
    site: 'Cerebellopontine Angle',
  },
]
