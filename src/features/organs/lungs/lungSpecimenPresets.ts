/**
 * The lung specimen rack — one radiograph per class in the 16-class label
 * space, so every class can be exercised without hunting for a file.
 *
 * Each plate is a real image from the training corpus, chosen because its
 * label is unambiguous: the fourteen NIH plates are single-label images, so
 * what the card says is exactly what the annotation says. COVID and TB come
 * from their own collections.
 *
 * The subtitle names the source, not a diagnosis — these are annotations, and
 * the card should not claim more than the label does.
 */

export interface LungSpecimenPreset {
  readonly plate: string
  readonly name: string
  readonly path: string
  readonly subtitle: string
  readonly site: string
}

const NIH = 'NIH ChestX-ray14 · single label'

export const LUNG_SPECIMEN_PRESETS: readonly LungSpecimenPreset[] = [
  {
    plate: 'Pl. A',
    name: 'Atelectasis',
    path: '/samples/lungs/atelectasis.png',
    subtitle: NIH,
    site: 'Lobar / segmental',
  },
  {
    plate: 'Pl. B',
    name: 'Cardiomegaly',
    path: '/samples/lungs/cardiomegaly.png',
    subtitle: NIH,
    site: 'Cardiomediastinal silhouette',
  },
  {
    plate: 'Pl. C',
    name: 'Effusion',
    path: '/samples/lungs/effusion.png',
    subtitle: NIH,
    site: 'Pleural space',
  },
  {
    plate: 'Pl. D',
    name: 'Infiltration',
    path: '/samples/lungs/infiltration.png',
    subtitle: NIH,
    site: 'Bilateral lung fields',
  },
  {
    plate: 'Pl. E',
    name: 'Mass',
    path: '/samples/lungs/mass.png',
    subtitle: NIH,
    site: 'Lung parenchyma',
  },
  {
    plate: 'Pl. F',
    name: 'Nodule',
    path: '/samples/lungs/nodule.png',
    subtitle: NIH,
    site: 'Lung parenchyma',
  },
  {
    plate: 'Pl. G',
    name: 'Pneumonia',
    path: '/samples/lungs/pneumonia.png',
    subtitle: NIH,
    site: 'Lobar consolidation',
  },
  {
    plate: 'Pl. H',
    name: 'Pneumothorax',
    path: '/samples/lungs/pneumothorax.png',
    subtitle: NIH,
    site: 'Pleural space',
  },
  {
    plate: 'Pl. I',
    name: 'Consolidation',
    path: '/samples/lungs/consolidation.png',
    subtitle: NIH,
    site: 'Lobar / segmental',
  },
  {
    plate: 'Pl. J',
    name: 'Edema',
    path: '/samples/lungs/edema.png',
    subtitle: NIH,
    site: 'Perihilar / bilateral',
  },
  {
    plate: 'Pl. K',
    name: 'Emphysema',
    path: '/samples/lungs/emphysema.png',
    subtitle: NIH,
    site: 'Upper lobes',
  },
  {
    plate: 'Pl. L',
    name: 'Fibrosis',
    path: '/samples/lungs/fibrosis.png',
    subtitle: NIH,
    site: 'Basal / peripheral',
  },
  {
    plate: 'Pl. M',
    name: 'Pleural Thickening',
    path: '/samples/lungs/pleural_thickening.png',
    subtitle: NIH,
    site: 'Pleural space',
  },
  {
    plate: 'Pl. N',
    name: 'Hernia',
    path: '/samples/lungs/hernia.png',
    subtitle: NIH,
    site: 'Diaphragmatic',
  },
  {
    plate: 'Pl. O',
    name: 'COVID',
    path: '/samples/lungs/covid.png',
    subtitle: 'COVID-19 Radiography Database',
    site: 'Bilateral / peripheral',
  },
  {
    plate: 'Pl. P',
    name: 'TB',
    path: '/samples/lungs/tb.png',
    subtitle: 'TB Chest Radiography Database',
    site: 'Apical / upper lobes',
  },
]

export default LUNG_SPECIMEN_PRESETS
