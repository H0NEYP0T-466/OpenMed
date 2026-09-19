import React from 'react'
import { Link } from 'react-router-dom'
import {
  Box,
  Brain as BrainIcon,
  Heart,
  Activity,
  Eye,
  Droplets,
  Scan,
  ArrowUpRight,
} from 'lucide-react'
import { RomanSection } from '../../../components/common/RomanSection'
import type { OrganDirectoryItem } from '../types'

const ALL_ORGANS: readonly OrganDirectoryItem[] = [
  {
    id: 'brain',
    name: 'Brain',
    latin: 'Encephalon',
    system: 'Central Nervous System (CNS)',
    modelFile: 'brain.glb',
    modality: 'Cranial Axial MRI (512×512)',
    classificationTask: '39-Class Histological & Sequence Subtyping (T1/T1C+/T2)',
    segmentationTask: 'BraTS Glioma Sub-regions (WT / TC / ET)',
    hotspotsCount: 5,
    active: true,
    iconType: 'brain',
  },
  {
    id: 'body',
    name: 'Whole Body Atlas',
    latin: 'Systema Corporis',
    system: 'Musculoskeletal & Visceral Systems',
    modelFile: 'body.glb',
    modality: 'Full-Body Biomechanical PBR',
    classificationTask: 'Multi-Organ Pathology Indexing & Triage',
    segmentationTask: '12 Anatomical Systems Exploded Volumetry',
    hotspotsCount: 12,
    iconType: 'body',
  },
  {
    id: 'heart',
    name: 'Heart',
    latin: 'Cor',
    system: 'Cardiovascular System',
    modelFile: 'heart.glb',
    modality: 'Echocardiography & Cine-MRI',
    classificationTask: '12-Lead ECG Arrhythmia & Ischemia Grading',
    segmentationTask: 'ACDC Left/Right Ventricular & Myocardial Volumetry',
    hotspotsCount: 5,
    iconType: 'heart',
  },
  {
    id: 'lungs',
    name: 'Lungs',
    latin: 'Pulmones',
    system: 'Respiratory System',
    modelFile: 'lungs.glb',
    modality: 'Chest Radiography & High-Resolution CT',
    classificationTask: 'CheXpert 14-Pathology & Pneumonia Differential',
    segmentationTask: 'Pulmonary Lobe & Airway Tree Instance Delineation',
    hotspotsCount: 5,
    iconType: 'lungs',
  },
  {
    id: 'kidney',
    name: 'Kidneys',
    latin: 'Ren',
    system: 'Urinary & Excretory System',
    modelFile: 'kidney.glb',
    modality: 'Abdominal Contrast CT',
    classificationTask: 'Renal Cell Carcinoma (RCC) Subtype Grading',
    segmentationTask: 'KiTS23 Kidney, Tumor & Cyst Parenchyma Volumetry',
    hotspotsCount: 4,
    iconType: 'kidney',
  },
  {
    id: 'liver',
    name: 'Liver',
    latin: 'Hepar',
    system: 'Digestive & Metabolic System',
    modelFile: 'liver.glb',
    modality: 'Tri-Phase Contrast Abdominal CT',
    classificationTask: 'Focal Liver Lesion (HCC vs Hemangioma vs FNH)',
    segmentationTask: 'LiTS Hepatic Parenchyma & Tumor Burden Volumetry',
    hotspotsCount: 4,
    iconType: 'liver',
  },
  {
    id: 'eye',
    name: 'Eye',
    latin: 'Oculus',
    system: 'Visual & Sensory System',
    modelFile: 'eye.glb',
    modality: 'Color Fundus Photography & OCT',
    classificationTask: 'Diabetic Retinopathy (5-Stage) & Glaucoma Screening',
    segmentationTask: 'Retinal Vessel Arborization & Optic Disc/Cup Ratio',
    hotspotsCount: 3,
    iconType: 'eye',
  },
  {
    id: 'blood',
    name: 'Blood & Vasculature',
    latin: 'Systema Vasorum',
    system: 'Circulatory System',
    modelFile: 'blood.glb',
    modality: 'CTA & Blood Smear Microscopy',
    classificationTask: 'ALL-IDB Leukocyte & Blast Cell Differential',
    segmentationTask: 'Coronary & Cerebrovascular Lumen Stenosis Segmentation',
    hotspotsCount: 4,
    iconType: 'blood',
  },
  {
    id: 'bone',
    name: 'Bone & Skeleton',
    latin: 'Systema Skeletale',
    system: 'Musculoskeletal System',
    modelFile: 'bone.glb',
    modality: 'Radiographs & Whole-Body CT',
    classificationTask: 'MURA Fracture Detection & Osteoarthritis Grading',
    segmentationTask: 'TotalSegmentator Vertebral Column & Cortical Bone',
    hotspotsCount: 6,
    iconType: 'bone',
  },
  {
    id: 'breast',
    name: 'Breast',
    latin: 'Mamma',
    system: 'Reproductive & Glandular System',
    modelFile: 'breast.glb',
    modality: 'Full-Field Digital Mammography (FFDM)',
    classificationTask: 'BI-RADS Density Assessment & Malignancy Classification',
    segmentationTask: 'Mammographic Soft Tissue Mass & Calcification Clusters',
    hotspotsCount: 3,
    iconType: 'breast',
  },
  {
    id: 'skin',
    name: 'Skin',
    latin: 'Integumentum Commune',
    system: 'Integumentary System',
    modelFile: 'skin.glb',
    modality: 'Dermoscopy & Clinical Macro Photography',
    classificationTask: 'ISIC 7-Class Malignancy Differential (Melanoma/Nevi)',
    segmentationTask: 'Dermoscopic Lesion Boundary & ABCD Metric Scoring',
    hotspotsCount: 4,
    iconType: 'skin',
  },
  {
    id: 'pancreas',
    name: 'Pancreas',
    latin: 'Pancreas',
    system: 'Endocrine & Digestive System',
    modelFile: 'pancreas.glb',
    modality: 'Dual-Phase Pancreatic CT',
    classificationTask: 'Pancreatic Adenocarcinoma (PDAC) vs Neuroendocrine',
    segmentationTask: 'MSD Pancreatic Parenchyma & Duct Dilatation Volumetry',
    hotspotsCount: 3,
    iconType: 'pancreas',
  },
  {
    id: 'intestine',
    name: 'Intestines & Colon',
    latin: 'Tractus Gastrointestinalis',
    system: 'Gastrointestinal System',
    modelFile: 'intestine.glb',
    modality: 'High-Definition Video Endoscopy',
    classificationTask: 'Colon Polyp Histology (Adenoma vs Hyperplastic)',
    segmentationTask: 'Kvasir-SEG Real-Time Polyp Mask Delineation',
    hotspotsCount: 5,
    iconType: 'intestine',
  },
]

export const Doc3DModelsDirectory: React.FC = () => {
  return (
    <section id="section-3d-models" className="doc-section-wrapper">
      <RomanSection
        index={0}
        of={5}
        title="3D Models - Interactive WebGL Anatomical Assets Registry"
        className="sp12"
      >
        <div className="doc-info-block">
          <div className="doc-block-header">
            <Box size={16} className="doc-block-icon" />
            <span className="doc-block-title">Master 3D Organ &amp; Biomechanical Model Index (13 Structures)</span>
          </div>
          <p className="doc-block-copy">
            OpenMed integrates 13 real-time 3D anatomical models rendered with Physically Based Rendering (PBR) materials,
            anisotropic lighting, stereotactic spatial landmarks, and interactive orbit controls.
            Select any anatomical asset below to enter its live 3D clinical workspace.
          </p>

          <div className="organ-cards-grid-all">
            {ALL_ORGANS.map((organ) => (
              <Link
                key={organ.id}
                to={organ.id === 'body' ? '/app' : `/app?organ=${organ.id}`}
                className={`organ-dir-card ${organ.active ? 'active-organ' : ''}`}
              >
                <div className="organ-card-top">
                  <span className={`organ-tag ${organ.active ? 'active' : ''}`}>
                    {organ.active ? 'Active Suite' : organ.system.split(' ')[0]}
                  </span>
                  <span className="organ-asset-name">{organ.modelFile}</span>
                </div>

                <div className="organ-card-body">
                  <div className={`organ-icon-wrap ${organ.iconType}`}>
                    {organ.iconType === 'brain' && <BrainIcon size={20} />}
                    {organ.iconType === 'heart' && <Heart size={20} />}
                    {organ.iconType === 'lungs' && <Activity size={20} />}
                    {organ.iconType === 'body' && <Box size={20} />}
                    {organ.iconType === 'eye' && <Eye size={20} />}
                    {organ.iconType === 'blood' && <Droplets size={20} />}
                    {!['brain', 'heart', 'lungs', 'body', 'eye', 'blood'].includes(organ.iconType) && (
                      <Scan size={20} />
                    )}
                  </div>
                  <div className="organ-card-text">
                    <h4 className="organ-name">
                      {organ.name} <span className="latin">{organ.latin}</span>
                    </h4>
                    <span className="organ-system-label">{organ.system}</span>
                  </div>
                </div>

                <div className="organ-tasks-preview">
                  <div className="task-preview-row">
                    <span className="task-k">Classify:</span>
                    <span className="task-v">{organ.classificationTask}</span>
                  </div>
                  <div className="task-preview-row">
                    <span className="task-k">Segment:</span>
                    <span className="task-v">{organ.segmentationTask}</span>
                  </div>
                </div>

                <div className="organ-card-footer">
                  <span className="organ-meta">{organ.hotspotsCount} Landmark Hotspots</span>
                  <span className="organ-jump-btn">
                    Launch 3D <ArrowUpRight size={12} />
                  </span>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </RomanSection>
    </section>
  )
}
