export type OrganId =
  | 'body'
  | 'brain'
  | 'lungs'
  | 'heart'
  | 'kidney'
  | 'eye'
  | 'blood'
  | 'liver'
  | 'bone'
  | 'breast'
  | 'skin'
  | 'pancreas'
  | 'intestine'

export type Modality =
  | 'MRI'
  | 'CT'
  | 'X-ray'
  | 'X-ray / CT'
  | 'Ultrasound / MRI'
  | 'Fundus'
  | 'Microscopy / Fundus'
  | 'Mammography / Histo'
  | 'Dermoscopy'
  | 'Multi-Modal Anatomical'

export type RenderMode = 'pbr' | 'xray' | 'wireframe' | 'segmentation'

export type CameraPreset = 'anterior' | 'lateral' | 'superior' | 'isometric'

export interface Hotspot {
  readonly id: string
  readonly label: string
  readonly latinTerm: string
  readonly detail: string
  readonly position: [number, number, number]
  readonly color: string
  readonly clinicalRelevance?: string
}

export interface OrganClinicalProfile {
  readonly system: string
  readonly poeticTitle: string
  readonly physiology: string
  readonly dailyFact: string
  readonly medicalNote: string
  readonly bloodSupply: string
  readonly commonConditions: readonly string[]
}

export interface OrganMetadata {
  readonly id: OrganId
  readonly name: string
  readonly anatomicalTerm: string
  readonly icon: string
  readonly classificationDataset: string
  readonly segmentationDataset: string
  readonly modality: Modality
  readonly modelFile: string
  readonly description: string
  readonly clinicalTasks: readonly string[]
  readonly accentColor: string
  readonly cameraDistance?: number
  readonly clinicalProfile: OrganClinicalProfile
  readonly hotspots: readonly Hotspot[]
}

export interface ViewerSettings {
  readonly renderMode: RenderMode
  readonly autoRotate: boolean
  readonly rotationSpeed: number
  readonly showLesionMask: boolean
  readonly wireframeOverlay: boolean
  readonly showHotspots: boolean
}

export const ORGANS_REGISTRY: Record<OrganId, OrganMetadata> = {
  heart: {
    id: 'heart',
    name: 'Heart',
    anatomicalTerm: 'Cor (Cardiovascular System)',
    icon: '❤️',
    classificationDataset: 'EchoNet-Dynamic (Ejection Fraction)',
    segmentationDataset: 'EchoNet-Dynamic + ACDC (Ventricles & Myocardium)',
    modality: 'Ultrasound / MRI',
    modelFile: '/models/organs/heart.glb',
    description: 'Anatomically precise four-chamber cardiac model with aorta, pulmonary trunk, coronary sulci, and textured myocardium.',
    clinicalTasks: ['Left Ventricle Ejection Fraction Estimation', 'Myocardial Infarction Segmentation', 'Cardiomegaly Risk Scoring'],
    accentColor: '#ee7c6a',
    cameraDistance: 1.7,
    clinicalProfile: {
      system: 'Cardiovascular System',
      poeticTitle: 'The Tireless Pump',
      physiology: 'Circulates oxygenated blood throughout the body, delivering vital metabolic substrates and oxygen to every living cell.',
      dailyFact: 'Beats approximately 100,000 times daily, pumping over 7,500 liters of blood.',
      medicalNote: 'Its intrinsic electrical conduction system (SA node, AV node, His-Purkinje network) orchestrates continuous synchronized contractions.',
      bloodSupply: 'Left and right coronary arteries arising from aortic sinuses.',
      commonConditions: ['Coronary Artery Disease', 'Myocardial Infarction', 'Arrhythmia', 'Heart Failure', 'Hypertrophic Cardiomyopathy'],
    },
    hotspots: [
      {
        id: 'aorta',
        label: 'Aorta',
        latinTerm: 'Aorta ascendens & Arcus aortae',
        detail: 'The primary systemic arterial trunk conveying high-pressure oxygenated blood from the left ventricle.',
        position: [-0.35, 1.65, 0.55],
        color: '#ee7c6a',
        clinicalRelevance: 'Screening target for aortic dissection, aneurysms, and coarctation.',
      },
      {
        id: 'left-atrium',
        label: 'Left Atrium',
        latinTerm: 'Atrium sinistrum',
        detail: 'Receives freshly oxygenated blood from the pulmonary veins prior to mitral ventricular filling.',
        position: [0.82, 0.65, 0.5],
        color: '#f2a33b',
        clinicalRelevance: 'Primary site of thrombus formation in atrial fibrillation.',
      },
      {
        id: 'right-atrium',
        label: 'Right Atrium',
        latinTerm: 'Atrium dextrum',
        detail: 'Receives deoxygenated systemic venous return via the superior and inferior vena cava.',
        position: [-0.9, 0.35, 0.55],
        color: '#6393d8',
        clinicalRelevance: 'Assessment of central venous pressure (CVP) and tricuspid regurgitation.',
      },
      {
        id: 'left-ventricle',
        label: 'Left Ventricle',
        latinTerm: 'Ventriculus sinister',
        detail: 'Thick muscular chamber generating systemic systolic pressure to perfuse the entire body.',
        position: [0.7, -0.75, 0.65],
        color: '#f2a33b',
        clinicalRelevance: 'Primary target for EchoNet-Dynamic ejection fraction (LVEF) AI prediction.',
      },
      {
        id: 'right-ventricle',
        label: 'Right Ventricle',
        latinTerm: 'Ventriculus dexter',
        detail: 'Pumps venous blood into the pulmonary arterial circulation under low vascular resistance.',
        position: [-0.65, -0.68, 0.66],
        color: '#ee7c6a',
        clinicalRelevance: 'Susceptible to pulmonary hypertension and right ventricular strain.',
      },
      {
        id: 'mitral',
        label: 'Mitral Valve',
        latinTerm: 'Valva atrioventricularis sinistra',
        detail: 'Bicuspid atrioventricular valve preventing retrograde systolic flow into the left atrium.',
        position: [0.18, -1.35, 0.48],
        color: '#d89bc4',
        clinicalRelevance: 'Evaluation of mitral regurgitation, stenosis, and prolapse on echocardiography.',
      },
    ],
  },
  brain: {
    id: 'brain',
    name: 'Brain',
    anatomicalTerm: 'Encephalon / Cerebrum',
    icon: '🧠',
    classificationDataset: 'OpenMed-BT12K · 12,626 Curated Scans (39 Classes)',
    segmentationDataset: 'BraTS 2024 + 12K Bounding Boxes & Histological Masks',
    modality: 'MRI',
    modelFile: '/models/organs/brain.glb',
    description: 'High-definition cortical topology and cerebellar hemispheres with photorealistic PBR gyri and sulci textures.',
    clinicalTasks: [
      '39-Class Histological & Sequence Subtyping (T1, T1C+, T2)',
      'Grad-CAM Convolutional Attention Heatmap Localization',
      '3D Stereotactic MNI Spatial Coordinate Projection',
      'Primary-Location Stratified Risk Scoring'
    ],
    accentColor: '#c58696',
    cameraDistance: 1.8,
    clinicalProfile: {
      system: 'Central Nervous System (CNS)',
      poeticTitle: 'The Universe Within',
      physiology: 'The primary central processor orchestrating consciousness, executive function, motor control, memory, and autonomic homeostasis.',
      dailyFact: 'Consumes roughly 20% of the body total glucose and oxygen despite representing only 2% of body mass.',
      medicalNote: 'Contains over 86 billion neurons interconnected by hundreds of trillions of synaptic junctions.',
      bloodSupply: 'Internal carotid arteries and vertebral arteries joining at the Circle of Willis.',
      commonConditions: ['Glioma / Glioblastoma', 'Ischemic Stroke', "Alzheimer Disease", 'Parkinson Disease', 'Multiple Sclerosis'],
    },
    hotspots: [
      {
        id: 'frontal',
        label: 'Frontal Lobe',
        latinTerm: 'Lobus frontalis',
        detail: 'Seat of executive cognition, working memory, voluntary motor control (precentral gyrus), and expressive language (Broca area).',
        position: [-0.7, 0.65, 0.8],
        color: '#ee7c6a',
        clinicalRelevance: 'Target for frontotemporal dementia staging and high-grade glioma resection planning.',
      },
      {
        id: 'parietal',
        label: 'Parietal Lobe',
        latinTerm: 'Lobus parietalis',
        detail: 'Integrates somatosensory inputs, spatial orientation, proprioception, and linguistic comprehension.',
        position: [0.15, 1.1, 0.65],
        color: '#f2a33b',
        clinicalRelevance: 'Evaluated in parietal cortical atrophy and sensory neglect syndromes.',
      },
      {
        id: 'temporal',
        label: 'Temporal Lobe',
        latinTerm: 'Lobus temporalis',
        detail: 'Houses the hippocampus and amygdala; fundamental for episodic memory consolidation and auditory processing.',
        position: [0.75, -0.1, 0.82],
        color: '#6393d8',
        clinicalRelevance: 'Hippocampal volume loss is the cornerstone biomarker for OASIS-3 Alzheimer classification.',
      },
      {
        id: 'cerebellum',
        label: 'Cerebellum',
        latinTerm: 'Cerebellum',
        detail: 'Coordinates smooth voluntary motor execution, procedural learning, and rapid vestibulo-ocular balance reflexes.',
        position: [0.72, -0.9, 0.55],
        color: '#d89bc4',
        clinicalRelevance: 'Assessed in cerebellar ataxias and posterior fossa tumors.',
      },
    ],
  },
  lungs: {
    id: 'lungs',
    name: 'Chest / Lungs',
    anatomicalTerm: 'Pulmones (Thoracic Cavity)',
    icon: '🫁',
    classificationDataset: 'NIH ChestX-ray14 + MIMIC-CXR',
    segmentationDataset: 'LIDC-IDRI (Pulmonary Nodules)',
    modality: 'X-ray / CT',
    modelFile: '/models/organs/lungs.glb',
    description: 'Bilateral pulmonary lobes featuring trachea, primary bronchi bifurcations, and textured visceral pleura.',
    clinicalTasks: ['14-Class Chest Pathology Screening', 'Pulmonary Nodule 3D Segmentation', 'Pneumonia / Atelectasis Detection'],
    accentColor: '#dd8f8b',
    cameraDistance: 1.9,
    clinicalProfile: {
      system: 'Respiratory System',
      poeticTitle: 'The Breath of Life',
      physiology: 'Executes rapid alveolar capillary gas exchange, replenishing blood oxygen and clearing carbon dioxide.',
      dailyFact: 'Moves upwards of 11,000 liters of atmospheric air daily through over 300 million alveoli.',
      medicalNote: 'Total alveolar surface area spans approximately 70-100 square meters-equivalent to half a tennis court.',
      bloodSupply: 'Pulmonary trunk for gas exchange; bronchial arteries from descending aorta for parenchyma nutrition.',
      commonConditions: ['Pneumonia', 'Pulmonary Embolism', 'COPD', 'Pulmonary Fibrosis', 'Non-Small Cell Lung Carcinoma (NSCLC)'],
    },
    hotspots: [
      {
        id: 'trachea',
        label: 'Trachea',
        latinTerm: 'Trachea',
        detail: 'Fibrocartilaginous tube lined with pseudostratified ciliated epithelium transporting inspired air.',
        position: [0, 1.6, 0.2],
        color: '#6393d8',
        clinicalRelevance: 'Evaluated for tracheal stenosis, malacia, and endotracheal tube placement.',
      },
      {
        id: 'right-lung',
        label: 'Right Lung',
        latinTerm: 'Pulmo dexter',
        detail: 'Consists of three distinct lobes (superior, middle, inferior) divided by horizontal and oblique fissures.',
        position: [-1.2, 0.1, 0.7],
        color: '#ee7c6a',
        clinicalRelevance: 'Primary site of aspiration pneumonia due to steeper right main bronchus angle.',
      },
      {
        id: 'left-lung',
        label: 'Left Lung',
        latinTerm: 'Pulmo sinister',
        detail: 'Comprises two lobes (superior and inferior) with the cardiac notch accommodating the apex of the heart.',
        position: [1.2, 0.1, 0.7],
        color: '#f2a33b',
        clinicalRelevance: 'Screened for lingular consolidations and left hilar lymphadenopathy.',
      },
      {
        id: 'bronchus',
        label: 'Primary Bronchus',
        latinTerm: 'Bronchus principalis',
        detail: 'Bifurcating airway conduit originating at the carina and branching into lobar and segmental bronchi.',
        position: [-0.03, 0.3, 0.35],
        color: '#d89bc4',
        clinicalRelevance: 'Crucial landmark for endobronchial ultrasound (EBUS) and biopsy navigation.',
      },
      {
        id: 'base',
        label: 'Lung Base & Diaphragm',
        latinTerm: 'Basis pulmonis',
        detail: 'Concave inferior pulmonary surface conforming to the convex dome of the respiratory diaphragm.',
        position: [-1.14, -1.2, 1],
        color: '#7fa88a',
        clinicalRelevance: 'Frequent location for pleural effusions, atelectasis, and lower lobe pneumonia.',
      },
    ],
  },
  kidney: {
    id: 'kidney',
    name: 'Kidneys',
    anatomicalTerm: 'Ren (Urinary System)',
    icon: '🫘',
    classificationDataset: 'KiTS Labels (Renal Mass Classification)',
    segmentationDataset: 'KiTS21 / KiTS23 (Kidney & Tumor & Cyst)',
    modality: 'CT',
    modelFile: '/models/organs/kidney.glb',
    description: 'Bilateral renal parenchyma with detailed hilum vasculature, renal veins/arteries, and ureter tracts.',
    clinicalTasks: ['Renal Cell Carcinoma (RCC) Segmentation', 'Cyst vs Solid Mass Differentiation', 'Surgical Pre-op Planning'],
    accentColor: '#f59e0b',
    cameraDistance: 1.6,
    clinicalProfile: {
      system: 'Urinary / Renal System',
      poeticTitle: 'The Master Filters',
      physiology: 'Maintains fluid balance, electrolyte homeostasis, acid-base equilibrium, blood pressure, and erythropoiesis.',
      dailyFact: 'Filters roughly 180 liters of blood plasma every 24 hours, reclaiming over 99% into circulation.',
      medicalNote: 'Each kidney contains roughly 1 million functional units called nephrons, consisting of glomeruli and convoluted tubules.',
      bloodSupply: 'Left and right renal arteries originating directly from abdominal aorta.',
      commonConditions: ['Renal Cell Carcinoma (RCC)', 'Chronic Kidney Disease (CKD)', 'Nephrolithiasis', 'Polycystic Kidney Disease'],
    },
    hotspots: [
      {
        id: 'cortex',
        label: 'Renal Cortex',
        latinTerm: 'Cortex renalis',
        detail: 'Outer metabolic zone containing renal corpuscles, Bowman capsules, and proximal/distal convoluted tubules.',
        position: [-0.9, 0.55, 0.7],
        color: '#ee7c6a',
        clinicalRelevance: 'Primary site of origin for Clear Cell Renal Cell Carcinoma (ccRCC) on KiTS datasets.',
      },
      {
        id: 'medulla',
        label: 'Renal Medulla & Pyramids',
        latinTerm: 'Medulla renalis',
        detail: 'Hypertonic inner zone housing loops of Henle and collecting ducts responsible for countercurrent urinary concentration.',
        position: [0.85, 0.2, 0.7],
        color: '#f2a33b',
        clinicalRelevance: 'High vulnerability to hypoxic injury during acute tubular necrosis (ATN).',
      },
      {
        id: 'ureter',
        label: 'Renal Pelvis & Ureter',
        latinTerm: 'Pelvis renalis & Ureter',
        detail: 'Funnel-shaped muscular conduit propelling urine from the minor/major calyces to the urinary bladder.',
        position: [0.4, -1.1, 0.5],
        color: '#6393d8',
        clinicalRelevance: 'Common site of acute calculus obstruction causing hydronephrosis and renal colic.',
      },
    ],
  },
  eye: {
    id: 'eye',
    name: 'Eye / Retina',
    anatomicalTerm: 'Bulbus Oculi (Ophthalmic)',
    icon: '👁️',
    classificationDataset: 'STARE + APTOS 2019 (Diabetic Retinopathy)',
    segmentationDataset: 'DRIVE + STARE + CHASE_DB1 (Retinal Vasculature)',
    modality: 'Fundus',
    modelFile: '/models/organs/eye.glb',
    description: 'Ocular globe showing detailed iris striations, pupil aperture, scleral micro-vasculature, and posterior optic nerve.',
    clinicalTasks: ['Diabetic Retinopathy Grading (5-stage)', 'Vessel Tree Tortuosity Segmentation', 'Glaucoma Cup-to-Disc Ratio'],
    accentColor: '#10b981',
    cameraDistance: 1.6,
    clinicalProfile: {
      system: 'Special Sensory System',
      poeticTitle: 'The Optical Window',
      physiology: 'Focuses incident photons onto the neural retina, triggering phototransduction cascades conveyed via optic axons.',
      dailyFact: 'Executes over 100,000 saccadic micro-movements per day to reconstruct a seamless visual visual field.',
      medicalNote: 'The retina is embryologically an outgrowth of the diencephalon-the only place where CNS micro-vessels can be viewed directly non-invasively.',
      bloodSupply: 'Central retinal artery and posterior ciliary arteries from ophthalmic artery.',
      commonConditions: ['Diabetic Retinopathy', 'Open-Angle Glaucoma', 'Age-Related Macular Degeneration (AMD)', 'Retinal Vein Occlusion'],
    },
    hotspots: [
      {
        id: 'cornea',
        label: 'Cornea',
        latinTerm: 'Cornea',
        detail: 'Transparent, avascular anterior refractive dome accounting for roughly two-thirds of total optical power.',
        position: [-0.94, 0.05, 1.47],
        color: '#6393d8',
        clinicalRelevance: 'Assessed for keratoconus, corneal dystrophies, and refractive surgery clearance.',
      },
      {
        id: 'iris',
        label: 'Iris & Pupil',
        latinTerm: 'Iris & Pupilla',
        detail: 'Pigmented contractile diaphragm regulating intraocular luminosity via sphincter and dilator pupillae muscles.',
        position: [-1.22, -0.53, 1.15],
        color: '#f2a33b',
        clinicalRelevance: 'Pupillary light reflex testing provides vital neuro-critical diagnostic feedback.',
      },
      {
        id: 'optic',
        label: 'Optic Nerve & Disc',
        latinTerm: 'Nervus opticus (CN II)',
        detail: 'Transmits over 1 million myelinated retinal ganglion cell axons across the lamina cribrosa into the intracranial space.',
        position: [1.61, -0.18, 0.54],
        color: '#d89bc4',
        clinicalRelevance: 'Calculation of cup-to-disc ratio (CDR) on fundus photos to detect glaucomatous neuropathy.',
      },
    ],
  },
  liver: {
    id: 'liver',
    name: 'Liver',
    anatomicalTerm: 'Hepar (Digestive / Metabolic)',
    icon: '🫀',
    classificationDataset: 'LiTS Lesion Labels (HCC vs Benign)',
    segmentationDataset: 'LiTS + SLIVER07 + AMOS',
    modality: 'CT',
    modelFile: '/models/organs/liver.glb',
    description: 'Hepatic parenchyma showing anatomical lobes, gallbladder attachment, and portal vascular confluence.',
    clinicalTasks: ['Hepatocellular Carcinoma (HCC) Lesion Segmentation', 'Liver Parenchyma Volume Quantification', 'Steatosis Staging'],
    accentColor: '#d97706',
    cameraDistance: 1.7,
    clinicalProfile: {
      system: 'Digestive / Metabolic System',
      poeticTitle: 'The Metabolic Powerhouse',
      physiology: 'Executes over 500 vital biochemical operations: drug metabolism, glycogen storage, bile synthesis, and clotting factor production.',
      dailyFact: 'Has the extraordinary biological capacity to regenerate back to full volume from just 25% of healthy remnant tissue.',
      medicalNote: 'Features a unique dual blood supply: 75% via the portal vein (nutrient-rich) and 25% via the hepatic artery (oxygen-rich).',
      bloodSupply: 'Hepatic portal vein and proper hepatic artery.',
      commonConditions: ['Hepatocellular Carcinoma (HCC)', 'Non-Alcoholic Fatty Liver Disease (NAFLD)', 'Cirrhosis', 'Hepatic Adenoma'],
    },
    hotspots: [
      {
        id: 'right-lobe',
        label: 'Right Hepatic Lobe',
        latinTerm: 'Lobus hepatis dexter',
        detail: 'Largest anatomical lobe comprising functional Couinaud segments V, VI, VII, and VIII.',
        position: [-0.75, 0.35, 0.75],
        color: '#ee7c6a',
        clinicalRelevance: 'Primary site for LiTS lesion annotation and radiofrequency ablation planning.',
      },
      {
        id: 'left-lobe',
        label: 'Left Hepatic Lobe',
        latinTerm: 'Lobus hepatis sinister',
        detail: 'Flatter anterior segment extending across the epigastrium (Couinaud segments II, III, and IV).',
        position: [0.85, 0.25, 0.75],
        color: '#f2a33b',
        clinicalRelevance: 'Common location for living-donor liver transplantation grafts.',
      },
      {
        id: 'portal',
        label: 'Portal Vein & Gallbladder',
        latinTerm: 'Vena portae hepatis & Vesica biliaris',
        detail: 'Confluence draining mesenteric venous blood into hepatic sinusoids alongside the bile storage reservoir.',
        position: [0.1, -0.3, 0.82],
        color: '#6393d8',
        clinicalRelevance: 'Assessed for portal vein thrombosis and cholecystitis.',
      },
    ],
  },
  skin: {
    id: 'skin',
    name: 'Skin / Integument',
    anatomicalTerm: 'Integumentum Commune',
    icon: '🩺',
    classificationDataset: 'ISIC 2017 + ISIC 2019 (7-class Dermoscopy)',
    segmentationDataset: 'ISIC 2016+2018+2019 Pooled (Lesion Boundary)',
    modality: 'Dermoscopy',
    modelFile: '/models/organs/skin.glb',
    description: 'High-detail cutaneous layered cross-section with epidermis, dermis, subcutaneous tissue, and follicular structures.',
    clinicalTasks: ['Melanoma vs Nevus Multi-Class Classification', 'Dermoscopic Border Irregularity Segmentation', 'Asymmetry & Color Variance Analysis'],
    accentColor: '#14b8a6',
    cameraDistance: 1.9,
    clinicalProfile: {
      system: 'Integumentary System',
      poeticTitle: 'The Living Protective Shield',
      physiology: 'The body largest organ by surface area, providing barrier defense against ultraviolet radiation, microbial invasion, and dehydration.',
      dailyFact: 'Sheds roughly 500 million dead epidermal cells each day, completely renewing its stratum corneum monthly.',
      medicalNote: 'Cutaneous melanocytes in the basal epidermis produce melanin pigment; their dysregulated proliferation triggers aggressive melanoma.',
      bloodSupply: 'Subpapillary and deep dermal vascular plexuses.',
      commonConditions: ['Malignant Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma', 'Dysplastic Nevi', 'Psoriasis'],
    },
    hotspots: [
      {
        id: 'epidermis',
        label: 'Epidermis',
        latinTerm: 'Epidermis',
        detail: 'Stratified squamous keratinized epithelium consisting of basale, spinosum, granulosum, and corneum layers.',
        position: [-0.05, 0.88, 1.4],
        color: '#ee7c6a',
        clinicalRelevance: 'Origin of melanoma in situ and superficial spreading melanoma targeted in ISIC competitions.',
      },
      {
        id: 'dermis',
        label: 'Dermis',
        latinTerm: 'Dermis / Corium',
        detail: 'Dense irregular connective tissue matrix rich in collagen, elastin, mechanoreceptors, and micro-vasculature.',
        position: [0.29, 0.05, 1.4],
        color: '#f2a33b',
        clinicalRelevance: 'Breslow thickness measurement here dictates malignant melanoma surgical margins and sentinel node biopsy.',
      },
      {
        id: 'hypodermis',
        label: 'Hypodermis',
        latinTerm: 'Tela subcutanea',
        detail: 'Adipose-rich subcutaneous layer insulating against temperature shifts and serving as shock-absorbing cushion.',
        position: [-0.39, -1.15, 1.4],
        color: '#6393d8',
        clinicalRelevance: 'Involvement indicates advanced Clark level V invasive disease.',
      },
      {
        id: 'follicle',
        label: 'Hair Follicle & Sebaceous Gland',
        latinTerm: 'Folliculus pili & Glandula sebacea',
        detail: 'Cutaneous adnexal invagination providing sebum lipid lubrication and follicular epidermal stem cell reserves.',
        position: [0.89, -0.44, 1.4],
        color: '#d89bc4',
        clinicalRelevance: 'Must be segmented or hair-removed in dermoscopy preprocessing to avoid false-positive boundary artifacts.',
      },
    ],
  },
  pancreas: {
    id: 'pancreas',
    name: 'Pancreas',
    anatomicalTerm: 'Pancreas (Endocrine / Exocrine)',
    icon: '🎗️',
    classificationDataset: 'NIH Pancreas CT (Adenocarcinoma)',
    segmentationDataset: 'MSD Task07 (Pancreas & Tumor)',
    modality: 'CT',
    modelFile: '/models/organs/pancreas.glb',
    description: 'Glandular parenchymal structure showing head, body, and tail with pancreatic ductal network.',
    clinicalTasks: ['Pancreatic Ductal Adenocarcinoma Segmentation', 'Parenchyma Atrophy Staging', 'Cystic Lesion Screening'],
    accentColor: '#eab308',
    cameraDistance: 1.7,
    clinicalProfile: {
      system: 'Endocrine & Exocrine System',
      poeticTitle: 'The Metabolic Regulator',
      physiology: 'Secretes enzymatic digestive juices (amylase, lipase, proteases) into duodenum and insulin/glucagon hormones into portal circulation.',
      dailyFact: 'Produces about 1.5 liters of pancreatic juice every day despite having only 2% of its mass dedicated to endocrine islets.',
      medicalNote: 'The Islets of Langerhans host alpha (glucagon), beta (insulin), delta (somatostatin), and PP cells controlling glucose setpoints.',
      bloodSupply: 'Splenic artery and pancreaticoduodenal branches.',
      commonConditions: ['Pancreatic Ductal Adenocarcinoma (PDAC)', 'Acute / Chronic Pancreatitis', 'Intraductal Papillary Mucinous Neoplasm (IPMN)'],
    },
    hotspots: [
      {
        id: 'head',
        label: 'Pancreatic Head',
        latinTerm: 'Caput pancreatis',
        detail: 'Expanded portion nestled within the C-loop of the duodenum and traversed by the distal common bile duct.',
        position: [-1.32, -0.36, 0.55],
        color: '#ee7c6a',
        clinicalRelevance: 'Location of over 70% of pancreatic adenocarcinoma, often presenting with painless obstructive jaundice.',
      },
      {
        id: 'body',
        label: 'Pancreatic Body',
        latinTerm: 'Corpus pancreatis',
        detail: 'Central transverse segment coursing anterior to the aorta, superior mesenteric vessels, and left renal vein.',
        position: [0.05, 0.25, 0.45],
        color: '#f2a33b',
        clinicalRelevance: 'Key site for evaluating parenchymal atrophy and ductal dilatation.',
      },
      {
        id: 'tail',
        label: 'Pancreatic Tail',
        latinTerm: 'Cauda pancreatis',
        detail: 'Narrow tapered terminus abutting the splenic hilum within the splenorenal ligament.',
        position: [1.55, 0.3, 0.35],
        color: '#6393d8',
        clinicalRelevance: 'Frequently harbors neuroendocrine tumors (NETs) such as insulinomas.',
      },
      {
        id: 'duct',
        label: 'Main Pancreatic Duct',
        latinTerm: 'Ductus pancreaticus (Wirsung)',
        detail: 'Axial drainage conduit uniting with common bile duct at the ampulla of Vater.',
        position: [-0.61, 0.39, 0.5],
        color: '#d89bc4',
        clinicalRelevance: 'Pathological ductal cutoff ("double duct sign") strongly indicates underlying malignancy.',
      },
    ],
  },
  intestine: {
    id: 'intestine',
    name: 'Intestines',
    anatomicalTerm: 'Intestinum (Gastrointestinal)',
    icon: '🥨',
    classificationDataset: 'Kvasir / HyperKvasir (Endoscopy)',
    segmentationDataset: 'Kvasir-SEG (Polyp Segmentation)',
    modality: 'CT',
    modelFile: '/models/organs/intestine.glb',
    description: 'Gastrointestinal tract showing small intestinal loops and colon segments with mesenteric vasculature.',
    clinicalTasks: ['Polyp & Colorectal Lesion Segmentation', 'Bowel Obstruction / Perforation Detection', 'Inflammatory Bowel Staging'],
    accentColor: '#f97316',
    cameraDistance: 1.9,
    clinicalProfile: {
      system: 'Gastrointestinal System',
      poeticTitle: 'The Nutrient Absorber',
      physiology: 'Digests macronutrients, absorbs vitamins and water, and hosts the diverse human intestinal microbiome.',
      dailyFact: 'Total length spans 7-8 meters with mucosal microvilli expanding surface area to roughly 30 square meters.',
      medicalNote: 'Its epithelial lining completely regenerates every 3-5 days-the highest cellular turnover rate in the human body.',
      bloodSupply: 'Superior and inferior mesenteric arteries.',
      commonConditions: ['Colorectal Adenomatous Polyps', 'Crohn Disease', 'Ulcerative Colitis', 'Small Bowel Obstruction', 'Diverticulitis'],
    },
    hotspots: [
      {
        id: 'duodenum',
        label: 'Duodenum',
        latinTerm: 'Duodenum',
        detail: 'Initial C-shaped proximal small intestine receiving acidic gastric chyme, bile, and pancreatic enzymes.',
        position: [0.6, 0.8, 0.75],
        color: '#f2a33b',
        clinicalRelevance: 'Common site for peptic ulcers and Brunner gland adenomas.',
      },
      {
        id: 'jejunum',
        label: 'Jejunum & Ileum',
        latinTerm: 'Jejunum & Ileum',
        detail: 'Extensive folded small bowel loops packed with nutrient-absorbing enterocytes and Peyer patches.',
        position: [-0.45, 0.1, 0.82],
        color: '#ee7c6a',
        clinicalRelevance: 'Target for video capsule endoscopy and inflammatory bowel disease (Crohn disease) staging.',
      },
      {
        id: 'colon',
        label: 'Large Intestine / Colon',
        latinTerm: 'Colon & Caecum',
        detail: 'Haustrated muscular tube reclaiming water, electrolytes, and compacting residual digestive waste.',
        position: [0.75, -0.55, 0.72],
        color: '#6393d8',
        clinicalRelevance: 'Screening ground for Kvasir-SEG polyp segmentation to prevent colorectal carcinoma progression.',
      },
    ],
  },
  blood: {
    id: 'blood',
    name: 'Blood / Vascular',
    anatomicalTerm: 'Sanguis & Systema Vasorum',
    icon: '🩸',
    classificationDataset: 'BCCD + PBC + NIH Malaria',
    segmentationDataset: 'DRIVE (Vessels) + BCCD (White Blood Cells)',
    modality: 'Microscopy / Fundus',
    modelFile: '/models/organs/blood.glb',
    description: 'Systemic vascular arborization highlighting arterial and venous circulatory pathways.',
    clinicalTasks: ['Blood Cell Differential Count (RBC/WBC/Platelet)', 'Malaria Parasitemia Detection', 'Microvascular Occlusion Analysis'],
    accentColor: '#ef4444',
    cameraDistance: 2.0,
    clinicalProfile: {
      system: 'Circulatory / Hematological System',
      poeticTitle: 'The River of Life',
      physiology: 'Conveys oxygen, humoral immunoglobulins, leukocytes, platelets, nutrients, and hormones across systemic capillary beds.',
      dailyFact: 'Circulates over 5 liters of whole blood through 96,000 kilometers of blood vessels every single minute.',
      medicalNote: 'Hematopoietic stem cells in red bone marrow generate 200 billion fresh erythrocytes every single day.',
      bloodSupply: 'Systemic aorta branching into continuous arterial, arteriolar, capillary, venular, and caval networks.',
      commonConditions: ['Anemia', 'Leukemia', 'Plasmodium falciparum Malaria', 'Thrombocytopenia', 'Atherosclerosis'],
    },
    hotspots: [
      {
        id: 'aorta-trunk',
        label: 'Major Arterial Trunk',
        latinTerm: 'Arteriae systemicae',
        detail: 'High-pressure distribution tree carrying oxygen-rich erythrocytes from cardiac ejection.',
        position: [0.0, 0.6, 0.2],
        color: '#ef4444',
        clinicalRelevance: 'Vasculature segmentation target on DRIVE and angiographic datasets.',
      },
      {
        id: 'venous-tree',
        label: 'Systemic Venous Return',
        latinTerm: 'Venae systemicae',
        detail: 'Low-pressure capacitance network returning deoxygenated blood and metabolic waste.',
        position: [-0.2, -0.4, 0.2],
        color: '#3b82f6',
        clinicalRelevance: 'Target for deep vein thrombosis (DVT) and thrombophlebitis screening.',
      },
      {
        id: 'capillaries',
        label: 'Microvascular Bed',
        latinTerm: 'Vasa capillaria',
        detail: 'Microscopic single-endothelial-cell exchange boundaries where erythrocytes discharge oxygen.',
        position: [0.3, -0.8, 0.2],
        color: '#ec4899',
        clinicalRelevance: 'BCCD white blood cell differential and microfilarial microscopy analysis.',
      },
    ],
  },
  bone: {
    id: 'bone',
    name: 'Bone / Skeleton',
    anatomicalTerm: 'Systema Skeletale (Vertebrae / Pelvis)',
    icon: '🦴',
    classificationDataset: 'MURA (Musculoskeletal Radiographs)',
    segmentationDataset: 'RSNA Bone Age + Vertebrae CT',
    modality: 'X-ray',
    modelFile: '/models/organs/bone.glb',
    description: 'Articulated vertebral column and skeletal structure with cortical bone trabeculae.',
    clinicalTasks: ['Fracture & Abnormality Detection', 'Pediatric Bone Age Regression', 'Vertebral Compression Scoring'],
    accentColor: '#e2e8f0',
    cameraDistance: 1.8,
    clinicalProfile: {
      system: 'Musculoskeletal System',
      poeticTitle: 'The Architectural Scaffold',
      physiology: 'Provides rigid structural biomechanical support, protects thoracic and neural viscera, and stores 99% of total body calcium.',
      dailyFact: 'Bone remodeling continuously replaces our entire skeleton roughly every 10 years through osteoblast and osteoclast activity.',
      medicalNote: 'Cortical bone constitutes 80% of skeletal mass, while trabecular (cancellous) bone provides metabolic calcium reserve and marrow housing.',
      bloodSupply: 'Nutrient arteries entering via nutrient foramina and periosteal vascular loops.',
      commonConditions: ['Osteoporosis', 'Traumatic Fractures', 'Vertebral Spondylolisthesis', 'Osteosarcoma', 'Osteoarthritis'],
    },
    hotspots: [
      {
        id: 'vertebral-body',
        label: 'Vertebral Body',
        latinTerm: 'Corpus vertebrae',
        detail: 'Cylindrical weight-bearing anterior element composed of dense cancellous bone encased in a thin cortical shell.',
        position: [0.0, 0.4, 0.2],
        color: '#e2e8f0',
        clinicalRelevance: 'Evaluation of osteoporotic wedge compression fractures and metastasis.',
      },
      {
        id: 'spinous-process',
        label: 'Spinous Process',
        latinTerm: 'Processus spinosus',
        detail: 'Posterior bony projection serving as the mechanical attachment anchor for interspinous ligaments and back muscles.',
        position: [0.0, 0.1, -0.6],
        color: '#94a3b8',
        clinicalRelevance: 'Palpatory anatomical landmark during lumbar puncture and epidural anesthesia.',
      },
      {
        id: 'spinal-canal',
        label: 'Vertebral Foramen & Canal',
        latinTerm: 'Canalis vertebralis',
        detail: 'Protective osseous conduit transmitting the spinal cord, meningeal sheaths, and spinal nerve roots.',
        position: [0.0, 0.2, -0.1],
        color: '#38bdf8',
        clinicalRelevance: 'Target for spinal canal stenosis and disc herniation diagnosis on CT/MRI.',
      },
    ],
  },
  breast: {
    id: 'breast',
    name: 'Breast / Mammary',
    anatomicalTerm: 'Mamma (Glandula Mammaria)',
    icon: '🤱',
    classificationDataset: 'BreakHis (Histopathology Benign/Malignant)',
    segmentationDataset: 'INbreast + DDSM (Mass & Microcalcification)',
    modality: 'Mammography / Histo',
    modelFile: '/models/organs/breast.glb',
    description: 'Mammary glandular tissue with anatomical ducts, lobules, and surrounding adipose structural matrix.',
    clinicalTasks: ['BIRADS Lesion Classification', 'Breast Mass & Architectural Distortion Segmentation', 'Histological Subtype Grading'],
    accentColor: '#ec4899',
    cameraDistance: 1.7,
    clinicalProfile: {
      system: 'Reproductive / Glandular System',
      poeticTitle: 'The Glandular Matrix',
      physiology: 'Modified tubuloalveolar exocrine sweat gland composed of glandular lobules, lactiferous ducts, and Cooper suspensory ligaments.',
      dailyFact: 'Sensitive to cyclic fluctuations in estrogen and progesterone, undergoing dynamic cellular micro-remodeling every menstrual cycle.',
      medicalNote: 'Most breast carcinomas originate in the terminal duct lobular units (TDLU), presenting as invasive ductal carcinoma (IDC).',
      bloodSupply: 'Internal thoracic artery (internal mammary) and lateral thoracic artery branches.',
      commonConditions: ['Invasive Ductal Carcinoma (IDC)', 'Ductal Carcinoma In Situ (DCIS)', 'Fibroadenoma', 'Phyllodes Tumor', 'Mastitis'],
    },
    hotspots: [
      {
        id: 'ducts',
        label: 'Lactiferous Ducts',
        latinTerm: 'Ductus lactiferi',
        detail: 'Branching epithelial channels converging at the ampulla and nipple, responsible for fluid drainage.',
        position: [0.0, 0.2, 0.5],
        color: '#ec4899',
        clinicalRelevance: 'Primary site of microcalcifications and DCIS on screening mammography.',
      },
      {
        id: 'lobules',
        label: 'Glandular Lobules (TDLU)',
        latinTerm: 'Lobuli glandulae mammariae',
        detail: 'Secretory alveoli clusters representing the functional glandular parenchyma.',
        position: [0.3, -0.2, 0.4],
        color: '#f43f5e',
        clinicalRelevance: 'Target for invasive lobular carcinoma (ILC) and BreakHis histopathology classification.',
      },
      {
        id: 'adipose',
        label: 'Adipose & Stroma',
        latinTerm: 'Corpus adiposum mammae',
        detail: 'Fatty and dense fibrous connective tissue matrix providing structural suspension via Cooper ligaments.',
        position: [-0.3, -0.3, 0.3],
        color: '#fbbf24',
        clinicalRelevance: 'Mammographic breast density classification (ACR BI-RADS categories A-D).',
      },
    ],
  },
  body: {
    id: 'body',
    name: 'Full Human Body',
    anatomicalTerm: 'Homo Sapiens (Whole Body)',
    icon: '🧍',
    classificationDataset: 'Multi-Organ Screening',
    segmentationDataset: 'Visible Human Project / HRA',
    modality: 'Multi-Modal Anatomical',
    modelFile: '/models/organs/human_body.glb',
    description: 'United anatomical human body macro coordinate model for localized organ discovery and multi-system triage.',
    clinicalTasks: ['Whole-body screening', 'Cross-organ triage', 'Anatomical mapping'],
    accentColor: '#38bdf8',
    cameraDistance: 2.2,
    clinicalProfile: {
      system: 'Whole Body Macro Anatomy',
      poeticTitle: 'The Living Synthesis',
      physiology: 'Holistic integration of 11 physiological organ systems operating in precise metabolic and neural synchrony.',
      dailyFact: 'Coordinates over 37 trillion cells across 78 distinct organs to maintain physiological homeostasis.',
      medicalNote: 'Macro coordinate referencing enables unified cross-organ triage and automated anatomical localization for CT/PET imaging.',
      bloodSupply: 'Central cardiovascular arborization branching to all peripheral extremities.',
      commonConditions: ['Polytrauma', 'Systemic Sepsis', 'Metastatic Disease', 'Multi-Organ Failure', 'Metabolic Syndrome'],
    },
    hotspots: [
      {
        id: 'head-point',
        label: 'Cranial & Neural Hub',
        latinTerm: 'Regio cephalica',
        detail: 'Hosts the brain, ocular globes, cranial nerves, and neuro-endocrine axes.',
        position: [0.0, 1.4, 0.0],
        color: '#c58696',
        clinicalRelevance: 'Brain MRI and fundus retinopathy assessment zone.',
      },
      {
        id: 'thoracic-point',
        label: 'Thoracic Visceral Hub',
        latinTerm: 'Cavitas thoracis',
        detail: 'Protected cage housing the bilateral lungs, tracheobronchial tree, heart, and great thoracic vessels.',
        position: [0.0, 0.7, 0.2],
        color: '#dd8f8b',
        clinicalRelevance: 'Chest X-ray, CT nodule, and echocardiography focal zone.',
      },
      {
        id: 'abdominal-point',
        label: 'Abdominal / Metabolic Hub',
        latinTerm: 'Cavitas abdominis',
        detail: 'Visceral center comprising the liver, kidneys, pancreas, and gastrointestinal loops.',
        position: [0.0, 0.1, 0.2],
        color: '#d97706',
        clinicalRelevance: 'Abdominal CT triage for hepatic, renal, and gastrointestinal malignancies.',
      },
      {
        id: 'pelvic-point',
        label: 'Pelvic & Skeletal Hub',
        latinTerm: 'Cavitas pelvis & Skeleton',
        detail: 'Bony pelvic ring and lower spinal column anchoring locomotion and lower pelvic viscera.',
        position: [0.0, -0.5, 0.1],
        color: '#e2e8f0',
        clinicalRelevance: 'Musculoskeletal fracture detection and bone age evaluation.',
      },
    ],
  },
}
