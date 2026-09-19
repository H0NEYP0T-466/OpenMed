export interface TumorInfo {
  name: string;
  description: string;
  commonLocations: string[];
  mriCharacteristics: string;
  grading: string;
  prognosis: string;
}

export const TUMOR_DATA: Record<string, TumorInfo> = {
  'Meningioma': {
    name: 'Meningioma',
    description: 'A typically slow-growing, benign tumor that forms from the meninges-the membranous layers surrounding the brain and spinal cord.',
    commonLocations: ['Parasagittal/falcine region', 'Convexity', 'Sphenoid wing', 'Olfactory groove', 'Posterior fossa'],
    mriCharacteristics: 'Usually isointense to gray matter on T1 and T2-weighted imaging. Shows strong, homogenous enhancement on T1C+ (contrast). Often exhibits a "dural tail" sign.',
    grading: 'WHO Grade 1 (80% of cases), Grade 2 (atypical, 15-20%), Grade 3 (anaplastic/malignant, 1-3%).',
    prognosis: 'Generally excellent for Grade 1 following gross total resection. Higher grades have increased recurrence rates.'
  },
  'Astrocytoma': {
    name: 'Astrocytoma',
    description: 'A type of glioma that develops from star-shaped glial cells (astrocytes) that support nerve cells.',
    commonLocations: ['Cerebral hemispheres (especially frontal and temporal lobes)', 'Cerebellum', 'Brainstem', 'Spinal cord'],
    mriCharacteristics: 'Varies widely by grade. Low-grade typically shows T2 hyperintensity without contrast enhancement. High-grade shows heterogenous appearance with irregular enhancement on T1C+ and surrounding edema.',
    grading: 'WHO Grades 1-4. Grade 4 is commonly referred to as Glioblastoma.',
    prognosis: 'Highly dependent on grade, age, and extent of resection. Lower grades have longer survival but tend to progress to higher grades over time.'
  },
  'Glioblastoma': {
    name: 'Glioblastoma',
    description: 'A fast-growing, aggressive type of central nervous system tumor that forms from supportive tissue of the brain and spinal cord. It is a Grade 4 Astrocytoma.',
    commonLocations: ['Cerebral hemispheres (frontal and temporal lobes are most common)', 'Can cross the corpus callosum ("butterfly glioma")'],
    mriCharacteristics: 'Heterogeneous mass with irregular thick peripheral enhancement on T1C+. Central necrosis (non-enhancing core). Extensive surrounding vasogenic edema appearing hyperintense on T2/FLAIR.',
    grading: 'WHO Grade 4.',
    prognosis: 'Poor, despite multimodal therapy. Median survival is typically 12-18 months with standard of care.'
  },
  'Schwannoma': {
    name: 'Schwannoma',
    description: 'A benign nerve sheath tumor composed entirely of Schwann cells, which normally produce the insulating myelin sheath covering peripheral nerves.',
    commonLocations: ['Cerebellopontine angle (Vestibular Schwannoma or Acoustic Neuroma)', 'Spinal nerve roots'],
    mriCharacteristics: 'Isointense to hypointense on T1, hyperintense on T2. Demonstrates avid enhancement on T1C+. May have cystic degeneration, especially in larger tumors.',
    grading: 'WHO Grade 1.',
    prognosis: 'Excellent following surgical resection or stereotactic radiosurgery. Can cause cranial nerve deficits due to local compression.'
  },
  'Normal': {
    name: 'Normal Brain',
    description: 'No gross pathological tumor masses detected. The brain parenchyma, ventricles, and extra-axial spaces appear within normal limits.',
    commonLocations: ['N/A'],
    mriCharacteristics: 'Normal gray-white matter differentiation. Symmetrical ventricles. No abnormal contrast enhancement.',
    grading: 'N/A',
    prognosis: 'N/A'
  },
  'Ependymoma - Subependymoma': {
    name: 'Ependymoma / Subependymoma',
    description: 'Tumors arising from ependymal cells lining the ventricles of the brain and the center of the spinal cord.',
    commonLocations: ['Fourth ventricle (most common in children)', 'Lateral ventricles', 'Spinal cord'],
    mriCharacteristics: 'Often intraventricular masses. Heterogeneous appearance with areas of cystic degeneration, necrosis, and calcification. Variable enhancement on T1C+.',
    grading: 'WHO Grades 1-3. Subependymoma is typically Grade 1.',
    prognosis: 'Variable depending on grade and location. Five-year survival rates range from 50-80%.'
  },
  'Medulloblastoma': {
    name: 'Medulloblastoma',
    description: 'The most common malignant embryonal pediatric brain tumor, originating in the cerebellum or posterior fossa.',
    commonLocations: ['Cerebellar vermis', 'Fourth ventricle roof'],
    mriCharacteristics: 'Typically a solid, enhancing mass in the posterior fossa. May restrict diffusion on DWI due to high cellularity. Prone to cerebrospinal fluid (CSF) dissemination.',
    grading: 'WHO Grade 4.',
    prognosis: 'Aggressive, but modern treatments achieve 5-year survival rates of 70-80% in standard-risk patients.'
  },
  'Oligodendroglioma': {
    name: 'Oligodendroglioma',
    description: 'A glioma originating from the oligodendrocytes, the cells that produce myelin in the central nervous system.',
    commonLocations: ['Frontal lobes (most common)', 'Temporal lobes'],
    mriCharacteristics: 'Cortical/subcortical mass. Often heterogeneous on T1/T2. Coarse, clumped calcifications are very common (present in 70-90%). Patchy enhancement on T1C+.',
    grading: 'WHO Grade 2 or Grade 3 (Anaplastic).',
    prognosis: 'Generally better prognosis than astrocytomas of the same grade, particularly if 1p/19q codeletion is present.'
  },
  'Hemangiopericytoma': {
    name: 'Hemangiopericytoma (Solitary Fibrous Tumor)',
    description: 'A rare mesenchymal tumor that can arise in the meninges, previously often misdiagnosed as an angioblastic meningioma.',
    commonLocations: ['Meninges (mimicking meningioma)', 'Dural sinuses'],
    mriCharacteristics: 'Isointense on T1, hyperintense on T2 (unlike meningiomas which are typically isointense on T2). Prominent flow voids are often seen. Avid, heterogeneous enhancement.',
    grading: 'WHO Grades 1-3.',
    prognosis: 'High rate of local recurrence and a distinct propensity for extracranial metastasis (especially to bone, lung, and liver).'
  },
  'Neurocytoma': {
    name: 'Neurocytoma',
    description: 'A rare, benign neuronal tumor typically found within the ventricular system.',
    commonLocations: ['Lateral ventricles (usually near the Foramen of Monro)', 'Third ventricle'],
    mriCharacteristics: 'Intraventricular lobulated mass. Heterogeneous appearance ("bubbly") due to cystic spaces. Moderate to strong enhancement. Calcifications are common.',
    grading: 'WHO Grade 2.',
    prognosis: 'Generally excellent prognosis following total surgical resection.'
  },
  'Dysembryoplastic Neuroepithelial Tumor': {
    name: 'Dysembryoplastic Neuroepithelial Tumor (DNET)',
    description: 'A benign, slow-growing glioneuronal tumor strongly associated with medically refractory epilepsy, usually occurring in children and young adults.',
    commonLocations: ['Temporal lobe (most common)', 'Frontal lobe'],
    mriCharacteristics: 'Cortical-based, multi-cystic ("bubbly") or multinodular appearance. T1 hypointense, T2 hyperintense. Usually does not enhance or shows only faint enhancement. Often causes remodeling of the inner table of the skull.',
    grading: 'WHO Grade 1.',
    prognosis: 'Excellent. Resection typically provides seizure control and cures the patient.'
  },
  'Germinoma': {
    name: 'Germinoma',
    description: 'A type of germ cell tumor in the brain, most often found in children and young adults.',
    commonLocations: ['Pineal region', 'Suprasellar region'],
    mriCharacteristics: 'Solid, well-circumscribed masses. Isointense on T1 and T2. They show vivid, homogenous contrast enhancement on T1C+. May restrict diffusion.',
    grading: 'WHO Grade' + ' (malignant but highly treatable).',
    prognosis: 'Highly sensitive to radiation and chemotherapy. Cure rates are exceptionally high (often >90%).'
  },
  'Ganglioglioma': {
    name: 'Ganglioglioma',
    description: 'A rare, slow-growing tumor that contains both neoplastic glial and neuronal components. Commonly associated with seizures.',
    commonLocations: ['Temporal lobe (most common)', 'Frontal and parietal lobes'],
    mriCharacteristics: 'Typically appears as a cystic mass with an enhancing mural nodule. Calcification is seen in up to 50% of cases. Overlying cortical dysplasia is common.',
    grading: 'WHO Grade 1 (vast majority).',
    prognosis: 'Excellent long-term prognosis. Surgical resection is primarily aimed at seizure control.'
  },
  'Pituitary': {
    name: 'Pituitary Adenoma / Neuroendocrine Tumor',
    description: 'An epithelial neoplasm arising from anterior pituitary gland cells in the sella turcica, commonly causing endocrine syndromes or chiasmatic visual field deficits.',
    commonLocations: ['Sella turcica', 'Suprasellar space', 'Cavernous sinus'],
    mriCharacteristics: 'Microadenomas (<10mm) appear hypointense on early dynamic T1C+. Macroadenomas (>10mm) demonstrate sellar remodeling, classic snowman/figure-8 appearance with suprasellar expansion, and vivid contrast enhancement.',
    grading: 'WHO Pituitary Neuroendocrine Tumor (PitNET); predominantly benign behavior.',
    prognosis: 'Excellent following endoscopic transsphenoidal resection or medical management (dopamine agonists).'
  }
};
