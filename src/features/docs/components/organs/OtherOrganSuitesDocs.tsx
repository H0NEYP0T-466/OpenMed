import React from 'react'
import { Workflow } from 'lucide-react'

export const OtherOrganSuitesDocs: React.FC = () => {
  return (
    <div id="model-other-organs" className="other-organs-boilerplate-section">
      <div className="other-section-header">
        <Workflow size={16} className="text-olive" />
        <h4 className="other-section-title">Additional Organ Suites · Task Architecture Specifications</h4>
      </div>

      <div className="other-suites-grid">
        {/* Kidneys */}
        <div className="other-suite-card">
          <div className="suite-card-top">
            <span className="suite-organ-name">Kidneys (Ren)</span>
            <span className="suite-tag">KiTS23 / RCC</span>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini class">Classification:</span>
            <p className="task-desc-mini">Renal Cell Carcinoma histological grading (Clear Cell vs Papillary vs Chromophobe).</p>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini seg">Segmentation:</span>
            <p className="task-desc-mini">KiTS volumetric delineation of renal parenchyma, tumor core, and fluid-filled cysts.</p>
          </div>
        </div>

        {/* Liver */}
        <div className="other-suite-card">
          <div className="suite-card-top">
            <span className="suite-organ-name">Liver (Hepar)</span>
            <span className="suite-tag">LiTS / HCC</span>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini class">Classification:</span>
            <p className="task-desc-mini">Multi-phase CT focal liver lesion differential (HCC vs Hemangioma vs FNH).</p>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini seg">Segmentation:</span>
            <p className="task-desc-mini">Total Liver Volume (TLV) and functional future liver remnant (FLR) volumetry.</p>
          </div>
        </div>

        {/* Eye */}
        <div className="other-suite-card">
          <div className="suite-card-top">
            <span className="suite-organ-name">Eye (Oculus)</span>
            <span className="suite-tag">Fundus / DR</span>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini class">Classification:</span>
            <p className="task-desc-mini">5-stage Diabetic Retinopathy grading (Messidor) and glaucoma suspect screening.</p>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini seg">Segmentation:</span>
            <p className="task-desc-mini">Retinal vessel caliber extraction and vertical Cup-to-Disc Ratio (CDR) calculation.</p>
          </div>
        </div>

        {/* Skin */}
        <div className="other-suite-card">
          <div className="suite-card-top">
            <span className="suite-organ-name">Skin (Integumentum)</span>
            <span className="suite-tag">ISIC / Melanoma</span>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini class">Classification:</span>
            <p className="task-desc-mini">Dermoscopic 7-class lesion differential: Melanoma vs Basal Cell Carcinoma vs Nevi.</p>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini seg">Segmentation:</span>
            <p className="task-desc-mini">Border contour extraction with automated ABCD dermatological scoring parameters.</p>
          </div>
        </div>

        {/* Breast */}
        <div className="other-suite-card">
          <div className="suite-card-top">
            <span className="suite-organ-name">Breast (Mamma)</span>
            <span className="suite-tag">CBIS-DDSM / BI-RADS</span>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini class">Classification:</span>
            <p className="task-desc-mini">Digital mammography BI-RADS density classification and malignancy risk estimation.</p>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini seg">Segmentation:</span>
            <p className="task-desc-mini">Soft tissue mass boundary delineation and microcalcification cluster spatial mapping.</p>
          </div>
        </div>

        {/* Bone */}
        <div className="other-suite-card">
          <div className="suite-card-top">
            <span className="suite-organ-name">Bone (Skeleton)</span>
            <span className="suite-tag">MURA / TotalSegmentator</span>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini class">Classification:</span>
            <p className="task-desc-mini">Multi-site radiographic fracture detection and Kellgren-Lawrence knee OA grading.</p>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini seg">Segmentation:</span>
            <p className="task-desc-mini">CT vertebral column individual instance segmentation (C1–L5) and pelvic ring reconstruction.</p>
          </div>
        </div>

        {/* Blood */}
        <div className="other-suite-card">
          <div className="suite-card-top">
            <span className="suite-organ-name">Blood &amp; Vasculature</span>
            <span className="suite-tag">ALL-IDB / CTA</span>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini class">Classification:</span>
            <p className="task-desc-mini">Peripheral smear white blood cell differential &amp; acute leukemia blast screening.</p>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini seg">Segmentation:</span>
            <p className="task-desc-mini">CTA Circle of Willis and coronary artery vessel lumen caliber and stenosis mapping.</p>
          </div>
        </div>

        {/* Pancreas */}
        <div className="other-suite-card">
          <div className="suite-card-top">
            <span className="suite-organ-name">Pancreas</span>
            <span className="suite-tag">MSD Pancreas / PDAC</span>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini class">Classification:</span>
            <p className="task-desc-mini">Pancreatic ductal adenocarcinoma vs neuroendocrine tumor vs IPMN differential.</p>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini seg">Segmentation:</span>
            <p className="task-desc-mini">Pancreatic parenchyma boundary and pancreatic duct dilatation volumetry.</p>
          </div>
        </div>

        {/* Intestine */}
        <div className="other-suite-card">
          <div className="suite-card-top">
            <span className="suite-organ-name">Intestine &amp; Colon</span>
            <span className="suite-tag">Kvasir-SEG / Endoscopy</span>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini class">Classification:</span>
            <p className="task-desc-mini">Real-time colonoscopy polyp histology classification (Adenomatous vs Hyperplastic).</p>
          </div>
          <div className="suite-task-item">
            <span className="task-label-mini seg">Segmentation:</span>
            <p className="task-desc-mini">Endoscopic polyp boundary mask delineation with automated CADe reticle tracking.</p>
          </div>
        </div>
      </div>
    </div>
  )
}
