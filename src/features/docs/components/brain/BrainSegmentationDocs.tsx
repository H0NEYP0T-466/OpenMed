import React from 'react'

export const BrainSegmentationDocs: React.FC = () => {
  return (
    <div className="task-block segmentation">
      <div className="task-header-strip">
        <div className="task-tag-group">
          <span className="task-type-tag seg">Task · Segmentation</span>
          <span className="task-name-text">BraTS Multi-Compartment Glioma Volumetric Delineation</span>
        </div>
        <span className="task-model-pill">3D nnU-Net / SegFormer-3D · Voxel Metric</span>
      </div>

      <div className="boilerplate-card">
        <p className="boilerplate-copy">
          The volumetric segmentation module processes multi-parametric 3D cranial MRI acquisitions
          (<strong>T1, T1-contrast [T1Gd], T2, and T2-FLAIR</strong>) to extract sub-voxel masks across three clinical sub-regions:
        </p>
        <div className="seg-targets-grid">
          <div className="seg-target-box">
            <span className="seg-badge wt">Whole Tumor (WT)</span>
            <span className="seg-formula">ED + NET + ET</span>
            <p className="seg-desc">Encompasses complete pathological abnormality including peritumoral edema and necrotic tissue.</p>
            <span className="seg-target-metric">Target Dice: ≥ 0.91 · HD95 &lt; 4.2mm</span>
          </div>
          <div className="seg-target-box">
            <span className="seg-badge tc">Tumor Core (TC)</span>
            <span className="seg-formula">NET + ET</span>
            <p className="seg-desc">Delineates the resectable tumor core (active enhancing neoplasm and central necrotic core).</p>
            <span className="seg-target-metric">Target Dice: ≥ 0.86 · HD95 &lt; 5.0mm</span>
          </div>
          <div className="seg-target-box">
            <span className="seg-badge et">Enhancing Tumor (ET)</span>
            <span className="seg-formula">ET (Hyperintense T1Gd)</span>
            <p className="seg-desc">Isolates hyper-vascularized active tumor rim exhibiting disrupted blood-brain barrier contrast leak.</p>
            <span className="seg-target-metric">Target Dice: ≥ 0.82 · HD95 &lt; 3.8mm</span>
          </div>
        </div>
      </div>
    </div>
  )
}
