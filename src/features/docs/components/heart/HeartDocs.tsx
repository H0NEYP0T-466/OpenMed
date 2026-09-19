import React from 'react'
import { Link } from 'react-router-dom'
import { Heart, ArrowUpRight } from 'lucide-react'

export const HeartDocs: React.FC = () => {
  return (
    <div id="model-heart" className="organ-model-dossier">
      <div className="organ-dossier-header">
        <div className="dossier-head-left">
          <div className="organ-badge heart">Organ · Heart (Cor)</div>
          <h3 className="dossier-organ-title">Cardiovascular Electro-Mechanical &amp; Chamber Volumetry Suite</h3>
          <span className="dossier-sub">Cardiovascular System · 4-Chamber Cine-MRI &amp; 12-Lead ECG</span>
        </div>
        <Link to="/app?organ=heart" className="btn-launch-workspace">
          <Heart size={14} />
          <span>Open Heart 3D Suite</span>
          <ArrowUpRight size={13} />
        </Link>
      </div>

      {/* Task 1: Classification */}
      <div className="task-block classification">
        <div className="task-header-strip">
          <div className="task-tag-group">
            <span className="task-type-tag class">Task · Classification</span>
            <span className="task-name-text">12-Lead ECG Arrhythmia, Ischemia &amp; Infarction Subtyping</span>
          </div>
          <span className="task-model-pill">1D-ResNet with Squeeze-and-Excitation · PTB-XL</span>
        </div>
        <div className="boilerplate-card">
          <p className="boilerplate-copy">
            Multi-lead temporal signal processor classifying 12-lead diagnostic electrocardiograms across 5 super-classes:
            <strong> Normal ECG (NORM), Myocardial Infarction (MI), ST/T-Change (STTC), Conduction Disturbance (CD), and Hypertrophy (HYP)</strong>.
            Features temporal attention pooling for localized ST-elevation and QT-interval dispersion analysis.
          </p>
        </div>
      </div>

      {/* Task 2: Segmentation */}
      <div className="task-block segmentation">
        <div className="task-header-strip">
          <div className="task-tag-group">
            <span className="task-type-tag seg">Task · Segmentation</span>
            <span className="task-name-text">Cine-MRI Multi-Chamber &amp; Myocardial Volumetric Contouring</span>
          </div>
          <span className="task-model-pill">ACDC Benchmark · 2D/3D Hybrid U-Net</span>
        </div>
        <div className="boilerplate-card">
          <p className="boilerplate-copy">
            Delineates the endocardial and epicardial contours of the <strong>Left Ventricle (LV), Right Ventricle (RV), and Myocardium (MYO)</strong>
            at both End-Diastole (ED) and End-Systole (ES). Automatically computes clinical cardiac metrics:
            End-Diastolic Volume (EDV), End-Systolic Volume (ESV), Stroke Volume (SV), Myocardial Mass, and Left Ventricular Ejection Fraction (LVEF%).
          </p>
        </div>
      </div>
    </div>
  )
}
