import React from 'react'
import { Link } from 'react-router-dom'
import { Activity, ArrowUpRight } from 'lucide-react'

export const LungsDocs: React.FC = () => {
  return (
    <div id="model-lungs" className="organ-model-dossier">
      <div className="organ-dossier-header">
        <div className="dossier-head-left">
          <div className="organ-badge lungs">Organ · Lungs (Pulmones)</div>
          <h3 className="dossier-organ-title">Pulmonary Radiography &amp; Airway Tree Volumetric Suite</h3>
          <span className="dossier-sub">Respiratory System · 5 Lobes &amp; Tracheobronchial Arborization</span>
        </div>
        <Link to="/app?organ=lungs" className="btn-launch-workspace">
          <Activity size={14} />
          <span>Open Lungs 3D Suite</span>
          <ArrowUpRight size={13} />
        </Link>
      </div>

      {/* Task 1: Classification */}
      <div className="task-block classification">
        <div className="task-header-strip">
          <div className="task-tag-group">
            <span className="task-type-tag class">Task · Classification</span>
            <span className="task-name-text">Frontal Chest Radiograph 14-Pathology Differential</span>
          </div>
          <span className="task-model-pill">DenseNet-121 / ConvNeXt-Base · CheXpert &amp; NIH CXR14</span>
        </div>
        <div className="boilerplate-card">
          <p className="boilerplate-copy">
            High-throughput radiograph screening identifying cardiomegaly, consolidation, pulmonary edema, pleural effusion,
            pneumothorax, atelectasis, and solitary pulmonary nodules with uncertainty-aware multi-label BCE loss formulation.
          </p>
        </div>
      </div>

      {/* Task 2: Segmentation */}
      <div className="task-block segmentation">
        <div className="task-header-strip">
          <div className="task-tag-group">
            <span className="task-type-tag seg">Task · Segmentation</span>
            <span className="task-name-text">Thoracic CT Pulmonary Lobar &amp; Airway Tree Delineation</span>
          </div>
          <span className="task-model-pill">LIDC-IDRI / COVID-19-CT · 3D V-Net</span>
        </div>
        <div className="boilerplate-card">
          <p className="boilerplate-copy">
            Extracts 5 anatomical lobes (Right Superior, Middle, Inferior; Left Superior, Inferior) and segments the segmental
            tracheobronchial tree. Calculates localized ground-glass opacification (GGO) percentages and emphysematous tissue destruction.
          </p>
        </div>
      </div>
    </div>
  )
}
