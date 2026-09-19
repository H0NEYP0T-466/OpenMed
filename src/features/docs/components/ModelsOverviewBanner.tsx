import React from 'react'
import { Cpu } from 'lucide-react'

export const ModelsOverviewBanner: React.FC = () => {
  return (
    <div className="models-overview-banner">
      <div className="banner-top">
        <Cpu size={18} className="banner-icon" />
        <h3 className="banner-title">Dual Diagnostic Paradigms: Classification &amp; Volumetric Segmentation</h3>
      </div>
      <p className="banner-copy">
        Clinical AI in OpenMed is structured into two complementary task modalities:
        <strong> Histological &amp; Sequence Classification</strong> (pathology detection, disease staging, sequence subtyping, and WHO grading)
        and <strong>Volumetric Segmentation</strong> (pixel-level mask delineation, multi-compartment lesion burden, and organ volumetry).
      </p>
    </div>
  )
}
