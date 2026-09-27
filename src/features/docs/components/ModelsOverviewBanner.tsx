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
        <strong> Histological Classification</strong> (pathology detection, WHO family assignment, and calibrated
        differential ranking)
        and <strong>Volumetric Segmentation</strong> (promptable pixel-level mask delineation, lesion burden, and organ volumetry).
        Classification heads are trained on modality-flattened label spaces: the brain suite, for example, collapses
        T1 / T1C+ / T2 acquisitions into a single 9-class WHO histology head rather than siloing them per sequence.
      </p>
    </div>
  )
}
