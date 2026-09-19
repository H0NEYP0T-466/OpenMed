import React from 'react'
import { Compass } from 'lucide-react'
import { RomanSection } from '../../../../components/common/RomanSection'

export const MniStereotacticDocs: React.FC = () => {
  return (
    <section id="section-stereotactic" className="doc-section-wrapper">
      <RomanSection
        index={3}
        of={5}
        title="Stereotaxis - MNI152 Coordinate Normalization System"
        className="sp12"
      >
        <div className="doc-info-block">
          <div className="doc-block-header">
            <Compass size={15} className="doc-block-icon" />
            <span className="doc-block-title">Spatial Normalization &amp; Anatomical Projection</span>
          </div>
          <p className="doc-block-copy">
            To bridge 2D MRI slice classification with spatial neuroanatomy, OpenMed maps all 38 anatomical sites from the
            curated dataset into the <strong>Montreal Neurological Institute (MNI152)</strong> stereotactic coordinate frame.
            Each region is assigned a 3D centroid $[X, Y, Z]$ in millimeters referenced from the anterior commissure.
          </p>

          <div className="mni-axes-diagram">
            <div className="mni-axis-card">
              <span className="axis-title">X Axis · Sagittal</span>
              <span className="axis-range">-80 mm to +80 mm</span>
              <span className="axis-note">Negative values represent Left Hemisphere; positive values represent Right Hemisphere.</span>
            </div>
            <div className="mni-axis-card">
              <span className="axis-title">Y Axis · Coronal</span>
              <span className="axis-range">-110 mm to +70 mm</span>
              <span className="axis-note">Negative values indicate Posterior (Occipital/Cerebellar); positive indicate Anterior (Frontal).</span>
            </div>
            <div className="mni-axis-card">
              <span className="axis-title">Z Axis · Axial</span>
              <span className="axis-range">-50 mm to +85 mm</span>
              <span className="axis-note">Negative values indicate Inferior (Skull Base/Brainstem); positive indicate Superior (Vertex).</span>
            </div>
          </div>
        </div>
      </RomanSection>
    </section>
  )
}
