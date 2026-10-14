import React from 'react'
import { Compass } from 'lucide-react'
import { RomanSection } from '../../../../components/common/RomanSection'

export const MniStereotacticDocs: React.FC = () => {
  return (
    <section id="section-stereotactic" className="doc-section-wrapper">
      <RomanSection
        index={3}
        of={4}
        title="Stereotactic Coordinates — MNI Convention, Illustrative"
        className="sp12"
      >
        <div className="doc-info-block">
          <div className="doc-block-header">
            <Compass size={15} className="doc-block-icon" />
            <span className="doc-block-title">Spatial Normalization &amp; Anatomical Projection</span>
          </div>
          <p className="doc-block-copy">
            To bridge 2D MRI slice classification with spatial neuroanatomy, OpenMed assigns each of its 50 anatomical
            sites an approximate centroid <strong>[X, Y, Z]</strong> in millimetres, expressed in MNI convention
            (X = right, Y = anterior, Z = superior) so the viewer can place a highlight on the 3D model.
          </p>
          <p className="doc-block-copy">
            <strong>These coordinates are illustrative.</strong> They are authored from standard neuroanatomy as
            representative positions for a typical brain — not measurements registered to a template, and not derived
            from the scanned patient. No image registration is performed. The mapping answers "which region does this
            tumour type typically occupy", not "where is this patient's lesion"; that question is answered by the
            segmentation track, not by this one.
          </p>

          <div className="mni-axes-diagram">
            <div className="mni-axis-card">
              <span className="axis-title">X Axis · Sagittal</span>
              <span className="axis-range">0 to +48 mm in this registry</span>
              <span className="axis-note">MNI convention: positive is Right Hemisphere, negative is Left. Every region here sits on the midline or the right hemisphere — the registry contains no left-sided entries.</span>
            </div>
            <div className="mni-axis-card">
              <span className="axis-title">Y Axis · Coronal</span>
              <span className="axis-range">−56 to +46 mm</span>
              <span className="axis-note">Negative values indicate Posterior (Occipital/Cerebellar); positive indicate Anterior (Frontal).</span>
            </div>
            <div className="mni-axis-card">
              <span className="axis-title">Z Axis · Axial</span>
              <span className="axis-range">−46 to +36 mm</span>
              <span className="axis-note">Negative values indicate Inferior (Skull Base/Brainstem); positive indicate Superior (Vertex).</span>
            </div>
          </div>
        </div>
      </RomanSection>
    </section>
  )
}
