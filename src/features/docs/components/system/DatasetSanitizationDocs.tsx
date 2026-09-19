import React from 'react'
import { ShieldCheck } from 'lucide-react'
import { RomanSection } from '../../../../components/common/RomanSection'

export const DatasetSanitizationDocs: React.FC = () => {
  return (
    <section id="section-dataset" className="doc-section-wrapper">
      <RomanSection
        index={4}
        of={5}
        title="Curatio - Dataset Integrity, Purging &amp; Verification"
        className="sp12"
      >
        <div className="doc-info-block">
          <div className="doc-block-header">
            <ShieldCheck size={15} className="doc-block-icon" />
            <span className="doc-block-title">Dataset Sanitization Protocol</span>
          </div>
          <p className="doc-block-copy">
            The source dataset comprises 12,643 brain MRI images with comprehensive spatial bounding boxes and lesion contours.
            During dataset audit, 17 corrupt entries were identified containing binary mask filenames rather than source scans
            (ending in <code>_mask.png</code>).
          </p>
          <div className="purging-stat-strip">
            <div className="p-stat">
              <span className="v">12,643</span>
              <span className="k">Total Catalogued Files</span>
            </div>
            <div className="p-stat">
              <span className="v red">-17</span>
              <span className="k">Corrupt Mask Entries Purged</span>
            </div>
            <div className="p-stat">
              <span className="v green">12,626</span>
              <span className="k">Net Verified Training Scans</span>
            </div>
          </div>
        </div>
      </RomanSection>
    </section>
  )
}
