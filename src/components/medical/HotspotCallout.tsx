import React from 'react'
import type { Hotspot } from '../../types/organ'
import { X } from 'lucide-react'
import './HotspotCallout.css'

interface HotspotCalloutProps {
  readonly hotspot: Hotspot
  readonly onClose: () => void
}

export const HotspotCallout: React.FC<HotspotCalloutProps> = ({ hotspot, onClose }) => {
  return (
    <div className="callout" role="dialog" aria-label={`${hotspot.label} - anatomical landmark`}>
      <div className="co-head">
        <span className="co-eyebrow">Anatomical Landmark</span>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Close landmark details">
          <X size={13} />
        </button>
      </div>

      <div className="co-title">
        <span className="co-swatch" style={{ backgroundColor: hotspot.color }} />
        <div>
          <h4 className="co-name">
            {hotspot.label}
            <span className="dot">.</span>
          </h4>
          <span className="co-latin">{hotspot.latinTerm}</span>
        </div>
      </div>

      <p className="co-detail">{hotspot.detail}</p>

      {hotspot.clinicalRelevance && (
        <div className="co-clinical">
          <span className="cc-k">AI diagnostic & pathology target</span>
          <p className="cc-v">{hotspot.clinicalRelevance}</p>
        </div>
      )}

      <div className="co-foot">
        <span>Fig. {hotspot.id} - OM-26</span>
        <span className="fin">fin.</span>
      </div>
    </div>
  )
}
