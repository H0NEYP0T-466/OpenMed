import React from 'react'
import type { Hotspot } from '../../types/organ'
import { MapPin, X, Stethoscope, Sparkles } from 'lucide-react'

interface HotspotCalloutProps {
  readonly hotspot: Hotspot
  readonly onClose: () => void
}

export const HotspotCallout: React.FC<HotspotCalloutProps> = ({ hotspot, onClose }) => {
  return (
    <div className="hotspot-callout-panel">
      <div className="callout-header">
        <div className="callout-badge" style={{ borderColor: hotspot.color, color: hotspot.color }}>
          <MapPin size={13} />
          <span>Anatomical Landmark</span>
        </div>
        <button
          type="button"
          className="callout-close-btn"
          onClick={onClose}
          aria-label="Close Landmark Details"
        >
          <X size={14} />
        </button>
      </div>

      <div className="callout-title-row">
        <div className="callout-dot" style={{ backgroundColor: hotspot.color }} />
        <div>
          <h4 className="callout-main-label">{hotspot.label}</h4>
          <span className="callout-latin-term">{hotspot.latinTerm}</span>
        </div>
      </div>

      <p className="callout-detail-text">{hotspot.detail}</p>

      {hotspot.clinicalRelevance && (
        <div className="callout-clinical-box">
          <div className="clinical-box-header">
            <Stethoscope size={13} className="text-cyan-400" />
            <span>AI Diagnostic & Pathology Target</span>
          </div>
          <p className="clinical-box-content">{hotspot.clinicalRelevance}</p>
        </div>
      )}

      <div className="callout-footer">
        <Sparkles size={12} className="text-purple-400" />
        <span>Terminologia Anatomica (TA2) Verified</span>
      </div>
    </div>
  )
}
