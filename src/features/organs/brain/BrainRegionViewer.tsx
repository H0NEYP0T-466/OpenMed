import React, { useState } from 'react'
import { MapPin, Crosshair, Compass, ShieldAlert } from 'lucide-react'
import type { BrainRegion3D } from './brainTypes'

interface BrainRegionViewerProps {
  readonly locations: readonly BrainRegion3D[]
  readonly basis?: string
  readonly tumourType?: string
}

export const BrainRegionViewer: React.FC<BrainRegionViewerProps> = ({
  locations,
  basis,
  tumourType,
}) => {
  const [selectedIdx, setSelectedIdx] = useState<number>(0)

  if (!locations || locations.length === 0) {
    const normal = tumourType === 'Normal'
    return (
      <div className="locus-empty-card">
        <MapPin size={20} className="locus-empty-icon" />
        <span className="locus-empty-title">
          {normal ? 'No Lesion Flagged' : 'Diffuse / Unlocalised Pathology'}
        </span>
        <span className="locus-empty-sub">
          {normal
            ? 'The classifier returned a no-tumour label, so no anatomical site is reported. Slice-level classification cannot exclude pathology.'
            : 'No representative presentation site is registered for this finding, so no anatomical centroid is shown.'}
        </span>
      </div>
    )
  }

  const activeRegion = locations[selectedIdx] ?? locations[0]
  const [x, y, z] = activeRegion.coordinates_3d || [0, 0, 0]

  return (
    <div className="locus-info-card">
      {/* Target Site Selector (if multi-focal) */}
      {locations.length > 1 && (
        <div className="locus-site-selector">
          <span className="selector-label">Involved Sites ({locations.length}):</span>
          <div className="selector-pills">
            {locations.map((loc, i) => (
              <button
                key={loc.name}
                type="button"
                className={`site-pill ${i === selectedIdx ? 'active' : ''}`}
                onClick={() => setSelectedIdx(i)}
              >
                <span>{loc.display_name || loc.name}</span>
                <span className="site-pill-pct">{Math.round(loc.probability * 100)}%</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Primary Locus Header */}
      <div className="locus-primary-head">
        <div className="locus-title-group">
          <div className="locus-eyebrow">
            <Compass size={13} />
            <span>Primary Anatomical Locus</span>
          </div>
          <h4 className="locus-name">{activeRegion.display_name || activeRegion.name}</h4>
          <span className="locus-lobe-badge">
            Lobe: <strong>{activeRegion.lobe || 'Cranial Territory'}</strong>
          </span>
        </div>

        <div className="locus-likelihood-badge">
          <span className="likelihood-label">Typical-site Prior</span>
          <span className="likelihood-num">{Math.round(activeRegion.probability * 100)}%</span>
        </div>
      </div>

      {/* Approximate MNI Centroid (population template) */}
      <div className="stereotactic-coords-box">
        <div className="coords-header">
          <Crosshair size={13} className="coords-icon" />
          <span className="coords-title">Approximate MNI Centroid (population template)</span>
        </div>

        <div className="coords-axes-grid">
          <div className="coord-axis-cell">
            <div className="axis-id">X · Sagittal</div>
            <div className="axis-val mono">
              {x >= 0 ? `+${x.toFixed(1)}` : x.toFixed(1)} <span className="unit">mm</span>
            </div>
            <div className="axis-dir">{x >= 0 ? 'Right Hemisphere' : 'Left Hemisphere'}</div>
          </div>

          <div className="coord-axis-cell">
            <div className="axis-id">Y · Coronal</div>
            <div className="axis-val mono">
              {y >= 0 ? `+${y.toFixed(1)}` : y.toFixed(1)} <span className="unit">mm</span>
            </div>
            <div className="axis-dir">{y >= 0 ? 'Anterior (Frontal)' : 'Posterior (Occipital)'}</div>
          </div>

          <div className="coord-axis-cell">
            <div className="axis-id">Z · Axial</div>
            <div className="axis-val mono">
              {z >= 0 ? `+${z.toFixed(1)}` : z.toFixed(1)} <span className="unit">mm</span>
            </div>
            <div className="axis-dir">{z >= 0 ? 'Superior (Vertex)' : 'Inferior (Skull Base)'}</div>
          </div>
        </div>
      </div>

      {/* Anatomical Significance & Functional Risk */}
      <div className="locus-anatomy-description">
        <div className="desc-top">
          <ShieldAlert size={14} className="desc-icon" />
          <span className="desc-title">Regional Anatomy & Functional Significance</span>
        </div>
        <p className="desc-copy">
          {activeRegion.description ||
            'Lesion centroid localized in subcortical white matter. Surrounding eloquent cortex and white matter tracts require surgical navigation margin verification.'}
        </p>
      </div>

      {basis && (
        <p className="locus-basis-note">{basis}</p>
      )}
    </div>
  )
}
