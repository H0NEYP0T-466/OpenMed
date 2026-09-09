import React from 'react'
import type { OrganId, OrganMetadata } from '../../types/organ'
import { ORGANS_REGISTRY } from '../../types/organ'

interface OrganSelectorProps {
  readonly selectedId: OrganId
  readonly onSelectOrgan: (id: OrganId) => void
}

export const OrganSelector: React.FC<OrganSelectorProps> = ({
  selectedId,
  onSelectOrgan,
}) => {
  const organsList: OrganMetadata[] = Object.values(ORGANS_REGISTRY)

  return (
    <div className="organ-selector-bar">
      <div className="organ-selector-header">
        <span className="selector-title">Clinical Organ Systems</span>
        <span className="selector-count">{organsList.length} Entities Online</span>
      </div>

      <div className="organ-pills-scroll">
        {organsList.map((organ) => {
          const isSelected = organ.id === selectedId
          return (
            <button
              key={organ.id}
              type="button"
              className={`organ-pill ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectOrgan(organ.id)}
              style={
                isSelected
                  ? {
                      borderColor: organ.accentColor,
                      boxShadow: `0 0 12px ${organ.accentColor}33`,
                    }
                  : undefined
              }
            >
              <span className="organ-pill-icon">{organ.icon}</span>
              <div className="organ-pill-content">
                <span className="organ-pill-name">{organ.name}</span>
                <span className="organ-pill-modality">{organ.modality}</span>
              </div>
              {isSelected && (
                <div
                  className="active-dot"
                  style={{ backgroundColor: organ.accentColor }}
                />
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
