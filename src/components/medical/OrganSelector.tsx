import React from 'react'
import type { OrganId } from '../../types/organ'
import { ORGANS_REGISTRY } from '../../types/organ'
import './OrganSelector.css'

interface OrganSelectorProps {
  readonly selectedId: OrganId
  readonly onSelectOrgan: (id: OrganId) => void
}

/**
 * Index of specimens - a structured table-of-contents grid.
 * Every entry visible at once; no horizontal scrolling.
 */
export const OrganSelector: React.FC<OrganSelectorProps> = ({
  selectedId,
  onSelectOrgan,
}) => {
  const organs = Object.values(ORGANS_REGISTRY)
  const selectedNo = organs.findIndex((organ) => organ.id === selectedId) + 1

  return (
    <div className="organ-index">
      <div className="index-head">
        <span className="eyebrow">Index of Specimens</span>
        <span className="page-of">
          <b>{String(selectedNo).padStart(3, '0')}</b> / {String(organs.length).padStart(3, '0')}
        </span>
      </div>

      <nav className="pill-grid" aria-label="Specimen index">
        {organs.map((organ, i) => {
          const isSelected = organ.id === selectedId
          return (
            <button
              key={organ.id}
              type="button"
              className={`pill ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectOrgan(organ.id)}
              aria-pressed={isSelected}
            >
              <span className="p-num">{String(i + 1).padStart(2, '0')}</span>
              <span className="p-name">
                {organ.name}
                {organ.id === 'body' && (
                  <span className="p-star" aria-hidden="true">★</span>
                )}
              </span>
              <span className="p-mod">{organ.modality}</span>
            </button>
          )
        })}
      </nav>
    </div>
  )
}
