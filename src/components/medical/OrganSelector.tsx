import React, { useEffect, useRef } from 'react'
import type { OrganId } from '../../types/organ'
import { ORGANS_REGISTRY } from '../../types/organ'

interface OrganSelectorProps {
  readonly selectedId: OrganId
  readonly onSelectOrgan: (id: OrganId) => void
}

export const OrganSelector: React.FC<OrganSelectorProps> = ({
  selectedId,
  onSelectOrgan,
}) => {
  const organs = Object.values(ORGANS_REGISTRY)
  const selectedNo = organs.findIndex((organ) => organ.id === selectedId) + 1
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const idx = organs.findIndex((organ) => organ.id === selectedId)
    const pill = scrollRef.current?.children[idx] as HTMLElement | undefined
    pill?.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId])

  return (
    <div className="organ-index">
      <div className="index-head">
        <span className="eyebrow">Index of Specimens</span>
        <span className="page-of">
          <b>{String(selectedNo).padStart(3, '0')}</b> / {String(organs.length).padStart(3, '0')}
        </span>
      </div>

      <nav ref={scrollRef} className="pill-scroll" aria-label="Specimen index">
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
              <span className="p-name">{organ.name}</span>
              <span className="p-mod">{organ.modality}</span>
            </button>
          )
        })}
      </nav>
    </div>
  )
}
