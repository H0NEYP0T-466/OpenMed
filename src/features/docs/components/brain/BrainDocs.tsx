import React from 'react'
import { Link } from 'react-router-dom'
import { Brain as BrainIcon, ArrowUpRight } from 'lucide-react'
import { BrainClassificationDocs } from './BrainClassificationDocs'
import { BrainSegmentationDocs } from './BrainSegmentationDocs'
import type { ArtifactCard } from '../../types'

interface BrainDocsProps {
  readonly onOpenArtifact: (artifact: ArtifactCard) => void
}

export const BrainDocs: React.FC<BrainDocsProps> = ({ onOpenArtifact }) => {
  return (
    <div id="model-brain" className="organ-model-dossier">
      <div className="organ-dossier-header">
        <div className="dossier-head-left">
          <div className="organ-badge brain">Organ · Brain (Encephalon)</div>
          <h3 className="dossier-organ-title">Cranial Neuro-Oncology &amp; Volumetric Suite</h3>
          <span className="dossier-sub">Central Nervous System (CNS) · 38 Stereotactic MNI Regions</span>
        </div>
        <Link to="/app?organ=brain" className="btn-launch-workspace">
          <BrainIcon size={14} />
          <span>Open Brain 3D Suite</span>
          <ArrowUpRight size={13} />
        </Link>
      </div>

      <BrainClassificationDocs onOpenArtifact={onOpenArtifact} />
      <BrainSegmentationDocs />
    </div>
  )
}
