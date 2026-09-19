import React, { useState, useEffect } from 'react'
import { Download, X } from 'lucide-react'
import type { ArtifactCard } from '../types'

interface ArtifactModalProps {
  readonly activeArtifact: ArtifactCard | null
  readonly onClose: () => void
}

export const ArtifactModal: React.FC<ArtifactModalProps> = ({ activeArtifact, onClose }) => {
  const [artifactTextContent, setArtifactTextContent] = useState<string | null>(null)

  useEffect(() => {
    if (activeArtifact?.isText && activeArtifact.previewPath) {
      setArtifactTextContent(null)
      fetch(activeArtifact.previewPath)
        .then((resp) => resp.text())
        .then((text) => setArtifactTextContent(text))
        .catch(() => setArtifactTextContent('Unable to load artifact text.'))
    } else {
      setArtifactTextContent(null)
    }
  }, [activeArtifact])

  if (!activeArtifact) return null

  return (
    <div className="artifact-modal-backdrop" onClick={onClose}>
      <div className="artifact-modal-window" onClick={(e) => e.stopPropagation()}>
        <div className="artifact-modal-header">
          <div>
            <span className="modal-category-tag">{activeArtifact.category}</span>
            <h3 className="modal-title">{activeArtifact.title}</h3>
            <span className="modal-subtitle">{activeArtifact.desc}</span>
          </div>
          <div className="modal-actions">
            {activeArtifact.previewPath && (
              <a
                href={activeArtifact.previewPath}
                download={activeArtifact.filename.split(' ')[0]}
                className="modal-icon-btn"
                title="Download File"
              >
                <Download size={15} />
              </a>
            )}
            <button
              type="button"
              className="modal-icon-btn close"
              onClick={onClose}
              title="Close Modal"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        <div className="artifact-modal-content">
          {activeArtifact.isText ? (
            <div className="artifact-text-viewer">
              <pre>
                <code>{artifactTextContent ?? 'Loading telemetry log…'}</code>
              </pre>
            </div>
          ) : activeArtifact.previewPath ? (
            <div className="artifact-image-viewer">
              <img src={activeArtifact.previewPath} alt={activeArtifact.title} />
            </div>
          ) : (
            <div className="artifact-empty-view">No visual preview available for this binary weight file.</div>
          )}
        </div>

        <div className="artifact-modal-footer">
          <span className="modal-footer-file">
            File: <code>{activeArtifact.filename}</code>
          </span>
          <span className="modal-footer-meta">
            Verified Evaluation Artifact · Location-Stratified 39-Class Split
          </span>
        </div>
      </div>
    </div>
  )
}
