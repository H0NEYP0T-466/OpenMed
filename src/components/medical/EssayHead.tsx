import React from 'react'
import './EssayHead.css'
import './essay.css'

interface EssayHeadProps {
  readonly badge: string
  readonly verified: string
  readonly title: string
  readonly serif?: string
  readonly latin: string
  readonly lead: string
}

/**
 * The specimen's title block - set between the masthead and the plate,
 * so the reader knows whose body they are looking at before the WebGL.
 */
export const EssayHead: React.FC<EssayHeadProps> = ({
  badge,
  verified,
  title,
  serif,
  latin,
  lead,
}) => (
  <div className="fold-wrap">
    <header className="essay-head">
      <div className="eh-left">
        <div className="badge-row">
          <span className="mono-badge">{badge}</span>
          <span className="veri-line">{verified}</span>
        </div>

        <h1 className="dossier-title">
          {title}
          <span className="dot">.</span>
          {serif && <span className="serif">{serif}</span>}
        </h1>
        <p className="dossier-latin">{latin}</p>
      </div>
      <p className="dossier-lead">{lead}</p>
    </header>
  </div>
)
