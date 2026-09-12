import React from 'react'
import { ATLAS_SYSTEMS } from '../../types/atlas'
import { RomanSection } from '../common/RomanSection'
import './essay.css'

const READING_LINES = [
  'Drag the plate to orbit the figure — scroll to zoom toward a structure.',
  'Run the dissection rail past 50% — the body unfolds into its inventory board.',
  'Hover the exploded plate to name any structure; click to pin the inspector.',
  'Toggle any of the fifteen systems from the rail to peel the figure apart.',
  'Structures with a dedicated plate deep-link into their organ workspace.',
]

/**
 * Centre-page essay for the whole-body atlas — what the plate is,
 * how to read it, and the fifteen systems it carries.
 * (The title block lives above the plate in EssayHead.)
 */
export const AtlasDossier: React.FC = () => {
  const sectionCount = 4

  return (
    <article className="organ-essay">
      <div className="dossier-grid">
        {/* I — De Tabula */}
        <RomanSection index={0} of={sectionCount} title="De Tabula — On the Plate" className="sp7">
          <p className="sec-copy">
            The figure streams in fifteen geometry chunks and is bound on the GPU as a single
            instanced plate: per-structure visibility, dissection offsets and selection are
            carried in state textures, which keeps 2,234 named parts interactive at full frame
            rate without a scene graph per bone.
          </p>
          <p className="sec-copy note">
            Dissection moves in two acts — first the systems fan out in situ, then the plate
            flattens into a printer&rsquo;s inventory board, each structure packed into its own
            cell and named on hover.
          </p>
        </RomanSection>

        {/* II — Reading the plate */}
        <RomanSection index={1} of={sectionCount} title="Lectiones — Reading the Plate" className="sp5">
          <div>
            {READING_LINES.map((line, i) => (
              <div key={line} className="target-row">
                <span className="t-idx">{String(i + 1).padStart(2, '0')}</span>
                <span className="t-arr" aria-hidden="true">→</span>
                <span className="t-name">{line}</span>
              </div>
            ))}
          </div>
        </RomanSection>

        {/* III — The fifteen systems */}
        <RomanSection index={2} of={sectionCount} title={`Systemata — The Fifteen Systems`} className="sp12">
          <div className="sys-cards">
            {ATLAS_SYSTEMS.map((sys) => (
              <div key={sys.id} className="sys-card">
                <div className="sc-head">
                  <span className="sc-swatch" style={{ backgroundColor: sys.color }} />
                  <span className="sc-name">{sys.name}</span>
                </div>
                <p className="sc-desc">{sys.description}</p>
              </div>
            ))}
          </div>
        </RomanSection>

        {/* IV — Colophon of sources */}
        <RomanSection index={3} of={sectionCount} title="Colophon — Sources" className="sp12">
          <div className="data-row">
            <span className="k">Source anatomy</span>
            <span className="v prose">
              BodyParts3D / Anatomica, © DBCLS &amp; University of Tokyo Life Science
              Integration Center — licensed CC BY 4.0.
            </span>
          </div>
          <div className="data-row">
            <span className="k">Interaction port</span>
            <span className="v prose">
              Dissection, naming and projected picking re-edited from ashemag/human-atlas (MIT)
              onto the OpenMed plate.
            </span>
          </div>
        </RomanSection>
      </div>

      <footer className="dossier-foot">
        <span>Anatomia Digitalis — the complete figure, edited for the press</span>
        <span className="fin">fin.</span>
      </footer>
    </article>
  )
}
