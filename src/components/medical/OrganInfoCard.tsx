import React from 'react'
import type { OrganMetadata, Hotspot } from '../../types/organ'
import { RomanSection } from '../common/RomanSection'

interface OrganInfoCardProps {
  readonly organ: OrganMetadata
  readonly plateNo: string
  readonly activeHotspot: Hotspot | null
  readonly onSelectHotspot: (hotspot: Hotspot | null) => void
}

const romanOfSpot = (i: number): string =>
  ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII'][i] ?? String(i + 1)

/**
 * The organ essay — full-width editorial spread rendered below the plate.
 * Six Roman-numbered sections per the Atelier Zero section-rule grammar.
 */
export const OrganInfoCard: React.FC<OrganInfoCardProps> = ({
  organ,
  plateNo,
  activeHotspot,
  onSelectHotspot,
}) => {
  const profile = organ.clinicalProfile
  const sectionCount = 6

  return (
    <article className="organ-essay">
      {/* Essay head */}
      <header className="essay-head">
        <div className="eh-left">
          <div className="badge-row">
            <span className="mono-badge">{organ.modality}</span>
            <span className="veri-line">Verified · Terminologia Anatomica TA2</span>
          </div>

          <h2 className="dossier-title">
            {organ.name}
            <span className="dot">.</span>
            {profile?.poeticTitle && <span className="serif">{profile.poeticTitle}</span>}
          </h2>
          <p className="dossier-latin">
            Plate Nº {plateNo} — {organ.anatomicalTerm}
          </p>
        </div>
        <p className="dossier-lead">{organ.description}</p>
      </header>

      <div className="dossier-grid">
        {/* I — Observatio */}
        <RomanSection index={0} of={sectionCount} title="Observatio — Physiology" className="sp7">
          <p className="sec-copy">{profile?.physiology ?? organ.description}</p>
          {profile?.medicalNote && <p className="sec-copy note">{profile.medicalNote}</p>}
        </RomanSection>

        {/* II — Constantia — daily fact as pull-quote */}
        {profile?.dailyFact && (
          <RomanSection index={1} of={sectionCount} title="Constantia — Physiological Fact" className="sp5">
            <figure className="pull-quote">
              <span className="pq-mark" aria-hidden="true">“</span>
              <p>{profile.dailyFact}</p>
              <figcaption>Nº Dies — {profile.system}</figcaption>
            </figure>
          </RomanSection>
        )}

        {/* III — Landmarks */}
        {organ.hotspots.length > 0 && (
          <RomanSection
            index={2}
            of={sectionCount}
            title={`Landmarks — ${organ.hotspots.length} Points`}
            className="sp12"
          >
            <p className="sec-hint">Select a point to locate it on the plate above.</p>
            <div className="lm-list">
              {organ.hotspots.map((spot, i) => {
                const isSelected = activeHotspot?.id === spot.id
                return (
                  <button
                    key={spot.id}
                    type="button"
                    className={`lm-item ${isSelected ? 'selected' : ''}`}
                    onClick={() => onSelectHotspot(isSelected ? null : spot)}
                    aria-pressed={isSelected}
                  >
                    <span className="lm-idx">{romanOfSpot(i)}</span>
                    <span className="lm-swatch" style={{ backgroundColor: spot.color }} />
                    <span className="lm-body">
                      <span className="lm-name">
                        {spot.label}
                        <span className="latin">{spot.latinTerm}</span>
                      </span>
                      <span className="lm-detail">{spot.detail}</span>
                    </span>
                  </button>
                )
              })}
            </div>
          </RomanSection>
        )}

        {/* IV — Data */}
        <RomanSection index={3} of={sectionCount} title="Data — Benchmarks" className="sp5">
          <div className="data-row">
            <span className="k">Classification</span>
            <span className="v">{organ.classificationDataset}</span>
          </div>
          <div className="data-row">
            <span className="k">Segmentation</span>
            <span className="v">{organ.segmentationDataset}</span>
          </div>
        </RomanSection>

        {/* V — Targetes */}
        <RomanSection index={4} of={sectionCount} title="Targetes — Diagnostic Tasks" className="sp7">
          <div>
            {organ.clinicalTasks.map((task, i) => (
              <div key={task} className="target-row">
                <span className="t-idx">{String(i + 1).padStart(2, '0')}</span>
                <span className="t-arr" aria-hidden="true">→</span>
                <span className="t-name">{task}</span>
              </div>
            ))}
          </div>
        </RomanSection>

        {/* VI — Vasa & Pathologiae */}
        <RomanSection index={5} of={sectionCount} title="Vasa & Pathologiae" className="sp12">
          {profile?.bloodSupply && (
            <div className="data-row">
              <span className="k">Vascular Supply</span>
              <span className="v prose">{profile.bloodSupply}</span>
            </div>
          )}
          {profile?.commonConditions && profile.commonConditions.length > 0 && (
            <div className="cond-row">
              {profile.commonConditions.map((cond) => (
                <span key={cond} className="cond-chip">{cond}</span>
              ))}
            </div>
          )}
        </RomanSection>
      </div>

      {/* Colophon footer */}
      <footer className="dossier-foot">
        <span>
          Asset — {organ.modelFile.split('/').pop()} · {profile?.system ?? 'Homo Sapiens'}
        </span>
        <span className="fin">fin.</span>
      </footer>
    </article>
  )
}
