import React from 'react'
import type { OrganMetadata, Hotspot } from '../../types/organ'

interface OrganInfoCardProps {
  readonly organ: OrganMetadata
  readonly plateNo: string
  readonly activeHotspot: Hotspot | null
  readonly onSelectHotspot: (hotspot: Hotspot | null) => void
}

const ROMANS = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII'] as const

const RomanSection: React.FC<{
  index: number
  of: number
  title: string
  children: React.ReactNode
}> = ({ index, of, title, children }) => (
  <section className="sec-block">
    <div className="sec-rule">
      <span className="roman">{ROMANS[index]}.</span>
      <span className="sec-title">{title}</span>
      <span className="page-of">
        {String(index + 1).padStart(3, '0')} / {String(of).padStart(3, '0')}
      </span>
    </div>
    {children}
  </section>
)

const romanOfSpot = (i: number): string => ROMANS[i] ?? String(i + 1)

export const OrganInfoCard: React.FC<OrganInfoCardProps> = ({
  organ,
  plateNo,
  activeHotspot,
  onSelectHotspot,
}) => {
  const profile = organ.clinicalProfile
  const sectionCount = 6

  return (
    <div className="dossier-body">
      {/* Plate header */}
      <header className="dossier-head">
        <div className="badge-row">
          <span className="mono-badge">{organ.modality}</span>
          <span className="veri-line">Verified · Terminologia Anatomica TA2</span>
        </div>

        <h2 className="dossier-title">
          {organ.name}
          <span className="dot">.</span>
          {profile?.poeticTitle && (
            <span className="serif">{profile.poeticTitle}</span>
          )}
        </h2>
        <p className="dossier-latin">
          Plate Nº {plateNo} — {organ.anatomicalTerm}
        </p>

        <p className="dossier-lead">
          {organ.description}
        </p>
      </header>

      {/* I — Observatio */}
      <RomanSection index={0} of={sectionCount} title="Observatio — Physiology">
        <p className="sec-copy">{profile?.physiology ?? organ.description}</p>
        {profile?.medicalNote && <p className="sec-copy">{profile.medicalNote}</p>}
      </RomanSection>

      {/* II — Constantia — daily fact as pull-quote */}
      {profile?.dailyFact && (
        <RomanSection index={1} of={sectionCount} title="Constantia — Physiological Fact">
          <figure className="pull-quote">
            <span className="pq-mark" aria-hidden="true">“</span>
            <p>{profile.dailyFact}</p>
            <figcaption>Nº Dies — {profile.system}</figcaption>
          </figure>
        </RomanSection>
      )}

      {/* III — Landmarks */}
      {organ.hotspots.length > 0 && (
        <RomanSection index={2} of={sectionCount} title={`Landmarks — ${organ.hotspots.length} Points`}>
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
      <RomanSection index={3} of={sectionCount} title="Data — Benchmarks">
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
      <RomanSection index={4} of={sectionCount} title="Targetes — Diagnostic Tasks">
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
      <RomanSection index={5} of={sectionCount} title="Vasa & Pathologiae">
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

      {/* Colophon footer */}
      <footer className="dossier-foot">
        <span>
          Asset — {organ.modelFile.split('/').pop()} · {profile?.system ?? 'Homo Sapiens'}
        </span>
        <span className="fin">fin.</span>
      </footer>
    </div>
  )
}
