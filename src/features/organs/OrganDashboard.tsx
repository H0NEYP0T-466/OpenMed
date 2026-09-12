import React, { useMemo, useState } from 'react'
import type { OrganId, ViewerSettings, Hotspot } from '../../types/organ'
import { ORGANS_REGISTRY } from '../../types/organ'
import { OrganViewer3D } from '../../components/medical/OrganViewer3D'
import { WholeBodyAtlasViewer } from '../../components/medical/WholeBodyAtlasViewer'
import { OrganSelector } from '../../components/medical/OrganSelector'
import { OrganInfoCard } from '../../components/medical/OrganInfoCard'

const ORGAN_ORDER: readonly OrganId[] = Object.keys(ORGANS_REGISTRY) as OrganId[]

export const OrganDashboard: React.FC = () => {
  const initialOrgan = (new URLSearchParams(window.location.search).get('organ') as OrganId) || 'body'
  const [selectedOrganId, setSelectedOrganId] = useState<OrganId>(initialOrgan)
  const [lastOrganId, setLastOrganId] = useState<OrganId>(initialOrgan !== 'body' ? initialOrgan : 'heart')
  const [activeHotspot, setActiveHotspot] = useState<Hotspot | null>(null)
  const [settings, setSettings] = useState<ViewerSettings>({
    renderMode: 'pbr',
    autoRotate: true,
    rotationSpeed: 0.8,
    showLesionMask: false,
    wireframeOverlay: false,
    showHotspots: true,
  })

  const currentOrgan = ORGANS_REGISTRY[selectedOrganId]
  const plateNo = useMemo(
    () => String(ORGAN_ORDER.indexOf(selectedOrganId) + 1).padStart(2, '0'),
    [selectedOrganId]
  )

  const handleSelectOrgan = (id: OrganId) => {
    setSelectedOrganId(id)
    setActiveHotspot(null)
    if (id !== 'body') setLastOrganId(id)
  }

  const isWholeBodyView = selectedOrganId === 'body'

  return (
    <div className="om-page">
      {/* Side rails — 36px fixed strips, rotated editorial labels */}
      <div className="om-rail left" aria-hidden="true">
        <span>OpenMed — Anatomia Digitalis · MMXXVI</span>
      </div>
      <div className="om-rail right" aria-hidden="true">
        <span>Terminologia Anatomica · TA2 · Lahore</span>
      </div>

      {/* Top metadata strip — Vol/Issue, Filed under, live status */}
      <div className="meta-strip">
        <span className="m-left">
          Vol. 01 / Issue Nº 26 — <b>Anatomia Digitalis</b>
        </span>
        <span className="m-mid">
          <span className="filed">Filed under</span> — Medical AI · 3D Anatomy · MICCAI Benchmarks
        </span>
        <span className="m-right">
          <span className="pulse-dot" aria-hidden="true" />
          Live build 0.1.0 · en-PK · 31.5204° N — 74.3587° E
        </span>
      </div>

      {/* Masthead */}
      <header className="masthead">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">Ø</div>
          <div style={{ minWidth: 0 }}>
            <div>
              <span className="brand-word">
                OpenMed<span className="dot">.</span>
              </span>
              <span className="brand-edition">FYP Edition · AI Hospital</span>
            </div>
            <span className="brand-tag">A multi-organ diagnostic annual — edited by M. Fezan</span>
          </div>
        </div>

        <div className="masthead-stats">
          <div className="stat-fig">
            <span className="fig">12</span>
            <span className="cap">Organs</span>
          </div>
          <div className="stat-fig">
            <span className="fig">
              20<em>+</em>
            </span>
            <span className="cap">AI Models</span>
          </div>
          <div className="stat-fig">
            <span className="fig">2,234</span>
            <span className="cap">Atlas Parts</span>
          </div>
        </div>

        <div className="masthead-actions">
          {isWholeBodyView ? (
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => handleSelectOrgan(lastOrganId)}
            >
              <span>Organ Workspaces</span>
              <span className="arr" aria-hidden="true">↗</span>
            </button>
          ) : (
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => handleSelectOrgan('body')}
            >
              <span>Whole-Body Atlas</span>
              <span className="nav-star" aria-hidden="true">★</span>
              <span className="arr" aria-hidden="true">↗</span>
            </button>
          )}
        </div>
      </header>

      {/* Main workspace — the plate, optionally with a dossier column */}
      <main className={`stage ${isWholeBodyView ? 'stage-atlas' : ''}`}>
        {isWholeBodyView ? (
          <section className="plate">
            <WholeBodyAtlasViewer onNavigateToOrgan={handleSelectOrgan} plateNo={plateNo} />
          </section>
        ) : (
          <>
            <section className="plate">
              <OrganViewer3D
                organ={currentOrgan}
                plateNo={plateNo}
                settings={settings}
                onUpdateSettings={setSettings}
                activeHotspot={activeHotspot}
                onSelectHotspot={setActiveHotspot}
              />
            </section>

            <aside className="dossier">
              <OrganInfoCard
                organ={currentOrgan}
                plateNo={plateNo}
                activeHotspot={activeHotspot}
                onSelectHotspot={setActiveHotspot}
              />
            </aside>
          </>
        )}
      </main>

      {/* Index of specimens — pill filters + clipped mega wordmark */}
      <footer className="index-bar">
        <div className="index-word" aria-hidden="true">
          OpenMed<span className="dot">.</span>
        </div>
        <OrganSelector
          selectedId={selectedOrganId}
          onSelectOrgan={handleSelectOrgan}
        />
      </footer>
    </div>
  )
}
