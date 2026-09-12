import React, { useCallback, useMemo, useRef, useState } from 'react'
import type { OrganId, ViewerSettings, Hotspot, CameraPreset } from '../../types/organ'
import { ORGANS_REGISTRY } from '../../types/organ'
import type { AtlasSceneState, Part, SystemId } from '../../types/atlas'
import { ATLAS_SYSTEMS, DEFAULT_ATLAS_SYSTEMS } from '../../types/atlas'
import { OrganViewer3D } from '../../components/medical/OrganViewer3D'
import { WholeBodyAtlasViewer } from '../../components/medical/WholeBodyAtlasViewer'
import { OrganViewportControls } from '../../components/medical/OrganViewportControls'
import { AtlasControlsRail } from '../../components/medical/AtlasControlsRail'
import { OrganSelector } from '../../components/medical/OrganSelector'
import { OrganInfoCard } from '../../components/medical/OrganInfoCard'
import { AtlasDossier } from '../../components/medical/AtlasDossier'
import { EssayHead } from '../../components/medical/EssayHead'
import { Maximize2, ExternalLink, X } from 'lucide-react'
import './OrganDashboard.css'

const ORGAN_ORDER: readonly OrganId[] = Object.keys(ORGANS_REGISTRY) as OrganId[]

interface OrganCameraApi {
  readonly setPreset: (preset: CameraPreset) => void
  readonly reset: () => void
}

const INITIAL_ATLAS_STATE: AtlasSceneState = {
  explode: 0,
  visibleSystems: DEFAULT_ATLAS_SYSTEMS,
  selectedPartId: null,
  isolate: false,
  viewAngle: 'three-quarter',
  autoRotate: true,
}

/** Quick organ map to jump from a whole-body structure to its OpenMed workspace */
const mapPartToOrgan = (name: string): OrganId | null => {
  const n = name.toLowerCase()
  if (n.includes('heart') || n.includes('myocardium') || n.includes('ventricle') || n.includes('atrium')) return 'heart'
  if (n.includes('brain') || n.includes('cerebrum') || n.includes('cerebellum') || n.includes('gyrus') || n.includes('cortex')) return 'brain'
  if (n.includes('lung') || n.includes('bronch') || n.includes('trachea')) return 'lungs'
  if (n.includes('kidney') || n.includes('renal') || n.includes('ureter')) return 'kidney'
  if (n.includes('liver') || n.includes('gallbladder') || n.includes('hepatic')) return 'liver'
  if (n.includes('eye') || n.includes('cornea') || n.includes('retina') || n.includes('optic')) return 'eye'
  if (n.includes('skin') || n.includes('epidermis') || n.includes('dermis')) return 'skin'
  if (n.includes('pancreas') || n.includes('pancreatic')) return 'pancreas'
  if (n.includes('intestine') || n.includes('colon') || n.includes('duodenum') || n.includes('ileum') || n.includes('jejunum')) return 'intestine'
  if (n.includes('bone') || n.includes('vertebra') || n.includes('femur') || n.includes('rib') || n.includes('skull') || n.includes('pelvis')) return 'bone'
  if (n.includes('arter') || n.includes('vein') || n.includes('aorta')) return 'blood'
  if (n.includes('breast') || n.includes('mammary')) return 'breast'
  return null
}

export const OrganDashboard: React.FC = () => {
  const [selectedOrganId, setSelectedOrganId] = useState<OrganId>('body')
  const [lastOrganId, setLastOrganId] = useState<OrganId>('heart')
  const [activeHotspot, setActiveHotspot] = useState<Hotspot | null>(null)
  const [settings, setSettings] = useState<ViewerSettings>({
    renderMode: 'pbr',
    autoRotate: true,
    rotationSpeed: 0.8,
    showLesionMask: false,
    wireframeOverlay: false,
    showHotspots: true,
  })

  const [atlasState, setAtlasState] = useState<AtlasSceneState>(INITIAL_ATLAS_STATE)
  const [selectedPart, setSelectedPart] = useState<Part | null>(null)

  const plateRef = useRef<HTMLElement>(null)
  const organCameraApi = useRef<OrganCameraApi | null>(null)

  const currentOrgan = ORGANS_REGISTRY[selectedOrganId]
  const isWholeBodyView = selectedOrganId === 'body'
  const plateNo = useMemo(
    () => String(ORGAN_ORDER.indexOf(selectedOrganId) + 1).padStart(2, '0'),
    [selectedOrganId]
  )

  const handleSelectOrgan = (id: OrganId) => {
    setSelectedOrganId(id)
    setActiveHotspot(null)
    if (id !== 'body') setLastOrganId(id)
  }

  // Selecting a landmark from the essay below scrolls the plate back into view.
  const handleSelectHotspot = useCallback((hotspot: Hotspot | null) => {
    setActiveHotspot(hotspot)
    if (hotspot) {
      plateRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [])

  const handleRegisterCamera = useCallback((api: OrganCameraApi) => {
    organCameraApi.current = api
  }, [])

  const handlePickPart = useCallback((part: Part) => {
    setSelectedPart((prev) => {
      const next = prev?.id === part.id ? null : part
      setAtlasState((s) => ({ ...s, selectedPartId: next ? next.id : null }))
      return next
    })
  }, [])

  const closeInspector = () => {
    setSelectedPart(null)
    setAtlasState((s) => ({ ...s, selectedPartId: null, isolate: false }))
  }

  const toggleAtlasSystem = (id: SystemId) => {
    setAtlasState((prev) => {
      const exists = prev.visibleSystems.includes(id)
      return {
        ...prev,
        visibleSystems: exists
          ? prev.visibleSystems.filter((s) => s !== id)
          : [...prev.visibleSystems, id],
      }
    })
  }

  const mappedOrganId = selectedPart ? mapPartToOrgan(selectedPart.name) : null
  const selectedSystemDef = selectedPart
    ? ATLAS_SYSTEMS.find((s) => s.id === selectedPart.system)
    : undefined

  return (
    <div className="om-page">
      {/* Side rails — 36px fixed strips, rotated editorial labels */}
      <div className="om-rail left" aria-hidden="true">
        <span>OpenMed — Anatomia Digitalis · MMXXVI</span>
      </div>
      <div className="om-rail right" aria-hidden="true">
        <span>Terminologia Anatomica · TA2 · Lahore</span>
      </div>

      {/* Masthead */}
      <header className="masthead">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">Ø</div>
          <span className="brand-word">
            OpenMed<span className="dot">.</span>
          </span>
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

      {/* Specimen title block — read it, then see it */}
      {isWholeBodyView ? (
        <EssayHead
          badge="3D Atlas · BodyParts3D 4.0"
          verified="Terminologia Anatomica concepts · CC BY 4.0"
          title="Corpus Integrum"
          serif="The Whole Figure"
          latin={`Plate Nº ${plateNo} — 2,234 named structures · 15 physiological systems`}
          lead="The centre page of the annual: one adult male reference body, segmented into every structure the anatomists of BodyParts3D chose to name — from the dura mater to the semitendinosus. Nothing here is a sculptor’s guess; each of the 2,234 meshes carries its own anatomical concept id, so naming, isolating and dissecting the figure stays truthful to the source atlas at every zoom."
        />
      ) : (
        <EssayHead
          badge={currentOrgan.modality}
          verified="Verified · Terminologia Anatomica TA2"
          title={currentOrgan.name}
          serif={currentOrgan.clinicalProfile?.poeticTitle}
          latin={`Plate Nº ${plateNo} — ${currentOrgan.anatomicalTerm}`}
          lead={currentOrgan.description}
        />
      )}

      {/* Stage — the plate stays clear; all controls live in the rail */}
      <main className={`stage ${isWholeBodyView ? 'stage-atlas' : ''}`}>
        <section className="plate" ref={plateRef}>
          {isWholeBodyView ? (
            <WholeBodyAtlasViewer
              plateNo={plateNo}
              state={atlasState}
              onPickPart={handlePickPart}
            />
          ) : (
            <OrganViewer3D
              organ={currentOrgan}
              plateNo={plateNo}
              settings={settings}
              activeHotspot={activeHotspot}
              onSelectHotspot={setActiveHotspot}
              onRegisterCamera={handleRegisterCamera}
            />
          )}

          {/* Selected-structure inspector (atlas only) */}
          {isWholeBodyView && selectedPart && (
            <div className="part-inspector" role="dialog" aria-label="Anatomical part inspector">
              <div className="co-head">
                <span className="co-eyebrow">Part Inspector</span>
                <button type="button" className="icon-btn" onClick={closeInspector} aria-label="Close inspector">
                  <X size={13} />
                </button>
              </div>

              <div>
                <h3 className="pi-name">
                  {selectedPart.name}
                  <span className="dot">.</span>
                </h3>
                <div className="pi-meta">
                  <span className="sys-chip">
                    <i style={{ backgroundColor: selectedSystemDef?.color ?? '#aebbb8' }} />
                    {selectedSystemDef?.name ?? selectedPart.system}
                  </span>
                  <span className="pi-latin">FMA · {selectedPart.conceptId}</span>
                </div>
                {selectedSystemDef && (
                  <p className="pi-desc">{selectedSystemDef.description}</p>
                )}
              </div>

              <div className="pi-actions">
                <button
                  type="button"
                  className={`btn-sm ${atlasState.isolate ? 'active' : ''}`}
                  onClick={() =>
                    setAtlasState((prev) => ({ ...prev, isolate: !prev.isolate }))
                  }
                >
                  <Maximize2 size={13} />
                  <span>{atlasState.isolate ? 'Show All Systems' : 'Isolate Structure'}</span>
                </button>

                {mappedOrganId && (
                  <button
                    type="button"
                    className="btn-sm hot"
                    onClick={() => handleSelectOrgan(mappedOrganId)}
                  >
                    <ExternalLink size={13} />
                    <span>Inspect in {ORGANS_REGISTRY[mappedOrganId].name} workspace</span>
                  </button>
                )}
              </div>
            </div>
          )}
        </section>

        {/* Controls rail — pushed off the model, onto paper */}
        <aside className="controls-rail">
          {isWholeBodyView ? (
            <AtlasControlsRail
              state={atlasState}
              onStateChange={setAtlasState}
              onToggleSystem={toggleAtlasSystem}
            />
          ) : (
            <OrganViewportControls
              settings={settings}
              onUpdateSettings={setSettings}
              onSetCameraPreset={(preset) => organCameraApi.current?.setPreset(preset)}
              onResetCamera={() => organCameraApi.current?.reset()}
              hotspotsCount={currentOrgan.hotspots?.length ?? 0}
            />
          )}
        </aside>
      </main>

      {/* The essay — below the plate, reached by scrolling */}
      <section className="dossier-flow">
        {isWholeBodyView ? (
          <AtlasDossier />
        ) : (
          <OrganInfoCard
            organ={currentOrgan}
            activeHotspot={activeHotspot}
            onSelectHotspot={handleSelectHotspot}
          />
        )}
      </section>

      {/* Index of specimens — structured table of contents */}
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
