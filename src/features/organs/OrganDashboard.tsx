import React, { useState } from 'react'
import type { OrganId, ViewerSettings, Hotspot } from '../../types/organ'
import { ORGANS_REGISTRY } from '../../types/organ'
import { OrganViewer3D } from '../../components/medical/OrganViewer3D'
import { WholeBodyAtlasViewer } from '../../components/medical/WholeBodyAtlasViewer'
import { OrganSelector } from '../../components/medical/OrganSelector'
import { OrganInfoCard } from '../../components/medical/OrganInfoCard'
import {
  Activity,
  Cpu,
  User,
  GitBranch,
  Layers
} from 'lucide-react'

export const OrganDashboard: React.FC = () => {
  const [selectedOrganId, setSelectedOrganId] = useState<OrganId>('body')
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

  const handleSelectOrgan = (id: OrganId) => {
    setSelectedOrganId(id)
    setActiveHotspot(null)
  }

  const isWholeBodyView = selectedOrganId === 'body'

  return (
    <div className="hospital-cockpit">
      {/* Top Navigation Bar */}
      <header className="cockpit-navbar">
        <div className="navbar-brand">
          <div className="brand-logo-gem">
            <Activity className="text-cyan-400" size={20} />
          </div>
          <div>
            <div className="brand-name-row">
              <span className="brand-title">OpenMed</span>
              <span className="brand-version-badge">v0.1 FYP</span>
            </div>
            <span className="brand-subtitle">AI Hospital • Multi-Organ Diagnostic Platform</span>
          </div>
        </div>

        <div className="navbar-stats">
          <div className="stat-pill">
            <Cpu size={14} className="text-cyan-400" />
            <span>12+ Organs Online</span>
          </div>
          <div className="stat-pill">
            <GitBranch size={14} className="text-purple-400" />
            <span>20+ AI Models</span>
          </div>
          <div className="stat-pill">
            <Layers size={14} className="text-emerald-400" />
            <span>BodyParts3D Atlas</span>
          </div>
        </div>

        <div className="navbar-actions">
          <button
            type="button"
            className={`view-mode-toggle ${isWholeBodyView ? 'active' : ''}`}
            onClick={() => handleSelectOrgan('body')}
          >
            <User size={15} />
            <span>Whole Body Macro Atlas</span>
          </button>
        </div>
      </header>

      {/* Main Clinical Diagnostic Workspace */}
      <main className="cockpit-main-workspace">
        {isWholeBodyView ? (
          /* Full Body Multi-System Atlas Viewer (from ashemag/human-atlas) */
          <section className="viewport-panel full-viewport-atlas">
            <WholeBodyAtlasViewer onNavigateToOrgan={handleSelectOrgan} />
          </section>
        ) : (
          /* Isolated Organ Diagnostic Workspace (with PBR textures & 3D Hotspot Points) */
          <>
            <section className="viewport-panel">
              <OrganViewer3D
                organ={currentOrgan}
                settings={settings}
                onUpdateSettings={setSettings}
                activeHotspot={activeHotspot}
                onSelectHotspot={setActiveHotspot}
              />
            </section>

            <aside className="telemetry-panel">
              <OrganInfoCard
                organ={currentOrgan}
                activeHotspot={activeHotspot}
                onSelectHotspot={setActiveHotspot}
              />
            </aside>
          </>
        )}
      </main>

      {/* Bottom Organ Selector Bar */}
      <footer className="cockpit-footer-bar">
        <OrganSelector
          selectedId={selectedOrganId}
          onSelectOrgan={handleSelectOrgan}
        />
      </footer>
    </div>
  )
}
