import React from 'react'
import type { AtlasSceneState, SystemId } from '../../types/atlas'
import { ATLAS_SYSTEMS } from '../../types/atlas'
import { RotateCw, Compass } from 'lucide-react'

interface AtlasControlsRailProps {
  readonly state: AtlasSceneState
  readonly onStateChange: React.Dispatch<React.SetStateAction<AtlasSceneState>>
  readonly onToggleSystem: (id: SystemId) => void
}

const VIEW_PRESETS = [
  { id: 'front', label: 'Anterior' },
  { id: 'back', label: 'Posterior' },
  { id: 'side', label: 'Lateral' },
  { id: 'three-quarter', label: 'Isometric' },
] as const

export const AtlasControlsRail: React.FC<AtlasControlsRailProps> = ({
  state,
  onStateChange,
  onToggleSystem,
}) => {
  const pct = Math.round(state.explode * 100)

  return (
    <>
      {/* Pl. I — Physiological systems */}
      <div className="rail-group">
        <div className="rail-head">
          <span className="rail-label">Pl. I — Systems</span>
          <span className="rail-count">{state.visibleSystems.length}/15</span>
        </div>
        <div className="rail-systems">
          {ATLAS_SYSTEMS.map((sys) => {
            const isVisible = state.visibleSystems.includes(sys.id)
            return (
              <button
                key={sys.id}
                type="button"
                className={`sys-item ${isVisible ? '' : 'off'}`}
                onClick={() => onToggleSystem(sys.id)}
                aria-pressed={isVisible}
                title={sys.description}
              >
                <span className="sys-swatch" style={{ backgroundColor: sys.color }} />
                <span className="sys-name">{sys.name}</span>
                <span className="sys-tick">{isVisible ? '✓' : '—'}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Pl. II — Dissection */}
      <div className="rail-group">
        <div className="rail-head">
          <span className="rail-label">Pl. II — Dissection</span>
          <span className="val-badge">{pct}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={state.explode}
          onChange={(e) =>
            onStateChange((prev) => ({ ...prev, explode: parseFloat(e.target.value) }))
          }
          className="range"
          aria-label="Anatomical dissection amount"
        />
        <div className="range-foot">
          <span>In situ</span>
          <span>Inventory board</span>
        </div>
        <p className="rail-hint">
          Past 50% the plate flattens into its inventory — hover any structure to read its name.
        </p>
      </div>

      {/* Pl. III — View */}
      <div className="rail-group">
        <div className="rail-head">
          <span className="rail-label">Pl. III — View</span>
        </div>
        <div className="rail-btns">
          {VIEW_PRESETS.map((preset) => (
            <button
              key={preset.id}
              type="button"
              className={`rail-btn ${state.viewAngle === preset.id ? 'active' : ''}`}
              onClick={() => onStateChange((prev) => ({ ...prev, viewAngle: preset.id }))}
            >
              <Compass size={13} className="ico" />
              <span>{preset.label}</span>
            </button>
          ))}
          <button
            type="button"
            className={`rail-btn ${state.autoRotate ? 'active' : ''}`}
            onClick={() => onStateChange((prev) => ({ ...prev, autoRotate: !prev.autoRotate }))}
          >
            <RotateCw size={13} className={`ico ${state.autoRotate ? 'spin-slow' : ''}`} />
            <span>Turntable</span>
          </button>
        </div>
      </div>
    </>
  )
}
