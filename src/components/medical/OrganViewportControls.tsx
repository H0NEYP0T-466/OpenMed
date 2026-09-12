import React from 'react'
import {
  RotateCw,
  Eye,
  Layers,
  Box,
  Compass,
  Sparkles,
  RefreshCw,
  Maximize2,
  MapPin
} from 'lucide-react'
import type { RenderMode, CameraPreset, ViewerSettings } from '../../types/organ'

interface OrganViewportControlsProps {
  readonly settings: ViewerSettings
  readonly onUpdateSettings: (updater: (prev: ViewerSettings) => ViewerSettings) => void
  readonly onSetCameraPreset: (preset: CameraPreset) => void
  readonly onResetCamera: () => void
  readonly hotspotsCount: number
}

export const OrganViewportControls: React.FC<OrganViewportControlsProps> = ({
  settings,
  onUpdateSettings,
  onSetCameraPreset,
  onResetCamera,
  hotspotsCount,
}) => {
  const setRenderMode = (mode: RenderMode) => {
    onUpdateSettings((prev) => ({ ...prev, renderMode: mode }))
  }

  const toggleAutoRotate = () => {
    onUpdateSettings((prev) => ({ ...prev, autoRotate: !prev.autoRotate }))
  }

  const toggleWireframeOverlay = () => {
    onUpdateSettings((prev) => ({ ...prev, wireframeOverlay: !prev.wireframeOverlay }))
  }

  const toggleHotspots = () => {
    onUpdateSettings((prev) => ({ ...prev, showHotspots: !prev.showHotspots }))
  }

  return (
    <div className="plate-hud">
      {/* Shading modes */}
      <div className="hud-group">
        <span className="hud-label">Pl. I — Shading</span>
        <div className="hud-chip" role="group" aria-label="Shading modes">
          <button
            type="button"
            className={`hud-btn ${settings.renderMode === 'pbr' ? 'active' : ''}`}
            onClick={() => setRenderMode('pbr')}
            title="Anatomical PBR Solid"
          >
            <Box size={13} className="ico" />
            <span>Solid</span>
          </button>

          <button
            type="button"
            className={`hud-btn ${settings.renderMode === 'xray' ? 'active' : ''}`}
            onClick={() => setRenderMode('xray')}
            title="Translucent X-Ray"
          >
            <Eye size={13} className="ico" />
            <span>X-Ray</span>
          </button>

          <button
            type="button"
            className={`hud-btn ${settings.renderMode === 'wireframe' ? 'active' : ''}`}
            onClick={() => setRenderMode('wireframe')}
            title="Topological Wireframe"
          >
            <Layers size={13} className="ico" />
            <span>Wireframe</span>
          </button>

          <button
            type="button"
            className={`hud-btn ${settings.renderMode === 'segmentation' ? 'active' : ''}`}
            onClick={() => setRenderMode('segmentation')}
            title="AI Segmentation Lesion Layer"
          >
            <Sparkles size={13} className="ico" />
            <span>AI Mask</span>
          </button>
        </div>
      </div>

      {/* Inspection & camera */}
      <div className="hud-group right">
        <span className="hud-label">Pl. II — Inspection</span>
        <div className="hud-chip" role="group" aria-label="Inspection controls">
          {hotspotsCount > 0 && (
            <button
              type="button"
              className={`hud-btn ${settings.showHotspots ? 'active' : ''}`}
              onClick={toggleHotspots}
              title="Toggle anatomical landmark points"
            >
              <MapPin size={13} className="ico" />
              <span>Points ({hotspotsCount})</span>
            </button>
          )}

          <button
            type="button"
            className="hud-btn"
            onClick={() => onSetCameraPreset('anterior')}
            title="Anterior (frontal plane)"
          >
            <Compass size={13} className="ico" />
            <span>Anterior</span>
          </button>

          <button
            type="button"
            className="hud-btn"
            onClick={() => onSetCameraPreset('lateral')}
            title="Lateral (sagittal plane)"
          >
            <Compass size={13} className="ico" />
            <span>Lateral</span>
          </button>

          <button
            type="button"
            className="hud-btn"
            onClick={() => onSetCameraPreset('superior')}
            title="Superior (axial plane)"
          >
            <Compass size={13} className="ico" />
            <span>Superior</span>
          </button>

          <button
            type="button"
            className={`hud-btn ${settings.autoRotate ? 'active' : ''}`}
            onClick={toggleAutoRotate}
            title="Toggle 360° auto-rotation"
          >
            <RotateCw size={13} className={`ico ${settings.autoRotate ? 'spin-slow' : ''}`} />
            <span>Rotate</span>
          </button>

          <button
            type="button"
            className={`hud-btn ${settings.wireframeOverlay ? 'active' : ''}`}
            onClick={toggleWireframeOverlay}
            title="Toggle wireframe grid overlay"
          >
            <Maximize2 size={13} className="ico" />
            <span>Grid</span>
          </button>

          <button
            type="button"
            className="hud-btn"
            onClick={onResetCamera}
            title="Recenter & reset camera"
          >
            <RefreshCw size={13} className="ico" />
            <span>Reset</span>
          </button>
        </div>
      </div>
    </div>
  )
}
