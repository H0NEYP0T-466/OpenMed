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
    <div className="viewport-controls-hud">
      {/* Top Left: Shading Modes */}
      <div className="controls-group">
        <span className="controls-group-label">Shading & Diagnostics</span>
        <div className="controls-buttons-row">
          <button
            type="button"
            className={`control-btn ${settings.renderMode === 'pbr' ? 'active' : ''}`}
            onClick={() => setRenderMode('pbr')}
            title="Anatomical PBR Solid"
          >
            <Box size={14} />
            <span>Anatomical</span>
          </button>

          <button
            type="button"
            className={`control-btn ${settings.renderMode === 'xray' ? 'active' : ''}`}
            onClick={() => setRenderMode('xray')}
            title="Translucent Holographic X-Ray"
          >
            <Eye size={14} />
            <span>X-Ray</span>
          </button>

          <button
            type="button"
            className={`control-btn ${settings.renderMode === 'wireframe' ? 'active' : ''}`}
            onClick={() => setRenderMode('wireframe')}
            title="Topological Wireframe"
          >
            <Layers size={14} />
            <span>Wireframe</span>
          </button>

          <button
            type="button"
            className={`control-btn lesion-btn ${settings.renderMode === 'segmentation' ? 'active' : ''}`}
            onClick={() => setRenderMode('segmentation')}
            title="AI Segmentation Lesion Layer"
          >
            <Sparkles size={14} />
            <span>AI Lesion Mask</span>
          </button>
        </div>
      </div>

      {/* Top Right: Camera Presets & Movement */}
      <div className="controls-group">
        <span className="controls-group-label">Inspection & Points</span>
        <div className="controls-buttons-row">
          {hotspotsCount > 0 && (
            <button
              type="button"
              className={`control-btn landmarks-btn ${settings.showHotspots ? 'active' : ''}`}
              onClick={toggleHotspots}
              title="Toggle Anatomical Landmarks & Points"
            >
              <MapPin size={14} />
              <span>Points ({hotspotsCount})</span>
            </button>
          )}

          <button
            type="button"
            className="control-btn"
            onClick={() => onSetCameraPreset('anterior')}
            title="Anterior (Frontal Plane)"
          >
            <Compass size={14} />
            <span>Anterior</span>
          </button>

          <button
            type="button"
            className="control-btn"
            onClick={() => onSetCameraPreset('lateral')}
            title="Lateral (Sagittal Plane)"
          >
            <Compass size={14} />
            <span>Lateral</span>
          </button>

          <button
            type="button"
            className="control-btn"
            onClick={() => onSetCameraPreset('superior')}
            title="Superior (Axial Plane)"
          >
            <Compass size={14} />
            <span>Superior</span>
          </button>

          <button
            type="button"
            className={`control-btn ${settings.autoRotate ? 'active' : ''}`}
            onClick={toggleAutoRotate}
            title="Toggle 360° Auto-Rotation"
          >
            <RotateCw size={14} className={settings.autoRotate ? 'spin-slow' : ''} />
            <span>Rotate</span>
          </button>

          <button
            type="button"
            className={`control-btn ${settings.wireframeOverlay ? 'active' : ''}`}
            onClick={toggleWireframeOverlay}
            title="Toggle Wireframe Grid Overlay"
          >
            <Maximize2 size={14} />
            <span>Grid</span>
          </button>

          <button
            type="button"
            className="control-btn reset-btn"
            onClick={onResetCamera}
            title="Recenter & Reset Camera"
          >
            <RefreshCw size={14} />
            <span>Reset</span>
          </button>
        </div>
      </div>
    </div>
  )
}
