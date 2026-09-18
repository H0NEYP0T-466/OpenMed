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
import './rail.css'

interface OrganViewportControlsProps {
  readonly settings: ViewerSettings
  readonly onUpdateSettings: (updater: (prev: ViewerSettings) => ViewerSettings) => void
  readonly onSetCameraPreset: (preset: CameraPreset) => void
  readonly onResetCamera: () => void
  readonly hotspotsCount: number
}

/**
 * Paper control rail - lives beside the plate, never on top of the model.
 */
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
    <>
      {/* Pl. I - Shading */}
      <div className="rail-group">
        <div className="rail-head">
          <span className="rail-label">Pl. I - Shading</span>
        </div>
        <div className="rail-btns" role="group" aria-label="Shading modes">
          <button
            type="button"
            className={`rail-btn ${settings.renderMode === 'pbr' ? 'active' : ''}`}
            onClick={() => setRenderMode('pbr')}
            title="Anatomical PBR solid"
          >
            <Box size={13} className="ico" />
            <span>Solid</span>
          </button>

          <button
            type="button"
            className={`rail-btn ${settings.renderMode === 'xray' ? 'active' : ''}`}
            onClick={() => setRenderMode('xray')}
            title="Translucent bone-paper X-ray"
          >
            <Eye size={13} className="ico" />
            <span>X-Ray</span>
          </button>

          <button
            type="button"
            className={`rail-btn ${settings.renderMode === 'wireframe' ? 'active' : ''}`}
            onClick={() => setRenderMode('wireframe')}
            title="Topological wireframe"
          >
            <Layers size={13} className="ico" />
            <span>Wireframe</span>
          </button>

          <button
            type="button"
            className={`rail-btn ${settings.renderMode === 'segmentation' ? 'active' : ''}`}
            onClick={() => setRenderMode('segmentation')}
            title="AI segmentation lesion layer"
          >
            <Sparkles size={13} className="ico" />
            <span>AI Mask</span>
          </button>
        </div>
      </div>

      {/* Pl. II - Annotation */}
      <div className="rail-group">
        <div className="rail-head">
          <span className="rail-label">Pl. II - Annotation</span>
        </div>
        <div className="rail-btns" role="group" aria-label="Annotation toggles">
          {hotspotsCount > 0 && (
            <button
              type="button"
              className={`rail-btn ${settings.showHotspots ? 'active' : ''}`}
              onClick={toggleHotspots}
              title="Toggle anatomical landmark points"
            >
              <MapPin size={13} className="ico" />
              <span>Landmarks ({hotspotsCount})</span>
            </button>
          )}

          <button
            type="button"
            className={`rail-btn ${settings.autoRotate ? 'active' : ''}`}
            onClick={toggleAutoRotate}
            title="Toggle turntable rotation"
          >
            <RotateCw size={13} className={`ico ${settings.autoRotate ? 'spin-slow' : ''}`} />
            <span>Turntable</span>
          </button>

          <button
            type="button"
            className={`rail-btn ${settings.wireframeOverlay ? 'active' : ''}`}
            onClick={toggleWireframeOverlay}
            title="Toggle wireframe grid overlay"
          >
            <Maximize2 size={13} className="ico" />
            <span>Grid Overlay</span>
          </button>
        </div>
      </div>

      {/* Pl. III - Planes */}
      <div className="rail-group">
        <div className="rail-head">
          <span className="rail-label">Pl. III - Planes</span>
        </div>
        <div className="rail-btns" role="group" aria-label="Camera planes">
          <button
            type="button"
            className="rail-btn"
            onClick={() => onSetCameraPreset('anterior')}
            title="Anterior (frontal plane)"
          >
            <Compass size={13} className="ico" />
            <span>Anterior</span>
          </button>

          <button
            type="button"
            className="rail-btn"
            onClick={() => onSetCameraPreset('lateral')}
            title="Lateral (sagittal plane)"
          >
            <Compass size={13} className="ico" />
            <span>Lateral</span>
          </button>

          <button
            type="button"
            className="rail-btn"
            onClick={() => onSetCameraPreset('superior')}
            title="Superior (axial plane)"
          >
            <Compass size={13} className="ico" />
            <span>Superior</span>
          </button>

          <button
            type="button"
            className="rail-btn"
            onClick={onResetCamera}
            title="Recenter & reset camera"
          >
            <RefreshCw size={13} className="ico" />
            <span>Reset</span>
          </button>
        </div>
      </div>
    </>
  )
}
