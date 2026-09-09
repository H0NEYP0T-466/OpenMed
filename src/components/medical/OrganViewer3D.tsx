import React, { useEffect, useRef, useState, useCallback } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js'
import type { OrganMetadata, ViewerSettings, CameraPreset, Hotspot } from '../../types/organ'
import {
  applyRenderMode,
  disposeObjectTree,
  buildEnvironmentMap,
  enhancePBRMaterials,
  normalizeModelToFitSize,
  fitCameraToNormalizedPivot,
  createHotspotSprite,
  FIT_SIZE
} from '../../utils/threeHelpers'
import { OrganViewportControls } from './OrganViewportControls'
import { HotspotCallout } from './HotspotCallout'
import { AlertCircle, Loader2 } from 'lucide-react'

interface OrganViewer3DProps {
  readonly organ: OrganMetadata
  readonly settings: ViewerSettings
  readonly onUpdateSettings: (updater: (prev: ViewerSettings) => ViewerSettings) => void
  readonly activeHotspot: Hotspot | null
  readonly onSelectHotspot: (hotspot: Hotspot | null) => void
}

export const OrganViewer3D: React.FC<OrganViewer3DProps> = ({
  organ,
  settings,
  onUpdateSettings,
  activeHotspot,
  onSelectHotspot,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)
  const organPivotRef = useRef<THREE.Group | null>(null)
  const hotspotsGroupRef = useRef<THREE.Group | null>(null)
  const animFrameIdRef = useRef<number | null>(null)
  const defaultDistanceRef = useRef<number>(7.5)

  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [loadingProgress, setLoadingProgress] = useState<number>(0)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [hoveredHotspot, setHoveredHotspot] = useState<Hotspot | null>(null)

  const settingsRef = useRef<ViewerSettings>(settings)
  useEffect(() => {
    settingsRef.current = settings
  }, [settings])

  const activeHotspotRef = useRef<Hotspot | null>(activeHotspot)
  useEffect(() => {
    activeHotspotRef.current = activeHotspot
  }, [activeHotspot])

  // Initialize Three.js Scene once
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const scene = new THREE.Scene()
    sceneRef.current = scene

    const width = container.clientWidth || 800
    const height = container.clientHeight || 600

    const camera = new THREE.PerspectiveCamera(36, width / height, 0.1, 100)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance',
      stencil: false,
    })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.05
    renderer.outputColorSpace = THREE.SRGBColorSpace

    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.055
    controls.rotateSpeed = 0.75
    controls.zoomSpeed = 0.9
    controlsRef.current = controls

    // Clinical Studio Lighting Rig
    scene.add(new THREE.AmbientLight(0xffffff, 0.45))
    scene.add(new THREE.HemisphereLight(0xfff8ee, 0x1a2233, 0.75))

    const keyLight = new THREE.DirectionalLight(0xfff3e7, 3.2)
    keyLight.position.set(4.8, 6.5, 6.8)
    scene.add(keyLight)

    const fillLight = new THREE.DirectionalLight(0xe6ecff, 1.1)
    fillLight.position.set(-4.5, 1.2, 5.2)
    scene.add(fillLight)

    const rimLight = new THREE.DirectionalLight(0xffb7a5, 1.5)
    rimLight.position.set(-4, 3.5, -5.5)
    scene.add(rimLight)

    const organWarmGlow = new THREE.PointLight(0xff8d70, 0.6, 12, 2)
    organWarmGlow.position.set(-2.5, -1.2, 3.0)
    scene.add(organWarmGlow)

    // Warm-to-cool gradient PMREM environment probe
    const envMap = buildEnvironmentMap(renderer)
    scene.environment = envMap

    // Subtle anatomical floor grid
    const grid = new THREE.GridHelper(12, 24, 0x1e293b, 0x0f172a)
    grid.position.y = -2.2
    scene.add(grid)

    // Raycaster for interactive Hotspot picking
    const raycaster = new THREE.Raycaster()
    const mouse = new THREE.Vector2()

    const onPointerDown = (event: PointerEvent) => {
      if (!container || !cameraRef.current || !hotspotsGroupRef.current) return
      const rect = container.getBoundingClientRect()
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

      raycaster.setFromCamera(mouse, cameraRef.current)
      const intersects = raycaster.intersectObjects(hotspotsGroupRef.current.children, true)

      if (intersects.length > 0) {
        const topHit = intersects[0].object
        const hotspot = topHit.userData['hotspot'] as Hotspot | undefined
        if (hotspot) {
          onSelectHotspot(hotspot)
        }
      }
    }

    const onPointerMove = (event: PointerEvent) => {
      if (!container || !cameraRef.current || !hotspotsGroupRef.current) return
      const rect = container.getBoundingClientRect()
      mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

      raycaster.setFromCamera(mouse, cameraRef.current)
      const intersects = raycaster.intersectObjects(hotspotsGroupRef.current.children, true)

      if (intersects.length > 0) {
        container.style.cursor = 'pointer'
        const topHit = intersects[0].object
        const hotspot = topHit.userData['hotspot'] as Hotspot | undefined
        if (hotspot) setHoveredHotspot(hotspot)
      } else {
        container.style.cursor = 'grab'
        setHoveredHotspot(null)
      }
    }

    const domElement = renderer.domElement
    domElement.addEventListener('pointerdown', onPointerDown)
    domElement.addEventListener('pointermove', onPointerMove)

    // Responsive resize handler
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width: w, height: h } = entry.contentRect
        if (w > 0 && h > 0) {
          camera.aspect = w / h
          camera.updateProjectionMatrix()
          renderer.setSize(w, h)
        }
      }
    })
    resizeObserver.observe(container)

    // Animation Render Loop
    const animate = () => {
      animFrameIdRef.current = requestAnimationFrame(animate)

      if (controlsRef.current) {
        controlsRef.current.update()
      }

      // Smooth auto-rotation if active
      const currentSettings = settingsRef.current
      if (currentSettings.autoRotate && organPivotRef.current) {
        organPivotRef.current.rotation.y += 0.0035 * currentSettings.rotationSpeed
      }

      renderer.render(scene, camera)
    }
    animate()

    return () => {
      domElement.removeEventListener('pointerdown', onPointerDown)
      domElement.removeEventListener('pointermove', onPointerMove)
      resizeObserver.disconnect()
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current)
      }
      if (organPivotRef.current) {
        disposeObjectTree(organPivotRef.current)
      }
      envMap.dispose()
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [onSelectHotspot]) // run once on mount

  // Load organ model whenever selected organ changes
  useEffect(() => {
    const scene = sceneRef.current
    const camera = cameraRef.current
    const controls = controlsRef.current
    const renderer = rendererRef.current
    if (!scene || !camera || !controls || !renderer) return

    setIsLoading(true)
    setLoadingProgress(0)
    setLoadError(null)

    // Clear previously loaded organ pivot
    if (organPivotRef.current) {
      scene.remove(organPivotRef.current)
      disposeObjectTree(organPivotRef.current)
      organPivotRef.current = null
      hotspotsGroupRef.current = null
    }

    const loader = new GLTFLoader()
    loader.setMeshoptDecoder(MeshoptDecoder)

    loader.load(
      organ.modelFile,
      (gltf) => {
        const model = gltf.scene

        // Create unified organ pivot group
        const organPivot = new THREE.Group()
        organPivot.name = 'organ-pivot'
        organPivotRef.current = organPivot

        // Enhance baked PBR materials & texture anisotropy
        const maxAnisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy())
        enhancePBRMaterials(model, maxAnisotropy)

        // Normalize model to standard FIT_SIZE space (3.8 units)
        normalizeModelToFitSize(model, FIT_SIZE)
        organPivot.add(model)

        // Create Hotspots layer
        const hotspotsGroup = new THREE.Group()
        hotspotsGroup.name = 'hotspots-group'
        hotspotsGroupRef.current = hotspotsGroup

        if (organ.hotspots && organ.hotspots.length > 0) {
          organ.hotspots.forEach((spot) => {
            const isSelected = activeHotspotRef.current?.id === spot.id
            const sprite = createHotspotSprite(spot.color, isSelected)
            sprite.position.set(spot.position[0], spot.position[1], spot.position[2])
            sprite.userData = { isHotspot: true, hotspot: spot }
            hotspotsGroup.add(sprite)
          })
        }

        hotspotsGroup.visible = settingsRef.current.showHotspots
        organPivot.add(hotspotsGroup)

        scene.add(organPivot)

        // Fit camera cleanly to normalized pivot
        const dist = organ.cameraDistance ? organ.cameraDistance * 4.2 : 7.5
        defaultDistanceRef.current = dist
        fitCameraToNormalizedPivot(camera, controls, dist)

        // Apply current render mode
        applyRenderMode(
          model,
          settingsRef.current.renderMode,
          organ.accentColor,
          settingsRef.current.wireframeOverlay
        )

        setIsLoading(false)
        setLoadingProgress(100)
      },
      (xhr) => {
        if (xhr.total > 0) {
          const percent = Math.round((xhr.loaded / xhr.total) * 100)
          setLoadingProgress(percent)
        }
      },
      (error) => {
        console.warn(`Failed to load ${organ.modelFile}:`, error)
        setLoadError(`Unable to load 3D file for ${organ.name}. Model is caching or unavailable.`)
        setIsLoading(false)
      }
    )
  }, [organ.id, organ.name, organ.modelFile, organ.accentColor, organ.cameraDistance, organ.hotspots])

  // Update hotspots visibility when setting changes
  useEffect(() => {
    if (hotspotsGroupRef.current) {
      hotspotsGroupRef.current.visible = settings.showHotspots
    }
  }, [settings.showHotspots])

  // Re-scale active hotspot sprite
  useEffect(() => {
    if (hotspotsGroupRef.current) {
      hotspotsGroupRef.current.children.forEach((child) => {
        if ((child as THREE.Sprite).isSprite) {
          const sprite = child as THREE.Sprite
          const spot = sprite.userData['hotspot'] as Hotspot | undefined
          const isSelected = spot?.id === activeHotspot?.id
          const scale = isSelected ? 0.32 : 0.22
          sprite.scale.set(scale, scale, 1)
        }
      })
    }
  }, [activeHotspot])

  // Update shaders/materials when settings change
  useEffect(() => {
    if (organPivotRef.current) {
      applyRenderMode(
        organPivotRef.current,
        settings.renderMode,
        organ.accentColor,
        settings.wireframeOverlay
      )
    }
  }, [settings.renderMode, settings.wireframeOverlay, organ.accentColor])

  // Camera Preset Actions
  const handleSetCameraPreset = useCallback((preset: CameraPreset) => {
    const camera = cameraRef.current
    const controls = controlsRef.current
    if (!camera || !controls) return

    const dist = defaultDistanceRef.current

    switch (preset) {
      case 'anterior':
        camera.position.set(0, 0, dist)
        break
      case 'lateral':
        camera.position.set(dist, 0, 0)
        break
      case 'superior':
        camera.position.set(0, dist, 0.001)
        break
      case 'isometric':
        camera.position.set(dist * 0.7, dist * 0.5, dist * 0.7)
        break
    }

    camera.lookAt(0, 0, 0)
    controls.target.set(0, 0, 0)
    controls.update()
  }, [])

  const handleResetCamera = useCallback(() => {
    handleSetCameraPreset('anterior')
  }, [handleSetCameraPreset])

  return (
    <div className="organ-viewer-container">
      {/* 3D WebGL Canvas mount point */}
      <div ref={containerRef} className="three-canvas-root" />

      {/* Loading Overlay */}
      {isLoading && (
        <div className="viewer-overlay loading-overlay">
          <div className="loading-card">
            <Loader2 className="animate-spin text-cyan-400" size={32} />
            <span className="loading-text">Loading 3D Anatomy: {organ.name}</span>
            <div className="progress-bar-track">
              <div
                className="progress-bar-fill"
                style={{ width: `${Math.max(loadingProgress, 8)}%` }}
              />
            </div>
            <span className="loading-subtext">
              {loadingProgress > 0 ? `${loadingProgress}% loaded` : 'Decompressing PBR Mesh...'}
            </span>
          </div>
        </div>
      )}

      {/* Error Overlay */}
      {loadError && !isLoading && (
        <div className="viewer-overlay error-overlay">
          <div className="error-card">
            <AlertCircle size={28} className="text-rose-400" />
            <span className="error-title">3D Model Offline</span>
            <p className="error-description">{loadError}</p>
          </div>
        </div>
      )}

      {/* Controls HUD */}
      <OrganViewportControls
        settings={settings}
        onUpdateSettings={onUpdateSettings}
        onSetCameraPreset={handleSetCameraPreset}
        onResetCamera={handleResetCamera}
        hotspotsCount={organ.hotspots?.length ?? 0}
      />

      {/* Hover Tooltip */}
      {hoveredHotspot && !activeHotspot && (
        <div className="hotspot-hover-tooltip">
          <div
            className="tooltip-dot"
            style={{ backgroundColor: hoveredHotspot.color }}
          />
          <div className="tooltip-text">
            <strong>{hoveredHotspot.label}</strong>
            <span>{hoveredHotspot.latinTerm}</span>
          </div>
        </div>
      )}

      {/* Active Selected Hotspot Callout */}
      {activeHotspot && (
        <HotspotCallout
          hotspot={activeHotspot}
          onClose={() => onSelectHotspot(null)}
        />
      )}
    </div>
  )
}
