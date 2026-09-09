import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type {
  SystemId,
  AtlasData,
  Part,
  AtlasSceneState,
  AtlasViewAngle,
} from '../../types/atlas'
import { ATLAS_SYSTEMS, DEFAULT_ATLAS_SYSTEMS } from '../../types/atlas'
import type { OrganId } from '../../types/organ'
import {
  createExplosionLayout,
  PointerTap,
  decodeModelResponse,
} from '../../utils/atlasHelpers'
import {
  Layers,
  RotateCw,
  Compass,
  Sliders,
  CheckSquare,
  Square,
  Maximize2,
  ExternalLink,
  Info,
  Loader2,
  AlertCircle,
  X
} from 'lucide-react'

interface WholeBodyAtlasViewerProps {
  readonly onNavigateToOrgan: (organId: OrganId) => void
}

export const WholeBodyAtlasViewer: React.FC<WholeBodyAtlasViewerProps> = ({
  onNavigateToOrgan,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [atlasData, setAtlasData] = useState<AtlasData | null>(null)
  const [loadingProgress, setLoadingProgress] = useState<number>(0)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  // Atlas Scene State
  const [sceneState, setSceneState] = useState<AtlasSceneState>({
    explode: 0,
    visibleSystems: DEFAULT_ATLAS_SYSTEMS,
    selectedPartId: null,
    isolate: false,
    viewAngle: 'three-quarter',
    autoRotate: true,
  })

  const [selectedPart, setSelectedPart] = useState<Part | null>(null)
  const sceneStateRef = useRef<AtlasSceneState>(sceneState)
  useEffect(() => {
    sceneStateRef.current = sceneState
  }, [sceneState])

  // Fetch atlas.json metadata on mount
  useEffect(() => {
    let active = true
    fetch('/models/atlas/atlas.json')
      .then((res) => {
        if (!res.ok) throw new Error('Could not load atlas metadata.')
        return res.json()
      })
      .then((data: AtlasData) => {
        if (active) {
          setAtlasData(data)
        }
      })
      .catch((err) => {
        if (active) {
          setErrorMessage(err.message || 'Failed to initialize BodyParts3D Atlas.')
          setIsLoading(false)
        }
      })

    return () => {
      active = false
    }
  }, [])

  // WebGL Scene Initialization & Chunk Assembly
  useEffect(() => {
    const container = containerRef.current
    if (!container || !atlasData) return

    let disposed = false
    let frame = 0
    let dirty = true
    let ready = false
    let amount = 0
    let lastView = ''
    let layoutKey = ''

    const abortController = new AbortController()

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor('#080d18')
    renderer.outputColorSpace = THREE.SRGBColorSpace
    renderer.toneMapping = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.15
    renderer.setSize(container.clientWidth, container.clientHeight)
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(
      34,
      container.clientWidth / container.clientHeight,
      0.005,
      100
    )
    camera.position.set(1.4, 1.05, 3.6)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, 0.85, 0)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.minDistance = 0.2
    controls.maxDistance = 35
    controls.addEventListener('change', () => {
      dirty = true
    })

    // Lighting
    scene.add(new THREE.AmbientLight(0xffffff, 0.55))
    scene.add(new THREE.HemisphereLight(0xfff8ee, 0x1e293b, 0.85))

    const keyLight = new THREE.DirectionalLight(0xfffaf4, 2.5)
    keyLight.position.set(-2, 4, 3)
    scene.add(keyLight)

    const rimLight = new THREE.DirectionalLight(0xe9f0ff, 1.8)
    rimLight.position.set(2, 2, -3)
    scene.add(rimLight)

    // Floor Platform
    const platform = new THREE.Mesh(
      new THREE.CylinderGeometry(0.7, 0.72, 0.03, 64),
      new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.8 })
    )
    platform.position.y = -0.016
    scene.add(platform)

    const grid = new THREE.GridHelper(8, 16, 0x1e293b, 0x0f172a)
    grid.position.y = -0.018
    scene.add(grid)

    // State Texture Buffers for GPU Shader Injections
    const width = THREE.MathUtils.ceilPowerOfTwo(atlasData.parts.length)
    const data = new Float32Array(width * 4)
    const partTexture = new THREE.DataTexture(data, width, 1, THREE.RGBAFormat, THREE.FloatType)
    partTexture.needsUpdate = true

    const selectedData = new Uint8Array(width * 4)
    const selectionTexture = new THREE.DataTexture(selectedData, width, 1)
    selectionTexture.needsUpdate = true

    const materials: THREE.Material[] = []
    const geometries: THREE.BufferGeometry[] = []
    const pickers: (THREE.Mesh | undefined)[] = []

    const centers = atlasData.parts.map((p) =>
      new THREE.Vector3()
        .fromArray(p.bounds[0])
        .add(new THREE.Vector3().fromArray(p.bounds[1]))
        .multiplyScalar(0.5)
    )
    const offsets: THREE.Vector3[] = []
    const bounds = atlasData.parts.map(
      (p) =>
        new THREE.Box3(
          new THREE.Vector3().fromArray(p.bounds[0]),
          new THREE.Vector3().fromArray(p.bounds[1])
        )
    )

    const materialFor = (system: string) => {
      const sysDef = ATLAS_SYSTEMS.find((s) => s.id === system)
      const m = new THREE.MeshStandardMaterial({
        color: sysDef?.color ?? '#aebbb8',
        metalness: 0.08,
        roughness: 0.52,
        side: THREE.DoubleSide,
        transparent: system === 'integumentary',
        opacity: system === 'integumentary' ? 0.16 : 1,
        depthWrite: system !== 'integumentary',
      })

      m.onBeforeCompile = (shader) => {
        shader.uniforms.partState = { value: partTexture }
        shader.uniforms.selectionState = { value: selectionTexture }
        shader.uniforms.stateWidth = { value: width }

        shader.vertexShader =
          'attribute float partIndex; uniform sampler2D partState; uniform sampler2D selectionState; uniform float stateWidth; varying float partVisible; varying float partSelected;\n' +
          shader.vertexShader

        shader.vertexShader = shader.vertexShader.replace(
          '#include <begin_vertex>',
          '#include <begin_vertex>\nvec2 stateUv = vec2((partIndex + 0.5) / stateWidth, 0.5); vec4 state = texture2D(partState, stateUv); transformed += state.xyz; partVisible = state.w; partSelected = texture2D(selectionState, stateUv).r;'
        )

        shader.fragmentShader =
          'varying float partVisible; varying float partSelected;\n' + shader.fragmentShader

        shader.fragmentShader = shader.fragmentShader.replace(
          '#include <clipping_planes_fragment>',
          '#include <clipping_planes_fragment>\nif (partVisible < 0.5) discard;'
        )

        shader.fragmentShader = shader.fragmentShader.replace(
          '#include <color_fragment>',
          '#include <color_fragment>\ndiffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.22, 0.74, 0.98), partSelected * 0.85);'
        )
      }

      materials.push(m)
      return m
    }

    const mats = new Map(ATLAS_SYSTEMS.map((s) => [s.id, materialFor(s.id)]))

    let loadedCount = 0

    const loadChunk = async (chunkIndex: number) => {
      const chunk = atlasData.chunks[chunkIndex]
      const fetchUrl = chunk.gzip || chunk.url
      const response = await fetch(fetchUrl, { signal: abortController.signal })
      const buffer = await decodeModelResponse(response, chunk.bytes, Boolean(chunk.gzip))
      if (disposed) return

      const groups = new Map<string, THREE.BufferGeometry[]>()

      atlasData.parts.forEach((p, i) => {
        if (p.chunk !== chunkIndex) return
        const g = new THREE.BufferGeometry()
        g.setAttribute(
          'position',
          new THREE.BufferAttribute(new Float32Array(buffer, p.positions, p.vertexCount * 3), 3)
        )
        g.setAttribute(
          'normal',
          new THREE.BufferAttribute(new Int16Array(buffer, p.normals, p.vertexCount * 3), 3, true)
        )
        g.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer, p.indices, p.indexCount), 1))
        g.boundingBox = bounds[i].clone()
        g.computeBoundingSphere()

        const pick = new THREE.Mesh(g)
        pick.matrixAutoUpdate = false
        pickers[i] = pick
        geometries.push(g)

        g.setAttribute('partIndex', new THREE.BufferAttribute(new Float32Array(p.vertexCount).fill(i), 1))

        const list = groups.get(p.system) ?? []
        list.push(g)
        groups.set(p.system, list)
      })

      groups.forEach((geomList, system) => {
        const merged = mergeGeometries(geomList, false)
        if (!merged) return
        geometries.push(merged)
        const mesh = new THREE.Mesh(merged, mats.get(system as SystemId))
        mesh.frustumCulled = false
        scene.add(mesh)
      })

      loadedCount++
      setLoadingProgress(Math.round((loadedCount / atlasData.chunks.length) * 100))
      dirty = true
    }

    // Load chunks progressively
    ;(async () => {
      try {
        let cursor = 0
        await Promise.all(
          Array.from({ length: 4 }, async () => {
            while (cursor < atlasData.chunks.length) {
              const i = cursor++
              await loadChunk(i)
            }
          })
        )
        if (!disposed) {
          ready = true
          setIsLoading(false)
          dirty = true
        }
      } catch (e) {
        if (!disposed) {
          setErrorMessage(e instanceof Error ? e.message : 'Could not load atlas geometry.')
          setIsLoading(false)
        }
      }
    })()

    // View Angle Camera Fitting
    const fit = (view: AtlasViewAngle, extent = 0) => {
      const normalDistance = 3.6
      const distance = THREE.MathUtils.lerp(normalDistance, 6.0, extent)
      const direction =
        view === 'front'
          ? new THREE.Vector3(0, 0.02, 1)
          : view === 'back'
          ? new THREE.Vector3(0, 0.02, -1)
          : view === 'side'
          ? new THREE.Vector3(1, 0.02, 0)
          : new THREE.Vector3(0.35, 0.06, 1).normalize()

      controls.target.set(0, 0.82, 0)
      camera.position.copy(controls.target).addScaledVector(direction, distance)
      controls.update()
      dirty = true
    }

    const resize = () => {
      if (!container) return
      camera.aspect = container.clientWidth / container.clientHeight
      camera.updateProjectionMatrix()
      renderer.setSize(container.clientWidth, container.clientHeight)
      dirty = true
    }
    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(container)

    // Raycasting & Tap Detection for 2,234 Anatomical Parts
    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    const tap = new PointerTap()
    const worldBox = new THREE.Box3()
    const hitPoint = new THREE.Vector3()

    const onPointerDown = (e: PointerEvent) => {
      tap.down(e.pointerId, e.clientX, e.clientY, e.pointerType === 'touch' ? 12 : 5)
    }

    const onPointerMove = (e: PointerEvent) => {
      tap.move(e.pointerId, e.clientX, e.clientY)
    }

    const onPointerUp = (e: PointerEvent) => {
      const validTap = tap.up(e.pointerId, e.clientX, e.clientY)
      if (!validTap || !ready) return

      const rect = renderer.domElement.getBoundingClientRect()
      pointer.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      )
      raycaster.setFromCamera(pointer, camera)

      let nearest = Infinity
      let found = -1

      pickers.forEach((mesh, i) => {
        if (!mesh || data[i * 4 + 3] < 0.5) return
        worldBox.copy(bounds[i]).translate(mesh.position)
        if (!raycaster.ray.intersectBox(worldBox, hitPoint)) return
        const hits = raycaster.intersectObject(mesh, false)
        if (hits[0] && hits[0].distance < nearest) {
          nearest = hits[0].distance
          found = i
        }
      })

      if (found >= 0) {
        const part = atlasData.parts[found]
        setSelectedPart(part)
        setSceneState((prev) => ({
          ...prev,
          selectedPartId: prev.selectedPartId === part.id ? null : part.id,
        }))
      }
    }

    const dom = renderer.domElement
    dom.addEventListener('pointerdown', onPointerDown)
    dom.addEventListener('pointermove', onPointerMove)
    dom.addEventListener('pointerup', onPointerUp)

    // Render Animation Loop
    const clock = new THREE.Clock()
    const animate = () => {
      if (disposed) return
      frame = requestAnimationFrame(animate)

      const dt = Math.min(clock.getDelta(), 0.05)
      const s = sceneStateRef.current

      controls.update()

      if (s.autoRotate) {
        scene.rotation.y += 0.003
        dirty = true
      }

      // Smooth interpolation for explosion / dissection slider
      const moving = Math.abs(amount - s.explode) > 0.0001
      if (moving) {
        amount = THREE.MathUtils.damp(amount, s.explode, 8, dt)
        dirty = true
      }

      const visible = new Set(s.visibleSystems)
      const selected = s.selectedPartId ? new Set([s.selectedPartId]) : new Set<string>()

      const visibleParts = atlasData.parts.filter((p) =>
        s.isolate ? selected.has(p.id) : visible.has(p.system) || selected.has(p.id)
      )

      const nextLayoutKey = visibleParts.map((p) => p.id).join(',') + ':' + camera.aspect.toFixed(3)

      if (nextLayoutKey !== layoutKey) {
        const layout = createExplosionLayout(visibleParts, camera.aspect)

        atlasData.parts.forEach((p, i) => {
          const cell = layout.cells.get(p.id)
          offsets[i] = cell ? new THREE.Vector3(cell.x, cell.y + 0.85, 0) : centers[i].clone()
        })
        layoutKey = nextLayoutKey
      }

      atlasData.parts.forEach((p, i) => {
        const c = centers[i]
        const destination = offsets[i]
        let dx = 0
        let dy = 0
        let dz = 0

        if (amount <= 0.45) {
          const t = amount / 0.45
          const group = ATLAS_SYSTEMS.findIndex((sys) => sys.id === p.system)
          const angle = (group / ATLAS_SYSTEMS.length) * Math.PI * 2
          dx = Math.sin(angle) * t * 0.48
          dy = (c.y - 0.85) * t * 0.28
          dz = Math.cos(angle) * t * 0.48
        } else {
          const t = (amount - 0.45) / 0.55
          const group = ATLAS_SYSTEMS.findIndex((sys) => sys.id === p.system)
          const angle = (group / ATLAS_SYSTEMS.length) * Math.PI * 2
          dx = THREE.MathUtils.lerp(Math.sin(angle) * 0.48, destination.x - c.x, t)
          dy = THREE.MathUtils.lerp((c.y - 0.85) * 0.28, destination.y - c.y, t)
          dz = THREE.MathUtils.lerp(Math.cos(angle) * 0.48, -c.z, t)
        }

        const isPartSelected = selected.has(p.id)
        data.set(
          [dx, dy, dz, (s.isolate ? isPartSelected : visible.has(p.system) || isPartSelected) ? 1 : 0],
          i * 4
        )
        selectedData[i * 4] = isPartSelected ? 255 : 0

        const mesh = pickers[i]
        if (mesh) {
          mesh.position.set(dx, dy, dz)
          mesh.updateMatrix()
          mesh.updateMatrixWorld(true)
        }
      })

      partTexture.needsUpdate = true
      selectionTexture.needsUpdate = true

      if (s.viewAngle !== lastView) {
        fit(s.viewAngle, amount)
        lastView = s.viewAngle
      }

      if (dirty) {
        renderer.render(scene, camera)
      }
    }
    animate()

    return () => {
      disposed = true
      abortController.abort()
      cancelAnimationFrame(frame)
      resizeObserver.disconnect()
      dom.removeEventListener('pointerdown', onPointerDown)
      dom.removeEventListener('pointermove', onPointerMove)
      dom.removeEventListener('pointerup', onPointerUp)

      geometries.forEach((g) => g.dispose())
      materials.forEach((m) => m.dispose())
      partTexture.dispose()
      selectionTexture.dispose()
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [atlasData])

  // System visibility toggle
  const toggleSystem = (id: SystemId) => {
    setSceneState((prev) => {
      const exists = prev.visibleSystems.includes(id)
      return {
        ...prev,
        visibleSystems: exists
          ? prev.visibleSystems.filter((s) => s !== id)
          : [...prev.visibleSystems, id],
      }
    })
  }

  // Quick organ map to jump to OpenMed dedicated workspace
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

  const mappedOrganId = selectedPart ? mapPartToOrgan(selectedPart.name) : null

  return (
    <div className="whole-body-atlas-root">
      {/* 3D Canvas Mount */}
      <div ref={containerRef} className="atlas-canvas-host" />

      {/* Top HUD Controls */}
      <div className="atlas-hud-top">
        {/* Dissection Explode Slider */}
        <div className="atlas-hud-card explode-control-card">
          <div className="explode-label-row">
            <div className="flex items-center gap-2">
              <Sliders size={13} className="text-cyan-400" />
              <span className="hud-title">Anatomical Dissection (Explode)</span>
            </div>
            <span className="explode-val-badge">
              {Math.round(sceneState.explode * 100)}%
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={sceneState.explode}
            onChange={(e) =>
              setSceneState((prev) => ({
                ...prev,
                explode: parseFloat(e.target.value),
              }))
            }
            className="explode-range-slider"
          />
        </div>

        {/* View Presets & Orbit */}
        <div className="atlas-hud-card view-presets-card">
          <div className="hud-buttons-row">
            <button
              type="button"
              className={`hud-btn ${sceneState.viewAngle === 'front' ? 'active' : ''}`}
              onClick={() => setSceneState((prev) => ({ ...prev, viewAngle: 'front' }))}
              title="Anterior (Frontal)"
            >
              <Compass size={13} />
              <span>Front</span>
            </button>

            <button
              type="button"
              className={`hud-btn ${sceneState.viewAngle === 'back' ? 'active' : ''}`}
              onClick={() => setSceneState((prev) => ({ ...prev, viewAngle: 'back' }))}
              title="Posterior (Back)"
            >
              <Compass size={13} />
              <span>Back</span>
            </button>

            <button
              type="button"
              className={`hud-btn ${sceneState.viewAngle === 'side' ? 'active' : ''}`}
              onClick={() => setSceneState((prev) => ({ ...prev, viewAngle: 'side' }))}
              title="Lateral (Side)"
            >
              <Compass size={13} />
              <span>Side</span>
            </button>

            <button
              type="button"
              className={`hud-btn ${sceneState.viewAngle === 'three-quarter' ? 'active' : ''}`}
              onClick={() => setSceneState((prev) => ({ ...prev, viewAngle: 'three-quarter' }))}
              title="Three-Quarter Isometric"
            >
              <Compass size={13} />
              <span>Iso</span>
            </button>

            <button
              type="button"
              className={`hud-btn ${sceneState.autoRotate ? 'active' : ''}`}
              onClick={() =>
                setSceneState((prev) => ({ ...prev, autoRotate: !prev.autoRotate }))
              }
              title="Toggle Auto-Rotation"
            >
              <RotateCw size={13} className={sceneState.autoRotate ? 'spin-slow' : ''} />
              <span>Spin</span>
            </button>
          </div>
        </div>
      </div>

      {/* Left Drawer: 15 Anatomical System Toggles */}
      <aside className="atlas-systems-drawer">
        <div className="systems-drawer-header">
          <div className="flex items-center gap-2">
            <Layers size={14} className="text-cyan-400" />
            <span className="drawer-title">Physiological Systems</span>
          </div>
          <span className="drawer-count">
            {sceneState.visibleSystems.length}/{ATLAS_SYSTEMS.length} Active
          </span>
        </div>

        <div className="systems-list-scroll">
          {ATLAS_SYSTEMS.map((sys) => {
            const isVisible = sceneState.visibleSystems.includes(sys.id)
            return (
              <button
                key={sys.id}
                type="button"
                className={`system-item-toggle ${isVisible ? 'visible' : 'hidden-sys'}`}
                onClick={() => toggleSystem(sys.id)}
              >
                <div
                  className="system-color-tag"
                  style={{ backgroundColor: sys.color }}
                />
                <span className="system-name">{sys.name}</span>
                {isVisible ? (
                  <CheckSquare size={13} className="text-cyan-400 ml-auto" />
                ) : (
                  <Square size={13} className="text-slate-600 ml-auto" />
                )}
              </button>
            )
          })}
        </div>
      </aside>

      {/* Selected Anatomical Structure Inspector */}
      {selectedPart && (
        <div className="atlas-part-inspector-card">
          <div className="inspector-header">
            <div className="inspector-badge">
              <Info size={13} className="text-cyan-400" />
              <span>Anatomical Part Inspector</span>
            </div>
            <button
              type="button"
              className="inspector-close"
              onClick={() => setSelectedPart(null)}
            >
              <X size={14} />
            </button>
          </div>

          <div className="inspector-body">
            <h3 className="inspector-part-name">{selectedPart.name}</h3>
            <div className="inspector-meta-row">
              <span className="inspector-system-tag">
                System: <strong>{selectedPart.system}</strong>
              </span>
              <span className="inspector-concept-tag">
                Concept: <code>{selectedPart.conceptId}</code>
              </span>
            </div>

            <div className="inspector-actions-row">
              <button
                type="button"
                className={`inspector-btn ${sceneState.isolate ? 'active' : ''}`}
                onClick={() =>
                  setSceneState((prev) => ({ ...prev, isolate: !prev.isolate }))
                }
              >
                <Maximize2 size={13} />
                <span>{sceneState.isolate ? 'Show All Systems' : 'Isolate Structure'}</span>
              </button>

              {mappedOrganId && (
                <button
                  type="button"
                  className="inspector-btn primary-workspace-btn"
                  onClick={() => onNavigateToOrgan(mappedOrganId)}
                >
                  <ExternalLink size={13} />
                  <span>Inspect in {mappedOrganId.toUpperCase()} Workspace</span>
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Loading Overlay */}
      {isLoading && (
        <div className="atlas-overlay loading-overlay">
          <div className="loading-card">
            <Loader2 className="animate-spin text-cyan-400" size={32} />
            <span className="loading-text">Assembling BodyParts3D Atlas</span>
            <div className="progress-bar-track">
              <div
                className="progress-bar-fill"
                style={{ width: `${Math.max(loadingProgress, 8)}%` }}
              />
            </div>
            <span className="loading-subtext">
              {loadingProgress > 0
                ? `${loadingProgress}% chunks assembled (2,234 parts)`
                : 'Loading progressive chunks...'}
            </span>
          </div>
        </div>
      )}

      {/* Error Overlay */}
      {errorMessage && !isLoading && (
        <div className="atlas-overlay error-overlay">
          <div className="error-card">
            <AlertCircle size={28} className="text-rose-400" />
            <span className="error-title">Atlas Initialization Error</span>
            <p className="error-description">{errorMessage}</p>
          </div>
        </div>
      )}
    </div>
  )
}
