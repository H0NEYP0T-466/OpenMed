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
import { ORGANS_REGISTRY } from '../../types/organ'
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
  Maximize2,
  ExternalLink,
  X
} from 'lucide-react'

interface WholeBodyAtlasViewerProps {
  readonly onNavigateToOrgan: (organId: OrganId) => void
  readonly plateNo: string
}

export const WholeBodyAtlasViewer: React.FC<WholeBodyAtlasViewerProps> = ({
  onNavigateToOrgan,
  plateNo,
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
    renderer.setClearColor('#16130d')
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

    // Lighting — warm paper key over an ink ground
    scene.add(new THREE.AmbientLight(0xfff6e3, 0.55))
    scene.add(new THREE.HemisphereLight(0xfff8ee, 0x241f16, 0.85))

    const keyLight = new THREE.DirectionalLight(0xfffaf4, 2.5)
    keyLight.position.set(-2, 4, 3)
    scene.add(keyLight)

    const rimLight = new THREE.DirectionalLight(0xe9f0ff, 1.8)
    rimLight.position.set(2, 2, -3)
    scene.add(rimLight)

    // Floor Platform
    const platform = new THREE.Mesh(
      new THREE.CylinderGeometry(0.7, 0.72, 0.03, 64),
      new THREE.MeshStandardMaterial({ color: 0x1c1812, roughness: 0.8 })
    )
    platform.position.y = -0.016
    scene.add(platform)

    const grid = new THREE.GridHelper(8, 16, 0x2a2620, 0x1d1a13)
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
          '#include <color_fragment>\ndiffuseColor.rgb = mix(diffuseColor.rgb, vec3(0.929, 0.435, 0.361), partSelected * 0.85);'
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
    <>
      {/* WebGL canvas mount */}
      <div ref={containerRef} className="plate-canvas" />

      {/* Corner brackets */}
      <span className="brk tl" aria-hidden="true" />
      <span className="brk tr" aria-hidden="true" />
      <span className="brk bl" aria-hidden="true" />
      <span className="brk br" aria-hidden="true" />

      {/* Plate caption */}
      <div className="plate-caption">
        <span className="p-num">
          Plate <i>Nº {plateNo}</i> — Corpus Integrum
        </span>
        <span className="p-hair" aria-hidden="true" />
        <span className="p-term">Homo Sapiens · 2,234 parts</span>
      </div>

      {/* Coordinate / mesh stamp */}
      <div className="plate-coord">
        FIG. {plateNo} / OM-26<span className="sep">·</span>15 physiological systems
        <span className="sep">·</span>tap a structure to inspect
      </div>

      {/* Dissection slider — top left */}
      <div className="atlas-card explode-card">
        <div className="explode-head">
          <span className="card-label">
            <Sliders size={12} style={{ verticalAlign: -2, marginRight: 7 }} />
            Dissection (Explode)
          </span>
          <span className="val-badge">{Math.round(sceneState.explode * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={sceneState.explode}
          onChange={(e) =>
            setSceneState((prev) => ({ ...prev, explode: parseFloat(e.target.value) }))
          }
          className="range"
          aria-label="Anatomical dissection amount"
        />
        <div className="range-foot">
          <span>In situ</span>
          <span>Plate layout</span>
        </div>
      </div>

      {/* View presets — top right */}
      <div style={{ position: 'absolute', top: 48, right: 20, zIndex: 35 }}>
        <div className="hud-chip" role="group" aria-label="Atlas view presets">
          <button
            type="button"
            className={`hud-btn ${sceneState.viewAngle === 'front' ? 'active' : ''}`}
            onClick={() => setSceneState((prev) => ({ ...prev, viewAngle: 'front' }))}
            title="Anterior (frontal)"
          >
            <Compass size={13} className="ico" />
            <span>Front</span>
          </button>
          <button
            type="button"
            className={`hud-btn ${sceneState.viewAngle === 'back' ? 'active' : ''}`}
            onClick={() => setSceneState((prev) => ({ ...prev, viewAngle: 'back' }))}
            title="Posterior (back)"
          >
            <Compass size={13} className="ico" />
            <span>Back</span>
          </button>
          <button
            type="button"
            className={`hud-btn ${sceneState.viewAngle === 'side' ? 'active' : ''}`}
            onClick={() => setSceneState((prev) => ({ ...prev, viewAngle: 'side' }))}
            title="Lateral (side)"
          >
            <Compass size={13} className="ico" />
            <span>Side</span>
          </button>
          <button
            type="button"
            className={`hud-btn ${sceneState.viewAngle === 'three-quarter' ? 'active' : ''}`}
            onClick={() => setSceneState((prev) => ({ ...prev, viewAngle: 'three-quarter' }))}
            title="Three-quarter isometric"
          >
            <Compass size={13} className="ico" />
            <span>Iso</span>
          </button>
          <button
            type="button"
            className={`hud-btn ${sceneState.autoRotate ? 'active' : ''}`}
            onClick={() => setSceneState((prev) => ({ ...prev, autoRotate: !prev.autoRotate }))}
            title="Toggle auto-rotation"
          >
            <RotateCw size={13} className={`ico ${sceneState.autoRotate ? 'spin-slow' : ''}`} />
            <span>Spin</span>
          </button>
        </div>
      </div>

      {/* Systems table of contents — left drawer */}
      <aside className="sys-drawer">
        <div className="sys-drawer-head">
          <span className="card-label">
            <Layers size={12} style={{ verticalAlign: -2, marginRight: 7 }} />
            Physiological Systems
          </span>
          <span className="count">{sceneState.visibleSystems.length}/{ATLAS_SYSTEMS.length}</span>
        </div>

        <div className="sys-scroll">
          {ATLAS_SYSTEMS.map((sys) => {
            const isVisible = sceneState.visibleSystems.includes(sys.id)
            return (
              <button
                key={sys.id}
                type="button"
                className={`sys-item ${isVisible ? '' : 'off'}`}
                onClick={() => toggleSystem(sys.id)}
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
      </aside>

      {/* Selected structure inspector */}
      {selectedPart && (
        <div className="part-inspector" role="dialog" aria-label="Anatomical part inspector">
          <div className="co-head">
            <span className="co-eyebrow">Part Inspector</span>
            <button
              type="button"
              className="icon-btn"
              onClick={() => setSelectedPart(null)}
              aria-label="Close inspector"
            >
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
                <i
                  style={{
                    backgroundColor:
                      ATLAS_SYSTEMS.find((s) => s.id === selectedPart.system)?.color ?? '#aebbb8',
                  }}
                />
                {selectedPart.system}
              </span>
              <span className="pi-latin">Concept {selectedPart.conceptId}</span>
            </div>
          </div>

          <div className="pi-actions">
            <button
              type="button"
              className={`btn-sm ${sceneState.isolate ? 'active' : ''}`}
              onClick={() => setSceneState((prev) => ({ ...prev, isolate: !prev.isolate }))}
            >
              <Maximize2 size={13} />
              <span>{sceneState.isolate ? 'Show All Systems' : 'Isolate Structure'}</span>
            </button>

            {mappedOrganId && (
              <button
                type="button"
                className="btn-sm hot"
                onClick={() => onNavigateToOrgan(mappedOrganId)}
              >
                <ExternalLink size={13} />
                <span>Inspect in {ORGANS_REGISTRY[mappedOrganId].name} workspace</span>
              </button>
            )}
          </div>
        </div>
      )}

      {/* Loading note */}
      {isLoading && (
        <div className="plate-overlay">
          <div className="plate-note">
            <span className="note-eyebrow">Off-press — assembling the macro plate</span>
            <span className="note-title">
              Imposing BodyParts3D<span className="dot">.</span>
            </span>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${Math.max(loadingProgress, 8)}%` }}
              />
            </div>
            <span className="progress-sub">
              {loadingProgress > 0
                ? `${loadingProgress}% of chunks bound (2,234 parts)`
                : 'Fetching progressive chunks…'}
            </span>
          </div>
        </div>
      )}

      {/* Error note */}
      {errorMessage && !isLoading && (
        <div className="plate-overlay">
          <div className="plate-note">
            <span className="note-eyebrow">Press halt</span>
            <span className="note-title">
              Atlas initialization error<span className="dot">.</span>
            </span>
            <p className="note-body">{errorMessage}</p>
          </div>
        </div>
      )}
    </>
  )
}
