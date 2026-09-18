import React, { useEffect, useRef } from 'react'
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
import { ATLAS_SYSTEMS } from '../../types/atlas'
import {
  createExplosionLayout,
  PointerTap,
  decodeModelResponse,
} from '../../utils/atlasHelpers'
import './plate.css'

interface WholeBodyAtlasViewerProps {
  readonly plateNo: string
  readonly state: AtlasSceneState
  readonly onPickPart: (part: Part) => void
}

interface PickTarget {
  index: number
  x: number
  y: number
  left: number
  right: number
  top: number
  bottom: number
}

const SYSTEM_NAMES = new Map<string, string>(ATLAS_SYSTEMS.map((s) => [s.id, s.name]))

export const WholeBodyAtlasViewer: React.FC<WholeBodyAtlasViewerProps> = ({
  plateNo,
  state,
  onPickPart,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [atlasData, setAtlasData] = React.useState<AtlasData | null>(null)
  const [loadingProgress, setLoadingProgress] = React.useState<number>(0)
  const [isLoading, setIsLoading] = React.useState<boolean>(true)
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null)

  const stateRef = useRef<AtlasSceneState>(state)
  useEffect(() => {
    stateRef.current = state
  }, [state])

  const pickRef = useRef(onPickPart)
  useEffect(() => {
    pickRef.current = onPickPart
  }, [onPickPart])

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
    let lastState: AtlasSceneState | null = null
    let packingWidth = 1
    let packingHeight = 1

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
    controls.maxDistance = 60
    controls.addEventListener('change', () => {
      dirty = true
    })

    // Lighting - warm paper key over an ink ground
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

    // View Angle Camera Fitting (adaptive for the exploded inventory board)
    const fit = (view: AtlasViewAngle, extent = 0) => {
      const cw = container.clientWidth || 1
      const ch = container.clientHeight || 1
      const availableAspect = Math.max(0.35, cw / ch)
      const atlasDistance =
        (Math.max(packingHeight, packingWidth / availableAspect) /
          (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)))) *
        1.12
      const normalDistance = 3.6
      const distance = THREE.MathUtils.lerp(normalDistance, Math.max(1.2, atlasDistance), extent)

      const effective: AtlasViewAngle = extent > 0.8 ? 'front' : view
      const direction =
        effective === 'front'
          ? new THREE.Vector3(0, 0.02, 1)
          : effective === 'back'
          ? new THREE.Vector3(0, 0.02, -1)
          : effective === 'side'
          ? new THREE.Vector3(1, 0.02, 0)
          : new THREE.Vector3(0.35, 0.06, 1).normalize()

      controls.target.set(extent > 0.1 ? -packingWidth * 0.1 : 0, extent > 0.1 ? 0.72 : 0.85, 0)
      controls.maxDistance = Math.max(60, distance * 2)
      camera.position.copy(controls.target).addScaledVector(direction, distance)
      controls.update()
      dirty = true
    }

    const resize = () => {
      camera.aspect = container.clientWidth / Math.max(1, container.clientHeight)
      camera.updateProjectionMatrix()
      renderer.setSize(container.clientWidth, container.clientHeight)
      layoutKey = ''
      fit(stateRef.current.viewAngle, amount)
      dirty = true
    }
    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(container)

    // ------------------------------------------------------------
    // Anatomical naming - projected 2D pick targets + hover label
    // (ported from ashemag/human-atlas scene.tsx)
    // ------------------------------------------------------------
    let targets: PickTarget[] = []
    const projected = new THREE.Vector3()

    const findTarget = (x: number, y: number, radius: number): number => {
      let best = -1
      let score = Infinity
      for (const t of targets) {
        const dx = Math.max(t.left - x, 0, x - t.right)
        const dy = Math.max(t.top - y, 0, y - t.bottom)
        const distance = Math.hypot(dx, dy)
        if (distance > radius) continue
        const candidate = distance + Math.hypot(t.x - x, t.y - y) * 0.025
        if (candidate < score) {
          score = candidate
          best = t.index
        }
      }
      return best
    }

    const rebuildTargets = () => {
      targets = []
      if (amount <= 0.45) return
      const cw = container.clientWidth
      const ch = container.clientHeight
      const hasSolid = atlasData.parts.some(
        (p, i) => p.system !== 'integumentary' && data[i * 4 + 3] > 0.5
      )
      atlasData.parts.forEach((p, i) => {
        if (data[i * 4 + 3] < 0.5 || (hasSolid && p.system === 'integumentary')) return
        let left = Infinity
        let right = -Infinity
        let top = Infinity
        let bottom = -Infinity
        for (let corner = 0; corner < 8; corner++) {
          projected
            .set(
              p.bounds[(corner & 1) ? 1 : 0][0] + data[i * 4],
              p.bounds[(corner & 2) ? 1 : 0][1] + data[i * 4 + 1],
              p.bounds[(corner & 4) ? 1 : 0][2] + data[i * 4 + 2]
            )
            .project(camera)
          const x = ((projected.x + 1) * cw) / 2
          const y = ((1 - projected.y) * ch) / 2
          left = Math.min(left, x)
          right = Math.max(right, x)
          top = Math.min(top, y)
          bottom = Math.max(bottom, y)
        }
        projected
          .set(
            centers[i].x + data[i * 4],
            centers[i].y + data[i * 4 + 1],
            centers[i].z + data[i * 4 + 2]
          )
          .project(camera)
        if (projected.z < -1 || projected.z > 1) return
        targets.push({
          index: i,
          x: ((projected.x + 1) * cw) / 2,
          y: ((1 - projected.y) * ch) / 2,
          left,
          right,
          top,
          bottom,
        })
      })
    }

    // Bone hover chip (imperative - avoids 60 Hz React state churn)
    const tip = document.createElement('div')
    tip.className = 'part-tip'
    tip.hidden = true
    tip.setAttribute('role', 'tooltip')
    const tipName = document.createElement('div')
    tipName.className = 'pt-name'
    const tipSys = document.createElement('div')
    tipSys.className = 'pt-sys'
    tip.append(tipName, tipSys)
    container.appendChild(tip)

    const hideTip = () => {
      tip.hidden = true
    }

    // Raycasting & Tap Detection for 2,234 Anatomical Parts
    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    const tap = new PointerTap()
    const worldBox = new THREE.Box3()
    const hitPoint = new THREE.Vector3()

    const onPointerDown = (e: PointerEvent) => {
      hideTip()
      tap.down(e.pointerId, e.clientX, e.clientY, e.pointerType === 'touch' ? 12 : 5)
    }

    const onPointerMove = (e: PointerEvent) => {
      tap.move(e.pointerId, e.clientX, e.clientY)

      // Hover naming while the plate is exploded (mouse only, no drag)
      if (e.buttons || amount < 0.5 || e.pointerType === 'touch' || !ready) {
        hideTip()
        return
      }
      const rect = renderer.domElement.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      const index = findTarget(x, y, 12)
      if (index < 0) {
        hideTip()
        renderer.domElement.style.cursor = 'grab'
        return
      }
      renderer.domElement.style.cursor = 'pointer'
      const part = atlasData.parts[index]
      tipName.textContent = part.name
      tipSys.textContent = SYSTEM_NAMES.get(part.system) ?? part.system
      tip.hidden = false
      tip.style.left = `${Math.max(8, Math.min(x + 14, rect.width - 260))}px`
      tip.style.top = `${Math.max(8, Math.min(y + 18, rect.height - 64))}px`
    }

    const onPointerLeave = () => {
      hideTip()
    }

    const onPointerUp = (e: PointerEvent) => {
      const validTap = tap.up(e.pointerId, e.clientX, e.clientY)
      if (!validTap || !ready) return
      hideTip()

      const rect = renderer.domElement.getBoundingClientRect()
      pointer.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1
      )
      raycaster.setFromCamera(pointer, camera)

      const hasSolid = atlasData.parts.some(
        (p, i) => p.system !== 'integumentary' && data[i * 4 + 3] > 0.5
      )

      let nearest = Infinity
      let found = -1

      pickers.forEach((mesh, i) => {
        if (!mesh || data[i * 4 + 3] < 0.5) return
        if (hasSolid && atlasData.parts[i].system === 'integumentary') return
        worldBox.copy(bounds[i]).translate(mesh.position)
        if (!raycaster.ray.intersectBox(worldBox, hitPoint)) return
        const hits = raycaster.intersectObject(mesh, false)
        if (hits[0] && hits[0].distance < nearest) {
          nearest = hits[0].distance
          found = i
        }
      })

      // When exploded, parts are flat cells on a board - use the projected targets
      if (found < 0 && amount > 0.45) {
        found = findTarget(
          e.clientX - rect.left,
          e.clientY - rect.top,
          e.pointerType === 'touch' ? 24 : 16
        )
      }

      if (found >= 0) {
        pickRef.current(atlasData.parts[found])
      }
    }

    const dom = renderer.domElement
    dom.addEventListener('pointerdown', onPointerDown)
    dom.addEventListener('pointermove', onPointerMove)
    dom.addEventListener('pointerup', onPointerUp)
    dom.addEventListener('pointerleave', onPointerLeave)

    // Render Animation Loop
    const clock = new THREE.Clock()
    const animate = () => {
      if (disposed) return
      frame = requestAnimationFrame(animate)

      const dt = Math.min(clock.getDelta(), 0.05)
      const s = stateRef.current
      if (s !== lastState) {
        lastState = s
        dirty = true
      }

      // Turntable lives on the camera (keeps world-space picking accurate)
      controls.autoRotate = s.autoRotate && !s.isolate && amount < 0.4
      controls.autoRotateSpeed = 0.7
      controls.enableRotate = amount < 0.8
      controls.mouseButtons.LEFT = amount < 0.8 ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN
      controls.update()

      // Smooth interpolation for explosion / dissection slider
      const moving = Math.abs(amount - s.explode) > 0.0001
      if (moving) {
        amount = THREE.MathUtils.damp(amount, s.explode, 8, dt)
        dirty = true
      }

      const visible = new Set(s.visibleSystems)
      const selected = s.selectedPartId ? new Set([s.selectedPartId]) : new Set<string>()

      const nextLayoutKey =
        (s.isolate ? 'iso:' : '') +
        s.visibleSystems.join(',') +
        ':' +
        (s.isolate ? s.selectedPartId ?? '' : '') +
        ':' +
        camera.aspect.toFixed(3)

      if (nextLayoutKey !== layoutKey) {
        const visibleParts = atlasData.parts.filter((p) =>
          s.isolate ? selected.has(p.id) : visible.has(p.system) || selected.has(p.id)
        )
        const layout = createExplosionLayout(visibleParts, camera.aspect)
        packingWidth = layout.width
        packingHeight = layout.height

        atlasData.parts.forEach((p, i) => {
          const cell = layout.cells.get(p.id)
          offsets[i] = cell ? new THREE.Vector3(cell.x, cell.y + 0.72, 0) : centers[i].clone()
        })
        layoutKey = nextLayoutKey
        if (amount > 0.05 && !s.isolate) {
          fit(s.viewAngle, Math.max(0, (amount - 0.3) / 0.7))
        }
      }

      if (moving || dirty) {
        atlasData.parts.forEach((p, i) => {
          const c = centers[i]
          const destination = offsets[i] ?? centers[i]
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
      }

      // Inventory board is read head-on; orbit view returns on re-assembly
      const effectiveView: AtlasViewAngle = amount > 0.5 && !s.isolate ? 'front' : s.viewAngle
      if (moving || effectiveView !== lastView) {
        fit(effectiveView, Math.max(0, (amount - 0.3) / 0.7))
        lastView = effectiveView
      }

      platform.visible = grid.visible = amount < 0.5

      if (dirty) {
        renderer.render(scene, camera)
        rebuildTargets()
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
      dom.removeEventListener('pointerleave', onPointerLeave)
      tip.remove()

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
          Plate <i>Nº {plateNo}</i> - Corpus Integrum
        </span>
        <span className="p-hair" aria-hidden="true" />
        <span className="p-term">Homo Sapiens · 2,234 parts</span>
      </div>

      {/* Coordinate / mesh stamp */}
      <div className="plate-coord">
        <span className="sep">·</span>15 physiological systems
        <span className="sep">·</span>hover past 50% dissection to name structures - click to inspect
      </div>

      {/* Loading note */}
      {isLoading && (
        <div className="plate-overlay">
          <div className="plate-note">
            <span className="note-eyebrow">Off-press - assembling the macro plate</span>
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
