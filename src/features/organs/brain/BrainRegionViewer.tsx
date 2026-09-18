import React, { useState, useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { Compass, RotateCw } from 'lucide-react'
import type { BrainRegion3D } from './brainTypes'

interface BrainRegionViewerProps {
  readonly locations: readonly BrainRegion3D[]
}

export const BrainRegionViewer: React.FC<BrainRegionViewerProps> = ({ locations }) => {
  const [viewMode, setViewMode] = useState<'3d' | '2d'>('3d')
  const [selectedIdx, setSelectedIdx] = useState<number>(0)
  const containerRef = useRef<HTMLDivElement>(null)

  const activeRegion = locations[selectedIdx] ?? locations[0]

  useEffect(() => {
    if (viewMode !== '3d') return
    const container = containerRef.current
    if (!container) return

    const width = container.clientWidth || 420
    const height = 300

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 100)
    camera.position.set(0, 1.8, 3.8)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.appendChild(renderer.domElement)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.06
    controls.rotateSpeed = 0.75
    controls.minDistance = 2.0
    controls.maxDistance = 5.5
    controls.enablePan = false

    // Warm museum lighting
    scene.add(new THREE.AmbientLight(0xfff5ea, 0.7))
    const key = new THREE.DirectionalLight(0xffeedd, 1.3)
    key.position.set(3.5, 4, 3)
    scene.add(key)
    const rim = new THREE.DirectionalLight(0xc58696, 1.0)
    rim.position.set(-3, -2, -3)
    scene.add(rim)

    const brainGroup = new THREE.Group()

    // Bilateral Hemispheres
    const hemiGeo = new THREE.SphereGeometry(1.0, 32, 24)
    hemiGeo.scale(0.82, 0.95, 1.15)
    const cortexMat = new THREE.MeshPhysicalMaterial({
      color: 0x3d3732,
      transparent: true,
      opacity: 0.32,
      roughness: 0.4,
      metalness: 0.1,
      transmission: 0.55,
      ior: 1.35,
    })

    const leftHemi = new THREE.Mesh(hemiGeo, cortexMat)
    leftHemi.position.set(-0.45, 0.1, 0)
    brainGroup.add(leftHemi)

    const rightHemi = leftHemi.clone()
    rightHemi.position.set(0.45, 0.1, 0)
    brainGroup.add(rightHemi)

    // Contour wireframes
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x8a7f75,
      wireframe: true,
      transparent: true,
      opacity: 0.12,
    })
    const leftWire = new THREE.Mesh(hemiGeo, wireMat)
    leftWire.position.copy(leftHemi.position)
    brainGroup.add(leftWire)
    const rightWire = new THREE.Mesh(hemiGeo, wireMat)
    rightWire.position.copy(rightHemi.position)
    brainGroup.add(rightWire)

    // Cerebellum & Stem
    const cerebGeo = new THREE.SphereGeometry(0.5, 20, 16)
    cerebGeo.scale(1.1, 0.6, 0.8)
    const cerebMesh = new THREE.Mesh(cerebGeo, cortexMat)
    cerebMesh.position.set(0, -0.65, -0.55)
    brainGroup.add(cerebMesh)

    const stemGeo = new THREE.CylinderGeometry(0.18, 0.14, 0.85, 16)
    const stemMesh = new THREE.Mesh(stemGeo, cortexMat)
    stemMesh.position.set(0, -0.85, -0.15)
    brainGroup.add(stemMesh)

    // Axial Drafting Plane Ring
    const ringGeo = new THREE.RingGeometry(1.5, 1.52, 48)
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x6e6358,
      transparent: true,
      opacity: 0.25,
      side: THREE.DoubleSide,
    })
    const ring = new THREE.Mesh(ringGeo, ringMat)
    ring.rotation.x = Math.PI / 2
    ring.position.y = -0.3
    scene.add(ring)

    // Pulsing Tumor Locus Beacons
    const beaconsGroup = new THREE.Group()
    const pulseObjects: THREE.Mesh[] = []

    locations.forEach((loc, idx) => {
      const [x, y, z] = loc.coordinates_3d || [0, 0, 0]
      const posX = x * 1.05
      const posY = z * 0.95
      const posZ = y * 1.15

      const isPrimary = idx === selectedIdx
      const colorHex = isPrimary ? 0xed6f5c : parseInt(loc.color.replace('#', '0x'), 16) || 0xc58696

      const sphereGeo = new THREE.SphereGeometry(isPrimary ? 0.12 : 0.08, 20, 20)
      const sphereMat = new THREE.MeshStandardMaterial({
        color: colorHex,
        emissive: colorHex,
        emissiveIntensity: isPrimary ? 0.95 : 0.4,
        roughness: 0.2,
      })
      const beaconSphere = new THREE.Mesh(sphereGeo, sphereMat)
      beaconSphere.position.set(posX, posY, posZ)
      beaconsGroup.add(beaconSphere)

      const haloGeo = new THREE.RingGeometry(0.13, 0.17, 32)
      const haloMat = new THREE.MeshBasicMaterial({
        color: colorHex,
        transparent: true,
        opacity: isPrimary ? 0.85 : 0.35,
        side: THREE.DoubleSide,
      })
      const haloMesh = new THREE.Mesh(haloGeo, haloMat)
      haloMesh.position.set(posX, posY, posZ)
      beaconsGroup.add(haloMesh)
      if (isPrimary) pulseObjects.push(haloMesh)
    })

    brainGroup.add(beaconsGroup)
    scene.add(brainGroup)

    let animId: number
    const clock = new THREE.Clock()

    const animate = () => {
      animId = requestAnimationFrame(animate)
      const elapsed = clock.getElapsedTime()

      controls.update()
      brainGroup.rotation.y = Math.sin(elapsed * 0.25) * 0.18

      pulseObjects.forEach((mesh) => {
        const s = 1.0 + Math.sin(elapsed * 4.0) * 0.3
        mesh.scale.set(s, s, s)
        mesh.lookAt(camera.position)
      })

      renderer.render(scene, camera)
    }
    animate()

    const handleResize = () => {
      if (!container) return
      const w = container.clientWidth || 420
      camera.aspect = w / height
      camera.updateProjectionMatrix()
      renderer.setSize(w, height)
    }
    window.addEventListener('resize', handleResize)

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', handleResize)
      controls.dispose()
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [viewMode, locations, selectedIdx])

  const regions2D = [
    { id: 'frontal', path: 'M40,100 C40,40 100,20 150,20 C160,20 180,25 180,60 C180,100 130,120 100,140 C60,150 40,130 40,100 Z', name: 'Frontal', centerX: 100, centerY: 60 },
    { id: 'parietal', path: 'M150,20 C220,20 260,50 280,100 C280,120 260,140 220,130 C180,120 180,60 150,20 Z', name: 'Parietal', centerX: 220, centerY: 60 },
    { id: 'occipital', path: 'M280,100 C290,130 280,170 240,180 C220,180 200,160 220,130 C240,110 260,100 280,100 Z', name: 'Occipital', centerX: 260, centerY: 140 },
    { id: 'temporal', path: 'M100,140 C130,120 180,120 220,130 C200,160 170,180 120,180 C90,180 80,160 100,140 Z', name: 'Temporal', centerX: 150, centerY: 150 },
    { id: 'cerebellum', path: 'M200,170 C240,160 260,180 250,220 C240,240 200,240 180,210 C180,190 190,180 200,170 Z', name: 'Cerebellum', centerX: 220, centerY: 200 },
    { id: 'brainstem', path: 'M150,170 C170,170 180,190 180,210 C180,250 160,280 140,280 C130,280 130,250 140,210 C140,190 140,170 150,170 Z', name: 'Brainstem', centerX: 150, centerY: 230 },
  ]

  const get2DProb = (name: string): number => {
    const loc = locations.find((l) => l.name.toLowerCase().includes(name.toLowerCase()))
    return loc ? loc.probability : 0
  }

  const get2DColor = (prob: number): string => {
    if (prob < 0.1) return 'rgba(21, 20, 15, 0.04)'
    if (prob < 0.3) return 'rgba(197, 134, 150, 0.25)'
    if (prob < 0.6) return 'rgba(197, 134, 150, 0.6)'
    return 'rgba(237, 111, 92, 0.9)'
  }

  return (
    <div className="locus-viewport-box">
      {/* Plate Header Bar */}
      <div className="locus-header">
        <span className="locus-title">
          <Compass size={12} className="ico" />
          <span>Spatial Localization · MNI Atlas</span>
        </span>
        <div className="locus-switch">
          <button
            type="button"
            className={`locus-toggle-btn ${viewMode === '3d' ? 'active' : ''}`}
            onClick={() => setViewMode('3d')}
          >
            3D Locus
          </button>
          <button
            type="button"
            className={`locus-toggle-btn ${viewMode === '2d' ? 'active' : ''}`}
            onClick={() => setViewMode('2d')}
          >
            Lobar Map
          </button>
        </div>
      </div>

      {/* Viewport Surface */}
      {viewMode === '3d' ? (
        <div className="locus-canvas-wrap">
          <div ref={containerRef} className="locus-canvas" />
          <div className="locus-hud-hint">
            <RotateCw size={10} />
            <span>Interactive orbit</span>
          </div>
        </div>
      ) : (
        <div className="locus-svg-wrap">
          <svg viewBox="0 0 320 300" className="locus-svg">
            <g stroke="var(--line)" strokeWidth="1" fill="none">
              {regions2D.map((r) => {
                const p = get2DProb(r.id)
                const f = get2DColor(p)
                const isHi = p > 0.4
                return (
                  <g key={r.id}>
                    <path
                      d={r.path}
                      fill={f}
                      stroke={isHi ? 'var(--accent)' : 'var(--line)'}
                      strokeWidth={isHi ? 1.5 : 1}
                    />
                    {p > 0.05 && (
                      <text
                        x={r.centerX}
                        y={r.centerY}
                        fill={isHi ? 'var(--bone)' : 'var(--ink)'}
                        fontSize="10"
                        fontFamily="var(--font-mono)"
                        textAnchor="middle"
                        fontWeight={isHi ? '700' : '500'}
                      >
                        {Math.round(p * 100)}%
                      </text>
                    )}
                  </g>
                )
              })}
            </g>
          </svg>
        </div>
      )}

      {/* Coordinate & Region Readout */}
      {activeRegion && (
        <div className="locus-readout">
          <div className="locus-data-row">
            <span className="k">Target Locus</span>
            <span className="v">
              <strong>{activeRegion.display_name || activeRegion.name}</strong>
              <em className="lobe-tag">{activeRegion.lobe}</em>
            </span>
          </div>
          <div className="locus-data-row">
            <span className="k">MNI Centroid</span>
            <span className="v mono">
              X: {activeRegion.coordinates_3d ? (activeRegion.coordinates_3d[0] >= 0 ? `+${activeRegion.coordinates_3d[0].toFixed(2)}` : activeRegion.coordinates_3d[0].toFixed(2)) : '0.00'} · 
              Y: {activeRegion.coordinates_3d ? (activeRegion.coordinates_3d[1] >= 0 ? `+${activeRegion.coordinates_3d[1].toFixed(2)}` : activeRegion.coordinates_3d[1].toFixed(2)) : '0.00'} · 
              Z: {activeRegion.coordinates_3d ? (activeRegion.coordinates_3d[2] >= 0 ? `+${activeRegion.coordinates_3d[2].toFixed(2)}` : activeRegion.coordinates_3d[2].toFixed(2)) : '0.00'}
            </span>
          </div>
          <div className="locus-data-row">
            <span className="k">Confidence</span>
            <span className="v mono">{Math.round(activeRegion.probability * 100)}% Likelihood</span>
          </div>

          {locations.length > 1 && (
            <div className="locus-chips">
              {locations.map((loc, i) => (
                <button
                  key={loc.name}
                  type="button"
                  className={`locus-chip ${i === selectedIdx ? 'selected' : ''}`}
                  onClick={() => setSelectedIdx(i)}
                >
                  {loc.display_name || loc.name} ({Math.round(loc.probability * 100)}%)
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
