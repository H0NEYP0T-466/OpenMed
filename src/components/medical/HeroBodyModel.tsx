import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { disposeObjectTree } from '../../utils/threeHelpers'

/**
 * The front-page body: one pre-baked low-poly GLB, turning on its own.
 *
 * Deliberately not the atlas viewer. The atlas streams 2,234 parts in 15 chunks
 * (33.6 MB, ~22 s to bind) because it has to support picking, isolation and
 * dissection. The front page needs none of that, so it loads a single 3.5 MB
 * asset instead — the same anatomy collapsed to 7% of the triangles, with the
 * reproductive system left out of this public-facing fold.
 *
 * No controls, no hotspots, no plate chrome, and no background box: the canvas
 * is transparent and the model sits directly on the page.
 */
const HERO_MODEL_URL = '/models/atlas/hero-body.glb'

/** Fraction of the container the model's longest axis should occupy. */
const FIT = 0.86

export const HeroBodyModel: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [status, setStatus] = useState<'loading' | 'ready' | 'error'>('loading')

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    let disposed = false
    let frame = 0

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setClearColor(0x000000, 0) // transparent — no plate box behind it
    container.appendChild(renderer.domElement)

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 100)
    camera.position.set(0, 0.12, 3.6)
    camera.lookAt(0, 0, 0)

    // Warm key over a cool rim, matching the plate lighting.
    scene.add(new THREE.AmbientLight(0xfff6e3, 1.05))
    const key = new THREE.DirectionalLight(0xfff2e0, 1.5)
    key.position.set(2.4, 3.4, 3.2)
    scene.add(key)
    const rim = new THREE.DirectionalLight(0xbcd8ff, 0.85)
    rim.position.set(-2.8, 1.2, -2.6)
    scene.add(rim)
    const fill = new THREE.DirectionalLight(0xffffff, 0.35)
    fill.position.set(0, -1.6, 1.4)
    scene.add(fill)

    const pivot = new THREE.Group()
    scene.add(pivot)

    const resize = () => {
      const w = container.clientWidth
      const h = container.clientHeight
      if (w === 0 || h === 0) return
      renderer.setSize(w, h, false)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(container)

    const loader = new GLTFLoader()
    loader.load(
      HERO_MODEL_URL,
      (gltf) => {
        if (disposed) {
          disposeObjectTree(gltf.scene)
          return
        }

        const model = gltf.scene
        // Centre on the origin and scale the longest axis to FIT.
        const box = new THREE.Box3().setFromObject(model)
        const size = box.getSize(new THREE.Vector3())
        const centre = box.getCenter(new THREE.Vector3())
        model.position.sub(centre)
        model.scale.setScalar(FIT / Math.max(size.x, size.y, size.z))
        model.traverse((child) => {
          if (child instanceof THREE.Mesh) child.frustumCulled = false
        })
        pivot.add(model)
        setStatus('ready')

        const tick = () => {
          if (disposed) return
          pivot.rotation.y += 0.0035
          renderer.render(scene, camera)
          frame = requestAnimationFrame(tick)
        }
        tick()
      },
      undefined,
      () => {
        if (!disposed) setStatus('error')
      },
    )

    return () => {
      disposed = true
      cancelAnimationFrame(frame)
      observer.disconnect()
      disposeObjectTree(scene)
      renderer.dispose()
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  return (
    <div className="hero-model" ref={containerRef} aria-hidden="true">
      {status === 'loading' && <span className="hero-model-note mono">Loading the plate…</span>}
      {status === 'error' && (
        <span className="hero-model-note mono">The plate could not be loaded.</span>
      )}
    </div>
  )
}
