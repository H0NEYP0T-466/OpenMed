import * as THREE from 'three'
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { RenderMode } from '../types/organ'

export const FIT_SIZE = 3.8

/**
 * Creates an equirectangular warm-to-cool ambient lighting gradient probe.
 * Bakes soft, realistic anatomical environmental reflections onto PBR materials.
 */
export function buildEnvironmentMap(renderer: THREE.WebGLRenderer): THREE.Texture {
  const width = 16
  const height = 32
  const data = new Uint8Array(width * height * 4)
  const top = new THREE.Color(0xfff3e4)
  const bottom = new THREE.Color(0x33252d)
  const mixed = new THREE.Color()

  for (let y = 0; y < height; y += 1) {
    mixed.copy(bottom).lerp(top, Math.pow(1 - y / (height - 1), 0.7))
    for (let x = 0; x < width; x += 1) {
      const i = (y * width + x) * 4
      data[i] = mixed.r * 255
      data[i + 1] = mixed.g * 255
      data[i + 2] = mixed.b * 255
      data[i + 3] = 255
    }
  }

  const source = new THREE.DataTexture(data, width, height)
  source.mapping = THREE.EquirectangularReflectionMapping
  source.colorSpace = THREE.SRGBColorSpace
  source.needsUpdate = true

  const pmrem = new THREE.PMREMGenerator(renderer)
  const environment = pmrem.fromEquirectangular(source).texture
  pmrem.dispose()
  source.dispose()
  return environment
}

/**
 * Normalizes model into standard coordinate space so hotspots line up accurately.
 */
export function normalizeModelToFitSize(
  model: THREE.Object3D,
  fitSize = FIT_SIZE
): { scale: number; center: THREE.Vector3 } {
  const box = new THREE.Box3().setFromObject(model)
  const size = box.getSize(new THREE.Vector3())
  const center = box.getCenter(new THREE.Vector3())
  const maxDim = Math.max(size.x, size.y, size.z, 0.001)
  const scale = fitSize / maxDim

  model.scale.setScalar(scale)
  model.position.copy(center.multiplyScalar(-scale))

  return { scale, center }
}

/**
 * Recenter camera to look at the normalized organ.
 */
export function fitCameraToNormalizedPivot(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  cameraDistance = 7.5
): void {
  camera.position.set(0, 0.4, cameraDistance)
  camera.lookAt(0, 0, 0)

  controls.target.set(0, 0, 0)
  controls.minDistance = 2.5
  controls.maxDistance = 18.0
  controls.update()
}

/**
 * Creates a glowing billboard sprite marker for anatomical landmarks.
 */
export function createHotspotSprite(colorHex: string, isSelected = false): THREE.Sprite {
  const size = 128
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')!
  const c = size / 2
  const tau = Math.PI * 2

  const color = new THREE.Color(colorHex)
  const r = Math.round(color.r * 255)
  const g = Math.round(color.g * 255)
  const b = Math.round(color.b * 255)

  // Outer radial glow
  const halo = ctx.createRadialGradient(c, c, size * 0.22, c, c, size * 0.5)
  halo.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${isSelected ? 0.7 : 0.4})`)
  halo.addColorStop(0.6, `rgba(${r}, ${g}, ${b}, 0.15)`)
  halo.addColorStop(1, 'rgba(0, 0, 0, 0)')
  ctx.fillStyle = halo
  ctx.beginPath()
  ctx.arc(c, c, c, 0, tau)
  ctx.fill()

  // High-contrast dark backing ring
  ctx.beginPath()
  ctx.arc(c, c, size * 0.3, 0, tau)
  ctx.fillStyle = 'rgba(11, 15, 25, 0.8)'
  ctx.fill()

  // Crisp white inner ring
  ctx.beginPath()
  ctx.arc(c, c, size * 0.26, 0, tau)
  ctx.fillStyle = 'rgba(255, 255, 255, 0.98)'
  ctx.fill()

  // Core saturated color dot
  ctx.beginPath()
  ctx.arc(c, c, size * 0.17, 0, tau)
  ctx.fillStyle = `rgb(${r}, ${g}, ${b})`
  ctx.fill()

  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace

  const material = new THREE.SpriteMaterial({
    map: texture,
    depthTest: false,
    depthWrite: false,
    transparent: true,
  })

  const sprite = new THREE.Sprite(material)
  const scale = isSelected ? 0.32 : 0.22
  sprite.scale.set(scale, scale, 1)

  return sprite
}

/**
 * Configure PBR materials with high-anisotropy textures and realistic organic response.
 */
export function enhancePBRMaterials(object: THREE.Object3D, maxAnisotropy = 8): void {
  object.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      const mesh = child as THREE.Mesh
      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]

      materials.forEach((mat) => {
        if (mat instanceof THREE.MeshStandardMaterial) {
          mat.roughness = THREE.MathUtils.clamp(mat.roughness ?? 0.5, 0.35, 0.65)
          mat.envMapIntensity = 0.85

          const maps = [
            mat.map,
            mat.normalMap,
            mat.roughnessMap,
            mat.metalnessMap,
            mat.aoMap,
            mat.emissiveMap,
          ]

          maps.forEach((texture) => {
            if (texture) {
              texture.anisotropy = maxAnisotropy
              texture.generateMipmaps = true
              texture.minFilter = THREE.LinearMipmapLinearFilter
              texture.magFilter = THREE.LinearFilter
              texture.needsUpdate = true
            }
          })

          mat.needsUpdate = true
        }
      })
    }
  })
}

/**
 * Apply clinical shading styles according to the chosen RenderMode.
 */
export function applyRenderMode(
  object: THREE.Object3D,
  mode: RenderMode,
  accentColor: string,
  showWireframeOverlay = false
): void {
  const accentThreeColor = new THREE.Color(accentColor)
  const medicalCyan = new THREE.Color('#38bdf8')
  const lesionAmber = new THREE.Color('#f43f5e')

  object.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      const mesh = child as THREE.Mesh

      if (!mesh.userData['originalMaterial']) {
        mesh.userData['originalMaterial'] = mesh.material
      }

      const original = mesh.userData['originalMaterial'] as THREE.Material | THREE.Material[]

      switch (mode) {
        case 'pbr': {
          if (Array.isArray(original)) {
            mesh.material = original.map((m) => {
              const clone = m.clone()
              if ('wireframe' in clone) (clone as THREE.MeshStandardMaterial).wireframe = showWireframeOverlay
              return clone
            })
          } else {
            const clone = original.clone()
            if ('wireframe' in clone) {
              (clone as THREE.MeshStandardMaterial).wireframe = showWireframeOverlay
            }
            mesh.material = clone
          }
          break
        }

        case 'xray': {
          mesh.material = new THREE.MeshPhysicalMaterial({
            color: medicalCyan,
            transparent: true,
            opacity: 0.38,
            roughness: 0.2,
            metalness: 0.1,
            transmission: 0.6,
            ior: 1.2,
            depthWrite: false,
            wireframe: showWireframeOverlay,
          })
          break
        }

        case 'wireframe': {
          mesh.material = new THREE.MeshBasicMaterial({
            color: accentThreeColor,
            wireframe: true,
            transparent: true,
            opacity: 0.85,
          })
          break
        }

        case 'segmentation': {
          mesh.material = new THREE.MeshStandardMaterial({
            color: lesionAmber,
            emissive: lesionAmber,
            emissiveIntensity: 0.55,
            roughness: 0.35,
            metalness: 0.1,
            wireframe: showWireframeOverlay,
          })
          break
        }
      }
    }
  })
}

/**
 * Clean up geometries and materials from GPU memory to prevent WebGL leaks.
 */
export function disposeObjectTree(object: THREE.Object3D): void {
  object.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      const mesh = child as THREE.Mesh
      mesh.geometry?.dispose()

      if (Array.isArray(mesh.material)) {
        mesh.material.forEach((mat) => mat.dispose())
      } else if (mesh.material) {
        mesh.material.dispose()
      }

      const orig = mesh.userData['originalMaterial']
      if (orig) {
        if (Array.isArray(orig)) {
          orig.forEach((mat: THREE.Material) => mat.dispose())
        } else if ((orig as THREE.Material).dispose) {
          (orig as THREE.Material).dispose()
        }
      }
    } else if ((child as THREE.Sprite).isSprite) {
      const sprite = child as THREE.Sprite
      sprite.material.map?.dispose()
      sprite.material.dispose()
    }
  })
}
