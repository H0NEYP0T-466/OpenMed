import React, { useCallback, useEffect, useState } from 'react'
import { RomanSection } from '../../../components/common/RomanSection'
import { SPECIMEN_PRESETS } from '../../organs/brain/specimenPresets'
import type { SpecimenPreset } from '../../organs/brain/specimenPresets'
import { DiagnosticThinkingHUD } from '../../../components/common/ThinkingOrb'
import { PromptableSegmentation } from '../PromptableSegmentation'
import { listLabModels, loadLabModel, unloadLabModel } from '../labApi'
import type { LabModelCard } from '../labApi'

/**
 * Instrument II — MedSAM2, inference only.
 *
 * The model is loaded when this tab mounts and released when it unmounts, which
 * is the lab's contract: nothing sits in memory for a tab nobody is looking at.
 * If the weights are absent the tab still renders and says why, rather than
 * showing a button that fails on click.
 */
const MODEL_ID = 'medsam2'

export const MedSAM2Tab: React.FC = () => {
  const [card, setCard] = useState<LabModelCard | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loadMs, setLoadMs] = useState<number | null>(null)
  const [loadingModel, setLoadingModel] = useState(false)

  useEffect(() => () => {
    if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl)
  }, [previewUrl])

  // Mount → load, unmount → unload.
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const { models } = await listLabModels()
        const found = models.find((model) => model.id === MODEL_ID) ?? null
        if (cancelled) return
        setCard(found)
        if (!found?.available) return
        if (!cancelled) setLoadingModel(true)
        const { load_ms } = await loadLabModel(MODEL_ID)
        if (!cancelled) {
          setLoadMs(load_ms)
          setLoadingModel(false)
        }
      } catch (cause) {
        if (!cancelled) {
          setLoadingModel(false)
          setError(cause instanceof Error ? cause.message : 'Lab unavailable.')
        }
      }
    })()
    return () => {
      cancelled = true
      void unloadLabModel(MODEL_ID)
    }
  }, [])

  const handleLoadPreset = useCallback(async (preset: SpecimenPreset) => {
    setError(null)
    setSelectedPreset(preset.plate)
    try {
      const response = await fetch(preset.path)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const blob = await response.blob()
      setFile(new File([blob], `${preset.name}.jpg`, { type: blob.type || 'image/jpeg' }))
      setPreviewUrl(preset.path)
    } catch {
      setError(`Unable to load specimen at ${preset.path}`)
      setFile(null)
      setPreviewUrl(null)
    }
  }, [])

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0]
    if (!selected) return
    if (!selected.type.startsWith('image/')) {
      setError('Selected file is not an image (JPEG, PNG).')
      return
    }
    setError(null)
    setSelectedPreset(null)
    setFile(selected)
    setPreviewUrl(URL.createObjectURL(selected))
  }, [])

  const unavailable = card && !card.available ? card.unavailable_reason : null

  return (
    <>
      {/* Same overlay as the brain and lungs workspaces: one loading language. */}
      <DiagnosticThinkingHUD isVisible={loadingModel} task="segmentation" />

      {unavailable && (
        <div className="lab-upload-error" role="status">
          MedSAM2 cannot run yet — {unavailable}
        </div>
      )}

      <RomanSection index={0} of={3} title="Specimen Rack">
        <div className="lab-rack">
          {SPECIMEN_PRESETS.map((preset) => (
            <button
              key={preset.plate}
              type="button"
              className={`lab-card ${selectedPreset === preset.plate ? 'is-selected' : ''}`}
              onClick={() => void handleLoadPreset(preset)}
            >
              <span className="lab-thumb">
                <img src={preset.path} alt={preset.name} loading="lazy" decoding="async" />
              </span>
              <span className="lab-card-body">
                <span className="lab-card-meta">
                  <span className="lab-plate">{preset.plate}</span>
                  <span className="lab-site">{preset.site}</span>
                </span>
                <span className="lab-name">{preset.name}</span>
                <span className="lab-sub">{preset.subtitle}</span>
              </span>
            </button>
          ))}
        </div>

        <div className="lab-upload">
          <label className="lab-upload-btn">
            <input type="file" accept="image/*" onChange={handleFileChange} />
            <span>Or upload a scan</span>
          </label>
          {error && <span className="lab-upload-error">{error}</span>}
        </div>
      </RomanSection>

      <RomanSection index={1} of={3} title="Box-Prompted Segmentation">
        <PromptableSegmentation
          modelId={MODEL_ID}
          file={file}
          previewUrl={previewUrl}
          disabledReason={unavailable}
          loadingModel={loadingModel}
        />
      </RomanSection>

      <RomanSection index={2} of={3} title="Provenance">
        <p className="lab-note">
          Engine: MedSAM2 · SAM2.1 Hiera-T at 512 px · 39 M parameters · inference only, no
          fine-tuning. Weights load when this tab opens and are released when you leave it
          {loadMs !== null ? ` (last load: ${loadMs.toFixed(0)} ms)` : ''}.
        </p>
        <p className="lab-note">
          MedSAM2 is the medical fine-tune of SAM2 and needs that fork&apos;s architecture —
          not the generic package. Served from
          backend/app/experimental/medsam2/ through the lab registry, so another promptable
          model is a package and one registration line, nothing more.
        </p>
      </RomanSection>
    </>
  )
}

export default MedSAM2Tab
