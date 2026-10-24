import React, { useCallback, useEffect, useState } from 'react'
import { RomanSection } from '../../../components/common/RomanSection'
import { AssistiveSegmentation } from '../../organs/brain/AssistiveSegmentation'
import { SPECIMEN_PRESETS } from '../../organs/brain/specimenPresets'
import type { SpecimenPreset } from '../../organs/brain/specimenPresets'

/**
 * Instrument I — assistive segmentation.
 *
 * LiteMedSAM is the engine here, not a settled part of the product: the
 * automated heatmap-prompt path was retired from the classifier run and a
 * replacement model is expected. Everything model-specific lives in
 * `backend/app/organs/brain/segmentation/experimental_lab/`.
 */
export const LiteMedSAMTab: React.FC = () => {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  // Same rule as the other workspaces: blob URLs are revoked when replaced or
  // on unmount, so uploads never pin their files in memory.
  useEffect(() => () => {
    if (previewUrl?.startsWith('blob:')) URL.revokeObjectURL(previewUrl)
  }, [previewUrl])
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleLoadPreset = useCallback(async (preset: SpecimenPreset) => {
    setError(null)
    setSelectedPreset(preset.plate)
    try {
      const resp = await fetch(preset.path)
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
      const blob = await resp.blob()
      setFile(new File([blob], `${preset.name}.jpg`, { type: 'image/jpeg' }))
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

  return (
    <>
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

      <RomanSection index={1} of={3} title="Assistive Segmentation">
        {file && previewUrl ? (
          <AssistiveSegmentation file={file} previewUrl={previewUrl} />
        ) : (
          <div className="lab-idle">Pick a specimen above, or upload a scan, to begin.</div>
        )}
      </RomanSection>

      <RomanSection index={2} of={3} title="Provenance">
        <p className="lab-note">
          Engine: LiteMedSAM · isolated at
          backend/app/organs/brain/segmentation/experimental_lab/ — the automated
          heatmap-prompt path is retired pending a retrained model. A click scores a median
          Dice of 0.917 against ground truth on the held-out split.
        </p>
      </RomanSection>
    </>
  )
}

export default LiteMedSAMTab
