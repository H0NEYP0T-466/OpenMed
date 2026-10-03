import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { AppShell } from '../../components/common/AppShell'
import { RomanSection } from '../../components/common/RomanSection'
import { AssistiveSegmentation } from '../organs/brain/AssistiveSegmentation'
import { SPECIMEN_PRESETS } from '../organs/brain/specimenPresets'
import type { SpecimenPreset } from '../organs/brain/specimenPresets'
import './ExperimentalLabPage.css'

/**
 * Experimental Laboratory — the home for assistive segmentation.
 *
 * LiteMedSAM is the current engine here, not a settled part of the product: the
 * automated heatmap-prompt path was retired from the classifier run and a
 * replacement model is expected. Everything model-specific lives in
 * `backend/app/organs/brain/segmentation/experimental_lab/`.
 *
 * The module itself is the same one that used to sit inside the brain
 * workspace — click or draw, and the mark becomes the prompt.
 */
export const ExperimentalLabPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleLoadPreset = async (preset: SpecimenPreset) => {
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
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0]
    if (!selected) return
    if (!selected.type.startsWith('image/')) {
      setError('Selected file is not an image (JPEG, PNG).')
      return
    }
    setError(null)
    setSelectedPreset(null)
    setFile(selected)
    setPreviewUrl(URL.createObjectURL(selected))
  }

  return (
    <AppShell
      cta={
        <Link className="btn btn-ghost" to="/app">
          <span>The Atlas</span>
          <span className="arr" aria-hidden="true">↗</span>
        </Link>
      }
    >
      <div className="lab-page">
        <header className="lab-head">
          <span className="lab-eyebrow mono">L · Experimental Laboratory</span>
          <h1 className="lab-title serif">
            Mark the lesion. The mark is the prompt<span className="dot">.</span>
          </h1>
          <p className="lab-lead">
            A clinician marks the suspicious region on the scan — a click or a drawn box — and the
            segmentation model returns a mask inside that mark. No classifier runs on this path, so
            the response is one model forward pass. Against ground truth on the held-out split, a
            click scores a median Dice of 0.917 where an automatically derived box scores 0.235.
          </p>
        </header>

        <RomanSection index={0} of={2} title="Specimen Rack">
          <div className="lab-rack">
            {SPECIMEN_PRESETS.map((preset) => (
              <button
                key={preset.plate}
                type="button"
                className={`lab-card ${selectedPreset === preset.plate ? 'is-selected' : ''}`}
                onClick={() => void handleLoadPreset(preset)}
              >
                <span className="lab-thumb">
                  <img
                    src={preset.path}
                    alt={preset.name}
                    loading="lazy"
                    decoding="async"
                  />
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

        <RomanSection index={1} of={2} title="Assistive Segmentation">
          {file && previewUrl ? (
            <AssistiveSegmentation file={file} previewUrl={previewUrl} />
          ) : (
            <div className="lab-idle">
              Pick a specimen above, or upload a scan, to begin.
            </div>
          )}
        </RomanSection>

        <footer className="lab-foot mono">
          Engine: LiteMedSAM · isolated at
          backend/app/organs/brain/segmentation/experimental_lab/ — the automated
          heatmap-prompt path is retired pending a retrained model.
        </footer>
      </div>
    </AppShell>
  )
}
