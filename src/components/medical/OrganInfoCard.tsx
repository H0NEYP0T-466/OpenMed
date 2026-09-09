import React from 'react'
import type { OrganMetadata, Hotspot } from '../../types/organ'
import {
  Activity,
  Database,
  Layers,
  FileCheck,
  CheckCircle2,
  Stethoscope,
  HeartPulse,
  Droplets,
  BookOpen,
  MapPin,
  Flame
} from 'lucide-react'

interface OrganInfoCardProps {
  readonly organ: OrganMetadata
  readonly activeHotspot: Hotspot | null
  readonly onSelectHotspot: (hotspot: Hotspot | null) => void
}

export const OrganInfoCard: React.FC<OrganInfoCardProps> = ({
  organ,
  activeHotspot,
  onSelectHotspot,
}) => {
  const profile = organ.clinicalProfile

  return (
    <div className="organ-info-card">
      {/* Header */}
      <div className="info-card-header">
        <div className="info-badge-row">
          <span
            className="modality-badge"
            style={{
              borderColor: `${organ.accentColor}66`,
              color: organ.accentColor,
              backgroundColor: `${organ.accentColor}15`,
            }}
          >
            <Activity size={12} />
            <span>{organ.modality}</span>
          </span>

          <span className="pipeline-status-badge">
            <CheckCircle2 size={12} className="text-emerald-400" />
            <span>AI Pipeline Verified</span>
          </span>
        </div>

        <div className="organ-title-block">
          <span className="organ-large-icon">{organ.icon}</span>
          <div>
            <div className="title-with-poetic">
              <h2 className="organ-main-title">{organ.name}</h2>
              {profile?.poeticTitle && (
                <span className="poetic-badge">"{profile.poeticTitle}"</span>
              )}
            </div>
            <p className="organ-latin-subtitle">{organ.anatomicalTerm}</p>
          </div>
        </div>
      </div>

      {/* Physiological Function */}
      <div className="clinical-overview-box">
        <div className="overview-header">
          <HeartPulse size={13} className="text-rose-400" />
          <span>Physiological Function</span>
        </div>
        <p className="overview-content">
          {profile?.physiology || organ.description}
        </p>
      </div>

      {/* Daily Clinical Stat & Fact */}
      {profile && (
        <div className="daily-fact-card">
          <div className="daily-fact-icon">
            <Flame size={16} className="text-amber-400" />
          </div>
          <div className="daily-fact-text">
            <span className="daily-fact-title">Physiological Constant</span>
            <p className="daily-fact-body">{profile.dailyFact}</p>
          </div>
        </div>
      )}

      {/* Anatomical Landmarks (Interactive Points) */}
      {organ.hotspots && organ.hotspots.length > 0 && (
        <div className="landmarks-section">
          <div className="landmarks-header">
            <div className="flex items-center gap-2">
              <MapPin size={13} className="text-cyan-400" />
              <span>Anatomical Landmarks ({organ.hotspots.length})</span>
            </div>
            <span className="landmarks-subhint">Click point to inspect</span>
          </div>

          <div className="landmarks-list">
            {organ.hotspots.map((spot) => {
              const isSelected = activeHotspot?.id === spot.id
              return (
                <button
                  key={spot.id}
                  type="button"
                  className={`landmark-item-btn ${isSelected ? 'selected' : ''}`}
                  onClick={() => onSelectHotspot(isSelected ? null : spot)}
                  style={
                    isSelected
                      ? {
                          borderColor: spot.color,
                          backgroundColor: `${spot.color}15`,
                        }
                      : undefined
                  }
                >
                  <div
                    className="landmark-bullet"
                    style={{ backgroundColor: spot.color }}
                  />
                  <div className="landmark-text">
                    <div className="landmark-name-row">
                      <strong className="landmark-title">{spot.label}</strong>
                      <span className="landmark-latin">{spot.latinTerm}</span>
                    </div>
                    <p className="landmark-brief">{spot.detail}</p>
                  </div>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Datasets Section */}
      <div className="datasets-grid">
        <div className="dataset-item">
          <div className="dataset-label">
            <Database size={13} className="text-cyan-400" />
            <span>Classification Benchmark</span>
          </div>
          <span className="dataset-value">{organ.classificationDataset}</span>
        </div>

        <div className="dataset-item">
          <div className="dataset-label">
            <Layers size={13} className="text-emerald-400" />
            <span>Segmentation Benchmark</span>
          </div>
          <span className="dataset-value">{organ.segmentationDataset}</span>
        </div>
      </div>

      {/* Clinical Diagnostic Tasks */}
      <div className="clinical-tasks-section">
        <div className="tasks-header">
          <Stethoscope size={13} className="text-purple-400" />
          <span>Active Diagnostic Targets</span>
        </div>
        <div className="tasks-tags-list">
          {organ.clinicalTasks.map((task, index) => (
            <span key={index} className="task-tag">
              <FileCheck size={11} className="text-cyan-400" />
              <span>{task}</span>
            </span>
          ))}
        </div>
      </div>

      {/* Clinical Conditions */}
      {profile?.commonConditions && profile.commonConditions.length > 0 && (
        <div className="conditions-section">
          <div className="conditions-header">
            <BookOpen size={13} className="text-indigo-400" />
            <span>Associated Clinical Pathologies</span>
          </div>
          <div className="conditions-chips">
            {profile.commonConditions.map((cond, idx) => (
              <span key={idx} className="condition-chip">
                {cond}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Blood Supply */}
      {profile?.bloodSupply && (
        <div className="blood-supply-box">
          <Droplets size={13} className="text-rose-400" />
          <span>Vascular Supply: <strong>{profile.bloodSupply}</strong></span>
        </div>
      )}

      {/* 3D Metadata Footer */}
      <div className="info-card-footer">
        <span className="footer-meta-item">
          System: <strong>{profile?.system || 'Homo Sapiens Anatomy'}</strong>
        </span>
        <span className="footer-meta-item">
          Asset: <code>{organ.modelFile.split('/').pop()}</code>
        </span>
      </div>
    </div>
  )
}
