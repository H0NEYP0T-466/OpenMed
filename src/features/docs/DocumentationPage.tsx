import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { AppShell } from '../../components/common/AppShell'
import { RomanSection } from '../../components/common/RomanSection'
import { Doc3DModelsDirectory } from './components/Doc3DModelsDirectory'
import { ModelsOverviewBanner } from './components/ModelsOverviewBanner'
import { BrainDocs } from './components/brain/BrainDocs'
import { HeartDocs } from './components/heart/HeartDocs'
import { LungsDocs } from './components/lungs/LungsDocs'
import { OtherOrganSuitesDocs } from './components/organs/OtherOrganSuitesDocs'
import { FastApiDocs } from './components/system/FastApiDocs'
import { MniStereotacticDocs } from './components/system/MniStereotacticDocs'
import { DatasetSanitizationDocs } from './components/system/DatasetSanitizationDocs'
import { ArtifactModal } from './components/ArtifactModal'
import type { ArtifactCard } from './types'
import './DocumentationPage.css'

export const DocumentationPage: React.FC = () => {
  const [activeArtifact, setActiveArtifact] = useState<ArtifactCard | null>(null)

  return (
    <AppShell
      cta={
        <Link to="/app?organ=brain" className="btn btn-primary">
          <span>Launch 3D Brain Atlas</span>
          <span className="arr" aria-hidden="true">↗</span>
        </Link>
      }
    >
      <main className="doc-page-layout">
        {/* Document Head */}
        <header className="doc-head-block">
          <div className="doc-eyebrow">
            <span>Academic &amp; Engineering Documentation</span>
            <span className="meta-sep">•</span>
            <span className="doc-meta-tag mono">OpenMed 2026 Core</span>
          </div>

          <h1 className="doc-main-title">
            Documenta &amp; Engineering Ledger
            <span className="serif">Architectural Specifications, Empirical Pathways &amp; Model Trajectories</span>
          </h1>

          <p className="doc-lead-copy">
            Rigorous mathematical, anatomical, and empirical documentation for OpenMed’s clinical AI architectures.
            This ledger organizes our entire multi-organ platform: 3D anatomical models, deep learning classification suites,
            volumetric segmentation pipelines, out-of-distribution failure mode discoveries, discarded architectural hypotheses,
            and production retraining blueprints.
          </p>

          {/* Quick Anchor Navigation */}
          <div className="doc-quick-nav">
            <a href="#section-3d-models" className="doc-nav-anchor active">
              I · 3D Models Directory (13 Organs)
            </a>
            <a href="#section-models-suite" className="doc-nav-anchor">
              II · Models &amp; Neural Suites
            </a>
            <a href="#model-brain" className="doc-nav-anchor">
              ↳ Brain (Classification &amp; Segmentation)
            </a>
            <a href="#model-heart" className="doc-nav-anchor">
              ↳ Heart (Classification &amp; Segmentation)
            </a>
            <a href="#model-lungs" className="doc-nav-anchor">
              ↳ Lungs (Classification &amp; Segmentation)
            </a>
            <a href="#model-other-organs" className="doc-nav-anchor">
              ↳ Other Organ Suites (Renal, Liver, Eye, etc.)
            </a>
            <a href="#section-api-infra" className="doc-nav-anchor">
              III · FastAPI Backend Service
            </a>
            <a href="#section-stereotactic" className="doc-nav-anchor">
              IV · MNI152 Stereotaxis
            </a>
            <a href="#section-dataset" className="doc-nav-anchor">
              V · Dataset Sanitization
            </a>
          </div>
        </header>

        {/* SECTION I: 3D ANATOMICAL MODELS DIRECTORY (ALL 13 ORGANS) */}
        <Doc3DModelsDirectory />

        {/* SECTION II: MODELS (NEURAL ARCHITECTURES & CLINICAL PIPELINES) */}
        <section id="section-models-suite" className="doc-section-wrapper">
          <RomanSection
            index={1}
            of={5}
            title="Models - Neural Architectures, Clinical Benchmarks & Task Suites"
            className="sp12"
          >
            <ModelsOverviewBanner />

            {/* Organ: Brain (Classification & Segmentation) */}
            <BrainDocs onOpenArtifact={setActiveArtifact} />

            {/* Organ: Heart (Classification & Segmentation) */}
            <HeartDocs />

            {/* Organ: Lungs (Classification & Segmentation) */}
            <LungsDocs />

            {/* Additional Organ Suites (Kidneys, Liver, Eye, Skin, etc.) */}
            <OtherOrganSuitesDocs />
          </RomanSection>
        </section>

        {/* SECTION III: FASTAPI PRODUCTION ARCHITECTURE */}
        <FastApiDocs />

        {/* SECTION IV: STEREOTACTIC MNI LOCALIZATION */}
        <MniStereotacticDocs />

        {/* SECTION V: DATASET CURATION & PURGING */}
        <DatasetSanitizationDocs />
      </main>

      {/* Artifact Lightbox Modal */}
      <ArtifactModal
        activeArtifact={activeArtifact}
        onClose={() => setActiveArtifact(null)}
      />
    </AppShell>
  )
}
