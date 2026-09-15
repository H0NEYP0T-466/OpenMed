import React from 'react'
import { Link } from 'react-router-dom'
import type { OrganId } from '../../types/organ'
import { ORGANS_REGISTRY } from '../../types/organ'
import { AppShell } from '../../components/common/AppShell'
import '../../components/medical/essay.css'
import '../../components/medical/OrganSelector.css'
import './LandingPage.css'

const ORGAN_ORDER = Object.keys(ORGANS_REGISTRY) as OrganId[]

const MOVEMENTS = [
  {
    roman: 'I',
    title: 'Paper Intake',
    term: 'Lectio',
    copy: 'A photograph of the handwritten clinic file goes in; structured text comes out — OCR over paper is the front door of Pakistani healthcare.',
    status: 'draft' as const,
  },
  {
    roman: 'II',
    title: 'Organ Routing',
    term: 'Distributio',
    copy: 'The scan reads itself: a classifier tells you which department owns this image before any diagnosis begins.',
    status: 'draft' as const,
  },
  {
    roman: 'III',
    title: 'Classification',
    term: 'Judicium',
    copy: 'Twelve organ departments, each with a fine-tuned vision ensemble grading the pathology — tumour type, stage, severity.',
    status: 'press' as const,
  },
  {
    roman: 'IV',
    title: 'Segmentation',
    term: 'Dissectio',
    copy: 'Not just what, but where: pixel-level masks trace every lesion boundary, later extruded into the 3D plate.',
    status: 'press' as const,
  },
  {
    roman: 'V',
    title: 'The Report',
    term: 'Scriptura',
    copy: 'A fine-tuned medical language model sets findings, impression and recommendation as a signed dispatch — always marked “clinician review required”.',
    status: 'draft' as const,
  },
  {
    roman: 'VI',
    title: 'Consultation',
    term: 'Disputatio',
    copy: 'A RAG assistant answers questions about the case, the report, and the literature behind it — with citations to every line.',
    status: 'draft' as const,
  },
]

const STATUS_LABEL = {
  live: 'Live',
  press: 'In Press',
  draft: 'Draft',
} as const

export const LandingPage: React.FC = () => (
  <AppShell
    cta={
      <Link className="btn btn-primary" to="/app">
        <span>Enter the Annual</span>
        <span className="nav-star" aria-hidden="true">★</span>
        <span className="arr" aria-hidden="true">↗</span>
      </Link>
    }
  >
    {/* Hero — the frontispiece fold */}
    <section className="lp-hero">
      <div className="lp-hero-type">
        <h1 className="dossier-title">
          An AI hospital<span className="dot">.</span>
        </h1>
        <p className="lp-hero-sub serif">edited like a medical annual</p>
        <p className="dossier-lead">
          OpenMed reads scans the way an editor reads copy twelve organ departments
          classify, segment, and report on what the image says, then typeset the verdict
          as a signed dispatch. Every plate in this volume is interactive, every claim
          carries its benchmark, and every structure is named by Terminologia Anatomica.
        </p>
        <div className="lp-cta-row">
          <Link className="btn btn-primary" to="/app">
            <span>Open the Atlas</span>
            <span className="arr" aria-hidden="true">↗</span>
          </Link>
          <a className="btn btn-ghost" href="#movements">
            <span>The consultation, in six movements</span>
            <span className="arr" aria-hidden="true">↓</span>
          </a>
        </div>
        <p className="lp-edition mono">
          This edition ships the atlas &amp; benchmark dossiers — the AI pipeline is in press.
        </p>
      </div>

    </section>

    {/* The consultation, in six movements */}
    <section className="lp-pipeline" id="movements">
      <div className="lp-sect-head">
        <div className="sec-rule">
          <span className="roman">II</span>
          <span className="sec-title">The Consultation</span>
          <span className="page-of">six movements</span>
        </div>
        <h2 className="lp-sect-title serif">From paper to diagnosis, the annual is printed in order.</h2>
      </div>
      <div className="lp-steps">
        {MOVEMENTS.map((m) => (
          <article className="lp-step" key={m.roman}>
            <div className="lp-step-top">
              <span className="lp-num serif">{m.roman}</span>
              <span className={`lp-status lp-status--${m.status}`}>{STATUS_LABEL[m.status]}</span>
            </div>
            <h3 className="lp-step-name">{m.title}</h3>
            <div className="lp-step-term mono">{m.term}</div>
            <p className="lp-step-copy">{m.copy}</p>
          </article>
        ))}
      </div>
    </section>

    {/* Departments — index of specimens, same pills as the workspace */}
    <section className="lp-departments">
      <div className="index-head">
        <div className="sec-rule">
          <span className="roman">III</span>
          <span className="sec-title">Index of Departments</span>
          <span className="page-of">13 plates</span>
        </div>
      </div>
      <div className="pill-grid">
        {ORGAN_ORDER.map((id, i) => {
          const organ = ORGANS_REGISTRY[id]
          return (
            <Link
              key={id}
              to={`/app?organ=${id}`}
              className="pill lp-pill"
              aria-label={`Enter the ${organ.name} plate`}
            >
              <span className="p-num serif">{String(i + 1).padStart(2, '0')}</span>
              <span className="p-name">
                {organ.name}
                {id === 'body' && <span className="p-star" aria-hidden="true">★</span>}
              </span>
              <span className="p-mod">{organ.modality}</span>
            </Link>
          )
        })}
      </div>
    </section>

    {/* Colophon */}
    <footer className="lp-foot">
      <div className="lp-wordmark" aria-hidden="true">
        OpenMed<span className="dot">.</span>
      </div>
      <div className="lp-foot-inner">
        <p className="lp-foot-line">
          <em className="serif">How many times has AI actually helped a doctor?</em>{' '}
          A verified ledger of the published answers: thirteen clinical disclosures, every
          number linked to its source.
        </p>
        <Link className="btn btn-ghost" to="/evidence">
          <span>Begin reading</span>
          <span className="nav-star" aria-hidden="true">↗</span>
        </Link>
      </div>
    </footer>
  </AppShell>
)
