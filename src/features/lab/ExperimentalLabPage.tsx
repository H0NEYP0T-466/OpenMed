import React, { useState } from 'react'
import { Link } from 'react-router-dom'
import { AppShell } from '../../components/common/AppShell'
import { LiteMedSAMTab } from './tabs/LiteMedSAMTab'
import type { LabInstrument } from './types'
import './ExperimentalLabPage.css'

/**
 * Experimental Laboratory — an instrument bench, not a page.
 *
 * Each tab is a self-contained instrument: it owns its state, its API calls and
 * its model lifecycle. This file only decides which one is mounted, so a new
 * instrument is one registry entry plus one component.
 */
const INSTRUMENTS: readonly LabInstrument[] = [
  {
    id: 'litemedsam',
    title: 'Assistive Segmentation',
    romanNumeral: 'I',
    description: 'Mark the lesion. The mark is the prompt.',
    component: LiteMedSAMTab,
  },
]

export const ExperimentalLabPage: React.FC = () => {
  const [activeId, setActiveId] = useState<string>(INSTRUMENTS[0].id)
  const active = INSTRUMENTS.find((instrument) => instrument.id === activeId) ?? INSTRUMENTS[0]
  const ActiveInstrument = active.component

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
            One instrument, one bench<span className="dot">.</span>
          </h1>
          <p className="lab-lead">
            Nothing here is part of the product yet. Each instrument runs one model against one
            question, loads only while its tab is open, and releases its memory the moment you
            leave. Mark a lesion yourself.
          </p>
        </header>

        <nav className="lab-tabs" aria-label="Laboratory instruments">
          {INSTRUMENTS.map((instrument) => (
            <button
              key={instrument.id}
              type="button"
              className={`lab-tab ${instrument.id === active.id ? 'is-active' : ''}`}
              aria-current={instrument.id === active.id ? 'page' : undefined}
              onClick={() => setActiveId(instrument.id)}
            >
              <span className="lab-tab-roman mono">{instrument.romanNumeral}</span>
              <span className="lab-tab-body">
                <span className="lab-tab-title">{instrument.title}</span>
                <span className="lab-tab-desc">{instrument.description}</span>
              </span>
            </button>
          ))}
        </nav>

        <div className="lab-panel" key={active.id}>
          <ActiveInstrument />
        </div>

        <footer className="lab-foot mono">
          Instrument {active.romanNumeral}
          {INSTRUMENTS.length > 1 ? ` of ${INSTRUMENTS.length}` : ''} · models load on tab select
          and unload on exit.
        </footer>
      </div>
    </AppShell>
  )
}
