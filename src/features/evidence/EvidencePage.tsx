import React, { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { AppShell } from '../../components/common/AppShell'
import { EssayHead } from '../../components/medical/EssayHead'
import '../../components/medical/essay.css'
import './EvidencePage.css'

const TALLY = 100_000_000

interface SourceRow {
  readonly org: string
  readonly stat: string
  readonly unit: string
  readonly note: string
  readonly url: string
  readonly label: string
}

const SOURCES: readonly SourceRow[] = [
  {
    org: 'Aidoc',
    stat: '60,000,000',
    unit: 'patient cases — every year',
    note: 'FDA-cleared radiology AI reading CTs and MRIs across nearly 2,000 hospitals; disclosed with its Series E round.',
    url: 'https://www.prnewswire.com/il/news-releases/aidoc-raises-150-million-series-e-led-by-goldman-sachs-to-scale-clinical-ai-for-earlier-safer-diagnoses-302757181.html',
    label: 'PR Newswire',
  },
  {
    org: 'Radiology Partners',
    stat: '20,000,000+',
    unit: 'examinations touched by clinical AI',
    note: 'One radiology practice group alone — plus 30M images processed and 116M reports passed through its language tooling.',
    url: 'https://radiologybusiness.com/topics/artificial-intelligence/radiology-partners-reaches-milestone-deploying-clinical-ai-across-more-20-million-exams',
    label: 'Radiology Business',
  },
  {
    org: 'U.S. FDA registry',
    stat: '1,016',
    unit: 'authorised AI/ML medical devices',
    note: 'Independently audited count as of Dec 2024 — every one of them cleared to operate on real patients in the U.S.',
    url: 'https://www.nature.com/articles/s41746-025-01800-1',
    label: 'Nature · npj Digital Medicine',
  },
  {
    org: 'Qure.ai × AstraZeneca',
    stat: '5,000,000',
    unit: 'people screened by AI chest X-ray',
    note: 'Lung-cancer risk assessment live in 20+ countries — largely in clinics with no radiologist on site.',
    url: 'https://www.qure.ai/us/news-press-coverages/A-new-era-for-lung-cancer-detection-AI-enabled-risk-assessment-reaches-5-million-people-around-the-world',
    label: 'Qure.ai press release',
  },
]

export const EvidencePage: React.FC = () => {
  const tallyRef = useRef<HTMLDivElement>(null)
  const [started, setStarted] = useState(false)
  const [value, setValue] = useState(0)

  useEffect(() => {
    const el = tallyRef.current
    if (!el) return
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setStarted(true)
          io.disconnect()
        }
      },
      { threshold: 0.35 }
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])

  useEffect(() => {
    if (!started) return
    const t0 = performance.now()
    const dur = 1900
    let raf = 0
    const tick = (t: number) => {
      const p = Math.min(1, (t - t0) / dur)
      const ease = 1 - Math.pow(1 - p, 3)
      setValue(Math.round(TALLY * ease))
      if (p < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [started])

  return (
    <AppShell
      cta={
        <Link className="btn btn-primary" to="/app">
          <span>Open the Atlas</span>
          <span className="arr" aria-hidden="true">↗</span>
        </Link>
      }
    >
      <EssayHead
        badge="Field Evidence · Ed. MMXXVI"
        verified="Figures as publicly disclosed · retrieved Sep 2026"
        title="The Tally"
        serif="When AI sat beside the doctor"
        latin={`Evidence Nº 01 — ${SOURCES.length} published tallies · audited by registry`}
        lead="No global registry counts every machine-assisted reading yet — so this page keeps its own ledger. Below are the deployments that publish numbers: add the disclosed years together, subtract nothing for the readers that never announce themselves, and the conservative floor for a single year still crosses one hundred million times a physician did not read that scan alone."
      />

      <main className="ev-sheet">
        <div className="ev-tally" ref={tallyRef}>
          <div className="ev-num mono" aria-live="polite">
            {value.toLocaleString('en-US')}
            <span className="ev-plus">+</span>
          </div>
          <p className="ev-cap">
            AI-assisted clinical reads per year — a conservative floor
            <br />
            <span className="ev-cap-sub">
              summed only from disclosed deployments above
            </span>
          </p>
        </div>

        <div className="ev-list-head">
          <div className="sec-rule">
            <span className="roman">I</span>
            <span className="sec-title">The Ledger</span>
            <span className="page-of">four sources</span>
          </div>
        </div>

        <div className="ev-rows">
          {SOURCES.map((s, i) => (
            <article className="ev-row" key={s.org}>
              <span className="ev-idx serif">{String(i + 1).padStart(2, '0')}</span>
              <div className="ev-body">
                <h3 className="ev-org">{s.org}</h3>
                <p className="ev-note">{s.note}</p>
                <a
                  className="ev-link mono"
                  href={s.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span>{s.label}</span>
                  <span className="arr" aria-hidden="true">↗</span>
                </a>
              </div>
              <div className="ev-stat">
                <div className="ev-stat-num">{s.stat}</div>
                <div className="ev-stat-unit">{s.unit}</div>
              </div>
            </article>
          ))}
        </div>

        <p className="ev-method mono">
          Method — company disclosures &amp; regulatory registries only; press
          claims are linked so every line can be audited. The honest caveat:
          these are annual rates, not a lifetime counter.
        </p>

        <div className="ev-foot">
          <Link className="btn btn-ghost" to="/">
            <span>Front page</span>
            <span className="arr" aria-hidden="true">↙</span>
          </Link>
          <Link className="btn btn-primary" to="/app">
            <span>See the atlas the numbers describe</span>
            <span className="arr" aria-hidden="true">↗</span>
          </Link>
        </div>
      </main>
    </AppShell>
  )
}
