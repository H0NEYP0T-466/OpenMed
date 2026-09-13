import React, { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { AppShell } from '../../components/common/AppShell'
import { EssayHead } from '../../components/medical/EssayHead'
import '../../components/medical/essay.css'
import './EvidencePage.css'

/* 152,437,366 — exact sum of the ten cumulative disclosures below;
   the headline rounds down. Annual flows are listed but never summed in. */
const TALLY = 152_000_000
const ANNUAL_FLOW = '≈ 190,000,000'

interface SourceRow {
  readonly org: string
  readonly stat: string
  readonly unit: string
  readonly basis: 'cumulative' | 'annual'
  readonly note: string
  readonly url: string
  readonly label: string
}

const SOURCES: readonly SourceRow[] = [
  {
    org: 'Aidoc',
    stat: '110,000,000',
    unit: 'patient cases analysed',
    basis: 'cumulative',
    note: 'FDA-cleared radiology AI in ~2,000 hospitals; disclosed “more than 110 million patient cases” with its Series E — over 60M of them in a single year.',
    url: 'https://www.prnewswire.com/news-releases/aidoc-raises-150-million-series-e-led-by-goldman-sachs-to-scale-clinical-ai-for-earlier-safer-diagnoses-302757181.html',
    label: 'PR Newswire · Apr 2026',
  },
  {
    org: 'Microsoft · Dragon Copilot',
    stat: '100,000,000',
    unit: 'encounters documented — each year',
    basis: 'annual',
    note: 'Ambient clinical AI: FY26 earnings call discloses a run-rate of over 100 million patient encounters automated this calendar year, 28M of them last quarter.',
    url: 'https://www.microsoft.com/en-us/investor/events/fy-2026/earnings-fy-2026-q4',
    label: 'Microsoft Investor Relations · Jul 2026',
  },
  {
    org: 'Infervision',
    stat: '19,000,000',
    unit: 'patients read by InferRead CT',
    basis: 'cumulative',
    note: 'China’s lung-CT AI: 55,000 cases a day across 380+ hospitals at the time of its FDA clearance — 19M patients already benefited.',
    url: 'https://www.prnewswire.com/news-releases/infervision-receives-fda-clearance-for-the-inferread-lung-ctai-product-301091145.html',
    label: 'PR Newswire · Jul 2020',
  },
  {
    org: 'ScreenPoint Medical',
    stat: '12,000,000',
    unit: 'mammograms processed',
    basis: 'cumulative',
    note: 'Transpara, acting as second reader in breast-screening programmes across 30+ countries.',
    url: 'https://www.prnewswire.com/news-releases/screenpoint-medical-secures-16m-to-lead-the-next-phase-of-ai-in-breast-cancer-care-302743926.html',
    label: 'PR Newswire · Apr 2026',
  },
  {
    org: 'Lunit',
    stat: '19,700,000',
    unit: 'images analysed — each year',
    basis: 'annual',
    note: 'Mammography & chest X-ray AI used by 9,500+ clinicians at 3,600+ sites worldwide (2025 sustainability report).',
    url: 'https://www.lunit.io/wp-content/uploads/2026-05-2025-Lunit-Sustainability-Report_ENG.pdf',
    label: 'Lunit Report · Dec 2025',
  },
  {
    org: 'DeepHealth · RadNet',
    stat: '10,000,000',
    unit: 'mammograms supported — each year',
    basis: 'annual',
    note: 'Breast-suite AI across the largest US outpatient radiology network; validated in the biggest real-world AI mammography study published to date.',
    url: 'https://www.radnet.com/about-radnet/news/deephealth-launches-breast-suite-elevating-breast-cancer-detection-risk-stratification-and-workflow',
    label: 'RadNet News · Dec 2025',
  },
  {
    org: 'Qure.ai × AstraZeneca',
    stat: '5,000,000',
    unit: 'chest X-rays screened',
    basis: 'cumulative',
    note: 'Lung-cancer risk AI in 20+ countries — largely in clinics with no radiologist on site.',
    url: 'https://www.qure.ai/us/news-press-coverages/A-new-era-for-lung-cancer-detection-AI-enabled-risk-assessment-reaches-5-million-people-around-the-world',
    label: 'Qure.ai press · Apr 2025',
  },
  {
    org: 'Mayo Clinic · AI-ECG',
    stat: '1,000,000+',
    unit: 'ECGs screened with AI by clinicians',
    basis: 'cumulative',
    note: 'The wLiv AI-ECG algorithm — born in Mayo’s lab, validated in its EAGLE trial across 45 hospitals — now used by clinicians over a million times since introduction.',
    url: 'https://newsnetwork.mayoclinic.org/discussion/ai-ecg-helps-physicians-detect-hidden-heart-condition-video/',
    label: 'Mayo Clinic News Network · Jul 2026',
  },
  {
    org: 'Kaiser Permanente N. Cal.',
    stat: '2,576,627',
    unit: 'encounters assisted by AI scribes',
    basis: 'cumulative',
    note: '7,260 physicians, Oct 2023–Dec 2024 — the largest published real-world study of ambient AI in clinical practice.',
    url: 'https://divisionofresearch.kaiserpermanente.org/ai-assisted-notetaking-gains-steady-support-from-kaiser-permanente-physicians/',
    label: 'Kaiser Division of Research · 2025',
  },
  {
    org: 'Ardent Health · Ambience',
    stat: '1,000,000+',
    unit: 'encounters supported',
    basis: 'cumulative',
    note: 'Hospital network crossing one million ambient-AI-assisted patient encounters within roughly a year of go-live.',
    url: 'https://finance.yahoo.com/healthcare/articles/ardent-health-surpasses-1-million-120000817.html',
    label: 'Press release · Jul 2026',
  },
  {
    org: 'Cleveland Clinic',
    stat: '1,000,000',
    unit: 'encounters documented & summarised',
    basis: 'cumulative',
    note: 'Ambient AI now writes the visit note itself in one of America’s flagship academic medical centers.',
    url: 'https://consultqd.clevelandclinic.org/less-typing-more-talking-how-ambient-ai-is-reshaping-clinical-workflow-at-cleveland-clinic',
    label: 'Consult QD · Aug 2025',
  },
  {
    org: 'Everida · ARDA (Google)',
    stat: '600,000+',
    unit: 'patients screened for blindness risk',
    basis: 'cumulative',
    note: 'Autonomous diabetic-retinopathy AI at Aravind Eye Care, Tamil Nadu — post-deployment results published in JAMA Network Open.',
    url: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11923701/',
    label: 'JAMA Netw Open · 2025',
  },
  {
    org: 'Vara · PRAIM study',
    stat: '260,739',
    unit: 'AI double-reads in a national program',
    basis: 'cumulative',
    note: 'Germany’s organised mammography screening ran AI as an additional reader — the first prospective nationwide deployment of its kind.',
    url: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC11922743/',
    label: 'Nature Medicine · 2025',
  },
]

const REGISTRY_NOTE =
  'Behind these rows stands the machinery: the FDA has authorised 1,451 AI-enabled medical devices (76% of them imaging); the WHO counts 3.6 billion imaging exams a year worldwide; and beyond clinician desks, consumers alone have now recorded 250 million self-ECGs on AliveCor devices — every one AI-annotated at capture — while Apple and Fitbit heart studies passively screened 874,996 more wrists.'

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
    const dur = 2300
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
        verified="Every figure disclosed by its own operator · retrieved Sep 2026"
        title="The Tally"
        serif="When AI sat beside the doctor"
        latin={`Evidence Nº 01 — ${SOURCES.length} published ledgers · audited by registry`}
        lead="There is no global registry of machine-assisted readings — so this page keeps its own, conservatively: it adds up only the cumulative figures that operators published themselves, and never counts a yearly flow twice. The ledger below is not a forecast. It is what has already happened, signed by the companies who did it."
      />

      <main className="ev-sheet">
        <div className="ev-tally" ref={tallyRef}>
          <div className="ev-num mono" aria-live="polite">
            {value.toLocaleString('en-US')}
            <span className="ev-plus">+</span>
          </div>
          <p className="ev-cap">
            cumulative AI-assisted clinical reads, disclosed and citable
            <br />
            <span className="ev-cap-sub">
              and roughly {ANNUAL_FLOW} more added every year — annual
              disclosures alone
            </span>
          </p>
        </div>

        <div className="ev-list-head">
          <div className="sec-rule">
            <span className="roman">I</span>
            <span className="sec-title">The Ledger</span>
            <span className="page-of">{SOURCES.length} sources</span>
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
                <div className={`ev-basis ev-basis--${s.basis}`}>
                  {s.basis === 'cumulative' ? 'cumulative' : 'per year'}
                </div>
              </div>
            </article>
          ))}
        </div>

        <p className="ev-method mono">
          Method — company disclosures &amp; peer-reviewed program studies only,
          every line linked; yearly flows excluded from the ledger total to
          avoid double counting. {REGISTRY_NOTE}
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
