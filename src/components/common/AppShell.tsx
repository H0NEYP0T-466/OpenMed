import React, { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import './AppShell.css'

interface AppShellProps {
  /** Masthead action. Optional — the front page carries none. */
  readonly cta?: React.ReactNode
  readonly children: React.ReactNode
}

const NAV_ITEMS = [
  { to: '/', label: 'Front Page', note: 'the annual, at a glance' },
  { to: '/app', label: 'The Atlas', note: '3D models · 12 plates' },
  {
    to: '/lab',
    label: 'Experimental Laboratory',
    note: 'assistive segmentation · click or draw',
  },
  { to: '/evidence', label: 'AI Helping Doctors', note: 'the verified tallies' },
  { to: '/docs', label: 'Documentation', note: 'academic benchmarks · training · ledger' },
]

/**
 * Page spine shared by every screen: three-line menu, fixed side rails,
 * masthead (brand → front page, stats, action slot).
 *
 * The menu is a sidebar rather than a dropdown so the list can keep growing —
 * it scrolls, and each entry has room for a label plus a note.
 */
export const AppShell: React.FC<AppShellProps> = ({ cta, children }) => {
  const [menuOpen, setMenuOpen] = useState(false)
  const { pathname } = useLocation()

  // Navigating away always closes it, whichever link was taken.
  useEffect(() => {
    setMenuOpen(false)
  }, [pathname])

  useEffect(() => {
    if (!menuOpen) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMenuOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [menuOpen])

  return (
    <div className="om-page">
      <header className="masthead">
        <div className="om-menu-wrap">
          <button
            type="button"
            className="om-menu-btn"
            aria-label={menuOpen ? 'Close navigation' : 'Open navigation'}
            aria-expanded={menuOpen}
            aria-controls="om-sidebar"
            onClick={() => setMenuOpen((v) => !v)}
          >
            <span className={`bar ${menuOpen ? 'x1' : ''}`} />
            <span className={`bar ${menuOpen ? 'hide' : ''}`} />
            <span className={`bar ${menuOpen ? 'x2' : ''}`} />
          </button>
        </div>

        {/* Slides in from the left. Closes on the burger again, on the pointer
            leaving the panel, on Escape, or on navigating. */}
        <aside
          id="om-sidebar"
          className={`om-sidebar ${menuOpen ? 'is-open' : ''}`}
          aria-label="Primary"
          aria-hidden={!menuOpen}
          onMouseLeave={() => setMenuOpen(false)}
        >
          <div className="om-sidebar-head">
            <span className="om-sidebar-title">Contents</span>
            <span className="om-sidebar-rule" aria-hidden="true" />
          </div>

          <nav className="om-sidebar-nav">
            {NAV_ITEMS.map((item) => (
              <Link key={item.to} to={item.to} className="om-menu-item">
                <span className="omi-label">{item.label}</span>
                <span className="omi-note mono">{item.note}</span>
              </Link>
            ))}
          </nav>

          <div className="om-sidebar-foot mono">
            <span>OpenMed</span>
            <span className="dot">.</span>
          </div>
        </aside>

        <Link to="/" className="brand" aria-label="OpenMed - return to the front page">
          <span className="brand-word">
            OpenMed<span className="dot">.</span>
          </span>
        </Link>

        <div className="masthead-stats">
          <div className="stat-fig">
            <span className="fig">12</span>
            <span className="cap">Organs</span>
          </div>
          <div className="stat-fig">
            <span className="fig">
              20<em>+</em>
            </span>
            <span className="cap">AI Models</span>
          </div>
          <div className="stat-fig">
            <span className="fig">2,234</span>
            <span className="cap">Atlas Parts</span>
          </div>
        </div>

        {cta && <div className="masthead-actions">{cta}</div>}
      </header>

      {children}
    </div>
  )
}
