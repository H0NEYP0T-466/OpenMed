import React, { useEffect, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import './AppShell.css'

interface AppShellProps {
  readonly cta: React.ReactNode
  readonly children: React.ReactNode
}

const NAV_ITEMS = [
  { to: '/', label: 'Front Page', note: 'the annual, at a glance' },
  { to: '/app', label: 'The Atlas', note: '3D models · 13 plates' },
  { to: '/evidence', label: 'AI Helping Doctors', note: 'the verified tallies' },
]

/**
 * Page spine shared by every screen: three-line menu, fixed side rails,
 * masthead (brand → front page, stats, action slot).
 */
export const AppShell: React.FC<AppShellProps> = ({ cta, children }) => {
  const [menuOpen, setMenuOpen] = useState(false)
  const { pathname } = useLocation()

  useEffect(() => {
    setMenuOpen(false)
  }, [pathname])

  return (
    <div className="om-page">
      <header className="masthead">
        <div className="om-menu-wrap">
          <button
            type="button"
            className="om-menu-btn"
            aria-label="Open navigation"
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((v) => !v)}
          >
            <span className={`bar ${menuOpen ? 'x1' : ''}`} />
            <span className={`bar ${menuOpen ? 'hide' : ''}`} />
            <span className={`bar ${menuOpen ? 'x2' : ''}`} />
          </button>

          {menuOpen && (
            <nav className="om-menu" aria-label="Primary">
              {NAV_ITEMS.map((item) => (
                <Link key={item.to} to={item.to} className="om-menu-item">
                  <span className="omi-label">{item.label}</span>
                  <span className="omi-note mono">{item.note}</span>
                </Link>
              ))}
            </nav>
          )}
        </div>

        <Link to="/" className="brand" aria-label="OpenMed — return to the front page">
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

        <div className="masthead-actions">{cta}</div>
      </header>

      {children}
    </div>
  )
}
