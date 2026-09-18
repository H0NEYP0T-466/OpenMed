import { HashRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { useEffect } from 'react'
import { LandingPage } from './features/landing/LandingPage'
import { OrganDashboard } from './features/organs/OrganDashboard'
import { EvidencePage } from './features/evidence/EvidencePage'
import { DocumentationPage } from './features/docs/DocumentationPage'

const ScrollToTop: React.FC = () => {
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo(0, 0) }, [pathname])
  return null
}

function App() {
  return (
    <HashRouter>
      <ScrollToTop />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/app" element={<OrganDashboard />} />
        <Route path="/evidence" element={<EvidencePage />} />
        <Route path="/docs" element={<DocumentationPage />} />
        <Route path="/documentation" element={<DocumentationPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </HashRouter>
  )
}

export default App
