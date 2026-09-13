import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource-variable/inter-tight'
import '@fontsource-variable/inter'
import '@fontsource-variable/jetbrains-mono'
import '@fontsource/playfair-display/500.css'
import '@fontsource/playfair-display/500-italic.css'
import './index.css'
import App from './App.tsx'
import { attachLiquidButtons } from './utils/liquidButtons'

attachLiquidButtons()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
