import { useState, useEffect } from 'react'
import { Routes, Route } from 'react-router-dom'
import ApiService from './services/api'
import Header from './components/layout/Header'
import StarField from './components/background/StarField'
import HomePage from './pages/HomePage'
import AboutPage from './pages/AboutPage'
import AITrainingPage from './pages/AITrainingPage'
import GPIPage from './pages/GPIPage'
import SearchPage from './pages/SearchPage'
import CatalogPage from './pages/CatalogPage'
import DatabasePage from './pages/DatabasePage'
import NotFoundPage from './pages/NotFoundPage'
import type { HealthStatus } from './types/api'

function App() {
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null)

  // Check API health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await ApiService.getHealth()
        setHealthStatus(health)
      } catch (err) {
        console.warn('API health check failed:', err)
      }
    }
    checkHealth()
  }, [])

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 dark:from-gray-900 dark:via-blue-900 dark:to-gray-900 light:from-blue-50 light:via-indigo-100 light:to-blue-50 transition-colors duration-500">
      {/* Animated Star Field Background */}
      <StarField />
      
      {/* Main Content */}
      <div className="relative z-10 min-h-screen">
        {/* Header */}
        <Header healthStatus={healthStatus} />

        {/* Routes */}
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/ai-training" element={<AITrainingPage />} />
          <Route path="/gpi" element={<GPIPage />} />
          <Route path="/search" element={<SearchPage />} />
          <Route path="/catalog" element={<CatalogPage />} />
          <Route path="/database" element={<DatabasePage />} />
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </div>
    </div>
  )
}

export default App
