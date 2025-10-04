import { useState, useEffect, lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import { typedApiClient } from './utils/typedApiClient'
import type { HealthStatus } from './types/api'
import Header from './components/layout/Header'
import StarBackground from './components/StarBackground'

// Lazy loading для оптимизации
const HomePage = lazy(() => import('./pages/HomePage'))
const AboutPage = lazy(() => import('./pages/AboutPage'))
const AITrainingPage = lazy(() => import('./pages/AITrainingPage'))
const GPIPage = lazy(() => import('./pages/GPIPage'))
const SearchPage = lazy(() => import('./pages/SearchPage'))
const CatalogPage = lazy(() => import('./pages/CatalogPage'))
const DatabasePage = lazy(() => import('./pages/DatabasePage'))
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'))

function App() {
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null)

  // Check API health with improved error handling
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await typedApiClient.getHealth()
        setHealthStatus(health)
      } catch (error) {
        if (import.meta.env.DEV) {
          console.warn('API health check failed:', error)
        }
        // Set degraded status on error
        setHealthStatus({ status: 'unhealthy', uptime: 0 })
      }
    }

    checkHealth()
    
    // Set up periodic health checks
    const interval = setInterval(checkHealth, 60000) // Every minute
    
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-black transition-colors duration-500">
      {/* Анимированный звездный фон */}
      <StarBackground />
      
      {/* Main Content */}
      <div className="relative z-20 min-h-screen">
        {/* Header */}
        <Header healthStatus={healthStatus} />

        {/* Routes с Suspense для lazy loading */}
        <Suspense fallback={
          <div className="flex items-center justify-center min-h-screen">
            <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div>
          </div>
        }>
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
        </Suspense>
      </div>
    </div>
  )
}

export default App
