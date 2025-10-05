import { useState, useEffect, lazy, Suspense } from 'react'
import { Routes, Route } from 'react-router-dom'
import { typedApiClient } from './utils/typedApiClient'
import type { HealthStatus } from './types/api'
import Header from './components/layout/Header'
import StarBackground from './components/StarBackground'
import SimpleStarBackground from './components/SimpleStarBackground'
import ErrorBoundary from './components/ErrorBoundary'

// Lazy loading для оптимизации
const HomePage = lazy(() => import('./pages/HomePage'))
const AboutPage = lazy(() => import('./pages/AboutPage'))
const GPIAnalysisPage = lazy(() => import('./pages/GPIAnalysisPage'))
const SearchPage = lazy(() => import('./pages/SearchPage'))
const CatalogPage = lazy(() => import('./pages/CatalogPage'))
const DatabasePage = lazy(() => import('./pages/DatabasePage'))
const AITrainingPage = lazy(() => import('./pages/AITrainingPage'))
const NotFoundPage = lazy(() => import('./pages/NotFoundPage'))

function App() {
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null)
  const [useSimpleBackground, setUseSimpleBackground] = useState(false)

  const handleSetUseSimpleBackground = (value: boolean | ((prev: boolean) => boolean)) => {
    setUseSimpleBackground(value)
  }

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
        setHealthStatus({ 
          status: 'unhealthy', 
          uptime: 0,
          timestamp: new Date().toISOString()
        })
      }
    }

    checkHealth()
    
    // Set up periodic health checks
    const interval = setInterval(checkHealth, 60000) // Every minute
    
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen bg-black transition-colors duration-500">
      {/* Выбираемый звездный фон */}
      {useSimpleBackground ? <SimpleStarBackground /> : <StarBackground />}
      
      {/* Main Content */}
      <div className="relative z-20 min-h-screen">
        {/* Header */}
        <Header 
          healthStatus={healthStatus} 
          useSimpleBackground={useSimpleBackground}
          setUseSimpleBackground={handleSetUseSimpleBackground}
        />

        {/* Routes с Suspense для lazy loading */}
        <ErrorBoundary>
          <Suspense fallback={
            <div className="flex items-center justify-center min-h-screen">
              <div className="flex flex-col items-center gap-4">
                <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-amber-500"></div>
                <p className="text-gray-400 text-lg">Загрузка...</p>
              </div>
            </div>
          }>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/about" element={<AboutPage useSimpleBackground={useSimpleBackground} />} />
              <Route path="/search" element={<SearchPage />} />
              <Route path="/gpi" element={<GPIAnalysisPage useSimpleBackground={useSimpleBackground} />} />
              <Route path="/catalog" element={<CatalogPage />} />
              <Route path="/database" element={<DatabasePage />} />
              <Route path="/ai-training" element={<AITrainingPage />} />
              <Route path="*" element={<NotFoundPage />} />
            </Routes>
          </Suspense>
        </ErrorBoundary>
      </div>
    </div>
  )
}

export default App
