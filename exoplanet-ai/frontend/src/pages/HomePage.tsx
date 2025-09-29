import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Telescope, Zap, Database, Info, BarChart3, Rocket, Brain, Globe } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import ApiService from '../services/api'
import type { HealthStatus } from '../types/api'

const HomePage: React.FC = () => {
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null)
  const { t } = useTranslation()

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
    <div className="min-h-screen">
      {/* Hero Section */}
      <div className="text-center py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-6xl md:text-7xl font-bold text-white mb-6 animate-float">
            {t('app.title')}
          </h1>
          <p className="text-2xl md:text-3xl text-gray-300 mb-4">
            {t('app.subtitle')}
          </p>
          <p className="text-lg text-gray-400 mb-12 max-w-4xl mx-auto">
            {t('app.description')}
          </p>
          
          {/* Decorative Elements */}
          <div className="flex justify-center items-center gap-4 mb-16">
            <div className="w-24 h-px bg-gradient-to-r from-transparent via-blue-400 to-transparent" />
            <Telescope className="w-8 h-8 text-blue-400 animate-pulse" />
            <div className="w-24 h-px bg-gradient-to-r from-transparent via-purple-400 to-transparent" />
          </div>
        </div>
      </div>

      {/* Project Features */}
      <div className="max-w-7xl mx-auto px-4 mb-16">
        <h2 className="text-4xl font-bold text-white text-center mb-12">
          {t('pages.home.features')}
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {/* BLS Analysis */}
          <div className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/20 hover:border-blue-400/50 transition-all duration-300 group">
            <div className="w-16 h-16 bg-blue-500/20 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <BarChart3 className="w-8 h-8 text-blue-400" />
            </div>
            <h3 className="text-2xl font-semibold text-white mb-4">{t('pages.features.bls.title')}</h3>
            <p className="text-gray-300 mb-4">
              {t('pages.features.bls.description')}
            </p>
            <ul className="text-gray-400 text-sm space-y-2">
              <li>• C++ ускорение для высокой производительности</li>
              <li>• Статистическая оценка значимости</li>
              <li>• Обработка данных TESS, Kepler, K2</li>
            </ul>
          </div>

          {/* GPI Method */}
          <div className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/20 hover:border-purple-400/50 transition-all duration-300 group">
            <div className="w-16 h-16 bg-purple-500/20 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Zap className="w-8 h-8 text-purple-400" />
            </div>
            <h3 className="text-2xl font-semibold text-white mb-4">{t('pages.features.gpi.title')}</h3>
            <p className="text-gray-300 mb-4">
              {t('pages.features.gpi.description')}
            </p>
            <ul className="text-gray-400 text-sm space-y-2">
              <li>• Обнаружение малых планет</li>
              <li>• ИИ-улучшенный анализ</li>
              <li>• Высокая чувствительность к шуму</li>
            </ul>
          </div>

          {/* Database */}
          <div className="bg-white/10 backdrop-blur-sm rounded-xl p-8 border border-white/20 hover:border-green-400/50 transition-all duration-300 group">
            <div className="w-16 h-16 bg-green-500/20 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
              <Database className="w-8 h-8 text-green-400" />
            </div>
            <h3 className="text-2xl font-semibold text-white mb-4">{t('pages.features.database.title')}</h3>
            <p className="text-gray-300 mb-4">
              {t('pages.features.database.description')}
            </p>
            <ul className="text-gray-400 text-sm space-y-2">
              <li>• Реальные данные NASA</li>
              <li>• Аналитика и статистика</li>
              <li>• История поисков</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Navigation Cards */}
      <div className="max-w-6xl mx-auto px-4 mb-16">
        <h2 className="text-3xl font-bold text-white text-center mb-8">
          {t('pages.home.getStarted')}
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Link to="/ai-training" className="group">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 hover:bg-white/20 transition-all duration-300 border border-white/20 hover:border-indigo-400/50 text-center">
              <Brain className="w-12 h-12 text-indigo-400 mb-4 mx-auto group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-semibold text-white mb-2">{t('navigation.aiTraining')}</h3>
              <p className="text-gray-300 text-sm">Тренировка нейронных сетей</p>
            </div>
          </Link>
          
          <Link to="/catalog" className="group">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 hover:bg-white/20 transition-all duration-300 border border-white/20 hover:border-green-400/50 text-center">
              <Globe className="w-12 h-12 text-green-400 mb-4 mx-auto group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-semibold text-white mb-2">{t('navigation.catalog')}</h3>
              <p className="text-gray-300 text-sm">Обзор экзопланет</p>
            </div>
          </Link>
          
          <Link to="/database" className="group">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 hover:bg-white/20 transition-all duration-300 border border-white/20 hover:border-purple-400/50 text-center">
              <Database className="w-12 h-12 text-purple-400 mb-4 mx-auto group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-semibold text-white mb-2">{t('navigation.database')}</h3>
              <p className="text-gray-300 text-sm">Метрики системы</p>
            </div>
          </Link>
          
          <Link to="/about" className="group">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 hover:bg-white/20 transition-all duration-300 border border-white/20 hover:border-yellow-400/50 text-center">
              <Info className="w-12 h-12 text-yellow-400 mb-4 mx-auto group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-semibold text-white mb-2">{t('navigation.about')}</h3>
              <p className="text-gray-300 text-sm">Наша миссия</p>
            </div>
          </Link>
        </div>
      </div>

      {/* Technology Stack */}
      <div className="max-w-6xl mx-auto px-4 mb-16">
        <h2 className="text-3xl font-bold text-white text-center mb-8">
          {t('pages.home.techStack')}
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <Rocket className="w-6 h-6 text-blue-400" />
              Frontend
            </h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="text-gray-300">
                <span className="text-blue-400">•</span> React 18 + TypeScript
              </div>
              <div className="text-gray-300">
                <span className="text-blue-400">•</span> TailwindCSS
              </div>
              <div className="text-gray-300">
                <span className="text-blue-400">•</span> React Router
              </div>
              <div className="text-gray-300">
                <span className="text-blue-400">•</span> Plotly.js
              </div>
            </div>
          </div>
          
          <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <Brain className="w-6 h-6 text-purple-400" />
              Backend
            </h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="text-gray-300">
                <span className="text-purple-400">•</span> FastAPI + Python
              </div>
              <div className="text-gray-300">
                <span className="text-purple-400">•</span> C++ Acceleration
              </div>
              <div className="text-gray-300">
                <span className="text-purple-400">•</span> SQLite Database
              </div>
              <div className="text-gray-300">
                <span className="text-purple-400">•</span> NASA APIs
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* System Status */}
      {healthStatus && (
        <div className="max-w-4xl mx-auto px-4 mb-16">
          <div className="bg-white/5 backdrop-blur-sm rounded-xl p-8 border border-white/10">
            <h3 className="text-2xl font-semibold text-white mb-6 text-center">Статус системы</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="w-4 h-4 bg-green-400 rounded-full mx-auto mb-3 animate-pulse" />
                <p className="text-white font-medium">API</p>
                <p className="text-gray-300 text-sm">{healthStatus.status}</p>
              </div>
              <div className="text-center">
                <div className="w-4 h-4 bg-blue-400 rounded-full mx-auto mb-3 animate-pulse" />
                <p className="text-white font-medium">Версия</p>
                <p className="text-gray-300 text-sm">{healthStatus.version}</p>
              </div>
              <div className="text-center">
                <div className="w-4 h-4 bg-purple-400 rounded-full mx-auto mb-3 animate-pulse" />
                <p className="text-white font-medium">База данных</p>
                <p className="text-gray-300 text-sm">Подключена</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="text-center pb-8 text-gray-400">
        <div className="max-w-4xl mx-auto space-y-4 px-4">
          <div className="flex justify-center items-center gap-4 mb-4">
            <div className="w-16 h-px bg-gradient-to-r from-transparent via-gray-500 to-transparent" />
            <Telescope className="w-5 h-5 text-gray-500" />
            <div className="w-16 h-px bg-gradient-to-r from-transparent via-gray-500 to-transparent" />
          </div>
          <p className="text-lg font-medium">
            {t('app.footer.powered')}
          </p>
          <p className="text-sm">
            {t('app.footer.description')}
          </p>
        </div>
      </footer>
    </div>
  )
}

export default HomePage
