import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Telescope, Zap, Database, Info, BarChart3, Rocket, Brain, Globe, Star, Sparkles, ArrowRight, Play, CheckCircle } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { motion } from 'framer-motion'
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
      <div className="relative text-center py-20 px-4 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-purple-900/20 via-transparent to-blue-900/20"></div>
        <div className="max-w-6xl mx-auto relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="mb-8"
          >
            <div className="inline-flex items-center gap-2 bg-gradient-to-r from-purple-500/20 to-blue-500/20 backdrop-blur-sm border border-purple-500/30 rounded-full px-6 py-2 mb-6">
              <Star className="w-4 h-4 text-purple-400" />
              <span className="text-purple-300 text-sm font-medium">Advanced Exoplanet Detection</span>
              <Sparkles className="w-4 h-4 text-blue-400" />
            </div>
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-6xl md:text-8xl font-bold bg-gradient-to-r from-white via-blue-200 to-purple-200 bg-clip-text text-transparent mb-6"
          >
            {t('app.title')}
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-2xl md:text-3xl text-gray-300 mb-4 font-light"
          >
            {t('app.subtitle')}
          </motion.p>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-lg text-gray-400 mb-12 max-w-4xl mx-auto leading-relaxed"
          >
            {t('app.description')}
          </motion.p>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16"
          >
            <Link to="/search" className="group">
              <motion.button
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="relative overflow-hidden bg-gradient-to-r from-purple-600 via-blue-600 to-purple-700 hover:from-purple-700 hover:via-blue-700 hover:to-purple-800 text-white font-bold py-4 px-8 rounded-xl transition-all duration-500 shadow-lg hover:shadow-purple-500/25"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                <div className="relative flex items-center gap-3">
                  <Play className="w-5 h-5" />
                  <span className="text-lg">Start Discovery</span>
                  <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                </div>
              </motion.button>
            </Link>
            
            <Link to="/about" className="group">
              <motion.button
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="bg-white/10 hover:bg-white/20 border border-white/20 hover:border-white/30 text-white font-semibold py-4 px-8 rounded-xl transition-all duration-300 backdrop-blur-sm"
              >
                <div className="flex items-center gap-2">
                  <Info className="w-5 h-5" />
                  <span>Learn More</span>
                </div>
              </motion.button>
            </Link>
          </motion.div>
          
          {/* Decorative Elements */}
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 1 }}
            className="flex justify-center items-center gap-6"
          >
            <motion.div 
              animate={{ width: [0, 100, 0] }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
              className="h-px bg-gradient-to-r from-transparent via-blue-400 to-transparent"
            />
            <motion.div
              animate={{ rotate: 360, scale: [1, 1.2, 1] }}
              transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
            >
              <Telescope className="w-8 h-8 text-blue-400" />
            </motion.div>
            <motion.div 
              animate={{ width: [0, 100, 0] }}
              transition={{ duration: 3, repeat: Infinity, ease: "easeInOut", delay: 1.5 }}
              className="h-px bg-gradient-to-r from-transparent via-purple-400 to-transparent"
            />
          </motion.div>
        </div>
      </div>

      {/* Project Features */}
      <div className="max-w-7xl mx-auto px-4 mb-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-5xl font-bold bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent mb-4">
            {t('pages.home.features')}
          </h2>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Cutting-edge algorithms and AI-powered analysis for exoplanet discovery
          </p>
        </motion.div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {/* BLS Analysis */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true }}
            whileHover={{ y: -10, scale: 1.02 }}
            className="group relative overflow-hidden bg-gradient-to-br from-blue-500/10 to-cyan-500/10 backdrop-blur-sm rounded-2xl p-8 border border-blue-500/20 hover:border-blue-400/50 transition-all duration-500"
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full -translate-y-16 translate-x-16"></div>
            <div className="relative">
              <motion.div 
                whileHover={{ rotate: 360, scale: 1.1 }}
                transition={{ duration: 0.6 }}
                className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-xl flex items-center justify-center mb-6 shadow-lg"
              >
                <BarChart3 className="w-8 h-8 text-blue-400" />
              </motion.div>
              <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-blue-300 transition-colors">{t('pages.features.bls.title')}</h3>
              <p className="text-gray-300 mb-6 leading-relaxed">
                {t('pages.features.bls.description')}
              </p>
              <div className="space-y-3">
                <div className="flex items-center gap-3 text-gray-400 text-sm">
                  <CheckCircle className="w-4 h-4 text-blue-400" />
                  <span>C++ ускорение для высокой производительности</span>
                </div>
                <div className="flex items-center gap-3 text-gray-400 text-sm">
                  <CheckCircle className="w-4 h-4 text-blue-400" />
                  <span>Статистическая оценка значимости</span>
                </div>
                <div className="flex items-center gap-3 text-gray-400 text-sm">
                  <CheckCircle className="w-4 h-4 text-blue-400" />
                  <span>Обработка данных TESS, Kepler, K2</span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* GPI Method */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            viewport={{ once: true }}
            whileHover={{ y: -10, scale: 1.02 }}
            className="group relative overflow-hidden bg-gradient-to-br from-purple-500/10 to-pink-500/10 backdrop-blur-sm rounded-2xl p-8 border border-purple-500/20 hover:border-purple-400/50 transition-all duration-500"
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-purple-500/5 rounded-full -translate-y-16 translate-x-16"></div>
            <div className="relative">
              <motion.div 
                whileHover={{ rotate: 360, scale: 1.1 }}
                transition={{ duration: 0.6 }}
                className="w-16 h-16 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center mb-6 shadow-lg"
              >
                <Zap className="w-8 h-8 text-purple-400" />
              </motion.div>
              <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-purple-300 transition-colors">{t('pages.features.gpi.title')}</h3>
              <p className="text-gray-300 mb-6 leading-relaxed">
                {t('pages.features.gpi.description')}
              </p>
              <div className="space-y-3">
                <div className="flex items-center gap-3 text-gray-400 text-sm">
                  <CheckCircle className="w-4 h-4 text-purple-400" />
                  <span>Обнаружение малых планет</span>
                </div>
                <div className="flex items-center gap-3 text-gray-400 text-sm">
                  <CheckCircle className="w-4 h-4 text-purple-400" />
                  <span>ИИ-улучшенный анализ</span>
                </div>
                <div className="flex items-center gap-3 text-gray-400 text-sm">
                  <CheckCircle className="w-4 h-4 text-purple-400" />
                  <span>Высокая чувствительность к шуму</span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Database */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            viewport={{ once: true }}
            whileHover={{ y: -10, scale: 1.02 }}
            className="group relative overflow-hidden bg-gradient-to-br from-green-500/10 to-emerald-500/10 backdrop-blur-sm rounded-2xl p-8 border border-green-500/20 hover:border-green-400/50 transition-all duration-500"
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-green-500/5 rounded-full -translate-y-16 translate-x-16"></div>
            <div className="relative">
              <motion.div 
                whileHover={{ rotate: 360, scale: 1.1 }}
                transition={{ duration: 0.6 }}
                className="w-16 h-16 bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded-xl flex items-center justify-center mb-6 shadow-lg"
              >
                <Database className="w-8 h-8 text-green-400" />
              </motion.div>
              <h3 className="text-2xl font-bold text-white mb-4 group-hover:text-green-300 transition-colors">{t('pages.features.database.title')}</h3>
              <p className="text-gray-300 mb-6 leading-relaxed">
                {t('pages.features.database.description')}
              </p>
              <div className="space-y-3">
                <div className="flex items-center gap-3 text-gray-400 text-sm">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Реальные данные NASA</span>
                </div>
                <div className="flex items-center gap-3 text-gray-400 text-sm">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>Аналитика и статистика</span>
                </div>
                <div className="flex items-center gap-3 text-gray-400 text-sm">
                  <CheckCircle className="w-4 h-4 text-green-400" />
                  <span>История поисков</span>
                </div>
              </div>
            </div>
          </motion.div>
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
              <p className="text-gray-300 text-sm">{t('navigation.descriptions.aiTraining')}</p>
            </div>
          </Link>
          
          <Link to="/catalog" className="group">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 hover:bg-white/20 transition-all duration-300 border border-white/20 hover:border-green-400/50 text-center">
              <Globe className="w-12 h-12 text-green-400 mb-4 mx-auto group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-semibold text-white mb-2">{t('navigation.catalog')}</h3>
              <p className="text-gray-300 text-sm">{t('navigation.descriptions.catalog')}</p>
            </div>
          </Link>
          
          <Link to="/database" className="group">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 hover:bg-white/20 transition-all duration-300 border border-white/20 hover:border-purple-400/50 text-center">
              <Database className="w-12 h-12 text-purple-400 mb-4 mx-auto group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-semibold text-white mb-2">{t('navigation.database')}</h3>
              <p className="text-gray-300 text-sm">{t('navigation.descriptions.database')}</p>
            </div>
          </Link>
          
          <Link to="/about" className="group">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 hover:bg-white/20 transition-all duration-300 border border-white/20 hover:border-yellow-400/50 text-center">
              <Info className="w-12 h-12 text-yellow-400 mb-4 mx-auto group-hover:scale-110 transition-transform" />
              <h3 className="text-xl font-semibold text-white mb-2">{t('navigation.about')}</h3>
              <p className="text-gray-300 text-sm">{t('navigation.descriptions.about')}</p>
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
            <h3 className="text-2xl font-semibold text-white mb-6 text-center">{t('systemStatus.title')}</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className="w-4 h-4 bg-green-400 rounded-full mx-auto mb-3 animate-pulse" />
                <p className="text-white font-medium">{t('systemStatus.api')}</p>
                <p className="text-gray-300 text-sm">{healthStatus.status}</p>
              </div>
              <div className="text-center">
                <div className="w-4 h-4 bg-blue-400 rounded-full mx-auto mb-3 animate-pulse" />
                <p className="text-white font-medium">{t('systemStatus.version')}</p>
                <p className="text-gray-300 text-sm">{healthStatus.version || 'Unknown'}</p>
              </div>
              <div className="text-center">
                <div className="w-4 h-4 bg-purple-400 rounded-full mx-auto mb-3 animate-pulse" />
                <p className="text-white font-medium">{t('systemStatus.database')}</p>
                <p className="text-gray-300 text-sm">{t('systemStatus.connected')}</p>
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
