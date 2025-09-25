import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Search, 
  BarChart3, 
  Brain, 
  Radar, 
  Zap, 
  Database,
  ArrowRight,
  Star,
<<<<<<< HEAD
  Globe
=======
  Globe,
  Rocket,
  Satellite
>>>>>>> 975c3a7 (Версия 1.5.1)
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import ApiService from '../services/api'
import { useAppStore } from '../store/useAppStore'
<<<<<<< HEAD
import LoadingSpinner from '../components/ui/LoadingSpinner'
=======
>>>>>>> 975c3a7 (Версия 1.5.1)

const features = [
  {
    icon: Brain,
    title: 'ИИ-анализ',
    description: 'Современные нейронные сети для точного обнаружения экзопланет',
    color: 'text-cosmic-400'
  },
  {
    icon: Radar,
    title: 'Данные миссий',
    description: 'Поддержка TESS, Kepler и K2 с автоматической загрузкой',
    color: 'text-primary-400'
  },
  {
    icon: BarChart3,
    title: 'Интерактивная визуализация',
    description: 'Современные графики кривых блеска и анализ транзитов',
    color: 'text-green-400'
  },
  {
    icon: Zap,
    title: 'Быстрый поиск',
    description: 'BLS алгоритм с ускорением и оптимизацией',
    color: 'text-yellow-400'
  },
  {
    icon: Database,
    title: 'База знаний',
    description: 'Накопление опыта и улучшение точности со временем',
    color: 'text-purple-400'
  },
  {
    icon: Globe,
    title: 'Открытая наука',
    description: 'Вклад в поиск обитаемых миров во Вселенной',
    color: 'text-blue-400'
  }
]

const stats = [
  { label: 'Проанализировано звезд', value: '50,000+', icon: Star },
  { label: 'Найдено кандидатов', value: '1,200+', icon: Globe },
  { label: 'Точность ИИ', value: '94.5%', icon: Brain },
  { label: 'Время анализа', value: '<30с', icon: Zap },
]

export default function HomePage() {
  const { setHealthStatus, addToast } = useAppStore()
  const [animationStep, setAnimationStep] = useState(0)

  // Health check query
  const { data: healthStatus, isError } = useQuery({
    queryKey: ['health'],
    queryFn: ApiService.getHealth,
    refetchInterval: 30000, // Check every 30 seconds
<<<<<<< HEAD
    onSuccess: (data) => {
      setHealthStatus(data)
    },
    onError: () => {
=======
  })

  // Handle success/error with useEffect
  useEffect(() => {
    if (healthStatus) {
      setHealthStatus(healthStatus)
    } else if (isError) {
>>>>>>> 975c3a7 (Версия 1.5.1)
      setHealthStatus({
        status: 'down',
        timestamp: new Date().toISOString(),
        services_available: false,
        scientific_libs: false,
        services: {
          bls: 'unavailable',
          data: 'unavailable',
          ai: 'unavailable'
        }
      })
    }
<<<<<<< HEAD
  })
=======
  }, [healthStatus, isError, setHealthStatus])
>>>>>>> 975c3a7 (Версия 1.5.1)

  useEffect(() => {
    const timer = setInterval(() => {
      setAnimationStep((prev) => (prev + 1) % 4)
    }, 2000)
    return () => clearInterval(timer)
  }, [])

  return (
<<<<<<< HEAD
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative overflow-hidden py-20 px-4">
        {/* Background Animation */}
        <div className="absolute inset-0 overflow-hidden">
          {[...Array(50)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-1 h-1 bg-white rounded-full opacity-20"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
              }}
              animate={{
                opacity: [0.2, 0.8, 0.2],
                scale: [1, 1.5, 1],
              }}
              transition={{
                duration: 3 + Math.random() * 2,
                repeat: Infinity,
                delay: Math.random() * 2,
              }}
            />
          ))}
        </div>

        <div className="max-w-7xl mx-auto relative z-10">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="mb-8"
            >
              <motion.div
                animate={{ rotate: animationStep * 90 }}
                transition={{ duration: 0.5 }}
                className="inline-block p-4 bg-cosmic-gradient rounded-2xl mb-6"
              >
                <Radar className="h-12 w-12 text-white" />
              </motion.div>
              
              <h1 className="text-5xl md:text-7xl font-bold mb-6">
                <span className="text-gradient">Exoplanet AI</span>
              </h1>
              
              <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto mb-8">
                Передовая система искусственного интеллекта для обнаружения экзопланет 
                с использованием анализа кривых блеска
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="flex flex-col sm:flex-row gap-4 justify-center mb-12"
            >
              <Link
                to="/search"
                className="btn-cosmic px-8 py-4 text-lg font-semibold rounded-xl flex items-center justify-center space-x-2 group"
              >
                <Search className="h-5 w-5" />
                <span>Начать поиск</span>
                <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
              </Link>
              
              <Link
                to="/about"
                className="btn-secondary px-8 py-4 text-lg font-semibold rounded-xl flex items-center justify-center space-x-2"
              >
                <span>Узнать больше</span>
              </Link>
            </motion.div>

            {/* Status indicator */}
            {healthStatus && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                className="inline-flex items-center space-x-2 px-4 py-2 bg-space-800/50 rounded-full border border-space-600"
              >
                <div className={`w-2 h-2 rounded-full ${
                  healthStatus.status === 'healthy' ? 'bg-green-500' : 
                  healthStatus.status === 'degraded' ? 'bg-yellow-500' : 'bg-red-500'
                }`} />
                <span className="text-sm text-gray-300">
                  Система {healthStatus.status === 'healthy' ? 'готова к работе' : 
                          healthStatus.status === 'degraded' ? 'работает с ограничениями' : 'недоступна'}
                </span>
              </motion.div>
            )}
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 px-4 bg-space-800/30">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
=======
    <div className="min-h-screen relative">
      {/* Hero Section */}
      <section className="relative py-32 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            {/* NASA Logo */}
            <div className="mb-12">
              <div className="w-24 h-24 mx-auto mb-8 bg-gradient-to-br from-blue-600 to-blue-800 rounded-full flex items-center justify-center">
                <Rocket className="h-12 w-12 text-white" />
              </div>
              
              <h1 className="text-6xl md:text-7xl font-light text-white mb-6 tracking-tight">
                Exoplanet AI
              </h1>
              
              <p className="text-xl text-gray-400 mb-12 max-w-2xl mx-auto leading-relaxed">
                NASA-grade artificial intelligence for discovering new worlds beyond our solar system
              </p>
            </div>
            
            {/* Status Indicators */}
            <div className="flex items-center justify-center space-x-8 mb-16">
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-gray-300">System Operational</span>
              </div>
              <div className="w-px h-6 bg-gray-600"></div>
              <div className="flex items-center space-x-3">
                <div className="w-3 h-3 bg-blue-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-gray-300">AI Models Ready</span>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="flex flex-col sm:flex-row gap-4 justify-center mb-12"
          >
            <Link
              to="/search"
              className="bg-cyan-500 hover:bg-cyan-600 text-white px-8 py-3 rounded-lg font-medium transition-colors flex items-center space-x-2"
            >
              <Search className="h-5 w-5" />
              <span>Start Search</span>
            </Link>
            
            <Link
              to="/catalog"
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-medium transition-colors flex items-center space-x-2"
            >
              <Database className="h-5 w-5" />
              <span>Browse Catalog</span>
            </Link>
          </motion.div>

          {/* NASA System Status */}
          {healthStatus && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="nasa-card inline-flex items-center space-x-3 px-6 py-3"
            >
              <div className={`w-3 h-3 rounded-full ${
                healthStatus.status === 'healthy' ? 'bg-green-400 status-success' : 
                healthStatus.status === 'degraded' ? 'bg-yellow-400 status-warning' : 'bg-red-400 status-error'
              } animate-pulse`} />
              <span className="text-sm font-mono uppercase tracking-wide">
                SYSTEM {healthStatus.status === 'healthy' ? 'NOMINAL' : 
                        healthStatus.status === 'degraded' ? 'DEGRADED' : 'OFFLINE'}
              </span>
            </motion.div>
          )}
        </div>
      </section>

      {/* Simple Statistics */}
      <section className="py-16 px-4">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold text-white mb-8">
              Mission Statistics
            </h2>
          </motion.div>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
>>>>>>> 975c3a7 (Версия 1.5.1)
            {stats.map((stat, index) => {
              const Icon = stat.icon
              return (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
<<<<<<< HEAD
                  className="text-center"
                >
                  <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-600/20 rounded-lg mb-4">
                    <Icon className="h-6 w-6 text-primary-400" />
                  </div>
                  <div className="text-2xl md:text-3xl font-bold text-white mb-2">
=======
                  className="nasa-card text-center"
                >
                  <Icon className="h-8 w-8 text-cyan-400 mx-auto mb-3" />
                  <div className="text-2xl font-bold text-white mb-1">
>>>>>>> 975c3a7 (Версия 1.5.1)
                    {stat.value}
                  </div>
                  <div className="text-sm text-gray-400">
                    {stat.label}
                  </div>
                </motion.div>
              )
            })}
          </div>
        </div>
      </section>

<<<<<<< HEAD
      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
=======
      {/* Features */}
      <section className="py-16 px-4">
        <div className="max-w-6xl mx-auto">
>>>>>>> 975c3a7 (Версия 1.5.1)
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
<<<<<<< HEAD
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Возможности системы
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Современные технологии машинного обучения для точного и быстрого 
              обнаружения экзопланет в данных космических миссий
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
=======
            className="text-center mb-12"
          >
            <h2 className="text-3xl font-bold text-white mb-4">
              Key Features
            </h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Advanced machine learning for precise exoplanet detection using space mission data
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
>>>>>>> 975c3a7 (Версия 1.5.1)
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
<<<<<<< HEAD
                  className="card hover:bg-space-700/50 transition-all duration-300 group"
                >
                  <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg mb-4 ${feature.color} bg-current/10`}>
                    <Icon className={`h-6 w-6 ${feature.color}`} />
                  </div>
                  
                  <h3 className="text-xl font-semibold text-white mb-3 group-hover:text-gradient transition-all">
                    {feature.title}
                  </h3>
                  
                  <p className="text-gray-400 group-hover:text-gray-300 transition-colors">
=======
                  className="nasa-card"
                >
                  <Icon className="h-8 w-8 text-cyan-400 mb-4" />
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-gray-400 text-sm">
>>>>>>> 975c3a7 (Версия 1.5.1)
                    {feature.description}
                  </p>
                </motion.div>
              )
            })}
          </div>
        </div>
      </section>

<<<<<<< HEAD
      {/* CTA Section */}
      <section className="py-20 px-4 bg-cosmic-gradient/10">
        <div className="max-w-4xl mx-auto text-center">
=======
      {/* Call to Action */}
      <section className="py-16 px-4">
        <div className="max-w-3xl mx-auto text-center">
>>>>>>> 975c3a7 (Версия 1.5.1)
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
<<<<<<< HEAD
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Готовы найти новые миры?
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              Присоединяйтесь к поиску экзопланет и помогите расширить наше понимание Вселенной
=======
            className="nasa-card p-8"
          >
            <Globe className="h-16 w-16 text-cyan-400 mx-auto mb-6" />
            
            <h2 className="text-3xl font-bold text-white mb-4">
              Ready to Discover New Worlds?
            </h2>
            <p className="text-gray-400 mb-8">
              Join the search for exoplanets and help expand our understanding of the Universe.
>>>>>>> 975c3a7 (Версия 1.5.1)
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/search"
<<<<<<< HEAD
                className="btn-cosmic px-8 py-4 text-lg font-semibold rounded-xl flex items-center justify-center space-x-2"
              >
                <Search className="h-5 w-5" />
                <span>Начать анализ</span>
              </Link>
              
              <Link
                to="/analysis"
                className="btn-secondary px-8 py-4 text-lg font-semibold rounded-xl flex items-center justify-center space-x-2"
              >
                <BarChart3 className="h-5 w-5" />
                <span>Посмотреть примеры</span>
=======
                className="bg-green-500 hover:bg-green-600 text-white px-6 py-3 rounded-lg font-medium transition-colors flex items-center space-x-2"
              >
                <Rocket className="h-5 w-5" />
                <span>Launch Search</span>
              </Link>
              
              <Link
                to="/bls"
                className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg font-medium transition-colors flex items-center space-x-2"
              >
                <BarChart3 className="h-5 w-5" />
                <span>BLS Analysis</span>
>>>>>>> 975c3a7 (Версия 1.5.1)
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}
