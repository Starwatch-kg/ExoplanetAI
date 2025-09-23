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
  Globe
} from 'lucide-react'
import { useQuery } from '@tanstack/react-query'
import ApiService from '../services/api'
import { useAppStore } from '../store/useAppStore'
import LoadingSpinner from '../components/ui/LoadingSpinner'

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
    onSuccess: (data) => {
      setHealthStatus(data)
    },
    onError: () => {
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
  })

  useEffect(() => {
    const timer = setInterval(() => {
      setAnimationStep((prev) => (prev + 1) % 4)
    }, 2000)
    return () => clearInterval(timer)
  }, [])

  return (
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
            {stats.map((stat, index) => {
              const Icon = stat.icon
              return (
                <motion.div
                  key={stat.label}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="text-center"
                >
                  <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-600/20 rounded-lg mb-4">
                    <Icon className="h-6 w-6 text-primary-400" />
                  </div>
                  <div className="text-2xl md:text-3xl font-bold text-white mb-2">
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

      {/* Features Section */}
      <section className="py-20 px-4">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
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
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                  className="card hover:bg-space-700/50 transition-all duration-300 group"
                >
                  <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg mb-4 ${feature.color} bg-current/10`}>
                    <Icon className={`h-6 w-6 ${feature.color}`} />
                  </div>
                  
                  <h3 className="text-xl font-semibold text-white mb-3 group-hover:text-gradient transition-all">
                    {feature.title}
                  </h3>
                  
                  <p className="text-gray-400 group-hover:text-gray-300 transition-colors">
                    {feature.description}
                  </p>
                </motion.div>
              )
            })}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-cosmic-gradient/10">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Готовы найти новые миры?
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              Присоединяйтесь к поиску экзопланет и помогите расширить наше понимание Вселенной
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                to="/search"
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
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}
