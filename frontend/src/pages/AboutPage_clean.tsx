import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import { 
  Brain, 
  Zap, 
  Database, 
  Radar, 
  Globe, 
  Award, 
  TrendingUp, 
  BookOpen, 
  Github, 
  Mail, 
  ExternalLink 
} from 'lucide-react'

const features = [
  {
    icon: Brain,
    title: 'Искусственный интеллект',
    description: 'Современные нейронные сети (CNN, LSTM, Transformers) для точного обнаружения экзопланет с минимальным количеством ложных срабатываний.',
    details: [
      'Ансамбль из трех типов моделей',
      'Transfer learning с данных Kepler на TESS',
      'Active learning с пользовательской обратной связью',
      'Оценка неопределенности предсказаний'
    ]
  },
  {
    icon: Zap,
    title: 'Высокая производительность',
    description: 'Оптимизированные алгоритмы с поддержкой GPU ускорения и параллельной обработки для быстрого анализа больших объемов данных.',
    details: [
      'GPU ускорение через CUDA',
      'Параллельная обработка множественных целей',
      'Оптимизированные алгоритмы BLS и GPI',
      'Кэширование промежуточных результатов'
    ]
  },
  {
    icon: Database,
    title: 'Интеграция с NASA',
    description: 'Прямая интеграция с архивами NASA MAST для автоматической загрузки и обработки данных телескопов TESS и Kepler.',
    details: [
      'Автоматическая загрузка light curves',
      'Поддержка TESS, Kepler, K2 данных',
      'Предобработка и нормализация данных',
      'Кэширование результатов анализа'
    ]
  }
]

const team = [
  {
    nameKey: 'about.aiSystem',
    roleKey: 'about.dataAnalysis',
    descriptionKey: 'about.aiSystemDesc',
    avatar: '🤖'
  },
  {
    nameKey: 'about.blsAlgorithm',
    roleKey: 'about.signalSearch',
    descriptionKey: 'about.blsAlgorithmDesc',
    avatar: '📊'
  },
  {
    nameKey: 'about.database',
    roleKey: 'about.knowledgeStorage',
    descriptionKey: 'about.databaseDesc',
    avatar: '💾'
  }
]

interface SystemStats {
  stars_analyzed: number
  candidates_found: number
  system_accuracy: number
  average_time_seconds: number
  total_searches: number
  uptime_hours: number
}

const defaultStats: SystemStats = {
  stars_analyzed: 0,
  candidates_found: 0,
  system_accuracy: 0,
  average_time_seconds: 0,
  total_searches: 0,
  uptime_hours: 0
}

export default function AboutPage() {
  const { t } = useTranslation()
  const [stats, setStats] = useState<SystemStats>(defaultStats)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const response = await fetch('/api/v1/system/stats')
        if (response.ok) {
          const data = await response.json()
          setStats(data)
        }
      } catch (error) {
        console.error('Failed to fetch stats:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchStats()
  }, [])

  const formatNumber = (num: number): string => {
    if (num === 0) {
      return '0'
    }
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M+'
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K+'
    }
    return num.toString()
  }

  const formatAccuracy = (accuracy: number): string => {
    if (accuracy === 0) {
      return 'N/A'
    }
    return `${accuracy.toFixed(1)}%`
  }

  const formatTime = (seconds: number): string => {
    if (seconds === 0) {
      return 'N/A'
    }
    return `<${seconds}s`
  }

  const statsData = [
    { 
      labelKey: 'about.stats.starsAnalyzed', 
      value: loading ? '...' : formatNumber(stats.stars_analyzed), 
      icon: Radar 
    },
    { 
      labelKey: 'about.stats.candidatesFound', 
      value: loading ? '...' : formatNumber(stats.candidates_found), 
      icon: Globe 
    },
    { 
      labelKey: 'about.stats.systemAccuracy', 
      value: loading ? '...' : formatAccuracy(stats.system_accuracy), 
      icon: Award 
    },
    { 
      labelKey: 'about.stats.averageTime', 
      value: loading ? '...' : formatTime(stats.average_time_seconds), 
      icon: TrendingUp 
    },
  ]

  // Animation variants
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1
      }
    }
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: {
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        ease: [0.25, 0.46, 0.45, 0.94]
      }
    }
  }

  const cardVariants = {
    hidden: { opacity: 0, y: 60, rotateX: -15 },
    visible: {
      opacity: 1,
      y: 0,
      rotateX: 0,
      transition: {
        duration: 0.8,
        type: "spring",
        stiffness: 100
      }
    },
    hover: {
      y: -10,
      rotateX: 5,
      scale: 1.02,
      transition: { duration: 0.3 }
    }
  }

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Hero Section */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="text-center mb-16"
        >
          <motion.div 
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ 
              duration: 1,
              delay: 0.2,
              type: "spring",
              stiffness: 200,
              damping: 20
            }}
            className="inline-block p-4 bg-cosmic-gradient rounded-2xl mb-6"
          >
            <Brain className="w-12 h-12 text-white" />
          </motion.div>
          <motion.h1 
            variants={itemVariants}
            className="text-4xl md:text-6xl font-bold text-white mb-6"
          >
            {t('about.title')}
          </motion.h1>
          <motion.p 
            variants={itemVariants}
            className="text-xl text-gray-300 max-w-3xl mx-auto"
          >
            {t('about.missionDescription')}
          </motion.p>
        </motion.div>

        {/* Stats */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16"
        >
          {statsData.map((stat, index) => {
            const Icon = stat.icon
            return (
              <motion.div
                key={stat.labelKey}
                variants={cardVariants}
                whileHover="hover"
                whileTap={{ scale: 0.95 }}
                className="card text-center cursor-pointer"
              >
                <motion.div 
                  initial={{ rotate: 0 }}
                  whileHover={{ rotate: 360 }}
                  transition={{ duration: 0.6 }}
                  className="inline-flex items-center justify-center w-12 h-12 bg-primary-600/20 rounded-lg mb-4"
                >
                  <Icon className="h-6 w-6 text-primary-400" />
                </motion.div>
                <motion.div 
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ 
                    duration: 0.5,
                    delay: 1.2 + index * 0.1,
                    type: "spring",
                    stiffness: 200
                  }}
                  className="text-2xl md:text-3xl font-bold text-white mb-2"
                >
                  {stat.value}
                </motion.div>
                <div className="text-sm text-gray-400">
                  {t(stat.labelKey)}
                </div>
              </motion.div>
            )
          })}
        </motion.div>

        {/* Features */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="grid md:grid-cols-3 gap-8 mb-16"
        >
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <motion.div 
                key={feature.title}
                variants={cardVariants}
                whileHover="hover"
                className="card hover:shadow-2xl transition-shadow duration-300"
              >
                <div className="flex items-center gap-3 mb-4">
                  <motion.div 
                    initial={{ scale: 0, rotate: -90 }}
                    animate={{ scale: 1, rotate: 0 }}
                    transition={{ 
                      duration: 0.6,
                      delay: 1.8 + index * 0.2,
                      type: "spring"
                    }}
                    whileHover={{ 
                      scale: 1.1,
                      rotate: 10,
                      transition: { duration: 0.2 }
                    }}
                    className="p-3 bg-primary-600/20 rounded-lg"
                  >
                    <Icon className="w-6 h-6 text-primary-400" />
                  </motion.div>
                  <motion.h3 
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.6, delay: 2.0 + index * 0.2 }}
                    className="text-xl font-semibold text-white"
                  >
                    {feature.title}
                  </motion.h3>
                </div>
                <motion.p 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.6, delay: 2.2 + index * 0.2 }}
                  className="text-gray-300 mb-4"
                >
                  {feature.description}
                </motion.p>
                <motion.ul 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.6, delay: 2.4 + index * 0.2 }}
                  className="space-y-2"
                >
                  {feature.details.map((detail, detailIndex) => (
                    <motion.li 
                      key={detailIndex}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ 
                        duration: 0.4, 
                        delay: 2.6 + index * 0.2 + detailIndex * 0.1 
                      }}
                      className="text-sm text-gray-400 flex items-center gap-2"
                    >
                      <span className="w-1.5 h-1.5 bg-primary-400 rounded-full"></span>
                      {detail}
                    </motion.li>
                  ))}
                </motion.ul>
              </motion.div>
            )
          })}
        </motion.div>

        {/* How it works */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="mb-16"
        >
          <motion.h2 
            variants={itemVariants}
            className="text-3xl font-bold text-white text-center mb-12"
          >
            {t('about.howItWorks')}
          </motion.h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            <motion.div 
              variants={cardVariants}
              whileHover="hover"
              className="card text-center"
            >
              <motion.div 
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.6, delay: 0.2, type: "spring" }}
                className="inline-flex items-center justify-center w-16 h-16 bg-primary-600/20 rounded-full mb-6"
              >
                <span className="text-2xl font-bold text-primary-400">1</span>
              </motion.div>
              <h3 className="text-xl font-semibold text-white mb-4">
                {t('about.dataLoading')}
              </h3>
              <p className="text-gray-300">
                {t('about.dataLoadingDesc')}
              </p>
            </motion.div>

            <motion.div 
              variants={cardVariants}
              whileHover="hover"
              className="card text-center"
            >
              <motion.div 
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.6, delay: 0.4, type: "spring" }}
                className="inline-flex items-center justify-center w-16 h-16 bg-cosmic-600/20 rounded-full mb-6"
              >
                <span className="text-2xl font-bold text-cosmic-400">2</span>
              </motion.div>
              <h3 className="text-xl font-semibold text-white mb-4">
                {t('about.blsAnalysis')}
              </h3>
              <p className="text-gray-300">
                {t('about.blsAnalysisDesc')}
              </p>
            </motion.div>

            <motion.div 
              variants={cardVariants}
              whileHover="hover"
              className="card text-center"
            >
              <motion.div 
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.6, delay: 0.6, type: "spring" }}
                className="inline-flex items-center justify-center w-16 h-16 bg-green-600/20 rounded-full mb-6"
              >
                <span className="text-2xl font-bold text-green-400">3</span>
              </motion.div>
              <h3 className="text-xl font-semibold text-white mb-4">
                {t('about.aiValidation')}
              </h3>
              <p className="text-gray-300">
                {t('about.aiValidationDesc')}
              </p>
            </motion.div>
          </div>
        </motion.div>

        {/* Team */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="mb-16"
        >
          <motion.h2 
            variants={itemVariants}
            className="text-3xl font-bold text-white text-center mb-12"
          >
            {t('about.systemComponents')}
          </motion.h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            {team.map((member, index) => (
              <motion.div
                key={member.nameKey}
                variants={cardVariants}
                whileHover="hover"
                className="card text-center"
              >
                <motion.div 
                  initial={{ scale: 0, rotate: 180 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{ 
                    duration: 0.8, 
                    delay: 0.2 + index * 0.1,
                    type: "spring"
                  }}
                  className="text-4xl mb-4"
                >
                  {member.avatar}
                </motion.div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  {t(member.nameKey)}
                </h3>
                <p className="text-cosmic-400 font-medium mb-3">
                  {t(member.roleKey)}
                </p>
                <p className="text-gray-300 text-sm">
                  {t(member.descriptionKey)}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Resources */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="visible"
          className="card text-center"
        >
          <motion.h2 
            variants={itemVariants}
            className="text-3xl font-bold text-white mb-6"
          >
            {t('about.resourcesTitle')}
          </motion.h2>
          
          <motion.p 
            variants={itemVariants}
            className="text-lg text-gray-300 mb-8 max-w-3xl mx-auto"
          >
            {t('about.resourcesDesc')}
          </motion.p>
          
          <motion.div 
            variants={containerVariants}
            className="grid md:grid-cols-3 gap-6"
          >
            <motion.a
              variants={cardVariants}
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              href="#"
              className="flex items-center justify-center space-x-2 p-4 bg-space-700/50 rounded-lg border border-space-600 hover:bg-space-700 transition-colors group"
            >
              <BookOpen className="h-5 w-5 text-primary-400" />
              <span className="text-white group-hover:text-primary-300">{t('about.documentation')}</span>
              <ExternalLink className="h-4 w-4 text-gray-400" />
            </motion.a>
            
            <motion.a
              variants={cardVariants}
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              href="#"
              className="flex items-center justify-center space-x-2 p-4 bg-space-700/50 rounded-lg border border-space-600 hover:bg-space-700 transition-colors group"
            >
              <Github className="h-5 w-5 text-primary-400" />
              <span className="text-white group-hover:text-primary-300">{t('about.github')}</span>
              <ExternalLink className="h-4 w-4 text-gray-400" />
            </motion.a>
            
            <motion.a
              variants={cardVariants}
              whileHover={{ scale: 1.05, y: -5 }}
              whileTap={{ scale: 0.95 }}
              href="#"
              className="flex items-center justify-center space-x-2 p-4 bg-space-700/50 rounded-lg border border-space-600 hover:bg-space-700 transition-colors group"
            >
              <Mail className="h-5 w-5 text-primary-400" />
              <span className="text-white group-hover:text-primary-300">{t('about.contacts')}</span>
              <ExternalLink className="h-4 w-4 text-gray-400" />
            </motion.a>
          </motion.div>
        </motion.div>
      </div>
    </div>
  )
}
