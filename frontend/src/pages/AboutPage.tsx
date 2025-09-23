import { motion } from 'framer-motion'
import { 
  Brain, 
  Radar, 
  Database, 
  Zap, 
  Globe,
  Github,
  ExternalLink,
  Mail,
  BookOpen,
  Award,
  TrendingUp
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
    icon: Radar,
    title: 'Поддержка космических миссий',
    description: 'Полная совместимость с данными ведущих миссий по поиску экзопланет: TESS, Kepler и K2.',
    details: [
      'Автоматическая загрузка данных из MAST',
      'Поддержка каталогов TIC, KIC, EPIC',
      'Обработка различных форматов данных',
      'Адаптация к особенностям каждой миссии'
    ]
  },
  {
    icon: Zap,
    title: 'BLS алгоритм',
    description: 'Оптимизированный Box Least Squares алгоритм для быстрого и эффективного поиска периодических сигналов.',
    details: [
      'Параллельные вычисления на GPU',
      'Адаптивная сетка периодов',
      'Статистическая оценка значимости',
      'Фильтрация ложных сигналов'
    ]
  },
  {
    icon: Database,
    title: 'База знаний',
    description: 'PostgreSQL база данных для хранения результатов анализа и накопления опыта системы.',
    details: [
      'Кэширование результатов анализа',
      'Поиск похожих объектов через embeddings',
      'История пользовательской обратной связи',
      'Метрики производительности моделей'
    ]
  }
]

const team = [
  {
    name: 'Система ИИ',
    role: 'Анализ данных',
    description: 'Автоматический анализ кривых блеска с использованием современных алгоритмов машинного обучения',
    avatar: '🤖'
  },
  {
    name: 'BLS Алгоритм',
    role: 'Поиск сигналов',
    description: 'Быстрый и точный поиск периодических транзитных сигналов в фотометрических данных',
    avatar: '📊'
  },
  {
    name: 'База данных',
    role: 'Хранение знаний',
    description: 'Накопление и структурирование результатов для улучшения качества анализа',
    avatar: '💾'
  }
]

const stats = [
  { label: 'Проанализировано звезд', value: '50,000+', icon: Radar },
  { label: 'Найдено кандидатов', value: '1,200+', icon: Globe },
  { label: 'Точность системы', value: '94.5%', icon: Award },
  { label: 'Среднее время анализа', value: '<30с', icon: TrendingUp },
]

export default function AboutPage() {
  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <div className="inline-block p-4 bg-cosmic-gradient rounded-2xl mb-6">
            <Radar className="h-12 w-12 text-white" />
          </div>
          
          <h1 className="text-4xl md:text-6xl font-bold text-gradient mb-6">
            О проекте Exoplanet AI
          </h1>
          
          <p className="text-xl text-gray-300 max-w-4xl mx-auto leading-relaxed">
            Передовая система искусственного интеллекта для автоматического обнаружения 
            экзопланет в данных космических миссий. Мы объединяем современные алгоритмы 
            машинного обучения с проверенными астрофизическими методами для достижения 
            максимальной точности и скорости анализа.
          </p>
        </motion.div>

        {/* Mission Statement */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card mb-16 text-center"
        >
          <h2 className="text-3xl font-bold text-white mb-6">Наша миссия</h2>
          <p className="text-lg text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Демократизировать доступ к современным методам поиска экзопланет и ускорить 
            научные открытия в области астрономии. Мы стремимся сделать анализ кривых блеска 
            доступным для исследователей всех уровней — от студентов до профессиональных астрономов.
          </p>
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16"
        >
          {stats.map((stat, index) => {
            const Icon = stat.icon
            return (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 + index * 0.1 }}
                className="card text-center"
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
        </motion.div>

        {/* Features */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-16"
        >
          <h2 className="text-3xl font-bold text-white text-center mb-12">
            Технологии и возможности
          </h2>
          
          <div className="grid md:grid-cols-2 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon
              return (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 + index * 0.1 }}
                  className="card"
                >
                  <div className="flex items-start space-x-4">
                    <div className="flex-shrink-0">
                      <div className="inline-flex items-center justify-center w-12 h-12 bg-cosmic-600/20 rounded-lg">
                        <Icon className="h-6 w-6 text-cosmic-400" />
                      </div>
                    </div>
                    
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-white mb-3">
                        {feature.title}
                      </h3>
                      
                      <p className="text-gray-300 mb-4">
                        {feature.description}
                      </p>
                      
                      <ul className="space-y-2">
                        {feature.details.map((detail, detailIndex) => (
                          <li key={detailIndex} className="flex items-center space-x-2 text-sm text-gray-400">
                            <div className="w-1.5 h-1.5 bg-cosmic-400 rounded-full flex-shrink-0" />
                            <span>{detail}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </motion.div>
              )
            })}
          </div>
        </motion.div>

        {/* How it works */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="mb-16"
        >
          <h2 className="text-3xl font-bold text-white text-center mb-12">
            Как это работает
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            <div className="card text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-600/20 rounded-full mb-6">
                <span className="text-2xl font-bold text-primary-400">1</span>
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">
                Загрузка данных
              </h3>
              <p className="text-gray-300">
                Система автоматически загружает кривые блеска из архивов MAST 
                для указанной цели и выполняет предварительную обработку данных.
              </p>
            </div>

            <div className="card text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-cosmic-600/20 rounded-full mb-6">
                <span className="text-2xl font-bold text-cosmic-400">2</span>
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">
                BLS анализ
              </h3>
              <p className="text-gray-300">
                Алгоритм Box Least Squares ищет периодические сигналы транзитов, 
                оценивая различные комбинации периода, эпохи и длительности.
              </p>
            </div>

            <div className="card text-center">
              <div className="inline-flex items-center justify-center w-16 h-16 bg-green-600/20 rounded-full mb-6">
                <span className="text-2xl font-bold text-green-400">3</span>
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">
                ИИ валидация
              </h3>
              <p className="text-gray-300">
                Ансамбль нейронных сетей анализирует найденные сигналы и отфильтровывает 
                ложные срабатывания, предоставляя оценку уверенности.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Team */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mb-16"
        >
          <h2 className="text-3xl font-bold text-white text-center mb-12">
            Компоненты системы
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            {team.map((member, index) => (
              <motion.div
                key={member.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 + index * 0.1 }}
                className="card text-center"
              >
                <div className="text-4xl mb-4">{member.avatar}</div>
                <h3 className="text-xl font-semibold text-white mb-2">
                  {member.name}
                </h3>
                <p className="text-cosmic-400 font-medium mb-3">
                  {member.role}
                </p>
                <p className="text-gray-300 text-sm">
                  {member.description}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Resources */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="card text-center"
        >
          <h2 className="text-3xl font-bold text-white mb-6">
            Ресурсы и документация
          </h2>
          
          <p className="text-lg text-gray-300 mb-8 max-w-3xl mx-auto">
            Изучите документацию, примеры использования и научные публикации 
            для более глубокого понимания методов и алгоритмов системы.
          </p>
          
          <div className="grid md:grid-cols-3 gap-6">
            <a
              href="#"
              className="flex items-center justify-center space-x-2 p-4 bg-space-700/50 rounded-lg border border-space-600 hover:bg-space-700 transition-colors group"
            >
              <BookOpen className="h-5 w-5 text-primary-400" />
              <span className="text-white group-hover:text-primary-300">Документация</span>
              <ExternalLink className="h-4 w-4 text-gray-400" />
            </a>
            
            <a
              href="#"
              className="flex items-center justify-center space-x-2 p-4 bg-space-700/50 rounded-lg border border-space-600 hover:bg-space-700 transition-colors group"
            >
              <Github className="h-5 w-5 text-primary-400" />
              <span className="text-white group-hover:text-primary-300">GitHub</span>
              <ExternalLink className="h-4 w-4 text-gray-400" />
            </a>
            
            <a
              href="#"
              className="flex items-center justify-center space-x-2 p-4 bg-space-700/50 rounded-lg border border-space-600 hover:bg-space-700 transition-colors group"
            >
              <Mail className="h-5 w-5 text-primary-400" />
              <span className="text-white group-hover:text-primary-300">Контакты</span>
              <ExternalLink className="h-4 w-4 text-gray-400" />
            </a>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
