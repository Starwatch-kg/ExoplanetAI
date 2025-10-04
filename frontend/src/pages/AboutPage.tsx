import { useState, useEffect } from 'react'
import { Brain, Zap, Database, Radar, Globe, Award, TrendingUp, Users, Clock, HardDrive } from 'lucide-react'
import { typedApiClient } from '../utils/typedApiClient'
import type { SystemStatistics } from '../types/api'

interface AboutPageProps {
  useSimpleBackground?: boolean
}

export default function AboutPage({ useSimpleBackground = false }: AboutPageProps) {
  const [stats, setStats] = useState<SystemStatistics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchStatistics = async () => {
      try {
        setLoading(true)
        const data = await typedApiClient.getStatistics()
        setStats(data)
        setError(null)
      } catch (err) {
        console.error('Failed to fetch statistics:', err)
        setError('Не удалось подключиться к серверу')
        setStats(null)
      } finally {
        setLoading(false)
      }
    }

    fetchStatistics()
  }, [])

  const formatNumber = (num: number): string => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M+'
    } else if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K+'
    }
    return num.toString()
  }

  const formatTime = (seconds: number): string => {
    if (seconds < 1) {
      return `${(seconds * 1000).toFixed(0)}ms`
    }
    return `${seconds.toFixed(1)}s`
  }
  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <div className="inline-block p-4 bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl mb-6">
            <Brain className="w-12 h-12 text-white" />
          </div>
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
            О проекте ExoplanetAI
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Передовая система искусственного интеллекта для обнаружения экзопланет с использованием данных телескопов TESS и Kepler
          </p>
        </div>

        {/* Stats */}
        <div className={`grid grid-cols-2 md:grid-cols-4 gap-6 mb-16 ${useSimpleBackground ? 'opacity-90' : 'opacity-100'}`}>
          <div className={`backdrop-blur-sm border rounded-xl p-6 text-center transition-all duration-300 ${useSimpleBackground ? 'bg-gray-800/30 border-gray-600 hover:bg-gray-700/30' : 'bg-gray-800/50 border-gray-700 hover:bg-gray-700/50 hover:shadow-blue-glow'}`}>
            <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg mb-4 ${useSimpleBackground ? 'bg-blue-500/15' : 'bg-blue-600/20 shadow-blue-glow'}`}>
              <Radar className={`h-6 w-6 ${useSimpleBackground ? 'text-blue-300' : 'text-blue-400 drop-shadow-blue-glow animate-pulse'}`} />
            </div>
            <div className="text-2xl md:text-3xl font-bold text-white mb-2">
              {loading ? '...' : (stats ? formatNumber(stats.stars_analyzed) : 'N/A')}
            </div>
            <div className="text-sm text-gray-400">
              Звезд проанализировано
            </div>
          </div>

          <div className={`backdrop-blur-sm border rounded-xl p-6 text-center transition-all duration-300 ${useSimpleBackground ? 'bg-gray-800/30 border-gray-600 hover:bg-gray-700/30' : 'bg-gray-800/50 border-gray-700 hover:bg-gray-700/50 hover:shadow-green-glow'}`}>
            <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg mb-4 ${useSimpleBackground ? 'bg-green-500/15' : 'bg-green-600/20 shadow-green-glow'}`}>
              <Globe className={`h-6 w-6 ${useSimpleBackground ? 'text-green-300' : 'text-green-400 drop-shadow-green-glow animate-pulse'}`} />
            </div>
            <div className="text-2xl md:text-3xl font-bold text-white mb-2">
              {loading ? '...' : (stats ? formatNumber(stats.candidates_found) : 'N/A')}
            </div>
            <div className="text-sm text-gray-400">
              Кандидатов найдено
            </div>
          </div>

          <div className={`backdrop-blur-sm border rounded-xl p-6 text-center transition-all duration-300 ${useSimpleBackground ? 'bg-gray-800/30 border-gray-600 hover:bg-gray-700/30' : 'bg-gray-800/50 border-gray-700 hover:bg-gray-700/50 hover:shadow-amber-glow'}`}>
            <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg mb-4 ${useSimpleBackground ? 'bg-amber-500/15' : 'bg-amber-600/20 shadow-amber-glow'}`}>
              <Award className={`h-6 w-6 ${useSimpleBackground ? 'text-amber-300' : 'text-amber-400 drop-shadow-glow animate-pulse'}`} />
            </div>
            <div className="text-2xl md:text-3xl font-bold text-white mb-2">
              {loading ? '...' : (stats ? `${stats.system_accuracy.toFixed(1)}%` : 'N/A')}
            </div>
            <div className="text-sm text-gray-400">
              Точность системы
            </div>
          </div>

          <div className={`backdrop-blur-sm border rounded-xl p-6 text-center transition-all duration-300 ${useSimpleBackground ? 'bg-gray-800/30 border-gray-600 hover:bg-gray-700/30' : 'bg-gray-800/50 border-gray-700 hover:bg-gray-700/50 hover:shadow-purple-glow'}`}>
            <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg mb-4 ${useSimpleBackground ? 'bg-purple-500/15' : 'bg-purple-600/20 shadow-purple-glow'}`}>
              <Clock className={`h-6 w-6 ${useSimpleBackground ? 'text-purple-300' : 'text-purple-400 drop-shadow-purple-glow animate-pulse'}`} />
            </div>
            <div className="text-2xl md:text-3xl font-bold text-white mb-2">
              {loading ? '...' : (stats ? formatTime(stats.average_processing_time_seconds) : 'N/A')}
            </div>
            <div className="text-sm text-gray-400">
              Среднее время анализа
            </div>
          </div>
        </div>

        {/* Additional Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 mb-16">
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 text-center hover:bg-gray-700/50 transition-colors">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-emerald-600/20 rounded-lg mb-4">
              <Award className="h-6 w-6 text-emerald-400" />
            </div>
            <div className="text-2xl md:text-3xl font-bold text-white mb-2">
              {loading ? '...' : (stats ? formatNumber(stats.confirmed_planets) : 'N/A')}
            </div>
            <div className="text-sm text-gray-400">
              Подтвержденных планет
            </div>
          </div>

          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 text-center hover:bg-gray-700/50 transition-colors">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-cyan-600/20 rounded-lg mb-4">
              <TrendingUp className="h-6 w-6 text-cyan-400" />
            </div>
            <div className="text-2xl md:text-3xl font-bold text-white mb-2">
              {loading ? '...' : (stats ? formatNumber(stats.total_searches) : 'N/A')}
            </div>
            <div className="text-sm text-gray-400">
              Всего поисков
            </div>
          </div>

          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 text-center hover:bg-gray-700/50 transition-colors">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-orange-600/20 rounded-lg mb-4">
              <Users className="h-6 w-6 text-orange-400" />
            </div>
            <div className="text-2xl md:text-3xl font-bold text-white mb-2">
              {loading ? '...' : (stats ? formatNumber(stats.active_users) : 'N/A')}
            </div>
            <div className="text-sm text-gray-400">
              Активных пользователей
            </div>
          </div>

          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 text-center hover:bg-gray-700/50 transition-colors">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-pink-600/20 rounded-lg mb-4">
              <HardDrive className="h-6 w-6 text-pink-400" />
            </div>
            <div className="text-2xl md:text-3xl font-bold text-white mb-2">
              {loading ? '...' : (stats ? `${stats.database_size_gb.toFixed(1)}GB` : 'N/A')}
            </div>
            <div className="text-sm text-gray-400">
              Размер базы данных
            </div>
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4 mb-8 text-center">
            <p className="text-red-400 text-sm">{error}</p>
            <p className="text-gray-500 text-xs mt-1">Статистика недоступна</p>
          </div>
        )}

        {/* Last updated */}
        {stats && !loading && (
          <div className="text-center mb-16">
            <p className="text-gray-500 text-sm">
              Последнее обновление: {new Date(stats.last_updated).toLocaleString('ru-RU')}
            </p>
          </div>
        )}

        {/* Features */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-blue-600/20 rounded-lg">
                <Brain className="w-6 h-6 text-blue-400" />
              </div>
              <h3 className="text-xl font-semibold text-white">
                Искусственный интеллект
              </h3>
            </div>
            <p className="text-gray-300 mb-4">
              Современные нейронные сети (CNN, LSTM, Transformers) для точного обнаружения экзопланет с минимальным количеством ложных срабатываний.
            </p>
            <ul className="space-y-2">
              <li className="text-sm text-gray-400 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full"></span>
                Ансамбль из трех типов моделей
              </li>
              <li className="text-sm text-gray-400 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full"></span>
                Transfer learning с данных Kepler на TESS
              </li>
              <li className="text-sm text-gray-400 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-blue-400 rounded-full"></span>
                Active learning с пользовательской обратной связью
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-amber-600/20 rounded-lg shadow-amber-glow">
                <Zap className="w-6 h-6 text-amber-400 drop-shadow-glow" />
              </div>
              <h3 className="text-xl font-semibold text-white">
                Высокая производительность
              </h3>
            </div>
            <p className="text-gray-300 mb-4">
              Оптимизированные алгоритмы с поддержкой GPU ускорения и параллельной обработки для быстрого анализа больших объемов данных.
            </p>
            <ul className="space-y-2">
              <li className="text-sm text-gray-400 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-amber-400 rounded-full shadow-amber-glow"></span>
                GPU ускорение через CUDA
              </li>
              <li className="text-sm text-gray-400 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-amber-400 rounded-full shadow-amber-glow"></span>
                Параллельная обработка множественных целей
              </li>
              <li className="text-sm text-gray-400 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-amber-400 rounded-full shadow-amber-glow"></span>
                Оптимизированные алгоритмы BLS и GPI
              </li>
            </ul>
          </div>

          <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-green-600/20 rounded-lg">
                <Database className="w-6 h-6 text-green-400" />
              </div>
              <h3 className="text-xl font-semibold text-white">
                Интеграция с NASA
              </h3>
            </div>
            <p className="text-gray-300 mb-4">
              Прямая интеграция с архивами NASA MAST для автоматической загрузки и обработки данных телескопов TESS и Kepler.
            </p>
            <ul className="space-y-2">
              <li className="text-sm text-gray-400 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-green-400 rounded-full"></span>
                Автоматическая загрузка light curves
              </li>
              <li className="text-sm text-gray-400 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-green-400 rounded-full"></span>
                Поддержка TESS, Kepler, K2 данных
              </li>
              <li className="text-sm text-gray-400 flex items-center gap-2">
                <span className="w-1.5 h-1.5 bg-green-400 rounded-full"></span>
                Предобработка и нормализация данных
              </li>
            </ul>
          </div>
        </div>

        {/* Contact */}
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-6">
            Свяжитесь с нами
          </h2>
          <p className="text-lg text-gray-300 mb-8 max-w-3xl mx-auto">
            Заинтересованы в сотрудничестве или хотите узнать больше о нашем проекте? Мы всегда рады новым контактам!
          </p>
          <button className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
            Связаться с командой
          </button>
        </div>
      </div>
    </div>
  )
}
