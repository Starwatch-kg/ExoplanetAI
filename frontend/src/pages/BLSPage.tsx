import { useState } from 'react'
import { motion } from 'framer-motion'
import { useForm } from 'react-hook-form'
import { 
  Zap, 
  Settings, 
  Target, 
  BarChart3,
  ChevronDown,
  Info,
  Clock,
  TrendingUp
} from 'lucide-react'
import ApiService from '../services/api'
import { useAppStore } from '../store/useAppStore'
import LoadingSpinner from '../components/ui/LoadingSpinner'

interface BLSFormData {
  target_name: string
  catalog: 'TIC' | 'KIC' | 'EPIC'
  mission: 'TESS' | 'Kepler' | 'K2'
  period_min: number
  period_max: number
  duration_min: number
  duration_max: number
  snr_threshold: number
  use_enhanced: boolean
}

interface BLSResult {
  target_name: string
  best_period: number
  best_t0: number
  best_duration: number
  best_power: number
  snr: number
  depth: number
  depth_err: number
  significance: number
  is_significant: boolean
  enhanced_analysis: boolean
  ml_confidence: number
  physical_validation: boolean
  processing_time_ms: number
  request_id?: string
  trace_id?: string
}

export default function BLSPage() {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [blsResults, setBLSResults] = useState<BLSResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const { addToast } = useAppStore()

  const { register, handleSubmit, watch, setValue, formState: { errors } } = useForm<BLSFormData>({
    defaultValues: {
      target_name: '',
      catalog: 'TIC',
      mission: 'TESS',
      period_min: 0.5,
      period_max: 20.0,
      duration_min: 0.05,
      duration_max: 0.3,
      snr_threshold: 7.0,
      use_enhanced: true
    }
  })

  const watchedCatalog = watch('catalog')
  const watchedEnhanced = watch('use_enhanced')

  const handleBLSAnalysis = async (data: BLSFormData) => {
    try {
      setIsLoading(true)
      addToast({
        type: 'info',
        title: 'Запуск BLS анализа',
        message: 'Загружаем данные и выполняем поиск транзитов...'
      })

      const result = await ApiService.analyzeBLS({
        target_name: data.target_name,
        catalog: data.catalog,
        mission: data.mission,
        period_min: data.period_min,
        period_max: data.period_max,
        duration_min: data.duration_min,
        duration_max: data.duration_max,
        snr_threshold: data.snr_threshold,
        use_enhanced: data.use_enhanced
      })

      setBLSResults(result)
      setIsLoading(false)
      
      if (result.is_significant) {
        addToast({
          type: 'success',
          title: 'Транзит обнаружен!',
          message: `Найден значимый сигнал с SNR ${result.snr.toFixed(1)}`
        })
      } else {
        addToast({
          type: 'info',
          title: 'Анализ завершен',
          message: 'Значимых транзитных сигналов не обнаружено'
        })
      }
    } catch (error: any) {
      setIsLoading(false)
      addToast({
        type: 'error',
        title: 'Ошибка BLS анализа',
        message: error.message || 'Произошла ошибка при выполнении BLS анализа'
      })
    }
  }

  const onSubmit = (data: BLSFormData) => {
    if (!data.target_name.trim()) {
      addToast({
        type: 'warning',
        title: 'Укажите цель',
        message: 'Введите название звезды или идентификатор'
      })
      return
    }
    handleBLSAnalysis(data)
  }

  // Auto-set mission based on catalog
  const handleCatalogChange = (catalog: string) => {
    setValue('catalog', catalog as any)
    if (catalog === 'TIC') setValue('mission', 'TESS')
    else if (catalog === 'KIC') setValue('mission', 'Kepler')
    else if (catalog === 'EPIC') setValue('mission', 'K2')
  }

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="nasa-title">
            BLS Анализ
          </h1>
          <p className="nasa-subtitle mb-6">
            Box Least Squares Transit Detection
          </p>
          <p className="nasa-text text-lg max-w-4xl mx-auto">
            Профессиональный алгоритм поиска транзитных экзопланет с использованием 
            метода наименьших квадратов и расширенной статистической обработки
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* BLS Configuration Panel */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="nasa-card sticky top-24">
              <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
                {/* Target Input */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    <Target className="inline h-4 w-4 mr-1" />
                    Цель для анализа
                  </label>
                  <input
                    {...register('target_name', { required: 'Укажите название цели' })}
                    type="text"
                    placeholder="TIC 441420236, HD 209458..."
                    className="nasa-input"
                  />
                  {errors.target_name && (
                    <p className="text-red-400 text-sm mt-1">{errors.target_name.message}</p>
                  )}
                </div>

                {/* Catalog and Mission */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Каталог
                    </label>
                    <select
                      {...register('catalog')}
                      onChange={(e) => handleCatalogChange(e.target.value)}
                      className="nasa-input"
                    >
                      <option value="TIC">TIC (TESS)</option>
                      <option value="KIC">KIC (Kepler)</option>
                      <option value="EPIC">EPIC (K2)</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Миссия
                    </label>
                    <select
                      {...register('mission')}
                      className="nasa-input"
                    >
                      <option value="TESS">TESS</option>
                      <option value="Kepler">Kepler</option>
                      <option value="K2">K2</option>
                    </select>
                  </div>
                </div>

                {/* Enhanced Analysis Toggle */}
                <div className="flex items-center justify-between p-4 bg-blue-900/20 rounded-lg border border-blue-500/30">
                  <div className="flex items-center space-x-3">
                    <TrendingUp className="h-5 w-5 text-blue-400" />
                    <div>
                      <p className="text-white font-medium">Расширенный анализ</p>
                      <p className="text-sm text-gray-400">Дополнительная валидация</p>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      {...register('use_enhanced')}
                      type="checkbox"
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-space-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                  </label>
                </div>

                {/* Advanced Settings */}
                <div>
                  <button
                    type="button"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center justify-between w-full p-3 bg-space-700/30 rounded-lg border border-space-600 hover:bg-space-700/50 transition-colors"
                  >
                    <div className="flex items-center space-x-2">
                      <Settings className="h-4 w-4" />
                      <span className="text-sm font-medium">Параметры BLS</span>
                    </div>
                    <ChevronDown className={`h-4 w-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`} />
                  </button>

                  {showAdvanced && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-4 space-y-4 p-4 bg-space-800/30 rounded-lg border border-space-600"
                    >
                      {/* Period Range */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Диапазон периодов (дни)
                        </label>
                        <div className="grid grid-cols-2 gap-2">
                          <input
                            {...register('period_min', { min: 0.1, max: 100 })}
                            type="number"
                            step="0.1"
                            placeholder="Мин"
                            className="nasa-input"
                          />
                          <input
                            {...register('period_max', { min: 0.1, max: 100 })}
                            type="number"
                            step="0.1"
                            placeholder="Макс"
                            className="nasa-input"
                          />
                        </div>
                      </div>

                      {/* Duration Range */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Длительность транзита (дни)
                        </label>
                        <div className="grid grid-cols-2 gap-2">
                          <input
                            {...register('duration_min', { min: 0.01, max: 1 })}
                            type="number"
                            step="0.01"
                            placeholder="Мин"
                            className="nasa-input"
                          />
                          <input
                            {...register('duration_max', { min: 0.01, max: 1 })}
                            type="number"
                            step="0.01"
                            placeholder="Макс"
                            className="nasa-input"
                          />
                        </div>
                      </div>

                      {/* SNR Threshold */}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-2">
                          Порог SNR
                        </label>
                        <input
                          {...register('snr_threshold', { min: 1, max: 20 })}
                          type="number"
                          step="0.1"
                          className="nasa-input"
                        />
                        <p className="text-xs text-gray-400 mt-1">
                          Минимальное отношение сигнал/шум для обнаружения
                        </p>
                      </div>
                    </motion.div>
                  )}
                </div>

                {/* Submit Button */}
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full nasa-btn nasa-btn-success py-3 flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <LoadingSpinner size="sm" />
                  ) : (
                    <>
                      <Zap className="h-5 w-5" />
                      <span>Запустить BLS анализ</span>
                    </>
                  )}
                </button>
              </form>

              {/* Info Panel */}
              <div className="mt-6 p-4 bg-blue-600/10 rounded-lg border border-blue-500/30">
                <div className="flex items-start space-x-2">
                  <Info className="h-4 w-4 text-blue-400 mt-0.5 flex-shrink-0" />
                  <div className="text-sm">
                    <p className="text-blue-300 font-medium mb-1">О BLS алгоритме</p>
                    <p className="text-gray-300">
                      Box Least Squares - это мощный статистический метод для обнаружения 
                      периодических транзитных сигналов в фотометрических данных.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Results Area */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-2"
          >
            {isLoading && (
              <div className="nasa-card text-center py-12">
                <LoadingSpinner 
                  size="lg" 
                  variant="cosmic"
                  message="Выполняем BLS анализ..."
                />
                <p className="text-gray-400 mt-4">
                  Поиск транзитных сигналов может занять несколько минут
                </p>
              </div>
            )}

            {blsResults && (
              <div className="space-y-6">
                {/* Results Summary */}
                <div className="nasa-card">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-xl font-semibold text-white">
                      BLS Результаты: {blsResults.target_name}
                    </h3>
                    <div className="flex items-center space-x-2">
                      <Clock className="h-4 w-4 text-gray-400" />
                      <span className="text-sm text-gray-400">
                        {(blsResults.processing_time_ms / 1000).toFixed(1)}с
                      </span>
                    </div>
                  </div>
                  
                  {/* Status Indicator */}
                  <div className={`p-4 rounded-lg border mb-6 ${
                    blsResults.is_significant 
                      ? 'bg-green-900/20 border-green-500/30' 
                      : 'bg-gray-900/20 border-gray-500/30'
                  }`}>
                    <div className="flex items-center space-x-3">
                      <div className={`w-4 h-4 rounded-full ${
                        blsResults.is_significant ? 'bg-green-400' : 'bg-gray-400'
                      }`} />
                      <div>
                        <p className={`font-medium ${
                          blsResults.is_significant ? 'text-green-300' : 'text-gray-300'
                        }`}>
                          {blsResults.is_significant 
                            ? '✅ Обнаружен значимый транзитный сигнал' 
                            : '❌ Значимых сигналов не найдено'}
                        </p>
                        <p className="text-sm text-gray-400">
                          SNR: {blsResults.snr.toFixed(2)} | 
                          Значимость: {(blsResults.significance * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Detailed Results */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    <div className="text-center">
                      <p className="text-3xl font-bold text-blue-400 font-mono">
                        {blsResults.best_period.toFixed(3)}
                      </p>
                      <p className="text-sm text-gray-400">Период (дни)</p>
                    </div>
                    <div className="text-center">
                      <p className="text-3xl font-bold text-green-400 font-mono">
                        {blsResults.snr.toFixed(1)}
                      </p>
                      <p className="text-sm text-gray-400">SNR</p>
                    </div>
                    <div className="text-center">
                      <p className="text-3xl font-bold text-yellow-400 font-mono">
                        {(blsResults.depth * 100).toFixed(3)}%
                      </p>
                      <p className="text-sm text-gray-400">Глубина</p>
                    </div>
                    <div className="text-center">
                      <p className="text-3xl font-bold text-purple-400 font-mono">
                        {(blsResults.best_duration * 24).toFixed(1)}
                      </p>
                      <p className="text-sm text-gray-400">Длительность (ч)</p>
                    </div>
                  </div>

                  {/* Additional Parameters */}
                  <div className="mt-6 grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-gray-400">Эпоха транзита</p>
                      <p className="text-white font-mono">{blsResults.best_t0.toFixed(3)} BJD</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Мощность сигнала</p>
                      <p className="text-white font-mono">{blsResults.best_power.toFixed(3)}</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Ошибка глубины</p>
                      <p className="text-white font-mono">±{(blsResults.depth_err * 100).toFixed(4)}%</p>
                    </div>
                  </div>

                  {/* Enhanced Analysis Results */}
                  {blsResults.enhanced_analysis && (
                    <div className="mt-6 p-4 bg-purple-900/20 rounded-lg border border-purple-500/30">
                      <h4 className="font-medium text-purple-300 mb-3">Расширенный анализ</h4>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="text-gray-400">ML уверенность</p>
                          <p className="text-white font-mono">{(blsResults.ml_confidence * 100).toFixed(1)}%</p>
                        </div>
                        <div>
                          <p className="text-gray-400">Физическая валидация</p>
                          <p className="text-white font-mono">
                            {blsResults.physical_validation ? '✅ Пройдена' : '❌ Не пройдена'}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {!isLoading && !blsResults && (
              <div className="nasa-card text-center py-12">
                <div className="mb-4">
                  <BarChart3 className="h-16 w-16 text-gray-600 mx-auto mb-4" />
                </div>
                <h3 className="text-xl font-semibold text-gray-400 mb-2">
                  Готов к BLS анализу
                </h3>
                <p className="text-gray-500">
                  Настройте параметры и запустите поиск транзитных сигналов
                </p>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  )
}
