import { useState } from 'react'
import { motion } from 'framer-motion'
import { useForm } from 'react-hook-form'
import { useEffect } from 'react'
import { 
  Search, 
  Settings, 
  Target, 
  Calendar,
  Zap,
  Brain,
  ChevronDown,
  Info
} from 'lucide-react'
import { SearchRequest, SearchResult } from '../types/api'
import ApiService from '../services/api'
import { useAppStore } from '../store/useAppStore'
import LoadingSpinner from '../components/ui/LoadingSpinner'
import LightCurveChart from '../components/charts/LightCurveChart'

interface SearchFormData {
  target_name: string
  catalog: 'TIC' | 'KIC' | 'EPIC'
  mission: 'TESS' | 'Kepler' | 'K2'
  period_min: number
  period_max: number
  duration_min: number
  duration_max: number
  snr_threshold: number
  use_ai: boolean
}

export default function SearchPage() {
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [searchResults, setSearchResults] = useState<SearchResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const { addToast, setIsSearching } = useAppStore()

  const { register, handleSubmit, watch, setValue, formState: { errors } } = useForm<SearchFormData>({
    defaultValues: {
      target_name: '',
      catalog: 'TIC',
      mission: 'TESS',
      period_min: 0.5,
      period_max: 20.0,
      duration_min: 0.05,
      duration_max: 0.3,
      snr_threshold: 7.0,
      use_ai: true
    }
  })

  const watchedCatalog = watch('catalog')
  const watchedUseAI = watch('use_ai')

  // Get available catalogs
  const [catalogsData, setCatalogsData] = useState<any>(null)
  
  useEffect(() => {
    ApiService.getCatalogs()
      .then(setCatalogsData)
      .catch(() => {
        addToast({
          type: 'error',
          title: 'Ошибка загрузки',
          message: 'Не удалось загрузить список каталогов'
        })
      })
  }, [])

  // Search function
  const handleSearch = async (data: SearchFormData) => {
    try {
      setIsLoading(true)
      setIsSearching(true)
      addToast({
        type: 'info',
        title: 'Начинаем поиск',
        message: 'Загружаем данные и выполняем анализ...'
      })

      const request: SearchRequest = {
        target_name: data.target_name,
        catalog: data.catalog,
        mission: data.mission,
        period_min: data.period_min,
        period_max: data.period_max,
        duration_min: data.duration_min,
        duration_max: data.duration_max,
        snr_threshold: data.snr_threshold
      }

      const result = data.use_ai 
        ? await ApiService.aiEnhancedSearch(request)
        : await ApiService.searchExoplanets(request)

      setSearchResults(result)
      setIsLoading(false)
      setIsSearching(false)
      addToast({
        type: 'success',
        title: 'Поиск завершен',
        message: `Найдено ${result.candidates?.length || 0} кандидатов`
      })
    } catch (error: any) {
      setIsLoading(false)
      setIsSearching(false)
      addToast({
        type: 'error',
        title: 'Ошибка поиска',
        message: error.message || 'Произошла ошибка при поиске экзопланет'
      })
    }
  }

  const onSubmit = (data: SearchFormData) => {
    if (!data.target_name.trim()) {
      addToast({
        type: 'warning',
        title: 'Укажите цель',
        message: 'Введите название звезды или идентификатор'
      })
      return
    }
    handleSearch(data)
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
          <h1 className="text-4xl md:text-5xl font-bold text-gradient mb-4">
            Поиск экзопланет
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Введите название звезды и настройте параметры поиска для обнаружения транзитных экзопланет
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Search Form */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="lg:col-span-1"
          >
            <div className="card sticky top-24">
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
                    placeholder="TIC 441420236, HD 209458, Kepler-452..."
                    className="input-field w-full"
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
                      className="input-field w-full"
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
                      className="input-field w-full"
                    >
                      <option value="TESS">TESS</option>
                      <option value="Kepler">Kepler</option>
                      <option value="K2">K2</option>
                    </select>
                  </div>
                </div>

                {/* AI Toggle */}
                <div className="flex items-center justify-between p-4 bg-space-700/30 rounded-lg border border-space-600">
                  <div className="flex items-center space-x-3">
                    <Brain className="h-5 w-5 text-cosmic-400" />
                    <div>
                      <p className="text-white font-medium">ИИ-анализ</p>
                      <p className="text-sm text-gray-400">Использовать нейронные сети</p>
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      {...register('use_ai')}
                      type="checkbox"
                      className="sr-only peer"
                    />
                    <div className="w-11 h-6 bg-space-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-cosmic-600"></div>
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
                      <span className="text-sm font-medium">Расширенные настройки</span>
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
                            className="input-field"
                          />
                          <input
                            {...register('period_max', { min: 0.1, max: 100 })}
                            type="number"
                            step="0.1"
                            placeholder="Макс"
                            className="input-field"
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
                            className="input-field"
                          />
                          <input
                            {...register('duration_max', { min: 0.01, max: 1 })}
                            type="number"
                            step="0.01"
                            placeholder="Макс"
                            className="input-field"
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
                          className="input-field w-full"
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
                  className="w-full btn-cosmic py-3 flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <LoadingSpinner size="sm" />
                  ) : (
                    <>
                      {watchedUseAI ? <Brain className="h-5 w-5" /> : <Search className="h-5 w-5" />}
                      <span>
                        {watchedUseAI ? 'ИИ-поиск экзопланет' : 'Поиск экзопланет'}
                      </span>
                    </>
                  )}
                </button>
              </form>

              {/* Info Panel */}
              <div className="mt-6 p-4 bg-primary-600/10 rounded-lg border border-primary-500/30">
                <div className="flex items-start space-x-2">
                  <Info className="h-4 w-4 text-primary-400 mt-0.5 flex-shrink-0" />
                  <div className="text-sm">
                    <p className="text-primary-300 font-medium mb-1">Совет</p>
                    <p className="text-gray-300">
                      Используйте ИИ-анализ для повышения точности обнаружения. 
                      Система автоматически отфильтрует ложные срабатывания.
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
              <div className="card text-center py-12">
                <LoadingSpinner 
                  size="lg" 
                  variant="cosmic"
                  message="Анализируем кривую блеска..."
                />
              </div>
            )}

            {searchResults && (
              <div className="space-y-6">
                {/* Results Summary */}
                <div className="card">
                  <h3 className="text-xl font-semibold text-white mb-4">
                    Результаты анализа: {searchResults.target_name}
                  </h3>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-cosmic-400">
                        {searchResults.candidates?.length || 0}
                      </p>
                      <p className="text-sm text-gray-400">Кандидатов найдено</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-primary-400">
                        {searchResults.bls_results?.snr?.toFixed(1) || 'N/A'}
                      </p>
                      <p className="text-sm text-gray-400">Лучший SNR</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-green-400">
                        {searchResults.bls_results?.best_period?.toFixed(2) || 'N/A'}
                      </p>
                      <p className="text-sm text-gray-400">Период (дни)</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-yellow-400">
                        {searchResults.ai_analysis?.confidence ? 
                          `${(searchResults.ai_analysis.confidence * 100).toFixed(1)}%` : 'N/A'}
                      </p>
                      <p className="text-sm text-gray-400">Уверенность ИИ</p>
                    </div>
                  </div>

                  {searchResults.ai_analysis?.explanation && (
                    <div className="p-4 bg-space-700/30 rounded-lg border border-space-600">
                      <h4 className="font-medium text-white mb-2">Объяснение ИИ:</h4>
                      <p className="text-gray-300 text-sm">
                        {searchResults.ai_analysis.explanation}
                      </p>
                    </div>
                  )}
                </div>

                {/* Light Curve Chart */}
                {searchResults.lightcurve_data && (
                  <LightCurveChart
                    data={searchResults.lightcurve_data}
                    candidates={searchResults.candidates}
                    height={500}
                  />
                )}
              </div>
            )}

            {!isLoading && !searchResults && (
              <div className="card text-center py-12">
                <div className="mb-4">
                  <Search className="h-16 w-16 text-gray-600 mx-auto mb-4" />
                </div>
                <h3 className="text-xl font-semibold text-gray-400 mb-2">
                  Готовы к поиску
                </h3>
                <p className="text-gray-500">
                  Введите название звезды и нажмите кнопку поиска для начала анализа
                </p>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  )
}
