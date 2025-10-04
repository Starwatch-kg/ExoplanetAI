import React, { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { motion } from 'framer-motion'
import { AlertCircle, Search, BarChart3, Settings, TrendingUp, Download, Info, Layers, Activity, Sparkles, Target, Clock, CheckCircle, Play, Star, ArrowRight } from 'lucide-react'
import SafePageWrapper from '../components/SafePageWrapper'

interface SearchParameters {
  target_name: string
  period_min: number
  period_max: number
  snr_threshold: number
  search_mode: 'bls' | 'ensemble' | 'hybrid'
}

interface SearchResult {
  search_completed: boolean
  unified_result: any
  search_info: any
  performance: any
  timestamp: string
}

const SearchPageContent: React.FC = () => {
  const { t } = useTranslation()
  const [parameters, setParameters] = useState<SearchParameters>({
    target_name: '',
    period_min: 0.5,
    period_max: 50.0,
    snr_threshold: 7.0,
    search_mode: 'bls'
  })
  
  const [result, setResult] = useState<SearchResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Безопасная функция для переводов с fallback
  const safeT = (key: string, fallback: string) => {
    try {
      const translation = t(key)
      return translation === key ? fallback : translation
    } catch {
      return fallback
    }
  }

  const searchMethods = [
    {
      id: 'bls',
      name: safeT('search.methods.bls.name', 'BLS Algorithm'),
      icon: BarChart3,
      description: safeT('search.methods.bls.description', 'Box Least Squares algorithm for transit detection'),
      features: ['Fast processing', 'High accuracy', 'Suitable for most cases'],
      bestFor: safeT('search.methods.bls.bestFor', 'Standard exoplanet searches'),
      processingTime: safeT('search.methods.bls.processingTime', '< 30 seconds'),
      accuracy: safeT('search.methods.bls.accuracy', '~85%'),
      color: 'blue'
    },
    {
      id: 'ensemble',
      name: safeT('search.methods.ensemble.name', 'Ensemble Method'),
      icon: Layers,
      description: safeT('search.methods.ensemble.description', 'Combined multiple algorithms for better accuracy'),
      features: ['Multiple algorithms', 'Higher accuracy', 'Robust detection'],
      bestFor: safeT('search.methods.ensemble.bestFor', 'Complex or weak signals'),
      processingTime: safeT('search.methods.ensemble.processingTime', '1-2 minutes'),
      accuracy: safeT('search.methods.ensemble.accuracy', '~92%'),
      color: 'purple'
    },
    {
      id: 'hybrid',
      name: safeT('search.methods.hybrid.name', 'Hybrid Approach'),
      icon: Settings,
      description: safeT('search.methods.hybrid.description', 'Adaptive method selection based on data characteristics'),
      features: ['Adaptive selection', 'Optimal performance', 'Smart processing'],
      bestFor: safeT('search.methods.hybrid.bestFor', 'Unknown or varied signals'),
      processingTime: safeT('search.methods.hybrid.processingTime', '30s - 2min'),
      accuracy: safeT('search.methods.hybrid.accuracy', '~90%'),
      color: 'green'
    }
  ]

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Валидация пользовательского ввода перед отправкой запроса
      if (!parameters.target_name || parameters.target_name.trim() === '') {
        throw new Error('Target name is required');
      }
      
      // Санитизация target_name для предотвращения XSS
      const sanitizedTargetName = parameters.target_name.replace(/[<>'"&]/g, (match) => {
        const escapeMap: Record<string, string> = {
          '<': '<',
          '>': '>',
          '"': '"',
          "'": '&#x27;',
          '&': '&'
        };
        return escapeMap[match] || match;
      });

      // Сначала получаем реальные данные из NASA API
      const dataResponse = await fetch('/api/v1/data/lightcurve', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'XMLHttpRequest'
        },
        body: JSON.stringify({
          target_name: sanitizedTargetName,
          mission: 'TESS'
        })
      })

      if (!dataResponse.ok) {
        throw new Error(`Failed to fetch lightcurve data: ${dataResponse.status}`)
      }

      const lightcurveData = await dataResponse.json()
      
      if (!lightcurveData.time_data || !lightcurveData.flux_data) {
        throw new Error('Invalid lightcurve data received')
      }

      // Теперь запускаем поиск с реальными данными
      const response = await fetch('/api/v1/search/unified', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'XMLHttpRequest'
        },
        body: JSON.stringify({
          target_name: sanitizedTargetName,
          time_data: lightcurveData.time_data,
          flux_data: lightcurveData.flux_data,
          flux_err_data: lightcurveData.flux_err_data,
          search_mode: parameters.search_mode,
          period_min: parameters.period_min,
          period_max: parameters.period_max,
          snr_threshold: parameters.snr_threshold,
          use_parallel: true
        })
      })

      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`)
      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Search failed')
    } finally {
      setLoading(false)
    }
  }

  const handleParameterChange = (key: keyof SearchParameters, value: any) => {
    setParameters(prev => ({ ...prev, [key]: value }))
  }

  const getColorClasses = (color: string) => {
    switch (color) {
      case 'blue':
        return {
          bg: 'bg-blue-500/20',
          border: 'border-blue-500/50',
          text: 'text-blue-400',
          hover: 'hover:bg-blue-500/30'
        }
      case 'purple':
        return { bg: 'bg-purple-500/20', border: 'border-purple-500/50', text: 'text-purple-400', hover: 'hover:bg-purple-500/30' }
      case 'green':
        return { bg: 'bg-green-500/20', border: 'border-green-500/50', text: 'text-green-400', hover: 'hover:bg-green-500/30' }
      default:
        return { bg: 'bg-gray-500/20', border: 'border-gray-500/50', text: 'text-gray-400', hover: 'hover:bg-gray-500/30' }
    }
  }

  const presetTargets = [
    { name: 'TIC 441420236', description: 'Known exoplanet host' },
    { name: 'TIC 307210830', description: 'TOI candidate' }
  ]

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
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
            className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-blue-500 via-purple-600 to-blue-700 rounded-full mb-6 shadow-lg"
          >
            <Search className="w-10 h-10 text-white" />
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-5xl font-bold bg-gradient-to-r from-white via-blue-200 to-purple-200 bg-clip-text text-transparent mb-4"
          >
            {t('search.title')}
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-xl text-gray-300 max-w-2xl mx-auto leading-relaxed"
          >
            {t('search.subtitle')}
          </motion.p>
          
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.8 }}
            className="flex justify-center items-center gap-4 mt-6"
          >
            <div className="flex items-center gap-2 bg-blue-500/20 backdrop-blur-sm border border-blue-500/30 rounded-full px-4 py-2">
              <Star className="w-4 h-4 text-blue-400" />
              <span className="text-blue-300 text-sm font-medium">Multi-Method Analysis</span>
            </div>
            <div className="flex items-center gap-2 bg-purple-500/20 backdrop-blur-sm border border-purple-500/30 rounded-full px-4 py-2">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <span className="text-purple-300 text-sm font-medium">AI-Enhanced</span>
            </div>
          </motion.div>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Method Selection & Parameters */}
          <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/20 shadow-xl">
            <div className="flex items-center gap-3 mb-8">
              <motion.div
                whileHover={{ rotate: 180 }}
                transition={{ duration: 0.3 }}
                className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center"
              >
                <Settings className="w-5 h-5 text-white" />
              </motion.div>
              <h2 className="text-2xl font-bold text-white">{t('search.configuration')}</h2>
            </div>

            {/* Method Selection */}
            <div className="space-y-4 mb-8">
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                <Target className="w-5 h-5 text-blue-400" />
                {t('search.searchMode')}
              </h3>
              <div className="space-y-3">
                {searchMethods.map((method, index) => {
                  const Icon = method.icon
                  const colors = getColorClasses(method.color)
                  const isSelected = parameters.search_mode === method.id

                  return (
                    <motion.div
                      key={method.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.5, delay: index * 0.1 }}
                      whileHover={{ scale: 1.02, y: -2 }}
                      whileTap={{ scale: 0.98 }}
                      className={`
                        group relative overflow-hidden backdrop-blur-sm rounded-xl p-5 
                        border-2 transition-all duration-500 cursor-pointer
                        ${isSelected 
                          ? `${colors.border} ${colors.bg} shadow-lg` 
                          : 'border-white/10 hover:border-white/20 bg-white/5 hover:bg-white/10'
                        }
                      `}
                      onClick={() => handleParameterChange('search_mode', method.id)}
                    >
                      <div className="absolute top-0 right-0 w-20 h-20 bg-white/5 rounded-full -translate-y-10 translate-x-10 group-hover:scale-110 transition-transform duration-500"></div>
                      <div className="relative flex items-start gap-4">
                        <motion.div 
                          whileHover={{ rotate: 360, scale: 1.1 }}
                          transition={{ duration: 0.6 }}
                          className={`flex-shrink-0 w-12 h-12 ${colors.bg} rounded-xl flex items-center justify-center shadow-lg`}
                        >
                          <Icon className={`w-6 h-6 ${colors.text}`} />
                        </motion.div>
                        <div className="flex-1">
                          <h4 className="text-lg font-bold text-white mb-2 group-hover:${colors.text.replace('text-', 'text-')} transition-colors">
                            {method.name}
                          </h4>
                          <p className="text-gray-300 text-sm mb-3 leading-relaxed">
                            {method.description}
                          </p>
                          <div className="flex items-center gap-4 text-xs">
                            <div className="flex items-center gap-1 text-gray-400">
                              <Clock className="w-3 h-3" />
                              <span>{method.processingTime}</span>
                            </div>
                            <div className="flex items-center gap-1 text-gray-400">
                              <TrendingUp className="w-3 h-3" />
                              <span>{method.accuracy}</span>
                            </div>
                          </div>
                        </div>
                        {isSelected && (
                          <motion.div 
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            className={`w-6 h-6 ${colors.bg} rounded-full flex items-center justify-center shadow-lg`}
                          >
                            <CheckCircle className={`w-4 h-4 ${colors.text}`} />
                          </motion.div>
                        )}
                      </div>
                    </motion.div>
                  )
                })}
              </div>
            </div>

            {/* Parameters Form */}
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="flex items-center gap-2 text-sm font-semibold text-gray-300 mb-3">
                  <Target className="w-4 h-4 text-blue-400" />
                  Target Name
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={parameters.target_name}
                    onChange={(e) => handleParameterChange('target_name', e.target.value)}
                    className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-400 transition-all duration-300"
                    placeholder="e.g., TIC 441420236"
                    required
                  />
                  <div className="absolute inset-y-0 right-0 flex items-center pr-3">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                  </div>
                </div>
                
                {/* Preset Targets */}
                <div className="mt-3">
                  <p className="text-xs font-medium text-gray-400 mb-2">Quick select:</p>
                  <div className="flex flex-wrap gap-2">
                    {presetTargets.map((target) => (
                      <motion.button
                        key={target.name}
                        type="button"
                        onClick={() => handleParameterChange('target_name', target.name)}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="text-xs bg-gradient-to-r from-blue-500/20 to-purple-500/20 hover:from-blue-500/30 hover:to-purple-500/30 text-blue-300 px-3 py-2 rounded-lg border border-blue-500/30 hover:border-blue-400/50 transition-all duration-300 backdrop-blur-sm"
                        title={target.description}
                      >
                        <div className="flex items-center gap-1">
                          <Star className="w-3 h-3" />
                          <span>{target.name}</span>
                        </div>
                      </motion.button>
                    ))}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Min Period (days)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={parameters.period_min}
                    onChange={(e) => handleParameterChange('period_min', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Max Period (days)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={parameters.period_max}
                    onChange={(e) => handleParameterChange('period_max', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  SNR Threshold
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={parameters.snr_threshold}
                  onChange={(e) => handleParameterChange('snr_threshold', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <motion.button
                type="submit"
                disabled={loading}
                whileHover={{ scale: 1.02, y: -2 }}
                whileTap={{ scale: 0.98 }}
                className="group relative overflow-hidden w-full bg-gradient-to-r from-blue-600 via-purple-600 to-blue-700 hover:from-blue-700 hover:via-purple-700 hover:to-blue-800 disabled:opacity-50 text-white font-bold py-4 px-6 rounded-xl transition-all duration-500 shadow-lg hover:shadow-blue-500/25"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                <div className="relative flex items-center justify-center gap-3">
                  {loading ? (
                    <>
                      <div className="animate-spin w-5 h-5 border-2 border-white/30 border-t-white rounded-full"></div>
                      <span className="text-lg">{t('search.searching')}</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-5 h-5" />
                      <span className="text-lg">{t('search.startSearch', { method: searchMethods.find(m => m.id === parameters.search_mode)?.name })}</span>
                      <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </>
                  )}
                </div>
              </motion.button>
            </form>
          </div>

          {/* Results Panel */}
          <motion.div 
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="lg:col-span-2 bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/20 shadow-xl"
          >
            <div className="flex items-center gap-3 mb-8">
              <motion.div
                animate={{ rotate: [0, 360] }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center"
              >
                <TrendingUp className="w-5 h-5 text-white" />
              </motion.div>
              <h2 className="text-2xl font-bold text-white">Search Results</h2>
            </div>

            {error && (
              <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 mb-6">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-red-400" />
                  <p className="text-red-300">{error}</p>
                </div>
              </div>
            )}

            {loading && (
              <div className="text-center py-12">
                <div className="animate-spin w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full mx-auto mb-4"></div>
                <p className="text-gray-300">{t('search.runningAnalysis', { method: searchMethods.find(m => m.id === parameters.search_mode)?.name })}</p>
                <p className="text-gray-400 text-sm mt-2">{t('search.mayTakeTime', { time: searchMethods.find(m => m.id === parameters.search_mode)?.processingTime })}</p>
              </div>
            )}

            {result && (
              <div className="space-y-6">
                {/* Detection Summary */}
                <div className="bg-green-500/20 border border-green-500/50 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-white mb-2">
                    ✅ Search Completed!
                  </h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-gray-300">Period: {result.unified_result?.best_period?.toFixed(3) || 'N/A'} days</p>
                      <p className="text-gray-300">SNR: {result.unified_result?.snr?.toFixed(2) || 'N/A'}</p>
                    </div>
                    <div>
                      <p className="text-gray-300">Confidence: {result.unified_result?.confidence?.toFixed(3) || 'N/A'}</p>
                      <p className="text-gray-300">Processing: {result.performance?.processing_time_seconds?.toFixed(1) || 'N/A'}s</p>
                    </div>
                  </div>
                </div>

                {/* Method Info */}
                <div className="bg-blue-500/10 rounded-lg p-4 border border-blue-500/30">
                  <h4 className="text-lg font-semibold text-white mb-2">Analysis Method</h4>
                  <p className="text-blue-300 text-sm">
                    Method: {result.search_info?.mode_used?.toUpperCase() || 'Unknown'}
                  </p>
                  <p className="text-gray-300 text-sm mt-1">
                    Data Points: {result.search_info?.data_points?.toLocaleString() || 'N/A'}
                  </p>
                </div>

                {/* Download Results */}
                <button
                  onClick={() => {
                    const dataStr = JSON.stringify(result, null, 2)
                    const dataBlob = new Blob([dataStr], {type: 'application/json'})
                    const url = URL.createObjectURL(dataBlob)
                    const link = document.createElement('a')
                    link.href = url
                    // Sanitize the target name for the filename to prevent path traversal
                    const sanitizedTargetName = parameters.target_name.replace(/[^a-zA-Z0-9\s\-_]/g, '_');
                    link.download = `unified_search_results_${sanitizedTargetName}_${Date.now()}.json`
                    link.click()
                  }}
                  className="w-full bg-white/10 hover:bg-white/20 text-white font-semibold py-2 px-4 rounded-lg transition-all duration-300 flex items-center justify-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download Results
                </button>
              </div>
            )}

            {!result && !loading && !error && (
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
                className="text-center py-16 text-gray-400"
              >
                <motion.div
                  animate={{ 
                    rotate: [0, 360],
                    scale: [1, 1.1, 1]
                  }}
                  transition={{ 
                    duration: 4,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }}
                  className="relative mx-auto mb-6 w-20 h-20"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-full"></div>
                  <Search className="w-12 h-12 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-blue-400" />
                </motion.div>
                <h3 className="text-lg font-semibold text-white mb-2">Ready for Analysis</h3>
                <p className="text-gray-400 mb-6">Configure search mode and parameters to start analysis</p>
                <div className="space-y-3">
                  <p className="text-sm font-medium text-gray-300">Available modes:</p>
                  <div className="flex flex-col sm:flex-row justify-center gap-3">
                    <div className="flex items-center gap-2 bg-blue-500/20 backdrop-blur-sm border border-blue-500/30 rounded-lg px-4 py-2">
                      <BarChart3 className="w-4 h-4 text-blue-400" />
                      <span className="text-blue-300 text-sm font-medium">BLS (Fast)</span>
                    </div>
                    <div className="flex items-center gap-2 bg-purple-500/20 backdrop-blur-sm border border-purple-500/30 rounded-lg px-4 py-2">
                      <Layers className="w-4 h-4 text-purple-400" />
                      <span className="text-purple-300 text-sm font-medium">Ensemble (Powerful)</span>
                    </div>
                    <div className="flex items-center gap-2 bg-green-500/20 backdrop-blur-sm border border-green-500/30 rounded-lg px-4 py-2">
                      <Activity className="w-4 h-4 text-green-400" />
                      <span className="text-green-300 text-sm font-medium">Hybrid (Smart)</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        </div>

        {/* Info Section */}
        <div className="mt-8 bg-blue-500/10 border border-blue-500/30 rounded-lg p-6">
          <div className="flex items-start gap-3">
            <Info className="w-6 h-6 text-blue-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">About Unified Search</h3>
              <p className="text-gray-300 text-sm leading-relaxed">
                Unified Search объединяет все методы поиска экзопланет в одном интерфейсе. 
                Выберите BLS для быстрого анализа, Ensemble для максимальной точности, 
                или Hybrid для автоматического выбора лучшего метода.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

const SearchPage: React.FC = () => {
  return (
    <SafePageWrapper>
      <SearchPageContent />
    </SafePageWrapper>
  )
}

export default SearchPage
