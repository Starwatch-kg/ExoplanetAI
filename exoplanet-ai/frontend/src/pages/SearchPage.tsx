import React, { useState } from 'react'
import { AlertCircle, Search, BarChart3, Settings, TrendingUp, Download, Info, Zap, Layers, Activity } from 'lucide-react'

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

const SearchPage: React.FC = () => {
  const [searchMode, setSearchMode] = useState<'bls' | 'ensemble' | 'hybrid'>('bls')
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

  const searchMethods = [
    {
      id: 'bls',
      name: 'Box Least Squares',
      icon: BarChart3,
      description: 'Быстрый классический поиск транзитов',
      features: [
        'Высокая скорость обработки',
        'Статистическая значимость',
        'C++ ускорение',
        'Проверенный алгоритм'
      ],
      bestFor: 'Обычные транзиты, чистые кривые блеска',
      processingTime: '< 30 секунд',
      accuracy: '~90%',
      color: 'blue'
    },
    {
      id: 'ensemble',
      name: 'Ultimate Ensemble',
      icon: Layers,
      description: 'Супер-мощный поиск с 6 методами одновременно',
      features: [
        '6 методов параллельно (BLS, GPI, TLS, Wavelet, Fourier, ML)',
        '89 признаков из 6 категорий',
        'Теория хаоса и информации',
        'Bootstrap валидация'
      ],
      bestFor: 'Сложные случаи, максимальная точность',
      processingTime: '< 120 секунд',
      accuracy: '~95%',
      color: 'purple'
    },
    {
      id: 'hybrid',
      name: 'Hybrid Search',
      icon: Activity,
      description: 'Умное объединение BLS и Ensemble с автовыбором',
      features: [
        'Параллельное выполнение BLS и Ensemble',
        'Автоматический выбор лучшего',
        'Сравнительный анализ',
        'Объяснение выбора метода'
      ],
      bestFor: 'Универсальное решение для всех случаев',
      processingTime: '< 90 секунд',
      accuracy: '~93%',
      color: 'green'
    }
  ]

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      // Сначала получаем реальные данные из NASA API
      const dataResponse = await fetch('/api/v1/data/lightcurve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_name: parameters.target_name,
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_name: parameters.target_name,
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
    if (key === 'search_mode') setSearchMode(value)
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
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full mb-4">
            <Search className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">
            Unified Exoplanet Search
          </h1>
          <p className="text-xl text-gray-300">
            Переключаемый поиск: BLS, Ensemble или Hybrid режим
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Method Selection & Parameters */}
          <div className="bg-white/10 dark:bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-white/20 dark:border-gray-700/50">
            <div className="flex items-center gap-2 mb-6">
              <Settings className="w-5 h-5 text-blue-400" />
              <h2 className="text-xl font-semibold text-white">Search Configuration</h2>
            </div>

            {/* Method Selection */}
            <div className="space-y-3 mb-6">
              <h3 className="text-lg font-semibold text-white">Search Mode</h3>
              {searchMethods.map((method) => {
                const Icon = method.icon
                const colors = getColorClasses(method.color)
                const isSelected = parameters.search_mode === method.id

                return (
                  <div
                    key={method.id}
                    className={`
                      relative bg-white/5 backdrop-blur-sm rounded-lg p-4 
                      border-2 transition-all duration-300 cursor-pointer
                      ${isSelected 
                        ? `${colors.border} ${colors.bg}` 
                        : 'border-white/10 hover:border-white/20'
                      }
                    `}
                    onClick={() => handleParameterChange('search_mode', method.id)}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`flex-shrink-0 w-10 h-10 ${colors.bg} rounded-lg flex items-center justify-center`}>
                        <Icon className={`w-5 h-5 ${colors.text}`} />
                      </div>
                      <div className="flex-1">
                        <h4 className="text-lg font-semibold text-white mb-1">
                          {method.name}
                        </h4>
                        <p className="text-gray-300 text-sm mb-2">
                          {method.description}
                        </p>
                        <div className="text-xs text-gray-400">
                          {method.processingTime} • {method.accuracy}
                        </div>
                      </div>
                      {isSelected && (
                        <div className={`w-5 h-5 ${colors.bg} rounded-full flex items-center justify-center`}>
                          <div className={`w-2 h-2 ${colors.text.replace('text-', 'bg-')} rounded-full`} />
                        </div>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Parameters Form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Target Name
                </label>
                <input
                  type="text"
                  value={parameters.target_name}
                  onChange={(e) => handleParameterChange('target_name', e.target.value)}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., TIC 441420236"
                  required
                />
                
                {/* Preset Targets */}
                <div className="mt-2">
                  <p className="text-xs text-gray-400 mb-2">Quick select:</p>
                  <div className="flex flex-wrap gap-2">
                    {presetTargets.map((target) => (
                      <button
                        key={target.name}
                        type="button"
                        onClick={() => handleParameterChange('target_name', target.name)}
                        className="text-xs bg-blue-500/20 hover:bg-blue-500/30 text-blue-300 px-2 py-1 rounded border border-blue-500/30 transition-colors"
                        title={target.description}
                      >
                        {target.name}
                      </button>
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

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 text-white font-semibold py-3 px-4 rounded-lg transition-all duration-300"
              >
                {loading ? 'Searching...' : `Start ${searchMethods.find(m => m.id === parameters.search_mode)?.name} Search`}
              </button>
            </form>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 bg-white/10 dark:bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-white/20 dark:border-gray-700/50">
            <div className="flex items-center gap-2 mb-6">
              <TrendingUp className="w-5 h-5 text-blue-400" />
              <h2 className="text-xl font-semibold text-white">Search Results</h2>
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
                <p className="text-gray-300">Running {searchMethods.find(m => m.id === parameters.search_mode)?.name} analysis...</p>
                <p className="text-gray-400 text-sm mt-2">This may take up to {searchMethods.find(m => m.id === parameters.search_mode)?.processingTime}</p>
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
                    link.download = `unified_search_results_${parameters.target_name}_${Date.now()}.json`
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
              <div className="text-center py-12 text-gray-400">
                <Search className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Configure search mode and parameters to start analysis</p>
                <div className="mt-4 text-sm">
                  <p className="mb-2">Available modes:</p>
                  <div className="flex justify-center gap-4">
                    <span className="text-blue-400">🔍 BLS (Fast)</span>
                    <span className="text-purple-400">🔥 Ensemble (Powerful)</span>
                    <span className="text-green-400">⚡ Hybrid (Smart)</span>
                  </div>
                </div>
              </div>
            )}
          </div>
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

export default SearchPage
