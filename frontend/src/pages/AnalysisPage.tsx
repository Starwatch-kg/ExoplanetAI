import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  BarChart3, 
  Brain, 
  Download, 
  Share2, 
  ThumbsUp, 
  ThumbsDown,
  Star,
  Globe,
  Zap,
  Target,
  Calendar,
  TrendingUp
} from 'lucide-react'
import ApiService from '../services/api'
import { useAppStore } from '../store/useAppStore'
import LightCurveChart from '../components/charts/LightCurveChart'
import PhaseFoldedChart from '../components/charts/PhaseFoldedChart'
import LoadingSpinner from '../components/ui/LoadingSpinner'

export default function AnalysisPage() {
  const { targetId } = useParams()
  const navigate = useNavigate()
  const [selectedCandidate, setSelectedCandidate] = useState(0)
  const [showAIDetails, setShowAIDetails] = useState(false)
  const [analysisData, setAnalysisData] = useState<any>(null)
  const [isLoading, setIsLoading] = useState(true)
  const { addToast } = useAppStore()

  // Load real analysis data
  useEffect(() => {
    const loadAnalysisData = async () => {
      if (!targetId) {
        navigate('/search')
        return
      }

      try {
        setIsLoading(true)
        
        // Get data from search results or perform new search
        const { searchResults } = useAppStore.getState()
        
        if (searchResults.length > 0 && searchResults[0].target_name === targetId) {
          setAnalysisData(searchResults[0])
        } else {
          // Perform new search for this target
          const result = await ApiService.aiEnhancedSearch({
            target_name: targetId,
            catalog: 'TIC',
            mission: 'TESS',
            period_min: 0.5,
            period_max: 20.0,
            duration_min: 0.05,
            duration_max: 0.3,
            snr_threshold: 7.0
          })
          setAnalysisData(result)
        }
        
        setIsLoading(false)
      } catch (error: any) {
        setIsLoading(false)
        addToast({
          type: 'error',
          title: 'Ошибка загрузки',
          message: error.message || 'Не удалось загрузить данные анализа'
        })
        navigate('/search')
      }
    }

    loadAnalysisData()
  }, [targetId, navigate, addToast])

  // Fallback data for demonstration
  const mockAnalysisData = {
    target_name: "TIC 441420236",
    analysis_timestamp: "2024-01-15T10:30:00Z",
    lightcurve_data: {
      time: Array.from({ length: 1000 }, (_, i) => i * 0.02),
      flux: Array.from({ length: 1000 }, (_, i) => 
        1 + 0.001 * Math.sin(i * 0.1) + 0.0005 * Math.random() - 
        (Math.abs((i * 0.02) % 4.2 - 2.1) < 0.1 ? 0.01 : 0)
      ),
      target_name: "TIC 441420236",
      mission: "TESS",
      sector: 15
    },
    bls_results: {
      best_period: 4.203,
      best_power: 0.85,
      best_duration: 0.12,
      best_t0: 2.1,
      snr: 12.4,
      depth: 0.01,
      depth_err: 0.001,
      significance: 0.95
    },
    candidates: [
      {
        period: 4.203,
        epoch: 2.1,
        duration: 0.12,
        depth: 0.01,
        snr: 12.4,
        significance: 0.95,
        is_planet_candidate: true,
        confidence: 0.89
      },
      {
        period: 8.406,
        epoch: 4.2,
        duration: 0.08,
        depth: 0.005,
        snr: 7.8,
        significance: 0.78,
        is_planet_candidate: false,
        confidence: 0.34
      }
    ],
    ai_analysis: {
      is_transit: true,
      confidence: 0.89,
      confidence_level: 'HIGH' as const,
      explanation: "Обнаружен четкий периодический сигнал с характеристиками транзита экзопланеты. Глубина и форма сигнала соответствуют прохождению планеты размером ~1.2 радиуса Земли.",
      model_predictions: {
        cnn: 0.92,
        lstm: 0.87,
        transformer: 0.88,
        ensemble: 0.89
      },
      uncertainty: 0.11,
      similar_targets: ["TIC 307210830", "KIC 8462852"]
    },
    physical_parameters: {
      planet_radius: 1.23,
      orbital_period: 4.203,
      equilibrium_temperature: 285,
      habitability_score: 0.72
    },
    status: 'success' as const
  }

  const handleFeedback = async (isPositive: boolean) => {
    if (!analysisData) return
    
    try {
      await ApiService.submitFeedback(
        analysisData.target_name,
        `anonymous_${Date.now()}`, // Анонимный пользователь с уникальным ID
        isPositive ? 'positive' : 'negative',
        isPositive ? 5 : 1,
        `User ${isPositive ? 'confirmed' : 'disputed'} the analysis result`
      )
      
      addToast({
        type: 'success',
        title: 'Спасибо за обратную связь!',
        message: 'Ваша оценка поможет улучшить точность системы'
      })
    } catch (error) {
      addToast({
        type: 'error',
        title: 'Ошибка',
        message: 'Не удалось отправить обратную связь'
      })
    }
  }

  const handleExport = async () => {
    if (!analysisData) return
    
    try {
      const exportData = await ApiService.exportResults(analysisData, 'json')
      
      // Create download link
      const blob = new Blob([JSON.stringify(exportData.data, null, 2)], { 
        type: 'application/json' 
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = exportData.filename || 'analysis_results.json'
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      
      addToast({
        type: 'success',
        title: 'Экспорт завершен',
        message: 'Результаты сохранены в файл'
      })
    } catch (error) {
      addToast({
        type: 'error',
        title: 'Ошибка экспорта',
        message: 'Не удалось экспортировать результаты'
      })
    }
  }

  const confidenceLevelColors = {
    'LOW': 'text-red-400 bg-red-400/10',
    'MEDIUM': 'text-yellow-400 bg-yellow-400/10',
    'HIGH': 'text-green-400 bg-green-400/10',
    'VERY_HIGH': 'text-emerald-400 bg-emerald-400/10'
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" message="Загружаем данные анализа..." />
      </div>
    )
  }

  if (!analysisData) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-white mb-4">Данные не найдены</h2>
          <p className="text-gray-400 mb-6">Не удалось загрузить результаты анализа</p>
          <button 
            onClick={() => navigate('/search')}
            className="btn-primary"
          >
            Вернуться к поиску
          </button>
        </div>
      </div>
    )
  }

  const data = analysisData || mockAnalysisData

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gradient mb-2">
                Анализ: {data.target_name}
              </h1>
              <p className="text-gray-300 flex items-center space-x-4">
                <span className="flex items-center space-x-1">
                  <Calendar className="h-4 w-4" />
                  <span>{new Date(data.analysis_timestamp).toLocaleString('ru')}</span>
                </span>
                <span className="flex items-center space-x-1">
                  <Target className="h-4 w-4" />
                  <span>
                    {data.lightcurve_data?.sector ? `Сектор ${data.lightcurve_data.sector}` : 
                     data.lightcurve_data?.mission || 'TESS'}
                  </span>
                </span>
              </p>
            </div>
            
            <div className="flex items-center space-x-2">
              <button
                onClick={handleExport}
                className="btn-secondary px-4 py-2 flex items-center space-x-2"
              >
                <Download className="h-4 w-4" />
                <span>Экспорт</span>
              </button>
              
              <button
                onClick={() => navigator.share?.({
                  title: `Анализ ${mockAnalysisData.target_name}`,
                  text: `Результаты анализа экзопланеты ${mockAnalysisData.target_name}`,
                  url: window.location.href
                })}
                className="btn-secondary px-4 py-2 flex items-center space-x-2"
              >
                <Share2 className="h-4 w-4" />
                <span>Поделиться</span>
              </button>
            </div>
          </div>
        </motion.div>

        {/* Summary Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        >
          <div className="card text-center">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-cosmic-600/20 rounded-lg mb-4">
              <Globe className="h-6 w-6 text-cosmic-400" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-1">
              {mockAnalysisData.candidates.filter(c => c.is_planet_candidate).length}
            </h3>
            <p className="text-gray-400 text-sm">Планетных кандидатов</p>
          </div>

          <div className="card text-center">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-600/20 rounded-lg mb-4">
              <TrendingUp className="h-6 w-6 text-primary-400" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-1">
              {mockAnalysisData.bls_results.snr.toFixed(1)}
            </h3>
            <p className="text-gray-400 text-sm">SNR</p>
          </div>

          <div className="card text-center">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-green-600/20 rounded-lg mb-4">
              <Brain className="h-6 w-6 text-green-400" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-1">
              {(mockAnalysisData.ai_analysis.confidence * 100).toFixed(1)}%
            </h3>
            <p className="text-gray-400 text-sm">Уверенность ИИ</p>
          </div>

          <div className="card text-center">
            <div className="inline-flex items-center justify-center w-12 h-12 bg-yellow-600/20 rounded-lg mb-4">
              <Star className="h-6 w-6 text-yellow-400" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-1">
              {mockAnalysisData.physical_parameters?.habitability_score ? 
                (mockAnalysisData.physical_parameters.habitability_score * 100).toFixed(0) : 'N/A'}
            </h3>
            <p className="text-gray-400 text-sm">Обитаемость (%)</p>
          </div>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Main Charts */}
          <div className="lg:col-span-2 space-y-6">
            {/* Light Curve */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <LightCurveChart
                data={mockAnalysisData.lightcurve_data}
                candidates={mockAnalysisData.candidates}
                height={400}
              />
            </motion.div>

            {/* Phase Folded Chart */}
            {mockAnalysisData.candidates[selectedCandidate] && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <PhaseFoldedChart
                  data={mockAnalysisData.lightcurve_data}
                  candidate={mockAnalysisData.candidates[selectedCandidate]}
                  height={400}
                />
              </motion.div>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* AI Analysis */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="card"
            >
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-white flex items-center space-x-2">
                  <Brain className="h-5 w-5 text-cosmic-400" />
                  <span>ИИ-анализ</span>
                </h3>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                  confidenceLevelColors[mockAnalysisData.ai_analysis.confidence_level]
                }`}>
                  {mockAnalysisData.ai_analysis.confidence_level}
                </span>
              </div>

              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm text-gray-400">Уверенность</span>
                    <span className="text-white font-medium">
                      {(mockAnalysisData.ai_analysis.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-space-700 rounded-full h-2">
                    <div 
                      className="bg-cosmic-gradient h-2 rounded-full transition-all duration-500"
                      style={{ width: `${mockAnalysisData.ai_analysis.confidence * 100}%` }}
                    />
                  </div>
                </div>

                <div className="p-3 bg-space-700/30 rounded-lg">
                  <p className="text-sm text-gray-300">
                    {mockAnalysisData.ai_analysis.explanation}
                  </p>
                </div>

                <button
                  onClick={() => setShowAIDetails(!showAIDetails)}
                  className="text-primary-400 hover:text-primary-300 text-sm font-medium"
                >
                  {showAIDetails ? 'Скрыть детали' : 'Показать детали модели'}
                </button>

                {showAIDetails && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    className="space-y-2"
                  >
                    {Object.entries(mockAnalysisData.ai_analysis.model_predictions).map(([model, score]) => (
                      <div key={model} className="flex justify-between items-center">
                        <span className="text-sm text-gray-400 capitalize">{model}</span>
                        <span className="text-sm text-white">{(score * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </motion.div>
                )}
              </div>

              {/* Feedback */}
              <div className="mt-6 pt-4 border-t border-space-600">
                <p className="text-sm text-gray-400 mb-3">Оцените результат:</p>
                <div className="flex space-x-2">
                  <button
                    onClick={() => handleFeedback(true)}
                    className="flex-1 btn-secondary py-2 flex items-center justify-center space-x-2 hover:bg-green-600/20 hover:border-green-500/30"
                  >
                    <ThumbsUp className="h-4 w-4" />
                    <span>Точно</span>
                  </button>
                  <button
                    onClick={() => handleFeedback(false)}
                    className="flex-1 btn-secondary py-2 flex items-center justify-center space-x-2 hover:bg-red-600/20 hover:border-red-500/30"
                  >
                    <ThumbsDown className="h-4 w-4" />
                    <span>Неточно</span>
                  </button>
                </div>
              </div>
            </motion.div>

            {/* Candidates List */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="card"
            >
              <h3 className="text-lg font-semibold text-white mb-4">
                Кандидаты ({mockAnalysisData.candidates.length})
              </h3>
              
              <div className="space-y-3">
                {mockAnalysisData.candidates.map((candidate, index) => (
                  <button
                    key={index}
                    onClick={() => setSelectedCandidate(index)}
                    className={`w-full p-3 rounded-lg border text-left transition-all ${
                      selectedCandidate === index
                        ? 'bg-primary-600/20 border-primary-500/50'
                        : 'bg-space-700/30 border-space-600 hover:bg-space-700/50'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-white">
                        Кандидат {index + 1}
                      </span>
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        candidate.is_planet_candidate 
                          ? 'bg-green-600/20 text-green-400'
                          : 'bg-gray-600/20 text-gray-400'
                      }`}>
                        {candidate.is_planet_candidate ? 'Планета' : 'Ложный'}
                      </span>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-gray-400">Период:</span>
                        <span className="text-white ml-1">{candidate.period.toFixed(2)}д</span>
                      </div>
                      <div>
                        <span className="text-gray-400">SNR:</span>
                        <span className="text-white ml-1">{candidate.snr.toFixed(1)}</span>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </motion.div>

            {/* Physical Parameters */}
            {mockAnalysisData.physical_parameters && (
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 }}
                className="card"
              >
                <h3 className="text-lg font-semibold text-white mb-4">
                  Физические параметры
                </h3>
                
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Радиус планеты:</span>
                    <span className="text-white">
                      {mockAnalysisData.physical_parameters.planet_radius?.toFixed(2)} R⊕
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Период:</span>
                    <span className="text-white">
                      {mockAnalysisData.physical_parameters.orbital_period?.toFixed(3)} дней
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Температура:</span>
                    <span className="text-white">
                      {mockAnalysisData.physical_parameters.equilibrium_temperature} K
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Обитаемость:</span>
                    <span className="text-white">
                      {mockAnalysisData.physical_parameters.habitability_score ? 
                        `${(mockAnalysisData.physical_parameters.habitability_score * 100).toFixed(0)}%` : 'N/A'}
                    </span>
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
