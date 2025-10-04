import React, { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { 
  Search, 
  Upload, 
  Play, 
  CheckCircle, 
  AlertCircle, 
  Clock,
  BarChart3,
  Star,
  FileText
} from 'lucide-react'
import Plot from 'react-plotly.js'

// Типы для единого API
interface UnifiedAnalysisRequest {
  target_name: string
  mission?: string
  auto_download?: boolean
}

interface UnifiedAnalysisResult {
  target_name: string
  analysis_timestamp: string
  processing_time_ms: number
  predicted_class: string
  confidence_score: number
  class_probabilities: Record<string, number>
  planet_parameters: {
    orbital_period_days: number
    transit_depth_ppm: number
    transit_duration_hours: number
    snr: number
    significance: number
  }
  star_info: {
    target_id: string
    ra: number
    dec: number
    magnitude: number
    stellar_temperature: number
    stellar_radius: number
    stellar_mass: number
    distance_pc: number
  }
  lightcurve_data: {
    points_count: number
    time_span_days: number
    cadence_minutes: number
    noise_level_ppm: number
  }
  plot_data: {
    time: number[]
    flux: number[]
    flux_err: number[]
    period_power: {
      periods: number[]
      power: number[]
    }
    best_period: number
    transit_times: number[]
  }
  data_source: string
  mission: string
  data_quality_score: number
  analysis_notes: string[]
}

const UnifiedAnalysisPage: React.FC = () => {
  const [analysisMode, setAnalysisMode] = useState<'search' | 'upload'>('search')
  const [targetName, setTargetName] = useState('')
  const [mission, setMission] = useState('TESS')
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<UnifiedAnalysisResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Функция анализа по имени объекта
  const analyzeByName = useCallback(async () => {
    if (!targetName.trim()) {
      setError('Введите название объекта')
      return
    }

    setIsAnalyzing(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/v1/analyze/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          target_name: targetName.trim(),
          mission: mission,
          auto_download: true
        } as UnifiedAnalysisRequest)
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Ошибка анализа: ${response.status} - ${errorText}`)
      }

      const analysisResult: UnifiedAnalysisResult = await response.json()
      setResult(analysisResult)
      
    } catch (err) {
      console.error('Analysis error:', err)
      setError(err instanceof Error ? err.message : 'Произошла ошибка при анализе')
    } finally {
      setIsAnalyzing(false)
    }
  }, [targetName, mission])

  // Функция анализа загруженного файла
  const analyzeUploadedFile = useCallback(async () => {
    if (!uploadedFile || !targetName.trim()) {
      setError('Выберите файл и введите название объекта')
      return
    }

    setIsAnalyzing(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', uploadedFile)
      formData.append('target_name', targetName.trim())

      const response = await fetch('/api/v1/analyze/analyze-file', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Ошибка анализа файла: ${response.status} - ${errorText}`)
      }

      const analysisResult: UnifiedAnalysisResult = await response.json()
      setResult(analysisResult)
      
    } catch (err) {
      console.error('File analysis error:', err)
      setError(err instanceof Error ? err.message : 'Произошла ошибка при анализе файла')
    } finally {
      setIsAnalyzing(false)
    }
  }, [uploadedFile, targetName])

  // Обработка загрузки файла
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setUploadedFile(file)
      setError(null)
    }
  }

  // Получение цвета для класса
  const getClassColor = (className: string) => {
    switch (className) {
      case 'Confirmed': return 'text-green-400'
      case 'Candidate': return 'text-yellow-400'
      case 'False Positive': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  // Получение цвета для уверенности
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400'
    if (confidence >= 0.6) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Заголовок */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-4">
            🌌 Автоматический анализ экзопланет
          </h1>
          <p className="text-gray-300 text-lg">
            Единый интерфейс для поиска и классификации экзопланет с использованием ИИ
          </p>
        </motion.div>

        {/* Панель выбора режима */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white/10 backdrop-blur-md rounded-xl p-6 mb-8"
        >
          <div className="flex gap-4 mb-6">
            <button
              onClick={() => setAnalysisMode('search')}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg transition-all ${
                analysisMode === 'search'
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-white/20 text-gray-300 hover:bg-white/30'
              }`}
            >
              <Search size={20} />
              Поиск по имени
            </button>
            <button
              onClick={() => setAnalysisMode('upload')}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg transition-all ${
                analysisMode === 'upload'
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-white/20 text-gray-300 hover:bg-white/30'
              }`}
            >
              <Upload size={20} />
              Загрузить файл
            </button>
          </div>

          {/* Форма поиска по имени */}
          {analysisMode === 'search' && (
            <div className="space-y-4">
              <div>
                <label className="block text-white mb-2">Название объекта</label>
                <input
                  type="text"
                  value={targetName}
                  onChange={(e) => setTargetName(e.target.value)}
                  placeholder="Например: TOI-715, TIC-441420236, Kepler-452b"
                  className="w-full px-4 py-3 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-white mb-2">Миссия</label>
                <select
                  value={mission}
                  onChange={(e) => setMission(e.target.value)}
                  className="w-full px-4 py-3 bg-white/20 border border-white/30 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="TESS">TESS</option>
                  <option value="Kepler">Kepler</option>
                  <option value="K2">K2</option>
                </select>
              </div>
            </div>
          )}

          {/* Форма загрузки файла */}
          {analysisMode === 'upload' && (
            <div className="space-y-4">
              <div>
                <label className="block text-white mb-2">Название объекта</label>
                <input
                  type="text"
                  value={targetName}
                  onChange={(e) => setTargetName(e.target.value)}
                  placeholder="Введите название для анализируемого объекта"
                  className="w-full px-4 py-3 bg-white/20 border border-white/30 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
              <div>
                <label className="block text-white mb-2">Файл кривой блеска (CSV/TXT)</label>
                <input
                  type="file"
                  accept=".csv,.txt"
                  onChange={handleFileUpload}
                  className="w-full px-4 py-3 bg-white/20 border border-white/30 rounded-lg text-white file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700"
                />
                {uploadedFile && (
                  <p className="text-green-400 mt-2">
                    <FileText size={16} className="inline mr-2" />
                    {uploadedFile.name}
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Кнопка анализа */}
          <div className="mt-6">
            <button
              onClick={analysisMode === 'search' ? analyzeByName : analyzeUploadedFile}
              disabled={isAnalyzing}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 px-6 rounded-lg font-semibold text-lg hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
            >
              {isAnalyzing ? (
                <>
                  <Clock className="animate-spin" size={20} />
                  Анализируем...
                </>
              ) : (
                <>
                  <Play size={20} />
                  Начать автоматический анализ
                </>
              )}
            </button>
          </div>

          {/* Ошибка */}
          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg flex items-center gap-2 text-red-400"
            >
              <AlertCircle size={20} />
              {error}
            </motion.div>
          )}
        </motion.div>

        {/* Результаты анализа */}
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Основные результаты */}
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                <CheckCircle className="text-green-400" />
                Результаты анализа: {result.target_name}
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Классификация */}
                <div className="bg-white/10 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-white mb-2">Классификация</h3>
                  <div className={`text-2xl font-bold ${getClassColor(result.predicted_class)}`}>
                    {result.predicted_class}
                  </div>
                  <div className={`text-lg ${getConfidenceColor(result.confidence_score)}`}>
                    Уверенность: {(result.confidence_score * 100).toFixed(1)}%
                  </div>
                </div>

                {/* Параметры планеты */}
                <div className="bg-white/10 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-white mb-2">Параметры планеты</h3>
                  <div className="space-y-1 text-sm text-gray-300">
                    <div>Период: {result.planet_parameters.orbital_period_days.toFixed(2)} дней</div>
                    <div>Глубина: {result.planet_parameters.transit_depth_ppm.toFixed(0)} ppm</div>
                    <div>Длительность: {result.planet_parameters.transit_duration_hours.toFixed(1)} ч</div>
                    <div>SNR: {result.planet_parameters.snr.toFixed(1)}</div>
                  </div>
                </div>

                {/* Качество данных */}
                <div className="bg-white/10 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-white mb-2">Качество данных</h3>
                  <div className="text-lg text-blue-400">
                    {(result.data_quality_score * 100).toFixed(0)}%
                  </div>
                  <div className="text-sm text-gray-300">
                    {result.lightcurve_data.points_count} точек данных
                  </div>
                  <div className="text-sm text-gray-300">
                    {result.data_source} / {result.mission}
                  </div>
                </div>
              </div>
            </div>

            {/* График кривой блеска */}
            <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                <BarChart3 />
                Кривая блеска
              </h3>
              <div className="bg-white rounded-lg p-4">
                <Plot
                  data={[
                    {
                      x: result.plot_data.time,
                      y: result.plot_data.flux,
                      type: 'scatter' as const,
                      mode: 'markers' as const,
                      marker: { size: 2, color: 'blue' },
                      name: 'Поток'
                    },
                    // Отметки транзитов
                    ...result.plot_data.transit_times.map((time, i) => ({
                      x: [time, time],
                      y: [Math.min(...result.plot_data.flux), Math.max(...result.plot_data.flux)],
                      type: 'scatter' as const,
                      mode: 'lines' as const,
                      line: { color: 'red', dash: 'dash' as const, width: 1 },
                      name: i === 0 ? 'Транзиты' : '',
                      showlegend: i === 0
                    }))
                  ]}
                  layout={{
                    title: { text: `Кривая блеска ${result.target_name}` },
                    xaxis: { title: { text: 'Время (дни)' } },
                    yaxis: { title: { text: 'Нормализованный поток' } },
                    height: 400,
                    margin: { t: 50, r: 50, b: 50, l: 50 }
                  }}
                  config={{
                    displayModeBar: true,
                    displaylogo: false,
                    modeBarButtonsToRemove: ['pan2d', 'lasso2d']
                  }}
                  style={{ width: '100%' }}
                />
              </div>
            </div>

            {/* Дополнительная информация */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Информация о звезде */}
              <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
                <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
                  <Star />
                  Информация о звезде
                </h3>
                <div className="space-y-2 text-gray-300">
                  <div>Координаты: {result.star_info.ra.toFixed(2)}°, {result.star_info.dec.toFixed(2)}°</div>
                  <div>Звездная величина: {result.star_info.magnitude.toFixed(1)}</div>
                  <div>Температура: {result.star_info.stellar_temperature.toFixed(0)} K</div>
                  <div>Радиус: {result.star_info.stellar_radius.toFixed(2)} R☉</div>
                  <div>Масса: {result.star_info.stellar_mass.toFixed(2)} M☉</div>
                  <div>Расстояние: {result.star_info.distance_pc.toFixed(1)} пк</div>
                </div>
              </div>

              {/* Заметки анализа */}
              <div className="bg-white/10 backdrop-blur-md rounded-xl p-6">
                <h3 className="text-xl font-bold text-white mb-4">Заметки анализа</h3>
                <div className="space-y-2">
                  {result.analysis_notes.map((note, index) => (
                    <div key={index} className="text-gray-300 text-sm">
                      • {note}
                    </div>
                  ))}
                </div>
                <div className="mt-4 text-xs text-gray-400">
                  Время обработки: {result.processing_time_ms.toFixed(0)} мс
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}

export default UnifiedAnalysisPage
