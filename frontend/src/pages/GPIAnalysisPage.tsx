import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Upload, 
  Play, 
  Settings, 
  BarChart3, 
  Zap,
  AlertCircle,
  Clock,
  Brain,
  Target
} from 'lucide-react'
import Plot from 'react-plotly.js'

interface GPIResult {
  id: string
  target_name: string
  method: 'GPI'
  result_class: string
  confidence: number
  parameters: {
    period: number
    depth: number
    snr: number
    significance: number
  }
  processing_time_ms: number
  timestamp: string
  plot_data?: {
    time: number[]
    flux: number[]
    model?: number[]
  }
}

interface GPIAnalysisPageProps {
  useSimpleBackground?: boolean
}

export default function GPIAnalysisPage({ useSimpleBackground = false }: GPIAnalysisPageProps) {
  const [targetName, setTargetName] = useState('')
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<GPIResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [analysisHistory, setAnalysisHistory] = useState<GPIResult[]>([])

  // GPI параметры
  const [gpiParams, setGpiParams] = useState({
    period_min: 1.0,
    period_max: 20.0,
    duration_min: 0.05,
    duration_max: 0.2,
    snr_threshold: 5.0,
    significance_threshold: 0.7
  })

  const handleAnalyze = async () => {
    if (!targetName.trim() && !uploadedFile) {
      setError('Please enter a target name or upload a file')
      return
    }

    setIsAnalyzing(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      
      if (uploadedFile) {
        formData.append('file', uploadedFile)
      } else {
        formData.append('target_name', targetName.trim())
      }
      
      // Добавляем GPI параметры
      Object.entries(gpiParams).forEach(([key, value]) => {
        formData.append(key, value.toString())
      })
      formData.append('method', 'GPI')

      const response = await fetch('/api/v1/analyze/gpi', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`)
      }

      const data = await response.json()
      
      const gpiResult: GPIResult = {
        id: Date.now().toString(),
        target_name: targetName || uploadedFile?.name || 'Unknown',
        method: 'GPI',
        result_class: data.predicted_class || 'Unknown',
        confidence: data.confidence_score || 0,
        parameters: {
          period: data.planet_parameters?.orbital_period_days || 0,
          depth: data.planet_parameters?.transit_depth_ppm || 0,
          snr: data.planet_parameters?.snr || 0,
          significance: data.significance || 0
        },
        processing_time_ms: data.processing_time_ms || 0,
        timestamp: new Date().toISOString(),
        plot_data: data.plot_data
      }

      setResult(gpiResult)
      setAnalysisHistory(prev => [gpiResult, ...prev.slice(0, 9)]) // Храним последние 10 результатов

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setUploadedFile(file)
      setTargetName('')
    }
  }

  const getResultColor = (resultClass: string) => {
    switch (resultClass.toLowerCase()) {
      case 'confirmed': return 'text-green-400 bg-green-500/20 border-green-500/30'
      case 'candidate': return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30'
      case 'false positive': return 'text-red-400 bg-red-500/20 border-red-500/30'
      default: return 'text-gray-400 bg-gray-500/20 border-gray-500/30'
    }
  }

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <div className="inline-block p-4 bg-gradient-to-r from-purple-600 to-pink-600 rounded-2xl mb-6">
            <Brain className="w-12 h-12 text-white" />
          </div>
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
            GPI Analysis
          </h1>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto">
            Advanced Gaussian Process Inference for exoplanet detection and characterization
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Input Panel */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            className={`backdrop-blur-sm border rounded-2xl p-8 ${useSimpleBackground ? 'bg-gray-800/30 border-gray-600' : 'bg-gray-800/50 border-gray-700'}`}
          >
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <Target className="w-6 h-6 text-purple-400" />
              Target Input
            </h2>

            {/* Target Name Input */}
            <div className="mb-6">
              <label className="block text-gray-300 text-sm font-medium mb-2">
                Target Name
              </label>
              <input
                type="text"
                value={targetName}
                onChange={(e) => setTargetName(e.target.value)}
                placeholder="e.g., TOI-715, Kepler-452b, TIC-441420236"
                className="w-full px-4 py-3 bg-gray-900/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-purple-500 focus:ring-2 focus:ring-purple-500/20 transition-all"
                disabled={!!uploadedFile}
              />
            </div>

            {/* File Upload */}
            <div className="mb-6">
              <label className="block text-gray-300 text-sm font-medium mb-2">
                Or Upload Light Curve File
              </label>
              <div className="relative">
                <input
                  type="file"
                  accept=".csv,.txt,.dat"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                  disabled={!!targetName.trim()}
                />
                <label
                  htmlFor="file-upload"
                  className={`flex items-center justify-center w-full px-4 py-8 border-2 border-dashed rounded-lg cursor-pointer transition-all ${
                    uploadedFile 
                      ? 'border-purple-500 bg-purple-500/10' 
                      : 'border-gray-600 hover:border-purple-500 hover:bg-purple-500/5'
                  }`}
                >
                  <div className="text-center">
                    <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-300">
                      {uploadedFile ? uploadedFile.name : 'Click to upload CSV/TXT file'}
                    </p>
                    <p className="text-gray-500 text-sm mt-1">
                      Supported formats: CSV, TXT, DAT
                    </p>
                  </div>
                </label>
              </div>
            </div>

            {/* GPI Parameters */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Settings className="w-5 h-5 text-purple-400" />
                GPI Parameters
              </h3>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-gray-400 text-sm mb-1">Period Min (days)</label>
                  <input
                    type="number"
                    value={gpiParams.period_min}
                    onChange={(e) => setGpiParams(prev => ({...prev, period_min: parseFloat(e.target.value)}))}
                    className="w-full px-3 py-2 bg-gray-900/50 border border-gray-600 rounded text-white text-sm"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-gray-400 text-sm mb-1">Period Max (days)</label>
                  <input
                    type="number"
                    value={gpiParams.period_max}
                    onChange={(e) => setGpiParams(prev => ({...prev, period_max: parseFloat(e.target.value)}))}
                    className="w-full px-3 py-2 bg-gray-900/50 border border-gray-600 rounded text-white text-sm"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-gray-400 text-sm mb-1">SNR Threshold</label>
                  <input
                    type="number"
                    value={gpiParams.snr_threshold}
                    onChange={(e) => setGpiParams(prev => ({...prev, snr_threshold: parseFloat(e.target.value)}))}
                    className="w-full px-3 py-2 bg-gray-900/50 border border-gray-600 rounded text-white text-sm"
                    step="0.1"
                  />
                </div>
                <div>
                  <label className="block text-gray-400 text-sm mb-1">Significance</label>
                  <input
                    type="number"
                    value={gpiParams.significance_threshold}
                    onChange={(e) => setGpiParams(prev => ({...prev, significance_threshold: parseFloat(e.target.value)}))}
                    className="w-full px-3 py-2 bg-gray-900/50 border border-gray-600 rounded text-white text-sm"
                    step="0.01"
                    min="0"
                    max="1"
                  />
                </div>
              </div>
            </div>

            {/* Analyze Button */}
            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing || (!targetName.trim() && !uploadedFile)}
              className="w-full bg-gradient-to-r from-purple-600 to-pink-600 text-white py-4 px-6 rounded-lg font-semibold flex items-center justify-center gap-3 hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
            >
              {isAnalyzing ? (
                <>
                  <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Start GPI Analysis
                </>
              )}
            </button>

            {/* Error Display */}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg flex items-center gap-3"
              >
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
                <p className="text-red-300">{error}</p>
              </motion.div>
            )}
          </motion.div>

          {/* Results Panel */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            className={`backdrop-blur-sm border rounded-2xl p-8 ${useSimpleBackground ? 'bg-gray-800/30 border-gray-600' : 'bg-gray-800/50 border-gray-700'}`}
          >
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <BarChart3 className="w-6 h-6 text-purple-400" />
              GPI Results
            </h2>

            <AnimatePresence mode="wait">
              {result ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="space-y-6"
                >
                  {/* Result Summary */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className={`p-4 rounded-lg border ${getResultColor(result.result_class)}`}>
                      <div className="text-sm opacity-75 mb-1">Classification</div>
                      <div className="font-bold text-lg">{result.result_class}</div>
                    </div>
                    <div className="p-4 rounded-lg border border-gray-600 bg-gray-700/30">
                      <div className="text-sm text-gray-400 mb-1">Confidence</div>
                      <div className="font-bold text-lg text-white">{(result.confidence * 100).toFixed(1)}%</div>
                    </div>
                  </div>

                  {/* Parameters */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-3 bg-gray-700/30 rounded-lg">
                      <div className="text-sm text-gray-400">Period</div>
                      <div className="font-semibold text-white">{result.parameters.period.toFixed(2)} days</div>
                    </div>
                    <div className="p-3 bg-gray-700/30 rounded-lg">
                      <div className="text-sm text-gray-400">Depth</div>
                      <div className="font-semibold text-white">{result.parameters.depth.toFixed(0)} ppm</div>
                    </div>
                    <div className="p-3 bg-gray-700/30 rounded-lg">
                      <div className="text-sm text-gray-400">SNR</div>
                      <div className="font-semibold text-white">{result.parameters.snr.toFixed(1)}</div>
                    </div>
                    <div className="p-3 bg-gray-700/30 rounded-lg">
                      <div className="text-sm text-gray-400">Processing Time</div>
                      <div className="font-semibold text-white">{result.processing_time_ms} ms</div>
                    </div>
                  </div>

                  {/* Plot */}
                  {result.plot_data && (
                    <div className="bg-gray-900/50 rounded-lg p-4">
                      <Plot
                        data={[
                          {
                            x: result.plot_data.time,
                            y: result.plot_data.flux,
                            type: 'scatter' as const,
                            mode: 'markers' as const,
                            marker: { size: 3, color: '#60A5FA' },
                            name: 'Light Curve'
                          } as any,
                          ...(result.plot_data.model ? [{
                            x: result.plot_data.time,
                            y: result.plot_data.model,
                            type: 'scatter' as const,
                            mode: 'lines' as const,
                            line: { color: '#F59E0B', width: 2 },
                            name: 'GPI Model'
                          } as any] : [])
                        ]}
                        layout={{
                          title: { text: 'GPI Analysis Result' },
                          xaxis: { title: { text: 'Time (days)' }, color: '#9CA3AF' },
                          yaxis: { title: { text: 'Relative Flux' }, color: '#9CA3AF' },
                          paper_bgcolor: 'rgba(0,0,0,0)',
                          plot_bgcolor: 'rgba(0,0,0,0)',
                          font: { color: '#9CA3AF' },
                          height: 300
                        } as any}
                        config={{ displayModeBar: false }}
                        className="w-full"
                      />
                    </div>
                  )}
                </motion.div>
              ) : (
                <div className="text-center text-gray-400 py-12">
                  <Zap className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Run GPI analysis to see results here</p>
                </div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>

        {/* Analysis History */}
        {analysisHistory.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            className={`mt-8 backdrop-blur-sm border rounded-2xl p-8 ${useSimpleBackground ? 'bg-gray-800/30 border-gray-600' : 'bg-gray-800/50 border-gray-700'}`}
          >
            <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
              <Clock className="w-6 h-6 text-purple-400" />
              Recent GPI Analyses
            </h2>
            <div className="space-y-3">
              {analysisHistory.map((analysis) => (
                <div key={analysis.id} className="flex items-center justify-between p-4 bg-gray-700/30 rounded-lg">
                  <div className="flex items-center gap-4">
                    <div className={`px-3 py-1 rounded-full text-sm ${getResultColor(analysis.result_class)}`}>
                      {analysis.result_class}
                    </div>
                    <div>
                      <div className="font-semibold text-white">{analysis.target_name}</div>
                      <div className="text-sm text-gray-400">
                        {new Date(analysis.timestamp).toLocaleString()} • {analysis.processing_time_ms}ms
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold text-white">{(analysis.confidence * 100).toFixed(1)}%</div>
                    <div className="text-sm text-gray-400">confidence</div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
