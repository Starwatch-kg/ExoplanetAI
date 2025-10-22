import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { useTranslation } from 'react-i18next'
import { AlertCircle, Zap, Settings, BarChart3, Download, Info } from 'lucide-react'

interface GPIParameters {
  target_name: string
  use_ai: boolean
  phase_sensitivity: number
  snr_threshold: number
  period_min: number
  period_max: number
}

interface GPIResult {
  target_name: string
  method: string
  exoplanet_detected: boolean
  detection_confidence: number
  gpi_analysis: any
  planetary_characterization: any
  ai_analysis?: any
  processing_time_ms: number
  status: string
}

const GPIPage: React.FC = () => {
  const { t } = useTranslation()
  const [parameters, setParameters] = useState<GPIParameters>({
    target_name: '',
    use_ai: true,
    phase_sensitivity: 1e-12,
    snr_threshold: 5.0,
    period_min: 0.1,
    period_max: 1000.0
  })
  
  const [result, setResult] = useState<GPIResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch('/api/v1/exoplanets/search?q=' + encodeURIComponent(parameters.target_name || 'TOI'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(parameters)
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'GPI analysis failed')
    } finally {
      setLoading(false)
    }
  }

  const handleParameterChange = (key: keyof GPIParameters, value: any) => {
    setParameters(prev => ({ ...prev, [key]: value }))
  }

  const generateSyntheticData = async () => {
    try {
      setLoading(true)
      const response = await fetch('/api/v1/analyze/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          num_points: 10000,
          observation_time: 365.0
        })
      })

      if (!response.ok) {
        throw new Error('Failed to generate synthetic data')
      }

      await response.json()
      
      // Use synthetic data for demonstration
      setParameters(prev => ({
        ...prev,
        target_name: `Synthetic-${Date.now()}`
      }))
      
      alert(t('gpi.syntheticGenerated'))
    } catch (err) {
      setError(err instanceof Error ? err.message : t('gpi.failedGenerate'))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.25, 0.46, 0.45, 0.94] }}
          className="text-center mb-8"
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
            className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-600 rounded-full mb-4"
          >
            <Zap className="w-8 h-8 text-white" />
          </motion.div>
          <motion.h1 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-4xl font-bold text-white mb-2"
          >
            {t('gpi.title')}
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-xl text-gray-300"
          >
            {t('gpi.subtitle')}
          </motion.p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Parameters Panel */}
          <motion.div 
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
            className="bg-white/10 dark:bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-white/20 dark:border-gray-700/50"
          >
            <div className="flex items-center gap-2 mb-6">
              <Settings className="w-5 h-5 text-purple-400" />
              <h2 className="text-xl font-semibold text-white">{t('gpi.parameters')}</h2>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Basic Parameters */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  {t('gpi.targetName')}
                </label>
                <input
                  type="text"
                  value={parameters.target_name}
                  onChange={(e) => handleParameterChange('target_name', e.target.value)}
                  className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  placeholder={t('gpi.targetPlaceholder')}
                  required
                />
              </div>

              {/* AI Enhancement */}
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="use_ai"
                  checked={parameters.use_ai}
                  onChange={(e) => handleParameterChange('use_ai', e.target.checked)}
                  className="w-4 h-4 text-purple-600 bg-white/10 border-white/20 rounded focus:ring-purple-500"
                />
                <label htmlFor="use_ai" className="text-sm font-medium text-gray-300">
                  {t('gpi.aiEnhancement')}
                </label>
              </div>

              {/* Advanced Parameters */}
              <div>
                <button
                  type="button"
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="flex items-center gap-2 text-purple-400 hover:text-purple-300 text-sm font-medium"
                >
                  <Settings className="w-4 h-4" />
                  {showAdvanced ? t('gpi.hideAdvanced') : t('gpi.showAdvanced')}
                </button>
              </div>

              {showAdvanced && (
                <div className="space-y-4 p-4 bg-white/5 rounded-lg border border-white/10">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      {t('gpi.phaseSensitivity')}
                    </label>
                    <input
                      type="number"
                      step="1e-15"
                      value={parameters.phase_sensitivity}
                      onChange={(e) => handleParameterChange('phase_sensitivity', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      {t('gpi.snrThreshold')}
                    </label>
                    <input
                      type="number"
                      step="0.1"
                      value={parameters.snr_threshold}
                      onChange={(e) => handleParameterChange('snr_threshold', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        {t('gpi.minPeriod')}
                      </label>
                      <input
                        type="number"
                        step="0.1"
                        value={parameters.period_min}
                        onChange={(e) => handleParameterChange('period_min', parseFloat(e.target.value))}
                        className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>

                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        {t('gpi.maxPeriod')}
                      </label>
                      <input
                        type="number"
                        step="1"
                        value={parameters.period_max}
                        onChange={(e) => handleParameterChange('period_max', parseFloat(e.target.value))}
                        className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-4 pt-4">
                <button
                  type="submit"
                  disabled={loading}
                  className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:opacity-50 text-white font-semibold py-2 px-4 rounded-lg transition-all duration-300"
                >
                  {loading ? t('gpi.analyzing') : t('gpi.startAnalysis')}
                </button>
                
                <button
                  type="button"
                  onClick={generateSyntheticData}
                  disabled={loading}
                  className="bg-white/10 hover:bg-white/20 disabled:opacity-50 text-white font-semibold py-2 px-4 rounded-lg transition-all duration-300"
                >
                  {t('gpi.generateSynthetic')}
                </button>
              </div>
            </form>
          </motion.div>

          {/* Results Panel */}
          <motion.div 
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 1.0 }}
            className="bg-white/10 dark:bg-gray-800/50 backdrop-blur-sm rounded-lg p-6 border border-white/20 dark:border-gray-700/50"
          >
            <div className="flex items-center gap-2 mb-6">
              <BarChart3 className="w-5 h-5 text-purple-400" />
              <h2 className="text-xl font-semibold text-white">{t('gpi.results')}</h2>
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
                <div className="animate-spin w-8 h-8 border-2 border-purple-400 border-t-transparent rounded-full mx-auto mb-4"></div>
                <p className="text-gray-300">Running GPI analysis...</p>
              </div>
            )}

            {result && (
              <div className="space-y-6">
                {/* Detection Summary */}
                <div className={`p-4 rounded-lg border ${
                  result.exoplanet_detected 
                    ? 'bg-green-500/20 border-green-500/50' 
                    : 'bg-yellow-500/20 border-yellow-500/50'
                }`}>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {result.exoplanet_detected ? '✅ Exoplanet Detected!' : '⚠️ No Clear Detection'}
                  </h3>
                  <p className="text-gray-300">
                    Confidence: {(result.detection_confidence * 100).toFixed(1)}%
                  </p>
                  <p className="text-gray-300">
                    Processing Time: {result.processing_time_ms.toFixed(1)}ms
                  </p>
                </div>

                {/* GPI Analysis Details */}
                {result.gpi_analysis && (
                  <div className="bg-white/5 rounded-lg p-4">
                    <h4 className="text-lg font-semibold text-white mb-3">GPI Analysis</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">Orbital Period</p>
                        <p className="text-white font-medium">
                          {result.gpi_analysis.orbital_period?.toFixed(2) || 'N/A'} days
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-400">SNR</p>
                        <p className="text-white font-medium">
                          {result.gpi_analysis.snr?.toFixed(2) || 'N/A'}
                        </p>
                      </div>
                      <div>
                        <p className="text-gray-400">Method</p>
                        <p className="text-white font-medium">
                          {result.gpi_analysis.method || 'GPI'}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* AI Analysis */}
                {result.ai_analysis && (
                  <div className="bg-white/5 rounded-lg p-4">
                    <h4 className="text-lg font-semibold text-white mb-3">AI Enhancement</h4>
                    <div className="text-sm">
                      <p className="text-gray-400">AI Confidence</p>
                      <p className="text-white font-medium">
                        {(result.ai_analysis.confidence * 100).toFixed(1)}%
                      </p>
                      <p className="text-gray-400 mt-2">Method</p>
                      <p className="text-white font-medium">
                        {result.ai_analysis.method}
                      </p>
                    </div>
                  </div>
                )}

                {/* Download Results */}
                <button
                  onClick={() => {
                    const dataStr = JSON.stringify(result, null, 2)
                    const dataBlob = new Blob([dataStr], {type: 'application/json'})
                    const url = URL.createObjectURL(dataBlob)
                    const link = document.createElement('a')
                    link.href = url
                    link.download = `gpi_results_${result.target_name}_${Date.now()}.json`
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
                <Zap className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>Configure parameters and start GPI analysis</p>
              </div>
            )}
          </motion.div>
        </div>

        {/* Info Section */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1.2 }}
          className="mt-8 bg-blue-500/10 border border-blue-500/30 rounded-lg p-6"
        >
          <div className="flex items-start gap-3">
            <Info className="w-6 h-6 text-blue-400 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-lg font-semibold text-white mb-2">About GPI Method</h3>
              <p className="text-gray-300 text-sm leading-relaxed">
                Gravitational Phase Interferometry (GPI) is a revolutionary method that analyzes 
                microscopic phase shifts in stellar gravitational fields caused by orbiting planets. 
                This technique is particularly effective for detecting small planets and works well 
                with noisy data where traditional methods might fail.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default GPIPage
