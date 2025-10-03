import React, { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { 
  Brain, 
  Play, 
  Square, 
  BarChart3, 
  Settings, 
  Download,
  AlertCircle,
  CheckCircle,
  Clock,
  TrendingUp,
  Database,
  Zap
} from 'lucide-react'

interface TrainingParams {
  model_type: string
  learning_rate: number
  batch_size: number
  epochs: number
  validation_split: number
  dataset_path: string
  use_augmentation: boolean
  early_stopping: boolean
}

interface TrainingStatus {
  is_training: boolean
  current_epoch: number
  total_epochs: number
  current_loss: number
  current_accuracy: number
  validation_loss: number
  validation_accuracy: number
  estimated_time_remaining: number
  status: string
}

const AITrainingPage: React.FC = () => {
  const { t } = useTranslation()
  const [trainingParams, setTrainingParams] = useState<TrainingParams>({
    model_type: 'cnn_lstm',
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 100,
    validation_split: 0.2,
    dataset_path: '',
    use_augmentation: true,
    early_stopping: true
  })

  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])

  const handleParamChange = (key: keyof TrainingParams, value: any) => {
    setTrainingParams(prev => ({ ...prev, [key]: value }))
  }

  const startTraining = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/v1/ai/train/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingParams)
      })

      if (!response.ok) {
        throw new Error('Failed to start training')
      }

      const result = await response.json()
      setLogs(prev => [...prev, `Training started: ${result.message}`])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start training')
    } finally {
      setLoading(false)
    }
  }

  const stopTraining = async () => {
    try {
      const response = await fetch('/api/v1/ai/train/stop', {
        method: 'POST'
      })

      if (!response.ok) {
        throw new Error('Failed to stop training')
      }

      setLogs(prev => [...prev, 'Training stopped by user'])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop training')
    }
  }

  const downloadModel = async () => {
    try {
      const response = await fetch('/api/v1/ai/model/download')
      
      if (!response.ok) {
        throw new Error('Failed to download model')
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'astromanas_model.h5'
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
      
      setLogs(prev => [...prev, 'Model downloaded successfully'])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to download model')
    }
  }

  // Poll training status
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch('/api/v1/ai/train/status')
        if (response.ok) {
          const status = await response.json()
          setTrainingStatus(status)
        }
      } catch (err) {
        console.warn('Failed to fetch training status:', err)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">
            {t('aiTraining.title')}
          </h1>
          <p className="text-xl text-gray-300">
            {t('aiTraining.subtitle')}
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Training Parameters */}
          <div className="lg:col-span-1">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20 mb-6">
              <div className="flex items-center gap-2 mb-6">
                <Settings className="w-5 h-5 text-indigo-400" />
                <h2 className="text-xl font-semibold text-white">{t('aiTraining.parameters')}</h2>
              </div>

              <div className="space-y-4">
                {/* Model Type */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    {t('aiTraining.modelType')}
                  </label>
                  <select
                    value={trainingParams.model_type}
                    onChange={(e) => handleParamChange('model_type', e.target.value)}
                    className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="cnn_lstm">CNN + LSTM</option>
                    <option value="transformer">Transformer</option>
                    <option value="resnet">ResNet</option>
                    <option value="efficientnet">EfficientNet</option>
                  </select>
                </div>

                {/* Learning Rate */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    {t('aiTraining.learningRate')}
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    value={trainingParams.learning_rate}
                    onChange={(e) => handleParamChange('learning_rate', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>

                {/* Batch Size */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    {t('aiTraining.batchSize')}
                  </label>
                  <select
                    value={trainingParams.batch_size}
                    onChange={(e) => handleParamChange('batch_size', parseInt(e.target.value))}
                    className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value={16}>16</option>
                    <option value={32}>32</option>
                    <option value={64}>64</option>
                    <option value={128}>128</option>
                  </select>
                </div>

                {/* Epochs */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    {t('aiTraining.epochs')}
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    value={trainingParams.epochs}
                    onChange={(e) => handleParamChange('epochs', parseInt(e.target.value))}
                    className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>

                {/* Validation Split */}
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    {t('aiTraining.validationSplit')}
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="0.5"
                    value={trainingParams.validation_split}
                    onChange={(e) => handleParamChange('validation_split', parseFloat(e.target.value))}
                    className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>

                {/* Checkboxes */}
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="use_augmentation"
                      checked={trainingParams.use_augmentation}
                      onChange={(e) => handleParamChange('use_augmentation', e.target.checked)}
                      className="w-4 h-4 text-indigo-600 bg-white/10 border-white/20 rounded focus:ring-indigo-500"
                    />
                    <label htmlFor="use_augmentation" className="text-sm font-medium text-gray-300">
                      {t('aiTraining.useAugmentation')}
                    </label>
                  </div>

                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      id="early_stopping"
                      checked={trainingParams.early_stopping}
                      onChange={(e) => handleParamChange('early_stopping', e.target.checked)}
                      className="w-4 h-4 text-indigo-600 bg-white/10 border-white/20 rounded focus:ring-indigo-500"
                    />
                    <label htmlFor="early_stopping" className="text-sm font-medium text-gray-300">
                      {t('aiTraining.earlyStopping')}
                    </label>
                  </div>
                </div>
              </div>

              {/* Control Buttons */}
              <div className="flex gap-2 mt-6">
                <button
                  onClick={startTraining}
                  disabled={loading || (trainingStatus?.is_training ?? false)}
                  className="flex-1 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 disabled:opacity-50 text-white font-semibold py-2 px-4 rounded-lg transition-all duration-300 flex items-center justify-center gap-2"
                >
                  <Play className="w-4 h-4" />
                  {t('aiTraining.startTraining')}
                </button>

                <button
                  onClick={stopTraining}
                  disabled={!trainingStatus?.is_training}
                  className="bg-red-600 hover:bg-red-700 disabled:opacity-50 text-white font-semibold py-2 px-4 rounded-lg transition-all duration-300"
                >
                  <Square className="w-4 h-4" />
                </button>

                <button
                  onClick={downloadModel}
                  className="bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg transition-all duration-300"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>

          {/* Training Status & Logs */}
          <div className="lg:col-span-2 space-y-6">
            {/* Training Status */}
            {trainingStatus && (
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
                <div className="flex items-center gap-2 mb-4">
                  <BarChart3 className="w-5 h-5 text-indigo-400" />
                  <h2 className="text-xl font-semibold text-white">{t('aiTraining.status')}</h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Clock className="w-4 h-4 text-blue-400" />
                      <span className="text-sm text-gray-300">{t('aiTraining.progress')}</span>
                    </div>
                    <div className="text-lg font-semibold text-white">
                      {trainingStatus.current_epoch} / {trainingStatus.total_epochs}
                    </div>
                    <div className="w-full bg-white/10 rounded-full h-2 mt-2">
                      <div 
                        className="bg-indigo-500 h-2 rounded-full transition-all duration-300"
                        style={{ width: `${(trainingStatus.current_epoch / trainingStatus.total_epochs) * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <TrendingUp className="w-4 h-4 text-green-400" />
                      <span className="text-sm text-gray-300">{t('aiTraining.accuracy')}</span>
                    </div>
                    <div className="text-lg font-semibold text-white">
                      {(trainingStatus.current_accuracy * 100).toFixed(2)}%
                    </div>
                  </div>

                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Zap className="w-4 h-4 text-yellow-400" />
                      <span className="text-sm text-gray-300">{t('aiTraining.loss')}</span>
                    </div>
                    <div className="text-lg font-semibold text-white">
                      {trainingStatus.current_loss.toFixed(4)}
                    </div>
                  </div>

                  <div className="bg-white/5 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <Database className="w-4 h-4 text-purple-400" />
                      <span className="text-sm text-gray-300">{t('aiTraining.valAccuracy')}</span>
                    </div>
                    <div className="text-lg font-semibold text-white">
                      {(trainingStatus.validation_accuracy * 100).toFixed(2)}%
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {trainingStatus.is_training ? (
                    <CheckCircle className="w-5 h-5 text-green-400" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-yellow-400" />
                  )}
                  <span className="text-gray-300">
                    {trainingStatus.is_training ? t('aiTraining.training') : t('aiTraining.idle')}
                  </span>
                </div>
              </div>
            )}

            {/* Error Display */}
            {error && (
              <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4">
                <div className="flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-red-400" />
                  <p className="text-red-300">{error}</p>
                </div>
              </div>
            )}

            {/* Training Logs */}
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
              <h2 className="text-xl font-semibold text-white mb-4">{t('aiTraining.logs')}</h2>
              <div className="bg-black/30 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
                {logs.length === 0 ? (
                  <div className="text-gray-400 italic">{t('aiTraining.noLogs')}</div>
                ) : (
                  logs.map((log, index) => (
                    <div key={index} className="text-green-400 mb-1">
                      [{new Date().toLocaleTimeString()}] {log}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AITrainingPage
