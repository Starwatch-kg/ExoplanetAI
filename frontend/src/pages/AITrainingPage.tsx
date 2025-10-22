import React, { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { motion } from 'framer-motion'
import { 
  Brain, 
  Play, 
  Square, 
  BarChart3, 
  Settings, 
  Download,
  AlertCircle,
  Clock,
  TrendingUp,
  Database,
  Zap,
  Sparkles,
  Target,
  ArrowRight,
  Activity,
  Cpu,
  Layers
} from 'lucide-react'
import SafePageWrapper from '../components/SafePageWrapper'

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

const AITrainingPageContent: React.FC = () => {
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
      a.download = 'vohymanas_model.h5'
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
        // Mock training status for demo
        const mockStatus: TrainingStatus = {
          is_training: false,
          current_epoch: 0,
          total_epochs: 100,
          current_loss: 0,
          current_accuracy: 0,
          validation_loss: 0,
          validation_accuracy: 0,
          estimated_time_remaining: 0,
          status: 'idle'
        }
        setTrainingStatus(mockStatus)
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
            className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-indigo-500 via-purple-600 to-pink-600 rounded-full mb-6 shadow-lg"
          >
            <Brain className="w-10 h-10 text-white" />
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-5xl font-bold bg-gradient-to-r from-white via-indigo-200 to-purple-200 bg-clip-text text-transparent mb-4"
          >
            {t('aiTraining.title')}
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed"
          >
            {t('aiTraining.subtitle')}
          </motion.p>
          
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.8 }}
            className="flex justify-center items-center gap-4 mt-6"
          >
            <div className="flex items-center gap-2 bg-indigo-500/20 backdrop-blur-sm border border-indigo-500/30 rounded-full px-4 py-2">
              <Cpu className="w-4 h-4 text-indigo-400" />
              <span className="text-indigo-300 text-sm font-medium">Deep Learning</span>
            </div>
            <div className="flex items-center gap-2 bg-purple-500/20 backdrop-blur-sm border border-purple-500/30 rounded-full px-4 py-2">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <span className="text-purple-300 text-sm font-medium">Neural Networks</span>
            </div>
          </motion.div>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Training Parameters */}
          <motion.div 
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="lg:col-span-1"
          >
            <div className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/20 shadow-xl mb-6">
              <div className="flex items-center gap-3 mb-8">
                <motion.div
                  whileHover={{ rotate: 180 }}
                  transition={{ duration: 0.3 }}
                  className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center"
                >
                  <Settings className="w-5 h-5 text-white" />
                </motion.div>
                <h2 className="text-2xl font-bold text-white">{t('aiTraining.parameters')}</h2>
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
              <div className="space-y-3 mt-8">
                <motion.button
                  onClick={startTraining}
                  disabled={loading || (trainingStatus?.is_training ?? false)}
                  whileHover={{ scale: 1.02, y: -2 }}
                  whileTap={{ scale: 0.98 }}
                  className="group relative overflow-hidden w-full bg-gradient-to-r from-indigo-600 via-purple-600 to-indigo-700 hover:from-indigo-700 hover:via-purple-700 hover:to-indigo-800 disabled:opacity-50 text-white font-bold py-4 px-6 rounded-xl transition-all duration-500 shadow-lg hover:shadow-indigo-500/25"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                  <div className="relative flex items-center justify-center gap-3">
                    <Play className="w-5 h-5" />
                    <span className="text-lg">{t('aiTraining.startTraining')}</span>
                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                  </div>
                </motion.button>

                <div className="flex gap-3">
                  <motion.button
                    onClick={stopTraining}
                    disabled={!trainingStatus?.is_training}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex-1 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 disabled:opacity-50 text-white font-semibold py-3 px-4 rounded-xl transition-all duration-300 flex items-center justify-center gap-2"
                  >
                    <Square className="w-4 h-4" />
                    <span>Stop</span>
                  </motion.button>

                  <motion.button
                    onClick={downloadModel}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="flex-1 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 text-white font-semibold py-3 px-4 rounded-xl transition-all duration-300 flex items-center justify-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    <span>Download</span>
                  </motion.button>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Training Status & Logs */}
          <motion.div 
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="lg:col-span-2 space-y-6"
          >
            {/* Training Status */}
            {trainingStatus && (
              <motion.div 
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.5 }}
                className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/20 shadow-xl"
              >
                <div className="flex items-center gap-3 mb-6">
                  <motion.div
                    animate={{ rotate: [0, 360] }}
                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                    className="w-8 h-8 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center"
                  >
                    <BarChart3 className="w-5 h-5 text-white" />
                  </motion.div>
                  <h2 className="text-2xl font-bold text-white">{t('aiTraining.status')}</h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <motion.div 
                    whileHover={{ scale: 1.02 }}
                    className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 backdrop-blur-sm rounded-xl p-6 border border-blue-500/20"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <Clock className="w-5 h-5 text-blue-400" />
                      <span className="text-sm font-semibold text-gray-300">{t('aiTraining.progress')}</span>
                    </div>
                    <div className="text-2xl font-bold text-white mb-3">
                      {trainingStatus ? `${trainingStatus.current_epoch} / ${trainingStatus.total_epochs}` : 'N/A'}
                    </div>
                    <div className="w-full bg-white/10 rounded-full h-3">
                      <motion.div 
                        initial={{ width: 0 }}
                        animate={{ width: trainingStatus ? `${(trainingStatus.current_epoch / trainingStatus.total_epochs) * 100}%` : '0%' }}
                        transition={{ duration: 0.5 }}
                        className="bg-gradient-to-r from-blue-500 to-cyan-500 h-3 rounded-full shadow-lg"
                      />
                    </div>
                  </motion.div>

                  <motion.div 
                    whileHover={{ scale: 1.02 }}
                    className="bg-gradient-to-br from-green-500/10 to-emerald-500/10 backdrop-blur-sm rounded-xl p-6 border border-green-500/20"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <TrendingUp className="w-5 h-5 text-green-400" />
                      <span className="text-sm font-semibold text-gray-300">{t('aiTraining.accuracy')}</span>
                    </div>
                    <div className="text-2xl font-bold text-white">
                      {trainingStatus ? `${(trainingStatus.current_accuracy * 100).toFixed(2)}%` : 'N/A'}
                    </div>
                  </motion.div>

                  <motion.div 
                    whileHover={{ scale: 1.02 }}
                    className="bg-gradient-to-br from-yellow-500/10 to-orange-500/10 backdrop-blur-sm rounded-xl p-6 border border-yellow-500/20"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <Zap className="w-5 h-5 text-yellow-400" />
                      <span className="text-sm font-semibold text-gray-300">{t('aiTraining.loss')}</span>
                    </div>
                    <div className="text-2xl font-bold text-white">
                      {trainingStatus ? trainingStatus.current_loss?.toFixed(4) || 'N/A' : 'N/A'}
                    </div>
                  </motion.div>

                  <motion.div 
                    whileHover={{ scale: 1.02 }}
                    className="bg-gradient-to-br from-purple-500/10 to-pink-500/10 backdrop-blur-sm rounded-xl p-6 border border-purple-500/20"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <Database className="w-5 h-5 text-purple-400" />
                      <span className="text-sm font-semibold text-gray-300">{t('aiTraining.valAccuracy')}</span>
                    </div>
                    <div className="text-2xl font-bold text-white">
                      {trainingStatus ? `${(trainingStatus.validation_accuracy * 100).toFixed(2)}%` : 'N/A'}
                    </div>
                  </motion.div>
                </div>

                <div className="flex items-center gap-3 p-4 bg-white/5 rounded-xl">
                  <motion.div
                    animate={trainingStatus?.is_training ? { scale: [1, 1.2, 1] } : {}}
                    transition={{ duration: 1, repeat: trainingStatus?.is_training ? Infinity : 0 }}
                  >
                    {trainingStatus?.is_training ? (
                      <Activity className="w-6 h-6 text-green-400" />
                    ) : (
                      <AlertCircle className="w-6 h-6 text-yellow-400" />
                    )}
                  </motion.div>
                  <span className="text-lg font-semibold text-white">
                    {trainingStatus?.is_training ? t('aiTraining.training') : t('aiTraining.idle')}
                  </span>
                  {trainingStatus?.is_training && (
                    <div className="flex gap-1 ml-auto">
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                    </div>
                  )}
                </div>
              </motion.div>
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
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/20 shadow-xl"
            >
              <div className="flex items-center gap-3 mb-6">
                <Layers className="w-6 h-6 text-green-400" />
                <h2 className="text-2xl font-bold text-white">{t('aiTraining.logs')}</h2>
              </div>
              <div className="bg-black/40 backdrop-blur-sm rounded-xl p-6 h-64 overflow-y-auto font-mono text-sm border border-green-500/20">
                {logs.length === 0 ? (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-center py-16 text-gray-400"
                  >
                    <Target className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p className="italic">{t('aiTraining.noLogs')}</p>
                  </motion.div>
                ) : (
                  logs.map((log, index) => (
                    <motion.div 
                      key={index} 
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.3, delay: index * 0.1 }}
                      className="text-green-400 mb-2 flex items-center gap-2"
                    >
                      <span className="text-green-600 text-xs">[{new Date().toLocaleTimeString()}]</span>
                      <span>{log}</span>
                    </motion.div>
                  ))
                )}
              </div>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}

const AITrainingPage: React.FC = () => {
  return (
    <SafePageWrapper>
      <AITrainingPageContent />
    </SafePageWrapper>
  )
}

export default AITrainingPage
