import React from 'react'
import { useTranslation } from 'react-i18next'
import { TrendingUp, Clock, Zap, Target } from 'lucide-react'

interface StatisticsData {
  period?: number
  depth?: number
  duration?: number
  snr?: number
  confidence?: number
}

interface StatisticsCardProps {
  data: StatisticsData
  className?: string
}

const StatisticsCard: React.FC<StatisticsCardProps> = ({ data, className = '' }) => {
  const { t } = useTranslation()

  const formatValue = (value: number | undefined, unit: string, decimals: number = 2): string => {
    if (value === undefined || value === null) return 'N/A'
    return `${value.toFixed(decimals)} ${unit}`
  }

  const getConfidenceColor = (confidence?: number): string => {
    if (!confidence) return 'text-gray-400'
    if (confidence >= 0.8) return 'text-green-400'
    if (confidence >= 0.6) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getConfidenceLabel = (confidence?: number): string => {
    if (!confidence) return 'Unknown'
    if (confidence >= 0.8) return 'High'
    if (confidence >= 0.6) return 'Medium'
    return 'Low'
  }

  return (
    <div className={`bg-white/5 dark:bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-white/10 dark:border-gray-700 ${className}`}>
      <h3 className="text-lg font-semibold text-white dark:text-gray-100 mb-4 flex items-center gap-2">
        <TrendingUp size={20} className="text-blue-400" />
        {t('results.statistics.title')}
      </h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Orbital Period */}
        <div className="flex items-center justify-between p-3 bg-white/5 dark:bg-gray-700/30 rounded-lg">
          <div className="flex items-center gap-2">
            <Clock size={16} className="text-purple-400" />
            <span className="text-sm text-gray-300 dark:text-gray-400">
              {t('results.statistics.period')}
            </span>
          </div>
          <span className="text-white dark:text-gray-100 font-mono text-sm">
            {formatValue(data.period, t('units.days'))}
          </span>
        </div>

        {/* Transit Depth */}
        <div className="flex items-center justify-between p-3 bg-white/5 dark:bg-gray-700/30 rounded-lg">
          <div className="flex items-center gap-2">
            <Target size={16} className="text-cyan-400" />
            <span className="text-sm text-gray-300 dark:text-gray-400">
              {t('results.statistics.depth')}
            </span>
          </div>
          <span className="text-white dark:text-gray-100 font-mono text-sm">
            {formatValue(data.depth ? data.depth * 100 : undefined, t('units.percent'), 3)}
          </span>
        </div>

        {/* Transit Duration */}
        <div className="flex items-center justify-between p-3 bg-white/5 dark:bg-gray-700/30 rounded-lg">
          <div className="flex items-center gap-2">
            <Clock size={16} className="text-green-400" />
            <span className="text-sm text-gray-300 dark:text-gray-400">
              {t('results.statistics.duration')}
            </span>
          </div>
          <span className="text-white dark:text-gray-100 font-mono text-sm">
            {formatValue(data.duration ? data.duration * 24 : undefined, t('units.hours'))}
          </span>
        </div>

        {/* Signal-to-Noise Ratio */}
        <div className="flex items-center justify-between p-3 bg-white/5 dark:bg-gray-700/30 rounded-lg">
          <div className="flex items-center gap-2">
            <Zap size={16} className="text-yellow-400" />
            <span className="text-sm text-gray-300 dark:text-gray-400">
              {t('results.statistics.snr')}
            </span>
          </div>
          <span className="text-white dark:text-gray-100 font-mono text-sm">
            {formatValue(data.snr, '', 1)}
          </span>
        </div>
      </div>

      {/* Detection Confidence */}
      {data.confidence !== undefined && (
        <div className="mt-4 p-4 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-lg border border-blue-500/20">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-300 dark:text-gray-400">
              {t('results.statistics.confidence')}
            </span>
            <div className="flex items-center gap-2">
              <span className={`text-sm font-semibold ${getConfidenceColor(data.confidence)}`}>
                {getConfidenceLabel(data.confidence)}
              </span>
              <span className="text-white dark:text-gray-100 font-mono text-sm">
                {(data.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
          
          {/* Confidence Bar */}
          <div className="mt-2 w-full bg-gray-700 rounded-full h-2">
            <div
              className={`h-2 rounded-full transition-all duration-500 ${
                data.confidence >= 0.8 ? 'bg-green-400' :
                data.confidence >= 0.6 ? 'bg-yellow-400' : 'bg-red-400'
              }`}
              style={{ width: `${data.confidence * 100}%` }}
            />
          </div>
        </div>
      )}
    </div>
  )
}

export default StatisticsCard
