import React from 'react'
import { Activity, Clock, Target, Zap, TrendingUp, AlertTriangle, CheckCircle } from 'lucide-react'
import { useTranslation } from 'react-i18next'

interface BLSDetailsProps {
  period: number
  depth: number
  duration: number
  snr: number
  significance: number
  isSignificant: boolean
  confidence: number
  className?: string
}

const BLSDetails: React.FC<BLSDetailsProps> = ({
  period,
  depth,
  duration,
  snr,
  significance,
  isSignificant,
  confidence,
  className = ''
}) => {
  const { t } = useTranslation()

  const getConfidenceLevel = (conf: number): { label: string; color: string; bgColor: string } => {
    if (conf >= 0.8) return { label: 'High', color: 'text-green-400', bgColor: 'bg-green-500/20 border-green-500/50' }
    if (conf >= 0.6) return { label: 'Medium', color: 'text-yellow-400', bgColor: 'bg-yellow-500/20 border-yellow-500/50' }
    return { label: 'Low', color: 'text-red-400', bgColor: 'bg-red-500/20 border-red-500/50' }
  }

  const confidenceInfo = getConfidenceLevel(confidence)

  return (
    <div className={`bg-white/10 dark:bg-gray-800/30 backdrop-blur-sm rounded-xl p-6 border border-white/20 dark:border-gray-700/50 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-600 rounded-lg">
            <Activity size={24} className="text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-white dark:text-gray-100">
              BLS Analysis Results
            </h2>
            <p className="text-sm text-gray-300 dark:text-gray-400">
              Box Least Squares transit detection
            </p>
          </div>
        </div>

        {/* Detection Status Badge */}
        <div className={`flex items-center gap-2 px-4 py-2 rounded-full border ${
          isSignificant 
            ? 'bg-green-500/20 border-green-500/50 text-green-300' 
            : 'bg-red-500/20 border-red-500/50 text-red-300'
        }`}>
          {isSignificant ? (
            <CheckCircle size={16} />
          ) : (
            <AlertTriangle size={16} />
          )}
          <span className="text-sm font-medium">
            {isSignificant ? '✅ Significant' : '❌ Not Significant'}
          </span>
        </div>
      </div>

      {/* Main Parameters Grid */}
      <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Orbital Period */}
        <div className="p-4 bg-white/5 dark:bg-gray-700/30 rounded-lg border border-white/10 dark:border-gray-600/30">
          <div className="flex items-center gap-2 mb-2">
            <Clock size={16} className="text-blue-400" />
            <p className="text-sm text-gray-400 dark:text-gray-500">Period</p>
          </div>
          <p className="text-2xl font-bold text-white dark:text-gray-100">
            {period.toFixed(2)}
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500">{t('units.days')}</p>
        </div>

        {/* Transit Depth */}
        <div className="p-4 bg-white/5 dark:bg-gray-700/30 rounded-lg border border-white/10 dark:border-gray-600/30">
          <div className="flex items-center gap-2 mb-2">
            <Target size={16} className="text-purple-400" />
            <p className="text-sm text-gray-400 dark:text-gray-500">Depth</p>
          </div>
          <p className="text-2xl font-bold text-white dark:text-gray-100">
            {(depth * 100).toFixed(3)}
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500">{t('units.percent')}</p>
        </div>

        {/* Duration */}
        <div className="p-4 bg-white/5 dark:bg-gray-700/30 rounded-lg border border-white/10 dark:border-gray-600/30">
          <div className="flex items-center gap-2 mb-2">
            <Clock size={16} className="text-green-400" />
            <p className="text-sm text-gray-400 dark:text-gray-500">Duration</p>
          </div>
          <p className="text-2xl font-bold text-white dark:text-gray-100">
            {(duration * 24).toFixed(1)}
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500">{t('units.hours')}</p>
        </div>

        {/* SNR */}
        <div className="p-4 bg-white/5 dark:bg-gray-700/30 rounded-lg border border-white/10 dark:border-gray-600/30">
          <div className="flex items-center gap-2 mb-2">
            <Zap size={16} className="text-yellow-400" />
            <p className="text-sm text-gray-400 dark:text-gray-500">SNR</p>
          </div>
          <p className="text-2xl font-bold text-white dark:text-gray-100">
            {snr.toFixed(1)}
          </p>
          <p className="text-xs text-gray-400 dark:text-gray-500">ratio</p>
        </div>
      </div>

      {/* Detailed Analysis */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Statistical Significance */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-white dark:text-gray-100 flex items-center gap-2">
            <TrendingUp size={18} className="text-cyan-400" />
            Statistical Analysis
          </h3>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-white/5 dark:bg-gray-700/30 rounded-lg">
              <span className="text-sm text-gray-300 dark:text-gray-400">Significance</span>
              <span className="font-mono text-white dark:text-gray-100">
                {significance.toFixed(2)}σ
              </span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-white/5 dark:bg-gray-700/30 rounded-lg">
              <span className="text-sm text-gray-300 dark:text-gray-400">Detection Threshold</span>
              <span className="font-mono text-gray-400 dark:text-gray-500">
                ≥ 3.0σ
              </span>
            </div>

            <div className="p-3 bg-white/5 dark:bg-gray-700/30 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-300 dark:text-gray-400">Signal Strength</span>
                <span className="text-sm text-gray-400 dark:text-gray-500">
                  {significance >= 3.0 ? 'Strong' : significance >= 2.0 ? 'Weak' : 'Noise'}
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className={`h-2 rounded-full transition-all duration-500 ${
                    significance >= 3.0 ? 'bg-green-400' :
                    significance >= 2.0 ? 'bg-yellow-400' : 'bg-red-400'
                  }`}
                  style={{ width: `${Math.min(significance / 5 * 100, 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Detection Confidence */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-white dark:text-gray-100 flex items-center gap-2">
            <Target size={18} className="text-pink-400" />
            Detection Confidence
          </h3>

          <div className={`p-4 rounded-lg border ${confidenceInfo.bgColor}`}>
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-gray-300 dark:text-gray-400">Overall Confidence</span>
              <div className="flex items-center gap-2">
                <span className={`text-sm font-semibold ${confidenceInfo.color}`}>
                  {confidenceInfo.label}
                </span>
                <span className="text-white dark:text-gray-100 font-mono text-sm">
                  {(confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
            
            <div className="w-full bg-gray-700 rounded-full h-3 mb-2">
              <div
                className={`h-3 rounded-full transition-all duration-500 ${
                  confidence >= 0.8 ? 'bg-green-400' :
                  confidence >= 0.6 ? 'bg-yellow-400' : 'bg-red-400'
                }`}
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
            
            <p className="text-xs text-gray-400 dark:text-gray-500">
              {confidence >= 0.8 
                ? 'Strong evidence for planetary transit'
                : confidence >= 0.6 
                ? 'Moderate evidence, requires follow-up'
                : 'Weak signal, likely false positive'
              }
            </p>
          </div>

          {/* Recommendation */}
          <div className={`p-3 rounded-lg border ${
            isSignificant 
              ? 'bg-green-500/10 border-green-500/30' 
              : 'bg-orange-500/10 border-orange-500/30'
          }`}>
            <p className={`text-sm font-medium ${
              isSignificant ? 'text-green-300' : 'text-orange-300'
            }`}>
              Recommendation:
            </p>
            <p className={`text-xs mt-1 ${
              isSignificant ? 'text-green-400' : 'text-orange-400'
            }`}>
              {isSignificant 
                ? 'Candidate for follow-up observations and validation'
                : 'Insufficient evidence for planetary transit detection'
              }
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default BLSDetails
