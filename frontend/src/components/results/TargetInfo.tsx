import React from 'react'
import { Star, MapPin, Eye, Calendar } from 'lucide-react'
// import { useTranslation } from 'react-i18next' // Для будущих переводов

interface TargetInfoProps {
  targetName: string
  catalog: string
  mission: string
  magnitude: number
  coordinates: {
    ra: number
    dec: number
  }
  observationDays: number
  dataPoints: number
  className?: string
}

const TargetInfo: React.FC<TargetInfoProps> = ({
  targetName,
  catalog,
  mission,
  magnitude,
  coordinates,
  observationDays,
  dataPoints,
  className = ''
}) => {
  // const { t } = useTranslation() // Для будущих переводов

  const formatCoordinates = (ra: number, dec: number): string => {
    return `${ra.toFixed(4)}°, ${dec.toFixed(4)}°`
  }

  return (
    <div className={`bg-white/10 dark:bg-gray-800/30 backdrop-blur-sm rounded-xl p-6 border border-white/20 dark:border-gray-700/50 ${className}`}>
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
          <Star size={24} className="text-white" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-white dark:text-gray-100">
            Target Information
          </h2>
          <p className="text-sm text-gray-300 dark:text-gray-400">
            Stellar object details from {catalog} catalog
          </p>
        </div>
      </div>

      {/* Target Details Grid */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Left Column */}
        <div className="space-y-4">
          {/* Target Name */}
          <div className="flex items-center justify-between p-4 bg-white/5 dark:bg-gray-700/30 rounded-lg border border-white/10 dark:border-gray-600/30">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/20 rounded-lg">
                <Star size={16} className="text-blue-400" />
              </div>
              <div>
                <p className="text-sm text-gray-400 dark:text-gray-500">Target</p>
                <p className="font-mono text-lg font-semibold text-white dark:text-gray-100">
                  {targetName}
                </p>
              </div>
            </div>
          </div>

          {/* Catalog & Mission */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-white/5 dark:bg-gray-700/30 rounded-lg border border-white/10 dark:border-gray-600/30 text-center">
              <p className="text-xs text-gray-400 dark:text-gray-500 mb-1">Catalog</p>
              <p className="font-semibold text-white dark:text-gray-100">{catalog}</p>
            </div>
            <div className="p-3 bg-white/5 dark:bg-gray-700/30 rounded-lg border border-white/10 dark:border-gray-600/30 text-center">
              <p className="text-xs text-gray-400 dark:text-gray-500 mb-1">Mission</p>
              <p className="font-semibold text-white dark:text-gray-100">{mission}</p>
            </div>
          </div>

          {/* Magnitude */}
          <div className="flex items-center justify-between p-4 bg-white/5 dark:bg-gray-700/30 rounded-lg border border-white/10 dark:border-gray-600/30">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-yellow-500/20 rounded-lg">
                <Eye size={16} className="text-yellow-400" />
              </div>
              <div>
                <p className="text-sm text-gray-400 dark:text-gray-500">Magnitude</p>
                <p className="text-lg font-semibold text-white dark:text-gray-100">
                  {magnitude.toFixed(1)} mag
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column */}
        <div className="space-y-4">
          {/* Coordinates */}
          <div className="flex items-center justify-between p-4 bg-white/5 dark:bg-gray-700/30 rounded-lg border border-white/10 dark:border-gray-600/30">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-green-500/20 rounded-lg">
                <MapPin size={16} className="text-green-400" />
              </div>
              <div>
                <p className="text-sm text-gray-400 dark:text-gray-500">Coordinates (RA, Dec)</p>
                <p className="font-mono text-sm font-semibold text-white dark:text-gray-100">
                  {formatCoordinates(coordinates.ra, coordinates.dec)}
                </p>
              </div>
            </div>
          </div>

          {/* Observation Stats */}
          <div className="grid grid-cols-2 gap-3">
            <div className="p-3 bg-white/5 dark:bg-gray-700/30 rounded-lg border border-white/10 dark:border-gray-600/30 text-center">
              <div className="flex items-center justify-center gap-2 mb-1">
                <Calendar size={14} className="text-purple-400" />
                <p className="text-xs text-gray-400 dark:text-gray-500">Days</p>
              </div>
              <p className="text-lg font-bold text-white dark:text-gray-100">
                {observationDays}
              </p>
            </div>
            <div className="p-3 bg-white/5 dark:bg-gray-700/30 rounded-lg border border-white/10 dark:border-gray-600/30 text-center">
              <p className="text-xs text-gray-400 dark:text-gray-500 mb-1">Data Points</p>
              <p className="text-lg font-bold text-white dark:text-gray-100">
                {dataPoints.toLocaleString()}
              </p>
            </div>
          </div>

          {/* Quality Indicator */}
          <div className="p-4 bg-gradient-to-r from-green-500/10 to-blue-500/10 rounded-lg border border-green-500/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-green-300 font-medium">Data Quality</p>
                <p className="text-xs text-green-400">
                  {dataPoints > 10000 ? 'Excellent' : dataPoints > 5000 ? 'Good' : 'Fair'} coverage
                </p>
              </div>
              <div className="flex items-center gap-1">
                {Array.from({ length: 5 }).map((_, i) => (
                  <div
                    key={i}
                    className={`w-2 h-2 rounded-full ${
                      i < (dataPoints > 15000 ? 5 : dataPoints > 10000 ? 4 : dataPoints > 5000 ? 3 : 2)
                        ? 'bg-green-400'
                        : 'bg-gray-600'
                    }`}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TargetInfo
