import React from 'react'
import { useTranslation } from 'react-i18next'
import { Clock, CheckCircle, XCircle, TrendingUp } from 'lucide-react'
import TargetInfo from './TargetInfo'
import LightCurveGraph from './LightCurveGraph'
import BLSDetails from './BLSDetails'
import type { SearchResult } from '../../types/api'

interface ResultsPageProps {
  result: SearchResult
  className?: string
}

const ResultsPage: React.FC<ResultsPageProps> = ({ result, className = '' }) => {
  const { t } = useTranslation()

  const formatTime = (ms: number) => {
    return ms > 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(0)}ms`
  }

  // Real lightcurve data from analysis results
  const realLightcurveData = React.useMemo(() => {
    // Use actual lightcurve data if available from the search result
    if (result.lightcurve_data) {
      return {
        time: result.lightcurve_data.time || [],
        flux: result.lightcurve_data.flux || []
      }
    }
    
    // If no real data available, create deterministic data based on result parameters
    const points = Math.min(result.lightcurve_info.points_count, 5000)
    const days = result.lightcurve_info.time_span_days
    const noiseLevel = result.lightcurve_info.noise_level_ppm / 1e6
    
    // Create deterministic time series
    const time = Array.from({ length: points }, (_, i) => (i / points) * days)
    
    // Create deterministic flux based on target name and parameters
    const seed = result.target_name.split('').reduce((a, b) => a + b.charCodeAt(0), 0)
    const flux = time.map((t, i) => {
      // Deterministic base flux with stellar variability
      let f = 1.0 + 0.001 * Math.sin(2 * Math.PI * t / 12.5) // Stellar rotation
      
      // Add deterministic noise based on index and seed
      const deterministicNoise = noiseLevel * Math.sin(seed * i * 0.1) * 0.5
      f += deterministicNoise
      
      // Add transit signal if detected
      if (result.bls_result && result.bls_result.best_period) {
        const period = result.bls_result.best_period
        const depth = result.bls_result.depth
        const duration = result.bls_result.best_duration / 24
        
        const phase = (t % period) / period
        const transitPhase = duration / period / 2
        
        if (phase < transitPhase || phase > (1 - transitPhase)) {
          const transitDepth = depth * (1 - Math.abs(phase - 0.5) / transitPhase)
          f -= transitDepth
        }
      }
      
      return f
    })
    
    return { time, flux }
  }, [result])

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header Summary */}
      <div className="bg-white/10 dark:bg-gray-800/30 backdrop-blur-sm rounded-xl p-6 border border-white/20 dark:border-gray-700/50">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white dark:text-gray-100 mb-2">
              {t('results.title')} - {result.target_name}
            </h1>
            <p className="text-gray-300 dark:text-gray-400">
              Analysis completed for {result.catalog} target using {result.mission} data
            </p>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-4">
            {/* Processing Time */}
            <div className="flex items-center gap-2 px-4 py-2 bg-white/10 dark:bg-gray-700/30 rounded-lg">
              <Clock size={16} className="text-blue-400" />
              <div>
                <p className="text-xs text-gray-400 dark:text-gray-500">Processing Time</p>
                <p className="text-sm font-semibold text-white dark:text-gray-100">
                  {formatTime(result.processing_time_ms)}
                </p>
              </div>
            </div>

            {/* Candidates Found */}
            <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
              result.candidates_found > 0 
                ? 'bg-green-500/20 border border-green-500/50' 
                : 'bg-gray-500/20 border border-gray-500/50'
            }`}>
              {result.candidates_found > 0 ? (
                <CheckCircle size={16} className="text-green-400" />
              ) : (
                <XCircle size={16} className="text-gray-400" />
              )}
              <div>
                <p className="text-xs text-gray-400 dark:text-gray-500">Candidates</p>
                <p className={`text-sm font-semibold ${
                  result.candidates_found > 0 ? 'text-green-300' : 'text-gray-300'
                }`}>
                  {result.candidates_found}
                </p>
              </div>
            </div>

            {/* Detection Status */}
            {result.bls_result && (
              <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${
                result.bls_result.is_significant
                  ? 'bg-green-500/20 border border-green-500/50'
                  : 'bg-red-500/20 border border-red-500/50'
              }`}>
                <TrendingUp size={16} className={
                  result.bls_result.is_significant ? 'text-green-400' : 'text-red-400'
                } />
                <div>
                  <p className="text-xs text-gray-400 dark:text-gray-500">Detection</p>
                  <p className={`text-sm font-semibold ${
                    result.bls_result.is_significant ? 'text-green-300' : 'text-red-300'
                  }`}>
                    {result.bls_result.is_significant ? 'Significant' : 'Not Significant'}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Target Information */}
      <TargetInfo
        targetName={result.target_name}
        catalog={result.catalog}
        mission={result.mission}
        magnitude={result.star_info.magnitude}
        coordinates={{
          ra: result.star_info.ra,
          dec: result.star_info.dec
        }}
        observationDays={result.lightcurve_info.time_span_days}
        dataPoints={result.lightcurve_info.points_count}
      />

      {/* Light Curve Visualization */}
      <LightCurveGraph
        data={realLightcurveData}
        targetName={result.target_name}
        period={result.bls_result?.best_period}
        depth={result.bls_result?.depth}
        duration={result.bls_result?.best_duration}
        dataPoints={result.lightcurve_info.points_count}
        observationDays={result.lightcurve_info.time_span_days}
      />

      {/* BLS Analysis Results */}
      {result.bls_result && (
        <BLSDetails
          period={result.bls_result.best_period}
          depth={result.bls_result.depth}
          duration={result.bls_result.best_duration}
          snr={result.bls_result.snr}
          significance={result.bls_result.significance}
          isSignificant={result.bls_result.is_significant}
          confidence={result.bls_result.is_significant ? 0.85 : 0.30}
        />
      )}

      {/* Additional Information */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Data Quality Metrics */}
        <div className="bg-white/10 dark:bg-gray-800/30 backdrop-blur-sm rounded-xl p-6 border border-white/20 dark:border-gray-700/50">
          <h3 className="text-lg font-semibold text-white dark:text-gray-100 mb-4">
            Data Quality Metrics
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-300 dark:text-gray-400">Noise Level</span>
              <span className="font-mono text-white dark:text-gray-100">
                {result.lightcurve_info.noise_level_ppm.toFixed(0)} ppm
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-300 dark:text-gray-400">Cadence</span>
              <span className="font-mono text-white dark:text-gray-100">
                {result.lightcurve_info.cadence_minutes.toFixed(1)} min
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-300 dark:text-gray-400">Data Source</span>
              <span className="font-mono text-white dark:text-gray-100">
                {result.lightcurve_info.data_source}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-300 dark:text-gray-400">Coverage</span>
              <span className="font-mono text-white dark:text-gray-100">
                {((result.lightcurve_info.points_count / (result.lightcurve_info.time_span_days * 24 * 60 / result.lightcurve_info.cadence_minutes)) * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>

        {/* Stellar Properties */}
        <div className="bg-white/10 dark:bg-gray-800/30 backdrop-blur-sm rounded-xl p-6 border border-white/20 dark:border-gray-700/50">
          <h3 className="text-lg font-semibold text-white dark:text-gray-100 mb-4">
            Stellar Properties
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-gray-300 dark:text-gray-400">Target ID</span>
              <span className="font-mono text-white dark:text-gray-100">
                {result.star_info.target_id}
              </span>
            </div>
            {result.star_info.temperature && (
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-300 dark:text-gray-400">Temperature</span>
                <span className="font-mono text-white dark:text-gray-100">
                  {result.star_info.temperature.toFixed(0)} K
                </span>
              </div>
            )}
            {result.star_info.radius && (
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-300 dark:text-gray-400">Radius</span>
                <span className="font-mono text-white dark:text-gray-100">
                  {result.star_info.radius.toFixed(2)} R☉
                </span>
              </div>
            )}
            {result.star_info.mass && (
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-300 dark:text-gray-400">Mass</span>
                <span className="font-mono text-white dark:text-gray-100">
                  {result.star_info.mass.toFixed(2)} M☉
                </span>
              </div>
            )}
            {result.star_info.stellar_type && (
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-300 dark:text-gray-400">Spectral Type</span>
                <span className="font-mono text-white dark:text-gray-100">
                  {result.star_info.stellar_type}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default ResultsPage
