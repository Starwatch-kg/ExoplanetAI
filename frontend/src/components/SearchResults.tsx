import React from 'react'
import { motion } from 'framer-motion'
import { 
  CheckCircle, 
  XCircle, 
  Star, 
  Globe, 
  Clock, 
  Zap, 
  Brain,
  BarChart3,
  Info,
  TrendingUp,
  Thermometer,
  Ruler,
  Weight
} from 'lucide-react'
import { SearchResult } from '../types/api'

interface SearchResultsProps {
  results: SearchResult
}

const SearchResults: React.FC<SearchResultsProps> = ({ results }) => {
  const formatNumber = (num: number, decimals: number = 2) => {
    return num.toFixed(decimals)
  }

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms}ms`
    return `${(ms / 1000).toFixed(1)}s`
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-400'
    if (confidence >= 0.6) return 'text-yellow-400'
    return 'text-red-400'
  }

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.9) return 'Very High'
    if (confidence >= 0.8) return 'High'
    if (confidence >= 0.6) return 'Medium'
    if (confidence >= 0.4) return 'Low'
    return 'Very Low'
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="bg-gray-900/50 rounded-xl border border-gray-800 p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-2xl font-semibold text-white flex items-center">
            <Star className="h-6 w-6 mr-3 text-blue-400" />
            {results.target_name}
          </h3>
          <div className="flex items-center space-x-4">
            <span className="px-3 py-1 bg-blue-900/50 text-blue-300 text-sm rounded-full border border-blue-800">
              {results.catalog}
            </span>
            <span className="px-3 py-1 bg-purple-900/50 text-purple-300 text-sm rounded-full border border-purple-800">
              {results.mission}
            </span>
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Candidates Found:</span>
            <span className="text-white ml-2 font-semibold">{results.candidates_found || 0}</span>
          </div>
          <div>
            <span className="text-gray-400">Processing Time:</span>
            <span className="text-white ml-2">{formatTime(results.processing_time_ms || 0)}</span>
          </div>
          <div>
            <span className="text-gray-400">Status:</span>
            <span className="text-green-400 ml-2 capitalize">{results.status || 'unknown'}</span>
          </div>
          <div>
            <span className="text-gray-400">Data Points:</span>
            <span className="text-white ml-2">{results.lightcurve_info?.points_count?.toLocaleString() || 'N/A'}</span>
          </div>
        </div>
      </motion.div>

      {/* Star Information */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
        className="bg-gray-900/50 rounded-xl border border-gray-800 p-6"
      >
        <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Star className="h-5 w-5 mr-2 text-yellow-400" />
          Stellar Properties
        </h4>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div className="text-center p-4 bg-black/30 rounded-lg">
            <Thermometer className="h-6 w-6 text-orange-400 mx-auto mb-2" />
            <div className="text-sm text-gray-400">Temperature</div>
            <div className="text-lg font-semibold text-white">{results.star_info?.temperature || 'N/A'}K</div>
          </div>
          
          <div className="text-center p-4 bg-black/30 rounded-lg">
            <Zap className="h-6 w-6 text-yellow-400 mx-auto mb-2" />
            <div className="text-sm text-gray-400">Magnitude</div>
            <div className="text-lg font-semibold text-white">{results.star_info?.magnitude ? formatNumber(results.star_info.magnitude) : 'N/A'}</div>
          </div>
          
          <div className="text-center p-4 bg-black/30 rounded-lg">
            <Ruler className="h-6 w-6 text-blue-400 mx-auto mb-2" />
            <div className="text-sm text-gray-400">Radius</div>
            <div className="text-lg font-semibold text-white">{results.star_info?.radius ? formatNumber(results.star_info.radius) + ' R☉' : 'N/A'}</div>
          </div>
          
          <div className="text-center p-4 bg-black/30 rounded-lg">
            <Weight className="h-6 w-6 text-purple-400 mx-auto mb-2" />
            <div className="text-sm text-gray-400">Mass</div>
            <div className="text-lg font-semibold text-white">{results.star_info?.mass ? formatNumber(results.star_info.mass) + ' M☉' : 'N/A'}</div>
          </div>
        </div>
        
        <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Target ID:</span>
            <span className="text-white ml-2 font-mono">{results.star_info?.target_id || 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-400">RA:</span>
            <span className="text-white ml-2">{results.star_info?.ra ? formatNumber(results.star_info.ra, 6) + '°' : 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-400">Dec:</span>
            <span className="text-white ml-2">{results.star_info?.dec ? formatNumber(results.star_info.dec, 6) + '°' : 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-400">Stellar Type:</span>
            <span className="text-white ml-2">{results.star_info?.stellar_type || 'N/A'}</span>
          </div>
        </div>
      </motion.div>

      {/* BLS Results */}
      {results.bls_result && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="bg-gray-900/50 rounded-xl border border-gray-800 p-6"
        >
          <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
            <BarChart3 className="h-5 w-5 mr-2 text-cyan-400" />
            BLS Analysis Results
          </h4>
          
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center p-4 bg-black/30 rounded-lg">
              <Clock className="h-6 w-6 text-blue-400 mx-auto mb-2" />
              <div className="text-sm text-gray-400">Period</div>
              <div className="text-lg font-semibold text-white">{formatNumber(results.bls_result.best_period)} days</div>
            </div>
            
            <div className="text-center p-4 bg-black/30 rounded-lg">
              <TrendingUp className="h-6 w-6 text-green-400 mx-auto mb-2" />
              <div className="text-sm text-gray-400">Power</div>
              <div className="text-lg font-semibold text-white">{formatNumber(results.bls_result.best_power)}</div>
            </div>
            
            <div className="text-center p-4 bg-black/30 rounded-lg">
              <Zap className="h-6 w-6 text-yellow-400 mx-auto mb-2" />
              <div className="text-sm text-gray-400">SNR</div>
              <div className="text-lg font-semibold text-white">{formatNumber(results.bls_result.snr)}</div>
            </div>
            
            <div className="text-center p-4 bg-black/30 rounded-lg">
              <Globe className="h-6 w-6 text-purple-400 mx-auto mb-2" />
              <div className="text-sm text-gray-400">Depth</div>
              <div className="text-lg font-semibold text-white">{formatNumber(results.bls_result.depth * 100, 3)}%</div>
            </div>
          </div>
          
          <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Duration:</span>
              <span className="text-white ml-2">{formatNumber(results.bls_result.best_duration)} hours</span>
            </div>
            <div>
              <span className="text-gray-400">T0:</span>
              <span className="text-white ml-2">{formatNumber(results.bls_result.best_t0)}</span>
            </div>
            <div>
              <span className="text-gray-400">Depth Error:</span>
              <span className="text-white ml-2">±{formatNumber(results.bls_result.depth_err * 100, 4)}%</span>
            </div>
            <div>
              <span className="text-gray-400">Significance:</span>
              <span className="text-white ml-2">{formatNumber(results.bls_result.significance)}σ</span>
            </div>
            <div>
              <span className="text-gray-400">ML Confidence:</span>
              <span className={`ml-2 font-semibold ${getConfidenceColor(results.bls_result.ml_confidence)}`}>
                {formatNumber(results.bls_result.ml_confidence * 100)}%
              </span>
            </div>
            <div className="flex items-center">
              <span className="text-gray-400">Significant:</span>
              {results.bls_result.is_significant ? (
                <CheckCircle className="h-4 w-4 text-green-400 ml-2" />
              ) : (
                <XCircle className="h-4 w-4 text-red-400 ml-2" />
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* AI Analysis Results */}
      {results.ai_result && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="bg-gray-900/50 rounded-xl border border-gray-800 p-6"
        >
          <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Brain className="h-5 w-5 mr-2 text-pink-400" />
            AI Analysis Results
          </h4>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-6 bg-black/30 rounded-lg">
              <div className="text-sm text-gray-400 mb-2">Prediction Score</div>
              <div className="text-3xl font-bold text-white mb-2">
                {formatNumber(results.ai_result.prediction * 100)}%
              </div>
              <div className={`text-sm font-medium ${getConfidenceColor(results.ai_result.prediction)}`}>
                {getConfidenceLabel(results.ai_result.prediction)}
              </div>
            </div>
            
            <div className="text-center p-6 bg-black/30 rounded-lg">
              <div className="text-sm text-gray-400 mb-2">Confidence</div>
              <div className="text-3xl font-bold text-white mb-2">
                {formatNumber(results.ai_result.confidence * 100)}%
              </div>
              <div className={`text-sm font-medium ${getConfidenceColor(results.ai_result.confidence)}`}>
                {getConfidenceLabel(results.ai_result.confidence)}
              </div>
            </div>
            
            <div className="text-center p-6 bg-black/30 rounded-lg">
              <div className="text-sm text-gray-400 mb-2">Planet Candidate</div>
              <div className="flex items-center justify-center mb-2">
                {results.ai_result.is_candidate ? (
                  <CheckCircle className="h-8 w-8 text-green-400" />
                ) : (
                  <XCircle className="h-8 w-8 text-red-400" />
                )}
              </div>
              <div className={`text-sm font-medium ${results.ai_result.is_candidate ? 'text-green-400' : 'text-red-400'}`}>
                {results.ai_result.is_candidate ? 'Yes' : 'No'}
              </div>
            </div>
          </div>
          
          <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-400">Model Used:</span>
              <span className="text-white ml-2 font-mono">{results.ai_result.model_used}</span>
            </div>
            <div>
              <span className="text-gray-400">Inference Time:</span>
              <span className="text-white ml-2">{formatTime(results.ai_result.inference_time_ms)}</span>
            </div>
          </div>
          
          {results.ai_result.error && (
            <div className="mt-4 p-3 bg-red-900/20 border border-red-800 rounded-lg">
              <div className="text-sm text-red-400">
                <strong>Error:</strong> {results.ai_result.error}
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* Light Curve Information */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="bg-gray-900/50 rounded-xl border border-gray-800 p-6"
      >
        <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Info className="h-5 w-5 mr-2 text-blue-400" />
          Light Curve Information
        </h4>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Data Points:</span>
            <span className="text-white ml-2">{results.lightcurve_info?.points_count?.toLocaleString() || 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-400">Time Span:</span>
            <span className="text-white ml-2">{results.lightcurve_info?.time_span_days ? formatNumber(results.lightcurve_info.time_span_days) + ' days' : 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-400">Cadence:</span>
            <span className="text-white ml-2">{results.lightcurve_info?.cadence_minutes ? formatNumber(results.lightcurve_info.cadence_minutes) + ' min' : 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-400">Noise Level:</span>
            <span className="text-white ml-2">{results.lightcurve_info?.noise_level_ppm ? formatNumber(results.lightcurve_info.noise_level_ppm) + ' ppm' : 'N/A'}</span>
          </div>
          <div>
            <span className="text-gray-400">Data Source:</span>
            <span className="text-white ml-2">{results.lightcurve_info?.data_source || 'N/A'}</span>
          </div>
        </div>
      </motion.div>

      {/* Physical Parameters (if available) */}
      {results.physical_parameters && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="bg-gray-900/50 rounded-xl border border-gray-800 p-6"
        >
          <h4 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Globe className="h-5 w-5 mr-2 text-green-400" />
            Estimated Planet Parameters
          </h4>
          
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            {results.physical_parameters.planet_radius && (
              <div>
                <span className="text-gray-400">Planet Radius:</span>
                <span className="text-white ml-2">{formatNumber(results.physical_parameters.planet_radius)} R⊕</span>
              </div>
            )}
            {results.physical_parameters.planet_mass && (
              <div>
                <span className="text-gray-400">Planet Mass:</span>
                <span className="text-white ml-2">{formatNumber(results.physical_parameters.planet_mass)} M⊕</span>
              </div>
            )}
            {results.physical_parameters.orbital_period && (
              <div>
                <span className="text-gray-400">Orbital Period:</span>
                <span className="text-white ml-2">{formatNumber(results.physical_parameters.orbital_period)} days</span>
              </div>
            )}
            {results.physical_parameters.equilibrium_temperature && (
              <div>
                <span className="text-gray-400">Equilibrium Temp:</span>
                <span className="text-white ml-2">{formatNumber(results.physical_parameters.equilibrium_temperature)} K</span>
              </div>
            )}
            {results.physical_parameters.habitability_score && (
              <div>
                <span className="text-gray-400">Habitability Score:</span>
                <span className="text-white ml-2">{formatNumber(results.physical_parameters.habitability_score * 100)}%</span>
              </div>
            )}
          </div>
        </motion.div>
      )}
    </div>
  )
}

export default SearchResults
