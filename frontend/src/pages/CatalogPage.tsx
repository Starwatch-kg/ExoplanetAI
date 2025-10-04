import React, { useState, useEffect } from 'react'
import { Search, Globe, Star, Thermometer, Ruler, Clock, MapPin, Sparkles, Target, TrendingUp } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { motion } from 'framer-motion'

interface ExoplanetData {
  id: string
  name: string
  host_star: string
  discovery_method: string
  discovery_year: number
  orbital_period_days: number
  radius_earth_radii: number
  mass_earth_masses: number
  equilibrium_temperature_k: number
  distance_parsecs: number
  confidence: number
  status: string
  habitable_zone: boolean
  created_at?: string
  updated_at?: string
}

interface CatalogResponse {
  total: number
  limit: number
  offset: number
  filters: {
    method?: string
    min_confidence?: number
    habitable_only: boolean
  }
  statistics: {
    confirmed_planets: number
    habitable_zone_planets: number
    average_confidence: number
  }
  exoplanets: ExoplanetData[]
  timestamp: string
  source: string
}

const CatalogPage: React.FC = () => {
  const { t } = useTranslation()
  const [exoplanets, setExoplanets] = useState<ExoplanetData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [selectedMethod, setSelectedMethod] = useState<string>('')
  const [minConfidence, setMinConfidence] = useState<number>(0)
  const [habitableOnly, setHabitableOnly] = useState(false)
  const [statistics, setStatistics] = useState<any>(null)
  const [currentPage, setCurrentPage] = useState(0)
  const [selectedPlanet, setSelectedPlanet] = useState<ExoplanetData | null>(null)

  const itemsPerPage = 12

  const fetchExoplanets = async () => {
    setLoading(true)
    setError(null)

    try {
      const params = new URLSearchParams({
        limit: itemsPerPage.toString(),
        offset: (currentPage * itemsPerPage).toString(),
        habitable_only: habitableOnly.toString()
      })

      if (selectedMethod) params.append('method', selectedMethod)
      if (minConfidence > 0) params.append('min_confidence', minConfidence.toString())

      const response = await fetch(`/api/v1/catalog/exoplanets?${params}`)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: CatalogResponse = await response.json()
      setExoplanets(data.exoplanets)
      setStatistics(data.statistics)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch exoplanet catalog')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchExoplanets()
  }, [currentPage, selectedMethod, minConfidence, habitableOnly])

  const filteredExoplanets = exoplanets.filter(planet =>
    planet.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    planet.host_star.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const getMethodColor = (method: string) => {
    switch (method.toLowerCase()) {
      case 'transit': return 'bg-blue-500/20 text-blue-300 border-blue-500/30'
      case 'radial_velocity': return 'bg-green-500/20 text-green-300 border-green-500/30'
      case 'microlensing': return 'bg-purple-500/20 text-purple-300 border-purple-500/30'
      case 'direct_imaging': return 'bg-orange-500/20 text-orange-300 border-orange-500/30'
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'confirmed': return 'bg-green-500/20 text-green-300 border-green-500/30'
      case 'candidate': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30'
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30'
    }
  }

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
            className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-green-500 via-blue-600 to-purple-600 rounded-full mb-6 shadow-lg"
          >
            <Globe className="w-10 h-10 text-white" />
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-5xl font-bold bg-gradient-to-r from-white via-green-200 to-blue-200 bg-clip-text text-transparent mb-4"
          >
            {t('catalog.title')}
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed"
          >
            {t('catalog.subtitle')}
          </motion.p>
          
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.8 }}
            className="flex justify-center items-center gap-4 mt-6"
          >
            <div className="flex items-center gap-2 bg-green-500/20 backdrop-blur-sm border border-green-500/30 rounded-full px-4 py-2">
              <Target className="w-4 h-4 text-green-400" />
              <span className="text-green-300 text-sm font-medium">Exoplanet Database</span>
            </div>
            <div className="flex items-center gap-2 bg-blue-500/20 backdrop-blur-sm border border-blue-500/30 rounded-full px-4 py-2">
              <Sparkles className="w-4 h-4 text-blue-400" />
              <span className="text-blue-300 text-sm font-medium">Real NASA Data</span>
            </div>
          </motion.div>
        </motion.div>

        {/* Statistics */}
        {statistics && (
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12"
          >
            <motion.div 
              whileHover={{ scale: 1.05, y: -5 }}
              className="group bg-gradient-to-br from-green-500/10 to-emerald-500/10 backdrop-blur-sm rounded-2xl p-8 border border-green-500/20 hover:border-green-400/50 transition-all duration-500 shadow-xl"
            >
              <div className="flex items-center gap-4">
                <motion.div 
                  whileHover={{ rotate: 360, scale: 1.1 }}
                  transition={{ duration: 0.6 }}
                  className="w-16 h-16 bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded-xl flex items-center justify-center shadow-lg"
                >
                  <Globe className="w-8 h-8 text-green-400" />
                </motion.div>
                <div>
                  <p className="text-3xl font-bold text-white group-hover:text-green-300 transition-colors">{statistics.confirmed_planets}</p>
                  <p className="text-gray-300 text-sm font-medium">Confirmed Planets</p>
                </div>
              </div>
            </motion.div>

            <motion.div 
              whileHover={{ scale: 1.05, y: -5 }}
              className="group bg-gradient-to-br from-blue-500/10 to-cyan-500/10 backdrop-blur-sm rounded-2xl p-8 border border-blue-500/20 hover:border-blue-400/50 transition-all duration-500 shadow-xl"
            >
              <div className="flex items-center gap-4">
                <motion.div 
                  whileHover={{ rotate: 360, scale: 1.1 }}
                  transition={{ duration: 0.6 }}
                  className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-xl flex items-center justify-center shadow-lg"
                >
                  <Star className="w-8 h-8 text-blue-400" />
                </motion.div>
                <div>
                  <p className="text-3xl font-bold text-white group-hover:text-blue-300 transition-colors">{statistics.habitable_zone_planets}</p>
                  <p className="text-gray-300 text-sm font-medium">Habitable Zone</p>
                </div>
              </div>
            </motion.div>

            <motion.div 
              whileHover={{ scale: 1.05, y: -5 }}
              className="group bg-gradient-to-br from-purple-500/10 to-pink-500/10 backdrop-blur-sm rounded-2xl p-8 border border-purple-500/20 hover:border-purple-400/50 transition-all duration-500 shadow-xl"
            >
              <div className="flex items-center gap-4">
                <motion.div 
                  whileHover={{ rotate: 360, scale: 1.1 }}
                  transition={{ duration: 0.6 }}
                  className="w-16 h-16 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center shadow-lg"
                >
                  <TrendingUp className="w-8 h-8 text-purple-400" />
                </motion.div>
                <div>
                  <p className="text-3xl font-bold text-white group-hover:text-purple-300 transition-colors">{(statistics.average_confidence * 100).toFixed(1)}%</p>
                  <p className="text-gray-300 text-sm font-medium">Avg Confidence</p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}

        {/* Filters */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/20 shadow-xl mb-12"
        >
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                {t('catalog.search')}
              </label>
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-12 pr-4 py-3 bg-white/10 border border-white/20 rounded-xl text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-400 transition-all duration-300"
                  placeholder={t('catalog.searchPlaceholder')}
                />
                <div className="absolute inset-y-0 right-0 flex items-center pr-3">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Discovery Method
              </label>
              <select
                value={selectedMethod}
                onChange={(e) => setSelectedMethod(e.target.value)}
                className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Methods</option>
                <option value="transit">Transit</option>
                <option value="radial_velocity">Radial Velocity</option>
                <option value="microlensing">Microlensing</option>
                <option value="direct_imaging">Direct Imaging</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Min Confidence
              </label>
              <select
                value={minConfidence}
                onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value={0}>Any Confidence</option>
                <option value={0.5}>50%+</option>
                <option value={0.7}>70%+</option>
                <option value={0.9}>90%+</option>
                <option value={0.95}>95%+</option>
              </select>
            </div>

            <div className="flex items-end">
              <label className="flex items-center gap-2 text-sm font-medium text-gray-300">
                <input
                  type="checkbox"
                  checked={habitableOnly}
                  onChange={(e) => setHabitableOnly(e.target.checked)}
                  className="w-4 h-4 text-blue-600 bg-white/10 border-white/20 rounded focus:ring-blue-500"
                />
                Habitable Zone Only
              </label>
            </div>
          </div>
        </motion.div>

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="animate-spin w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-gray-300">Loading exoplanet catalog...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 mb-8">
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {/* Exoplanet Grid */}
        {!loading && !error && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredExoplanets.map((planet, index) => (
              <motion.div
                key={planet.id || planet.name || index}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: Math.random() * 0.3 }}
                whileHover={{ scale: 1.05, y: -10 }}
                whileTap={{ scale: 0.95 }}
                className="group bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-6 border border-white/20 hover:border-white/40 transition-all duration-500 cursor-pointer shadow-xl hover:shadow-2xl"
                onClick={() => setSelectedPlanet(planet)}
              >
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-1">
                      {planet.name}
                    </h3>
                    <p className="text-gray-300 text-sm">
                      Host: {planet.host_star}
                    </p>
                  </div>
                  {planet.habitable_zone && (
                    <div className="w-3 h-3 bg-green-400 rounded-full" title="Habitable Zone" />
                  )}
                </div>

                <div className="space-y-2 mb-4">
                  <div className="flex items-center gap-2 text-sm">
                    <Clock className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-300">
                      {planet.orbital_period_days ? planet.orbital_period_days.toFixed(1) : 'N/A'} days
                    </span>
                  </div>
                  
                  <div className="flex items-center gap-2 text-sm">
                    <Ruler className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-300">
                      {planet.radius_earth_radii ? planet.radius_earth_radii.toFixed(2) : 'N/A'} R⊕
                    </span>
                  </div>
                  
                  <div className="flex items-center gap-2 text-sm">
                    <Thermometer className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-300">
                      {planet.equilibrium_temperature_k ? planet.equilibrium_temperature_k.toFixed(0) : 'N/A'} K
                    </span>
                  </div>
                  
                  <div className="flex items-center gap-2 text-sm">
                    <MapPin className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-300">
                      {planet.distance_parsecs ? planet.distance_parsecs.toFixed(1) : 'N/A'} pc
                    </span>
                  </div>
                </div>

                <div className="flex flex-wrap gap-2 mb-4">
                  <span className={`text-xs px-2 py-1 rounded border ${getMethodColor(planet.discovery_method || 'Unknown')}`}>
                    {planet.discovery_method ? planet.discovery_method.replace('_', ' ') : 'Unknown'}
                  </span>
                  <span className={`text-xs px-2 py-1 rounded border ${getStatusColor(planet.status)}`}>
                    {planet.status}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">
                    Discovered {planet.discovery_year}
                  </span>
                  <span className="text-xs text-gray-300">
                    {(planet.confidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
              </motion.div>
            ))}
          </div>
        )}

        {/* Pagination */}
        {!loading && !error && filteredExoplanets.length > 0 && (
          <div className="flex justify-center mt-8 gap-2">
            <motion.button
              onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
              disabled={currentPage === 0}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-3 bg-gradient-to-r from-blue-600/20 to-purple-600/20 hover:from-blue-600/30 hover:to-purple-600/30 border border-blue-500/30 hover:border-blue-400/50 disabled:opacity-50 text-white rounded-xl transition-all duration-300 backdrop-blur-sm"
            >
              Previous
            </motion.button>
            <span className="px-6 py-3 text-gray-300 font-medium">
              Page {currentPage + 1}
            </span>
            <motion.button
              onClick={() => setCurrentPage(currentPage + 1)}
              disabled={filteredExoplanets.length < itemsPerPage}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              className="px-6 py-3 bg-gradient-to-r from-blue-600/20 to-purple-600/20 hover:from-blue-600/30 hover:to-purple-600/30 border border-blue-500/30 hover:border-blue-400/50 disabled:opacity-50 text-white rounded-xl transition-all duration-300 backdrop-blur-sm"
            >
              Next
            </motion.button>
          </div>
        )}

        {/* Planet Detail Modal */}
        {selectedPlanet && (
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
            <div className="bg-gray-900/95 backdrop-blur-sm rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-white/20">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <h2 className="text-2xl font-bold text-white mb-2">
                    {selectedPlanet.name}
                  </h2>
                  <p className="text-gray-300">
                    Orbiting {selectedPlanet.host_star}
                  </p>
                </div>
                <button
                  onClick={() => setSelectedPlanet(null)}
                  className="text-gray-400 hover:text-white text-2xl"
                >
                  ×
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Physical Properties</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Radius:</span>
                        <span className="text-white">{selectedPlanet.radius_earth_radii.toFixed(2)} Earth radii</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Mass:</span>
                        <span className="text-white">{selectedPlanet.mass_earth_masses.toFixed(2)} Earth masses</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Temperature:</span>
                        <span className="text-white">{selectedPlanet.equilibrium_temperature_k.toFixed(0)} K</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Orbital Properties</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Period:</span>
                        <span className="text-white">{selectedPlanet.orbital_period_days.toFixed(2)} days</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Distance:</span>
                        <span className="text-white">{selectedPlanet.distance_parsecs.toFixed(1)} parsecs</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Discovery Info</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Method:</span>
                        <span className="text-white">{selectedPlanet.discovery_method.replace('_', ' ')}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Year:</span>
                        <span className="text-white">{selectedPlanet.discovery_year}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Status:</span>
                        <span className="text-white">{selectedPlanet.status}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Confidence:</span>
                        <span className="text-white">{(selectedPlanet.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold text-white mb-3">Habitability</h3>
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${selectedPlanet.habitable_zone ? 'bg-green-400' : 'bg-red-400'}`} />
                      <span className="text-white text-sm">
                        {selectedPlanet.habitable_zone ? 'In Habitable Zone' : 'Outside Habitable Zone'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default CatalogPage
