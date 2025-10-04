import React, { useState, useEffect } from 'react'
import { Database, BarChart3, Clock, TrendingUp, Search, Trash2, RefreshCw, Target, Activity, Zap, CheckCircle } from 'lucide-react'
import { useTranslation } from 'react-i18next'
import { motion } from 'framer-motion'

interface DatabaseStats {
  total_exoplanets: number
  confirmed_exoplanets: number
  habitable_zone_planets: number
  average_confidence: number
  total_searches: number
  searches_last_24h: number
  searches_by_method: Record<string, number>
}

interface SearchHistoryItem {
  id: number
  target_name: string
  catalog: string
  mission: string
  method: string
  exoplanet_detected: boolean
  detection_confidence: number
  processing_time_ms: number
  result_data: any
  created_at: string
}

interface SystemMetric {
  id: number
  timestamp: string
  service_name: string
  metric_name: string
  metric_value: number
  metadata: any
}

const DatabasePage: React.FC = () => {
  const { t } = useTranslation()
  const [stats, setStats] = useState<DatabaseStats | null>(null)
  const [searchHistory, setSearchHistory] = useState<SearchHistoryItem[]>([])
  const [metrics, setMetrics] = useState<SystemMetric[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'stats' | 'history' | 'metrics'>('stats')
  const [selectedMethod, setSelectedMethod] = useState<string>('')

  const fetchDatabaseStats = async () => {
    try {
      const response = await fetch('/api/v1/database/statistics')
      if (!response.ok) throw new Error('Failed to fetch statistics')
      const data = await response.json()
      setStats(data.database_statistics)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch database statistics')
    }
  }

  const fetchSearchHistory = async () => {
    try {
      const params = new URLSearchParams({
        limit: '50'
      })
      if (selectedMethod) params.append('method', selectedMethod)

      const response = await fetch(`/api/v1/database/search-history?${params}`)
      if (!response.ok) throw new Error('Failed to fetch search history')
      const data = await response.json()
      setSearchHistory(data.search_history)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch search history')
    }
  }

  const fetchMetrics = async () => {
    try {
      const response = await fetch('/api/v1/database/metrics?hours=24')
      if (!response.ok) throw new Error('Failed to fetch metrics')
      const data = await response.json()
      setMetrics(data.metrics)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch metrics')
    }
  }

  const cleanupOldData = async () => {
    try {
      const response = await fetch('/api/v1/database/cleanup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ days: 30 })
      })
      if (!response.ok) throw new Error('Failed to cleanup data')
      alert('Old data cleaned up successfully!')
      fetchData()
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Failed to cleanup data')
    }
  }

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    
    try {
      await Promise.all([
        fetchDatabaseStats(),
        fetchSearchHistory(),
        fetchMetrics()
      ])
    } catch (err) {
      // Error handling is done in individual functions
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  useEffect(() => {
    if (activeTab === 'history') {
      fetchSearchHistory()
    }
  }, [selectedMethod, activeTab])

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  const getMethodColor = (method: string | undefined) => {
    if (!method) return 'bg-gray-500/20 text-gray-300 border-gray-500/30'
    
    switch (method.toLowerCase()) {
      case 'bls': return 'bg-blue-500/20 text-blue-300 border-blue-500/30'
      case 'gpi': return 'bg-purple-500/20 text-purple-300 border-purple-500/30'
      case 'unified': return 'bg-green-500/20 text-green-300 border-green-500/30'
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
            className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-r from-indigo-500 via-purple-600 to-pink-600 rounded-full mb-6 shadow-lg"
          >
            <Database className="w-10 h-10 text-white" />
          </motion.div>
          
          <motion.h1 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-5xl font-bold bg-gradient-to-r from-white via-indigo-200 to-purple-200 bg-clip-text text-transparent mb-4"
          >
            {t('database.title')}
          </motion.h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
            className="text-xl text-gray-300 max-w-3xl mx-auto leading-relaxed"
          >
            {t('database.subtitle')}
          </motion.p>
          
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.8 }}
            className="flex justify-center items-center gap-4 mt-6"
          >
            <div className="flex items-center gap-2 bg-indigo-500/20 backdrop-blur-sm border border-indigo-500/30 rounded-full px-4 py-2">
              <Target className="w-4 h-4 text-indigo-400" />
              <span className="text-indigo-300 text-sm font-medium">Data Analytics</span>
            </div>
            <div className="flex items-center gap-2 bg-purple-500/20 backdrop-blur-sm border border-purple-500/30 rounded-full px-4 py-2">
              <Activity className="w-4 h-4 text-purple-400" />
              <span className="text-purple-300 text-sm font-medium">Real-time Monitoring</span>
            </div>
          </motion.div>
        </motion.div>

        {/* Tab Navigation */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="flex justify-center mb-12"
        >
          <div className="bg-gradient-to-r from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-2 border border-white/20 shadow-xl">
            <div className="flex gap-2">
              {[
                { id: 'stats', label: t('database.statistics'), icon: BarChart3 },
                { id: 'history', label: t('database.searchHistory'), icon: Clock },
                { id: 'metrics', label: t('database.systemMetrics'), icon: TrendingUp }
              ].map((tab) => {
                const Icon = tab.icon
                return (
                  <motion.button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className={`flex items-center gap-2 px-6 py-3 rounded-xl transition-all duration-300 font-medium ${
                      activeTab === tab.id
                        ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-lg'
                        : 'text-gray-300 hover:text-white hover:bg-white/10'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    {tab.label}
                  </motion.button>
                )
              })}
            </div>
          </div>
        </motion.div>

        {/* Loading State */}
        {loading && (
          <div className="text-center py-12">
            <div className="animate-spin w-8 h-8 border-2 border-indigo-400 border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-gray-300">Loading database information...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4 mb-8">
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {/* Statistics Tab */}
        {!loading && activeTab === 'stats' && stats && (
          <div className="space-y-8">
            {/* Overview Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
                whileHover={{ scale: 1.05, y: -5 }}
                className="group bg-gradient-to-br from-blue-500/10 to-cyan-500/10 backdrop-blur-sm rounded-2xl p-8 border border-blue-500/20 hover:border-blue-400/50 transition-all duration-500 shadow-xl"
              >
                <div className="flex items-center gap-4">
                  <motion.div 
                    whileHover={{ rotate: 360, scale: 1.1 }}
                    transition={{ duration: 0.6 }}
                    className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-xl flex items-center justify-center shadow-lg"
                  >
                    <Database className="w-8 h-8 text-blue-400" />
                  </motion.div>
                  <div>
                    <p className="text-3xl font-bold text-white group-hover:text-blue-300 transition-colors">{stats.total_exoplanets}</p>
                    <p className="text-gray-300 text-sm font-medium">Total Exoplanets</p>
                  </div>
                </div>
              </motion.div>

              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                whileHover={{ scale: 1.05, y: -5 }}
                className="group bg-gradient-to-br from-green-500/10 to-emerald-500/10 backdrop-blur-sm rounded-2xl p-8 border border-green-500/20 hover:border-green-400/50 transition-all duration-500 shadow-xl"
              >
                <div className="flex items-center gap-4">
                  <motion.div 
                    whileHover={{ rotate: 360, scale: 1.1 }}
                    transition={{ duration: 0.6 }}
                    className="w-16 h-16 bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded-xl flex items-center justify-center shadow-lg"
                  >
                    <CheckCircle className="w-8 h-8 text-green-400" />
                  </motion.div>
                  <div>
                    <p className="text-3xl font-bold text-white group-hover:text-green-300 transition-colors">{stats.confirmed_exoplanets}</p>
                    <p className="text-gray-300 text-sm font-medium">Confirmed</p>
                  </div>
                </div>
              </motion.div>

              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
                whileHover={{ scale: 1.05, y: -5 }}
                className="group bg-gradient-to-br from-purple-500/10 to-pink-500/10 backdrop-blur-sm rounded-2xl p-8 border border-purple-500/20 hover:border-purple-400/50 transition-all duration-500 shadow-xl"
              >
                <div className="flex items-center gap-4">
                  <motion.div 
                    whileHover={{ rotate: 360, scale: 1.1 }}
                    transition={{ duration: 0.6 }}
                    className="w-16 h-16 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl flex items-center justify-center shadow-lg"
                  >
                    <Search className="w-8 h-8 text-purple-400" />
                  </motion.div>
                  <div>
                    <p className="text-3xl font-bold text-white group-hover:text-purple-300 transition-colors">{stats.total_searches}</p>
                    <p className="text-gray-300 text-sm font-medium">Total Searches</p>
                  </div>
                </div>
              </motion.div>

              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
                whileHover={{ scale: 1.05, y: -5 }}
                className="group bg-gradient-to-br from-orange-500/10 to-red-500/10 backdrop-blur-sm rounded-2xl p-8 border border-orange-500/20 hover:border-orange-400/50 transition-all duration-500 shadow-xl"
              >
                <div className="flex items-center gap-4">
                  <motion.div 
                    whileHover={{ rotate: 360, scale: 1.1 }}
                    transition={{ duration: 0.6 }}
                    className="w-16 h-16 bg-gradient-to-br from-orange-500/20 to-red-500/20 rounded-xl flex items-center justify-center shadow-lg"
                  >
                    <Zap className="w-8 h-8 text-orange-400" />
                  </motion.div>
                  <div>
                    <p className="text-3xl font-bold text-white group-hover:text-orange-300 transition-colors">{stats.searches_last_24h}</p>
                    <p className="text-gray-300 text-sm font-medium">Last 24h</p>
                  </div>
                </div>
              </motion.div>
            </div>

            {/* Search Methods Chart */}
            <motion.div 
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.6 }}
              className="bg-gradient-to-br from-white/10 to-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/20 shadow-xl"
            >
              <h3 className="text-xl font-semibold text-white mb-6">Searches by Method</h3>
              <div className="space-y-4">
                {Object.entries(stats.searches_by_method).map(([method, count]) => {
                  const total = Object.values(stats.searches_by_method).reduce((a, b) => a + b, 0)
                  const percentage = total > 0 ? (count / total) * 100 : 0
                  
                  return (
                    <div key={method} className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-gray-300 capitalize">{method}</span>
                        <span className="text-white font-medium">{count} ({percentage.toFixed(1)}%)</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${method === 'bls' ? 'bg-blue-500' : 'bg-purple-500'}`}
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </motion.div>

            {/* Action Buttons */}
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.8 }}
              className="flex justify-center gap-4"
            >
              <motion.button
                onClick={fetchData}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="group flex items-center gap-3 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 text-white font-bold py-4 px-8 rounded-xl transition-all duration-300 shadow-lg hover:shadow-indigo-500/25"
              >
                <RefreshCw className="w-5 h-5 group-hover:rotate-180 transition-transform duration-500" />
                <span>Refresh Data</span>
              </motion.button>
              <motion.button
                onClick={cleanupOldData}
                whileHover={{ scale: 1.05, y: -2 }}
                whileTap={{ scale: 0.95 }}
                className="group flex items-center gap-3 bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-700 hover:to-pink-700 text-white font-bold py-4 px-8 rounded-xl transition-all duration-300 shadow-lg hover:shadow-red-500/25"
              >
                <Trash2 className="w-5 h-5 group-hover:scale-110 transition-transform duration-300" />
                <span>Cleanup Old Data</span>
              </motion.button>
            </motion.div>
          </div>
        )}

        {/* Search History Tab */}
        {!loading && activeTab === 'history' && (
          <div className="space-y-6">
            {/* Filter */}
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-4 border border-white/20">
              <div className="flex items-center gap-4">
                <label className="text-sm font-medium text-gray-300">Filter by method:</label>
                <select
                  value={selectedMethod}
                  onChange={(e) => setSelectedMethod(e.target.value)}
                  className="px-3 py-1 bg-white/10 border border-white/20 rounded text-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  <option value="">All Methods</option>
                  <option value="bls">BLS</option>
                  <option value="gpi">GPI</option>
                </select>
              </div>
            </div>

            {/* History List */}
            <div className="space-y-4">
              {searchHistory.map((item) => (
                <div key={item.id} className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white">{item.target_name}</h3>
                      <p className="text-gray-300 text-sm">
                        {item.catalog} • {item.mission} • {formatDate(item.created_at)}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`text-xs px-2 py-1 rounded border ${getMethodColor(item.method)}`}>
                        {item.method.toUpperCase()}
                      </span>
                      <span className={`text-xs px-2 py-1 rounded border ${
                        item.exoplanet_detected 
                          ? 'bg-green-500/20 text-green-300 border-green-500/30'
                          : 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30'
                      }`}>
                        {item.exoplanet_detected ? 'Detected' : 'No Detection'}
                      </span>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-gray-400">Confidence</p>
                      <p className="text-white font-medium">{(item.detection_confidence * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Processing Time</p>
                      <p className="text-white font-medium">{item.processing_time_ms.toFixed(1)}ms</p>
                    </div>
                    <div>
                      <p className="text-gray-400">Result Data</p>
                      <p className="text-white font-medium">
                        {typeof item.result_data === 'object' ? 'Available' : 'N/A'}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* System Metrics Tab */}
        {!loading && activeTab === 'metrics' && (
          <div className="space-y-6">
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
              <h3 className="text-xl font-semibold text-white mb-6">System Metrics (Last 24h)</h3>
              
              <div className="space-y-4">
                {metrics.map((metric) => (
                  <div key={metric.id} className="flex items-center justify-between p-4 bg-white/5 rounded-lg">
                    <div>
                      <p className="text-white font-medium">{metric.service_name}</p>
                      <p className="text-gray-300 text-sm">{metric.metric_name}</p>
                      <p className="text-gray-400 text-xs">{formatDate(metric.timestamp)}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-white font-bold text-lg">{metric.metric_value.toFixed(2)}</p>
                      {metric.metadata && typeof metric.metadata === 'object' && (
                        <p className="text-gray-400 text-xs">
                          {Object.keys(metric.metadata).length} metadata fields
                        </p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default DatabasePage
