import React, { useState, useEffect } from 'react'
import { Database, BarChart3, Clock, TrendingUp, Search, Trash2, RefreshCw } from 'lucide-react'
import { useTranslation } from 'react-i18next'

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

  const getMethodColor = (method: string) => {
    switch (method.toLowerCase()) {
      case 'bls': return 'bg-blue-500/20 text-blue-300 border-blue-500/30'
      case 'gpi': return 'bg-purple-500/20 text-purple-300 border-purple-500/30'
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30'
    }
  }

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full mb-4">
            <Database className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-white mb-2">
            {t('database.title')}
          </h1>
          <p className="text-xl text-gray-300">
            {t('database.subtitle')}
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-1 border border-white/20">
            <div className="flex gap-1">
              {[
                { id: 'stats', label: t('database.statistics'), icon: BarChart3 },
                { id: 'history', label: t('database.searchHistory'), icon: Clock },
                { id: 'metrics', label: t('database.systemMetrics'), icon: TrendingUp }
              ].map((tab) => {
                const Icon = tab.icon
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id as any)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-md transition-all duration-300 ${
                      activeTab === tab.id
                        ? 'bg-white/20 text-white'
                        : 'text-gray-300 hover:text-white hover:bg-white/10'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    {tab.label}
                  </button>
                )
              })}
            </div>
          </div>
        </div>

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
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center">
                    <Database className="w-6 h-6 text-blue-400" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white">{stats.total_exoplanets}</p>
                    <p className="text-gray-300 text-sm">Total Exoplanets</p>
                  </div>
                </div>
              </div>

              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-green-500/20 rounded-lg flex items-center justify-center">
                    <BarChart3 className="w-6 h-6 text-green-400" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white">{stats.confirmed_exoplanets}</p>
                    <p className="text-gray-300 text-sm">Confirmed</p>
                  </div>
                </div>
              </div>

              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-purple-500/20 rounded-lg flex items-center justify-center">
                    <Search className="w-6 h-6 text-purple-400" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white">{stats.total_searches}</p>
                    <p className="text-gray-300 text-sm">Total Searches</p>
                  </div>
                </div>
              </div>

              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 bg-orange-500/20 rounded-lg flex items-center justify-center">
                    <Clock className="w-6 h-6 text-orange-400" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-white">{stats.searches_last_24h}</p>
                    <p className="text-gray-300 text-sm">Last 24h</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Search Methods Chart */}
            <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 border border-white/20">
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
            </div>

            {/* Action Buttons */}
            <div className="flex justify-center gap-4">
              <button
                onClick={fetchData}
                className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                Refresh Data
              </button>
              <button
                onClick={cleanupOldData}
                className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                Cleanup Old Data
              </button>
            </div>
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
