import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Filter, 
  Search, 
  Download, 
  Eye, 
  Calendar,
  Target,
  TrendingUp,
  Brain,
  ChevronDown,
  SortAsc,
  SortDesc,
  RefreshCw
} from 'lucide-react'
import { FilterOptions, PaginationState } from '../types/ui'
import { SearchResult } from '../types/api'
import ApiService from '../services/api'
import { useAppStore } from '../store/useAppStore'
import LoadingSpinner from '../components/ui/LoadingSpinner'

// Mock data for demonstration
const mockResults: SearchResult[] = [
  {
    target_name: "TIC 441420236",
    analysis_timestamp: "2024-01-15T10:30:00Z",
    lightcurve_data: {
      time: [],
      flux: [],
      target_name: "TIC 441420236",
      mission: "TESS",
      sector: 15
    },
    bls_results: {
      best_period: 4.203,
      best_power: 0.85,
      best_duration: 0.12,
      best_t0: 2.1,
      snr: 12.4,
      depth: 0.01,
      depth_err: 0.001,
      significance: 0.95
    },
    candidates: [
      {
        period: 4.203,
        epoch: 2.1,
        duration: 0.12,
        depth: 0.01,
        snr: 12.4,
        significance: 0.95,
        is_planet_candidate: true,
        confidence: 0.89
      }
    ],
    ai_analysis: {
      is_transit: true,
      confidence: 0.89,
      confidence_level: 'HIGH',
      explanation: "Обнаружен четкий периодический сигнал",
      model_predictions: { cnn: 0.92, lstm: 0.87, transformer: 0.88, ensemble: 0.89 },
      uncertainty: 0.11
    },
    status: 'success'
  },
  {
    target_name: "KIC 8462852",
    analysis_timestamp: "2024-01-14T15:45:00Z",
    lightcurve_data: {
      time: [],
      flux: [],
      target_name: "KIC 8462852",
      mission: "Kepler",
      quarter: 8
    },
    bls_results: {
      best_period: 0.0,
      best_power: 0.12,
      best_duration: 0.0,
      best_t0: 0.0,
      snr: 3.2,
      depth: 0.0,
      depth_err: 0.0,
      significance: 0.15
    },
    candidates: [],
    ai_analysis: {
      is_transit: false,
      confidence: 0.23,
      confidence_level: 'LOW',
      explanation: "Сигнал не соответствует характеристикам транзита",
      model_predictions: { cnn: 0.25, lstm: 0.21, transformer: 0.23, ensemble: 0.23 },
      uncertainty: 0.77
    },
    status: 'success'
  },
  {
    target_name: "EPIC 249622103",
    analysis_timestamp: "2024-01-13T09:20:00Z",
    lightcurve_data: {
      time: [],
      flux: [],
      target_name: "EPIC 249622103",
      mission: "K2",
      quarter: 12
    },
    bls_results: {
      best_period: 7.856,
      best_power: 0.67,
      best_duration: 0.18,
      best_t0: 3.2,
      snr: 8.9,
      depth: 0.008,
      depth_err: 0.002,
      significance: 0.78
    },
    candidates: [
      {
        period: 7.856,
        epoch: 3.2,
        duration: 0.18,
        depth: 0.008,
        snr: 8.9,
        significance: 0.78,
        is_planet_candidate: true,
        confidence: 0.67
      }
    ],
    ai_analysis: {
      is_transit: true,
      confidence: 0.67,
      confidence_level: 'MEDIUM',
      explanation: "Умеренно уверенное обнаружение транзита",
      model_predictions: { cnn: 0.71, lstm: 0.63, transformer: 0.68, ensemble: 0.67 },
      uncertainty: 0.33
    },
    status: 'success'
  }
]

export default function ResultsPage() {
  const [results, setResults] = useState<SearchResult[]>([])
  const [filteredResults, setFilteredResults] = useState<SearchResult[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [showFilters, setShowFilters] = useState(false)
  const [sortBy, setSortBy] = useState<'date' | 'confidence' | 'target_name' | 'snr'>('date')
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc')
  const [isLoading, setIsLoading] = useState(true)
  const [totalResults, setTotalResults] = useState(0)
  const { addToast } = useAppStore()
  
  const [filters, setFilters] = useState<FilterOptions>({
    catalog: [],
    mission: [],
    dateRange: { start: null, end: null },
    confidenceRange: { min: 0, max: 100 },
    sortBy: 'date',
    sortOrder: 'desc'
  })

  // Load real results from API
  useEffect(() => {
    loadResults()
  }, [])

  const loadResults = async () => {
    try {
      setIsLoading(true)
      
      // Try to get results from API
      const response = await fetch('http://localhost:8000/api/results')
      if (response.ok) {
        const data = await response.json()
        setResults(data.results || [])
        setFilteredResults(data.results || [])
        setTotalResults(data.total || data.results?.length || 0)
      } else {
        // Fallback to store data
        const { searchResults } = useAppStore.getState()
        setResults(searchResults)
        setFilteredResults(searchResults)
        setTotalResults(searchResults.length)
      }
      
      setIsLoading(false)
    } catch (error) {
      console.error('Failed to load results:', error)
      
      // Fallback to store data
      const { searchResults } = useAppStore.getState()
      setResults(searchResults)
      setFilteredResults(searchResults)
      setTotalResults(searchResults.length)
      setIsLoading(false)
      
      addToast({
        type: 'warning',
        title: 'Ограниченные данные',
        message: 'Показаны только результаты текущей сессии'
      })
    }
  }

  const [pagination, setPagination] = useState<PaginationState>({
    page: 1,
    pageSize: 10,
    total: mockResults.length,
    hasNext: false,
    hasPrevious: false
  })

  const handleSearch = (query: string) => {
    setSearchQuery(query)
    const filtered = results.filter(result =>
      result.target_name.toLowerCase().includes(query.toLowerCase())
    )
    setFilteredResults(filtered)
  }

  const handleSort = (field: typeof sortBy) => {
    const newOrder = field === sortBy && sortOrder === 'asc' ? 'desc' : 'asc'
    setSortBy(field)
    setSortOrder(newOrder)
    
    const sorted = [...filteredResults].sort((a, b) => {
      let aVal: any, bVal: any
      
      switch (field) {
        case 'date':
          aVal = new Date(a.analysis_timestamp)
          bVal = new Date(b.analysis_timestamp)
          break
        case 'confidence':
          aVal = a.ai_analysis?.confidence || 0
          bVal = b.ai_analysis?.confidence || 0
          break
        case 'target_name':
          aVal = a.target_name
          bVal = b.target_name
          break
        case 'snr':
          aVal = a.bls_results.snr
          bVal = b.bls_results.snr
          break
        default:
          return 0
      }
      
      if (newOrder === 'asc') {
        return aVal > bVal ? 1 : -1
      } else {
        return aVal < bVal ? 1 : -1
      }
    })
    
    setFilteredResults(sorted)
  }

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.8) return 'bg-green-600/20 text-green-400 border-green-500/30'
    if (confidence >= 0.6) return 'bg-yellow-600/20 text-yellow-400 border-yellow-500/30'
    return 'bg-red-600/20 text-red-400 border-red-500/30'
  }

  const getStatusBadge = (hasTransit: boolean) => {
    return hasTransit 
      ? 'bg-cosmic-600/20 text-cosmic-400 border-cosmic-500/30'
      : 'bg-gray-600/20 text-gray-400 border-gray-500/30'
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" message="Загружаем результаты анализов..." />
      </div>
    )
  }

  return (
    <div className="min-h-screen py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl md:text-5xl font-bold text-gradient mb-4">
            Результаты анализа
          </h1>
          <p className="text-xl text-gray-300">
            История поиска экзопланет и детальные результаты анализа ({totalResults} анализов)
          </p>
        </motion.div>

        {/* Search and Filters */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card mb-8"
        >
          <div className="flex flex-col lg:flex-row gap-4">
            {/* Search */}
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Поиск по названию цели..."
                  value={searchQuery}
                  onChange={(e) => handleSearch(e.target.value)}
                  className="input-field w-full pl-10"
                />
              </div>
            </div>

            {/* Filter Toggle */}
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="btn-secondary px-4 py-2 flex items-center space-x-2"
            >
              <Filter className="h-4 w-4" />
              <span>Фильтры</span>
              <ChevronDown className={`h-4 w-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
            </button>

            {/* Refresh */}
            <button 
              onClick={loadResults}
              disabled={isLoading}
              className="btn-secondary px-4 py-2 flex items-center space-x-2"
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              <span>Обновить</span>
            </button>

            {/* Export */}
            <button className="btn-primary px-4 py-2 flex items-center space-x-2">
              <Download className="h-4 w-4" />
              <span>Экспорт</span>
            </button>
          </div>

          {/* Filters Panel */}
          {showFilters && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-6 pt-6 border-t border-space-600"
            >
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Миссия
                  </label>
                  <select className="input-field w-full">
                    <option value="">Все миссии</option>
                    <option value="TESS">TESS</option>
                    <option value="Kepler">Kepler</option>
                    <option value="K2">K2</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Статус
                  </label>
                  <select className="input-field w-full">
                    <option value="">Все результаты</option>
                    <option value="transit">Транзит обнаружен</option>
                    <option value="no_transit">Транзит не обнаружен</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Уверенность ИИ
                  </label>
                  <select className="input-field w-full">
                    <option value="">Любая</option>
                    <option value="high">Высокая (&gt;80%)</option>
                    <option value="medium">Средняя (60-80%)</option>
                    <option value="low">Низкая (&lt;60%)</option>
                  </select>
                </div>
              </div>
            </motion.div>
          )}
        </motion.div>

        {/* Results Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
        >
          <div className="card text-center py-4">
            <h3 className="text-2xl font-bold text-white mb-1">
              {filteredResults.length}
            </h3>
            <p className="text-gray-400 text-sm">Всего анализов</p>
          </div>

          <div className="card text-center py-4">
            <h3 className="text-2xl font-bold text-cosmic-400 mb-1">
              {filteredResults.filter(r => r.ai_analysis?.is_transit).length}
            </h3>
            <p className="text-gray-400 text-sm">Транзиты найдены</p>
          </div>

          <div className="card text-center py-4">
            <h3 className="text-2xl font-bold text-green-400 mb-1">
              {filteredResults.filter(r => (r.ai_analysis?.confidence || 0) > 0.8).length}
            </h3>
            <p className="text-gray-400 text-sm">Высокая уверенность</p>
          </div>

          <div className="card text-center py-4">
            <h3 className="text-2xl font-bold text-primary-400 mb-1">
              {filteredResults.reduce((sum, r) => sum + (r.candidates?.length || 0), 0)}
            </h3>
            <p className="text-gray-400 text-sm">Всего кандидатов</p>
          </div>
        </motion.div>

        {/* Results Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card overflow-hidden"
        >
          {/* Table Header */}
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-space-700/30 border-b border-space-600">
                <tr>
                  <th className="px-6 py-3 text-left">
                    <button
                      onClick={() => handleSort('target_name')}
                      className="flex items-center space-x-1 text-sm font-medium text-gray-300 hover:text-white"
                    >
                      <span>Цель</span>
                      {sortBy === 'target_name' && (
                        sortOrder === 'asc' ? <SortAsc className="h-4 w-4" /> : <SortDesc className="h-4 w-4" />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-3 text-left">
                    <button
                      onClick={() => handleSort('date')}
                      className="flex items-center space-x-1 text-sm font-medium text-gray-300 hover:text-white"
                    >
                      <span>Дата анализа</span>
                      {sortBy === 'date' && (
                        sortOrder === 'asc' ? <SortAsc className="h-4 w-4" /> : <SortDesc className="h-4 w-4" />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-3 text-left">Миссия</th>
                  <th className="px-6 py-3 text-left">
                    <button
                      onClick={() => handleSort('snr')}
                      className="flex items-center space-x-1 text-sm font-medium text-gray-300 hover:text-white"
                    >
                      <span>SNR</span>
                      {sortBy === 'snr' && (
                        sortOrder === 'asc' ? <SortAsc className="h-4 w-4" /> : <SortDesc className="h-4 w-4" />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-3 text-left">Кандидаты</th>
                  <th className="px-6 py-3 text-left">
                    <button
                      onClick={() => handleSort('confidence')}
                      className="flex items-center space-x-1 text-sm font-medium text-gray-300 hover:text-white"
                    >
                      <span>ИИ уверенность</span>
                      {sortBy === 'confidence' && (
                        sortOrder === 'asc' ? <SortAsc className="h-4 w-4" /> : <SortDesc className="h-4 w-4" />
                      )}
                    </button>
                  </th>
                  <th className="px-6 py-3 text-left">Статус</th>
                  <th className="px-6 py-3 text-right">Действия</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-space-600">
                {filteredResults.map((result, index) => (
                  <motion.tr
                    key={result.target_name + result.analysis_timestamp}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="hover:bg-space-700/20 transition-colors"
                  >
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        <Target className="h-4 w-4 text-gray-400" />
                        <span className="font-medium text-white">{result.target_name}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2 text-gray-300">
                        <Calendar className="h-4 w-4" />
                        <span className="text-sm">
                          {new Date(result.analysis_timestamp).toLocaleDateString('ru')}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="px-2 py-1 bg-primary-600/20 text-primary-300 rounded-full text-xs font-medium">
                        {result.lightcurve_data.mission}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-1">
                        <TrendingUp className="h-4 w-4 text-gray-400" />
                        <span className="text-white font-medium">
                          {result.bls_results.snr.toFixed(1)}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className="text-white font-medium">
                        {result.candidates?.length || 0}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        <Brain className="h-4 w-4 text-cosmic-400" />
                        <span className={`px-2 py-1 rounded-full text-xs font-medium border ${
                          getConfidenceBadge(result.ai_analysis?.confidence || 0)
                        }`}>
                          {result.ai_analysis?.confidence ? 
                            `${(result.ai_analysis.confidence * 100).toFixed(0)}%` : 'N/A'}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium border ${
                        getStatusBadge(result.ai_analysis?.is_transit || false)
                      }`}>
                        {result.ai_analysis?.is_transit ? 'Транзит' : 'Нет транзита'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <Link
                        to={`/analysis/${encodeURIComponent(result.target_name)}`}
                        className="inline-flex items-center space-x-1 text-primary-400 hover:text-primary-300 text-sm font-medium"
                      >
                        <Eye className="h-4 w-4" />
                        <span>Подробнее</span>
                      </Link>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {filteredResults.length > 10 && (
            <div className="px-6 py-4 border-t border-space-600 flex items-center justify-between">
              <div className="text-sm text-gray-400">
                Показано {Math.min(10, filteredResults.length)} из {filteredResults.length} результатов
              </div>
              <div className="flex items-center space-x-2">
                <button className="btn-secondary px-3 py-1 text-sm">
                  Предыдущая
                </button>
                <span className="text-gray-400 text-sm">1 из 1</span>
                <button className="btn-secondary px-3 py-1 text-sm">
                  Следующая
                </button>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
}
